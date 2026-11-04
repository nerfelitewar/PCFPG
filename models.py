"""v3 models: BoxNet (graph transformer -> room boxes) and RefineNet (boxes -> per-room segmentation) + helpers."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from common import S, T, M

N_IN = 2 + M + T      # boundary, front door, M slot-box masks, T type masks


def conv(i, o, s=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))


def node_feats(ntype, entr, share=None, known=None):
    """Per-room features: type one-hot, entrance flag, relative size (optional, masked when unknown), size-known flag."""
    valid = ntype >= 0
    oh = F.one_hot(ntype.clamp(min=0), T).float() * valid[..., None]
    if share is None:
        share = torch.zeros_like(entr); kn = torch.zeros_like(entr)
    else:
        kn = known.float()[:, None].expand_as(entr) * valid
        share = share * kn
    return torch.cat([oh, entr[..., None], (share * 8)[..., None], kn[..., None]], -1), valid


# ----------------------------- stage 1: BoxNet -----------------------------
class Layer(nn.Module):
    """Pre-LN transformer block with ADAPTIVE LAYER NORM (AdaLN-Zero style).
    A per-room conditioning vector c (type, entrance, relative size, size-known flag) produces shift/scale/gate for
    the self-attention (graph-biased), the cross-attention (boundary) and the FFN. So the room's target size directly
    rescales every sub-layer instead of being a single scalar mixed into the input token."""
    def __init__(s, d, h):
        super().__init__(); s.h = h; s.d = d
        s.sa = nn.MultiheadAttention(d, h, batch_first=True); s.ca = nn.MultiheadAttention(d, h, batch_first=True)
        s.n1, s.n2, s.n3 = [nn.LayerNorm(d, elementwise_affine=False) for _ in range(3)]
        s.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        s.eb = nn.Embedding(3, h)                      # 0 none / 1 adjacent / 2 door  -> one attention bias per head
        s.mod = nn.Sequential(nn.SiLU(), nn.Linear(d, 9 * d))
        nn.init.zeros_(s.mod[1].weight); nn.init.zeros_(s.mod[1].bias)
        with torch.no_grad(): s.mod[1].bias[2 * d:3 * d] = s.mod[1].bias[5 * d:6 * d] = s.mod[1].bias[8 * d:9 * d] = 1.0   # gates = 1 -> plain pre-LN at init

    def forward(s, x, mem, edge, valid, c):
        B, N, _ = x.shape
        sh1, sc1, g1, sh2, sc2, g2, sh3, sc3, g3 = s.mod(c).chunk(9, -1)
        bias = s.eb(edge).permute(0, 3, 1, 2) + (~valid)[:, None, None, :] * -1e4
        h = s.n1(x) * (1 + sc1) + sh1; x = x + g1 * s.sa(h, h, h, attn_mask=bias.reshape(B * s.h, N, N), need_weights=False)[0]
        h = s.n2(x) * (1 + sc2) + sh2; x = x + g2 * s.ca(h, mem, mem, need_weights=False)[0]
        h = s.n3(x) * (1 + sc3) + sh3; return x + g3 * s.ff(h)


class BoxNet(nn.Module):
    """cond (B,2,S,S) + room types/entrance/edges -> boxes (B,M,4) as (x0,y0,x1,y1) in [0,1].
    Boundary CNN (with coordinate channels) -> 16x16 tokens; room tokens attend to each other (graph-biased) and to the boundary.
    Deep supervision: every layer predicts boxes (shared head), so early layers learn useful layouts too."""
    def __init__(s, d=192, heads=6, layers=4, zdim=8):
        super().__init__(); s.zdim = zdim
        s.cnn = nn.Sequential(conv(4, 32), conv(32, 64, 2), conv(64, 128, 2), nn.Conv2d(128, d, 1))      # 64x64 -> 16x16 tokens
        g = (torch.arange(64) + 0.5) / 64
        s.register_buffer("coords", torch.stack(torch.meshgrid(g, g, indexing="ij")[::-1])[None])        # (1,2,64,64): x, y
        s.pos = nn.Parameter(torch.randn(1, 256, d) * 0.02)
        s.inp = nn.Linear(T + 3 + zdim, d)
        s.cmlp = nn.Sequential(nn.Linear(T + 3, d), nn.SiLU(), nn.Linear(d, d))       # per-room conditioning for AdaLN
        s.layers = nn.ModuleList([Layer(d, heads) for _ in range(layers)])
        s.nf = nn.LayerNorm(d); s.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 4))

    def decode(s, x):
        o = torch.sigmoid(s.head(s.nf(x)))
        cx, cy, w, h = o.unbind(-1); w, h = w.clamp(min=0.03), h.clamp(min=0.03)
        return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1).clamp(0, 1)

    def forward(s, cond, ntype, entr, edge, share=None, known=None, z=None, return_all=False):
        f, valid = node_feats(ntype, entr, share, known)
        if z is None: z = torch.randn(*ntype.shape, s.zdim, device=cond.device)   # breaks symmetry / gives variety
        c = F.avg_pool2d(cond.float(), 2)
        mem = s.cnn(torch.cat([c, s.coords.expand(len(c), -1, -1, -1)], 1)).flatten(2).transpose(1, 2) + s.pos
        x = s.inp(torch.cat([f, z], -1)); c = s.cmlp(f); outs = []
        for L in s.layers:
            x = L(x, mem, edge.long(), valid, c); outs.append(s.decode(x))
        return outs if return_all else outs[-1]


# ----------------------------- stage 2: RefineNet -----------------------------
class RefineNet(nn.Module):
    """U-Net: boundary + door + room-box masks -> logits over {outside, room slot 1..M} per pixel."""
    def __init__(s, base=32):
        super().__init__()
        ch = [base, base * 2, base * 4, base * 8, base * 16]
        blk = lambda i, o: nn.Sequential(conv(i, o), conv(o, o))
        s.enc = nn.ModuleList([blk(N_IN, ch[0])] + [blk(ch[i], ch[i + 1]) for i in range(4)])
        s.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in range(4)])
        s.dec = nn.ModuleList([blk(ch[i] * 2, ch[i]) for i in range(4)])
        s.head = nn.Conv2d(ch[0], M + 1, 1); s.pool = nn.MaxPool2d(2)

    def forward(s, x):
        sk = []
        for i, e in enumerate(s.enc):
            x = e(x if i == 0 else s.pool(x)); sk.append(x)
        x = sk.pop()
        for i in reversed(range(4)): x = s.dec[i](torch.cat([s.up[i](x), sk[i]], 1))
        return s.head(x)


# ----------------------------- helpers -----------------------------
def render_slots(boxes, ntype, soft=0.0):
    """boxes (B,M,4) normalized -> (B, M+T, S, S): one filled-box mask per room slot + one summed mask per room type.
    soft>0: box edges become sigmoid ramps of that width (fraction of the canvas) -> RefineNet stops trusting exact edges."""
    g = (torch.arange(S, device=boxes.device) + 0.5) / S
    x0, y0, x1, y1 = [boxes[..., i][..., None, None] for i in range(4)]
    if soft > 0:
        mx = torch.sigmoid((g.view(1, 1, 1, S) - x0) / soft) * torch.sigmoid((x1 - g.view(1, 1, 1, S)) / soft)
        my = torch.sigmoid((g.view(1, 1, S, 1) - y0) / soft) * torch.sigmoid((y1 - g.view(1, 1, S, 1)) / soft)
    else:
        mx = ((g.view(1, 1, 1, S) >= x0) & (g.view(1, 1, 1, S) <= x1)).float()
        my = ((g.view(1, 1, S, 1) >= y0) & (g.view(1, 1, S, 1) <= y1)).float()
    valid = (ntype >= 0)
    slots = (mx * my) * valid[..., None, None]
    oh = F.one_hot(ntype.clamp(min=0), T).float() * valid[..., None]
    return torch.cat([slots, torch.einsum("bmt,bmhw->bthw", oh, slots).clamp(max=1)], 1)


def refine_input(cond, boxes, ntype, soft=0.0):
    return torch.cat([cond.float(), render_slots(boxes, ntype, soft)], 1)


def edge_weights(inst, w=2.0):
    """1 + w on pixels within 1 px of a room border (from the GT instance map)."""
    b = torch.zeros_like(inst, dtype=torch.float32)
    d = (inst[:, 1:, :] != inst[:, :-1, :]).float(); b[:, 1:, :] += d; b[:, :-1, :] += d
    d = (inst[:, :, 1:] != inst[:, :, :-1]).float(); b[:, :, 1:] += d; b[:, :, :-1] += d
    return 1 + w * F.max_pool2d((b[:, None] > 0).float(), 3, 1, 1)[:, 0]


def aniso_tv(logits):
    """Anisotropic (L1) total variation of the class probabilities: |dP/dx| + |dP/dy|. Costs a diagonal edge sqrt(2)x more than an
    axis-aligned one of the same length, so it pushes room borders to be straight and axis-parallel."""
    p = logits.float().softmax(1)
    return (p[:, :, 1:, :] - p[:, :, :-1, :]).abs().mean() + (p[:, :, :, 1:] - p[:, :, :, :-1]).abs().mean()


def dihedral(cond, boxes, inst=None, k=0, flip=False):
    """Random-rotation/flip augmentation applied identically to images and box coordinates."""
    b = boxes
    if flip:
        cond = cond.flip(3); inst = inst.flip(2) if inst is not None else None
        b = torch.stack([1 - b[..., 2], b[..., 1], 1 - b[..., 0], b[..., 3]], -1)
    for _ in range(k % 4):
        cond = torch.rot90(cond, 1, (2, 3)); inst = torch.rot90(inst, 1, (1, 2)) if inst is not None else None
        b = torch.stack([b[..., 1], 1 - b[..., 2], b[..., 3], 1 - b[..., 0]], -1)
    return cond, b, inst


@torch.no_grad()
def align_perm(pred, gt, sig):
    """perm[b,i] = GT slot matched to predicted slot i (identity except inside groups of indistinguishable rooms)."""
    p = pred.detach().cpu().numpy(); g = gt.cpu().numpy(); sg = sig.cpu().numpy()
    perm = np.tile(np.arange(p.shape[1]), (len(p), 1))
    for b in range(len(p)):
        vals, cnt = np.unique(sg[b][sg[b] >= 0], return_counts=True)
        for v in vals[cnt > 1]:
            idx = np.where(sg[b] == v)[0]
            r, c = linear_sum_assignment(np.abs(p[b][idx][:, None, :] - g[b][idx][None, :, :]).sum(-1))
            perm[b, idx[r]] = idx[c]
    return torch.as_tensor(perm, device=pred.device)


def align_gt(pred, gt, sig):
    """Permute GT boxes among rooms the model cannot tell apart (same Weisfeiler-Lehman signature)."""
    return torch.gather(gt, 1, align_perm(pred, gt, sig)[..., None].expand(-1, -1, 4))


def relabel_inst(inst, perm):
    """Relabel an instance map (B,H,W) so that predicted slot i carries the pixels of GT slot perm[i]."""
    B, Mm = perm.shape
    lut = torch.zeros(B, Mm + 1, dtype=torch.long, device=inst.device)
    lut.scatter_(1, perm + 1, torch.arange(1, Mm + 1, device=inst.device)[None].expand(B, -1))
    return lut.gather(1, inst.flatten(1)).view_as(inst)


def noisy_share(share, valid, sigma=0.2):
    """Training-time size input: jittered (so approximate sizes at inference are fine) and renormalised."""
    s = share * torch.exp(sigma * torch.randn_like(share)) * valid
    return s / s.sum(1, keepdim=True).clamp(min=1e-6)


def group_sig(sig, share, known):
    """If sizes are given, rooms with different sizes are distinguishable -> refine the matching groups."""
    return torch.where((sig >= 0) & known[:, None], sig * 1000 + (share * 60).round().long(), sig)


def box_loss(pred, gt, valid, reduce=True):
    """5*L1 + 2*(1-GIoU) on valid rooms. reduce=True -> (scalar loss, mean IoU float); reduce=False -> per-plan (B,) tensors."""
    l1 = (pred - gt).abs().sum(-1)
    ix = (torch.min(pred[..., 2], gt[..., 2]) - torch.max(pred[..., 0], gt[..., 0])).clamp(min=0)
    iy = (torch.min(pred[..., 3], gt[..., 3]) - torch.max(pred[..., 1], gt[..., 1])).clamp(min=0)
    inter = ix * iy
    ap = (pred[..., 2] - pred[..., 0]) * (pred[..., 3] - pred[..., 1]); ag = (gt[..., 2] - gt[..., 0]) * (gt[..., 3] - gt[..., 1])
    union = (ap + ag - inter).clamp(min=1e-6); iou = inter / union
    cw = torch.max(pred[..., 2], gt[..., 2]) - torch.min(pred[..., 0], gt[..., 0])
    ch = torch.max(pred[..., 3], gt[..., 3]) - torch.min(pred[..., 1], gt[..., 1])
    carea = (cw * ch).clamp(min=1e-6); giou = iou - (carea - union) / carea
    per = 5 * l1 + 2 * (1 - giou); v = valid.float(); cnt = v.sum(1).clamp(min=1)
    if reduce: return per[valid].mean(), iou[valid].mean().item()
    return (per * v).sum(1) / cnt, (iou * v).sum(1) / cnt


def adjacency_loss(pred, edge, valid, reduce=True):
    """Rooms joined by a door/adjacency edge must have touching boxes (gap ~ one wall thickness)."""
    x0, y0, x1, y1 = pred.unbind(-1)
    dx = torch.max(x0[:, :, None] - x1[:, None, :], x0[:, None, :] - x1[:, :, None]).clamp(min=0)
    dy = torch.max(y0[:, :, None] - y1[:, None, :], y0[:, None, :] - y1[:, :, None]).clamp(min=0)
    gap = (torch.sqrt(dx ** 2 + dy ** 2 + 1e-8) - 0.012).clamp(min=0)
    m = ((edge >= 1) & valid[:, :, None] & valid[:, None, :]).float()
    if reduce: return (gap * m).sum() / m.sum().clamp(min=1)
    return (gap * m).sum((1, 2)) / m.sum((1, 2)).clamp(min=1)


def pairwise_overlap_loss(pred, valid, tau=0.3, reduce=True):
    """Inter-room box repulsion:  L = sum_{i<j} relu( |B_i ∩ B_j| / min(|B_i|,|B_j|) - tau ).
    Hinge with tolerance tau because REAL boxes overlap (L-shaped rooms have overlapping bounding boxes; ResPlan GT has mean
    excess occupancy 0.25) — a raw sum of IoUs would pull predictions away from the ground truth."""
    x0, y0, x1, y1 = pred.unbind(-1)
    iw = (torch.min(x1[:, :, None], x1[:, None, :]) - torch.max(x0[:, :, None], x0[:, None, :])).clamp(min=0)
    ih = (torch.min(y1[:, :, None], y1[:, None, :]) - torch.max(y0[:, :, None], y0[:, None, :])).clamp(min=0)
    a = (x1 - x0) * (y1 - y0); r = iw * ih / torch.min(a[:, :, None], a[:, None, :]).clamp(min=1e-6)
    Mm = pred.shape[1]; up = torch.triu(torch.ones(Mm, Mm, device=pred.device), 1)[None]
    m = valid[:, :, None].float() * valid[:, None, :].float() * up
    if reduce: return (F.relu(r - tau) * m).sum() / m.sum().clamp(min=1)
    return (F.relu(r - tau) * m).sum((1, 2)) / m.sum((1, 2)).clamp(min=1)


def soft_occupancy(pred, res=32, temp=0.012):
    """Differentiable rasterisation of boxes: (B,M,res,res) with sigmoid edges."""
    g = (torch.arange(res, device=pred.device) + 0.5) / res
    x0, y0, x1, y1 = [pred[..., i][..., None] for i in range(4)]
    mx = torch.sigmoid((g - x0) / temp) * torch.sigmoid((x1 - g) / temp)
    my = torch.sigmoid((g - y0) / temp) * torch.sigmoid((y1 - g) / temp)
    return my[..., :, None] * mx[..., None, :]


def _tiling(pred, ntype, footprint, tol, res):
    """Layout prior on the SUM of soft boxes O(p): rooms must cover the footprint, stay inside it (balconies excepted)
    and not stack more than (1+tol)-fold.  L = mean_fp relu(1-O) + mean_out O_nb + mean_fp relu(O-1-tol)."""
    valid = (ntype >= 0); occ_all = soft_occupancy(pred, res) * valid[..., None, None]
    nb = valid & (ntype != 4); occ_nb = soft_occupancy(pred, res) * nb[..., None, None]
    O, Onb = occ_all.sum(1), occ_nb.sum(1)
    fp = F.adaptive_avg_pool2d(footprint.float()[:, None], res)[:, 0]; out = 1 - fp
    cov = (F.relu(1 - O) * fp).sum((1, 2)) / fp.sum((1, 2)).clamp(min=1)
    outside = (Onb * out).sum((1, 2)) / out.sum((1, 2)).clamp(min=1)
    over = (F.relu(O - 1 - tol) * fp).sum((1, 2)) / fp.sum((1, 2)).clamp(min=1)
    return cov + outside + over


def tiling_loss(pred, ntype, footprint, ref=None, tol=0.5, res=32, reduce=True):
    """GT-anchored hinge: relu(T(pred) - T(ref)). Real layouts already overlap/leave gaps (T(GT)~0.26), so only layouts that tile
    WORSE than the ground truth are penalised -> zero gradient at the correct answer, no bias."""
    l = _tiling(pred, ntype, footprint, tol, res)
    if ref is not None: l = F.relu(l - _tiling(ref, ntype, footprint, tol, res).detach())
    return l.mean() if reduce else l
