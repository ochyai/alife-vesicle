"""
ALife v7: Transformer Vesicle Artificial Life System
====================================================
Transformer-brained cells communicate via vesicle diffusion.
Genome-modulated attention, relational features, nonlinear vesicle exchange,
nutrient depletion/regeneration, aging, crowding, speciation pressure,
predation, sexual recombination, transparent lens visuals, motion trails.

Controls: SPC:pause  V:vesicle  D:panel  R:reset  +/-:speed  Z/X:zoom
          S:stats  F5:record  F12:screenshot  F11:fullscreen  click:nutrient  ESC:quit
"""

import math, random, os, time, sys
import pygame, torch, torch.nn as nn, torch.nn.functional as F
try:
    import tracemalloc as _tm
    _tm.start()
    _TM = True
except Exception:
    _TM = False

# ━━ Device ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

if "--gpu" in sys.argv:
    DEV = _pick_device()
else:
    DEV = torch.device("cpu")  # CPU faster for N<1000; use --gpu for large sims
print(f"[ALife] device: {DEV}")

# ━━ Layout ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TW, TH = 1440, 800
PW = 360; SW = TW - PW  # sim 1080 × 800
FPS = 60

# ━━ Dims ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DIM = 32; GDIM = 16; NH = 4; K_N = 6
VES_TOKENS = 3  # nearby vesicle tokens in attention
SL = 1 + K_N + VES_TOKENS  # total sequence length: self + neighbors + vesicle tokens
ADIM = 2 + 1 + 1 + DIM  # dx,dy,emit,alpha,vesicle_content

# ━━ Sim ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MAX_C = 700; MAX_V = 3000; INIT_N = 300; MAX_NUTS = 64
INIT_E = 120.; DIV_E = 150.; MUT = 0.04
VSPD = 1.3; VLIFE = 150; VCOST = 0.12
ARAD = 26.; DECAY = 0.005; MCOST = 0.01
NNUTS = 8; NRAD = 160; NGAIN = 0.6
PASSIVE_REGEN = 0.004  # background sustain everywhere

# ━━ Visual ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BG = (3, 3, 12); PBG = (8, 8, 18); FADE = 12

SWt = torch.tensor([float(SW), float(TH)], device=DEV)


# ━━ Color ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def hsv(h, s, v):
    """h:0-360 s:0-1 v:0-1 → (R,G,B) 0-255."""
    h60 = h / 60.0
    i = int(h60) % 6
    f = h60 - int(h60)
    p, q, t = v*(1-s), v*(1-s*f), v*(1-s*(1-f))
    r, g, b = [(v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q)][i]
    return int(r*255), int(g*255), int(b*255)

def pheno_rgb(p, bri=0.7):
    """Map phenotype vector → vivid HSV color. Full 360° hue, high saturation."""
    v = torch.tanh(p[:4]).cpu().tolist() if len(p) >= 4 else [0]*4
    hue = (v[0] * 130 + v[1] * 90 + 180) % 360
    sat = 0.75 + 0.25 * abs(v[2])                    # very saturated always
    return hsv(hue, min(1., sat), min(1., max(.15, bri)))

def pheno_rgb_np(p4, bri=0.7):
    """Map 4-element numpy array → vivid HSV color (no torch)."""
    v = [max(-1., min(1., math.tanh(p4[k]))) for k in range(min(4, len(p4)))]
    while len(v) < 4: v.append(0.)
    hue = (v[0] * 130 + v[1] * 90 + 180) % 360
    sat = 0.75 + 0.25 * abs(v[2])
    return hsv(hue, min(1., sat), min(1., max(.15, bri)))

def dim_col(v):
    """Single value → vivid diverging color for DNA barcode. High contrast."""
    v = max(-1., min(1., v))
    if v >= 0:
        return hsv(20 + 40*v, .85, .35 + .65*v)     # dark amber → bright gold
    a = -v
    return hsv(190 + 50*a, .85, .35 + .65*a)         # dark teal → bright cyan

def wrapped_dist(a, b):
    """Wrapped Euclidean distance. a:(N,2) b:(M,2) → (N,M)."""
    d = (a.unsqueeze(1) - b.unsqueeze(0)).abs()          # (N,M,2)
    d = torch.min(d, SWt - d)
    return d.norm(dim=-1)                                  # (N,M)

def wrapped_diff(a, b):
    """Wrapped difference vectors. a:(N,2) b:(M,2) → (N,M,2). Signed."""
    d = a.unsqueeze(1) - b.unsqueeze(0)                    # (N,M,2)
    # wrap: if |d| > half-world, subtract world size with correct sign
    half = SWt / 2
    d = d - SWt * (d > half).float() + SWt * (d < -half).float()
    return d


# ━━ Brain: single multi-head attention layer (upgraded) ━━━━
class Brain(nn.Module):
    def __init__(self):
        super().__init__()
        self.gp = nn.Linear(GDIM, DIM)
        self.ln = nn.LayerNorm(DIM)
        hd = DIM // NH
        self.hd = hd
        self.Wq = nn.Linear(DIM, DIM, bias=False)
        self.Wk = nn.Linear(DIM, DIM, bias=False)
        self.Wv = nn.Linear(DIM, DIM, bias=False)
        self.proj = nn.Linear(DIM, DIM)
        self.ln2 = nn.LayerNorm(DIM)
        self.ff = nn.Sequential(nn.Linear(DIM, DIM*2), nn.GELU(), nn.Linear(DIM*2, DIM))
        self.head = nn.Linear(DIM, ADIM)
        # Change 6: Relational feature projection
        self.rel_proj = nn.Linear(5, DIM)
        # Change 7: Genome-based per-head Q/K scaling
        self.genome_qk = nn.Linear(GDIM, NH * 2)

    @torch.no_grad()
    def forward(self, s, g, ns, ves_tokens=None, mask=None):
        """
        s:  (N, DIM)          — self state
        g:  (N, GDIM)         — genome
        ns: (N, K, DIM)       — neighbor state tokens (with relational features added)
        ves_tokens: (N, VES_TOKENS, DIM) or None — nearby vesicle content tokens
        mask: (N, SL) bool    — True = masked (padded) position
        → (N, ADIM)
        """
        x = self.ln(s + self.gp(g))
        N = x.shape[0]

        # Build full token sequence: [self_token, neighbor_tokens, vesicle_tokens]
        self_tok = x.unsqueeze(1)  # (N, 1, DIM)
        tokens = [self_tok, ns]
        if ves_tokens is not None:
            tokens.append(ves_tokens)
        seq = torch.cat(tokens, dim=1)  # (N, SL, DIM)
        _, SL_actual, D = seq.shape

        # Q from self token only; K, V from full sequence
        Q = self.Wq(x).view(N, 1, NH, self.hd).permute(0, 2, 1, 3)    # (N, NH, 1, hd)
        Kk = self.Wk(seq).view(N, SL_actual, NH, self.hd).permute(0, 2, 1, 3)  # (N, NH, SL, hd)
        Vv = self.Wv(seq).view(N, SL_actual, NH, self.hd).permute(0, 2, 1, 3)  # (N, NH, SL, hd)

        # Change 7: Genome Q/K scaling
        scales = torch.sigmoid(self.genome_qk(g))  # (N, NH*2)
        q_scale = scales[:, :NH].view(N, NH, 1, 1)      # (N, NH, 1, 1)
        k_scale = scales[:, NH:].view(N, NH, 1, 1)      # (N, NH, 1, 1)
        Q = Q * q_scale
        Kk = Kk * k_scale

        attn = (Q @ Kk.transpose(-2, -1)) * (self.hd ** -.5)  # (N, NH, 1, SL)

        # Change 5: Attention masking for padded positions
        if mask is not None:
            # mask shape: (N, SL_actual), True = masked
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)  # (N, 1, 1, SL)
            attn = attn.masked_fill(mask_expanded, -1e9)

        ctx = (attn.softmax(-1) @ Vv).permute(0, 2, 1, 3).reshape(N, D)
        ctx = self.proj(ctx)
        x = x + ctx
        x = x + self.ff(self.ln2(x))
        return self.head(x)


# ━━ Nutrient ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class Nut:
    __slots__ = ['x','y','sig','radius']
    def __init__(self, x, y):
        self.x, self.y, self.sig, self.radius = x, y, torch.randn(DIM, device=DEV), NRAD
    @property
    def color(self): return pheno_rgb(self.sig, .45)
    @property
    def bright(self): return pheno_rgb(self.sig, .8)


# ━━ World (fully tensorized, upgraded) ━━━━━━━━━━━━━━━━━━━━━
class World:
    def __init__(self):
        self.brain = Brain().to(DEV)
        # Cell pool
        self.cpos = torch.zeros(MAX_C, 2, device=DEV)
        self.cvel = torch.zeros(MAX_C, 2, device=DEV)
        self.cstate = torch.zeros(MAX_C, DIM, device=DEV)
        self.cgenome = torch.zeros(MAX_C, GDIM, device=DEV)
        self.cenergy = torch.zeros(MAX_C, device=DEV)
        self.cage = torch.zeros(MAX_C, dtype=torch.long, device=DEV)
        self.cgen = torch.zeros(MAX_C, dtype=torch.long, device=DEV)
        self.calive = torch.zeros(MAX_C, dtype=torch.bool, device=DEV)
        self.cclust = torch.zeros(MAX_C, dtype=torch.long, device=DEV)
        self.cflash = torch.zeros(MAX_C, device=DEV)
        # Previous positions for motion trail (3 history frames)
        self.cprev1 = torch.zeros(MAX_C, 2, device=DEV)
        self.cprev2 = torch.zeros(MAX_C, 2, device=DEV)
        self.cprev3 = torch.zeros(MAX_C, 2, device=DEV)
        # Change 8: Recent absorption counter (exponential decay)
        self.crecent = torch.zeros(MAX_C, device=DEV)
        # Vesicle pool
        self.vpos = torch.zeros(MAX_V, 2, device=DEV)
        self.vprev = torch.zeros(MAX_V, 2, device=DEV)  # previous vesicle position for visual use
        self.vvel = torch.zeros(MAX_V, 2, device=DEV)
        self.vcont = torch.zeros(MAX_V, DIM, device=DEV)
        self.vlife = torch.zeros(MAX_V, device=DEV)
        self.valive = torch.zeros(MAX_V, dtype=torch.bool, device=DEV)
        # Change 9: Nutrient energy
        self.nut_energy = torch.full((MAX_NUTS,), 100.0, device=DEV)
        # Change 17: Cached nutrient tensors
        self.nut_pos = torch.zeros(0, 2, device=DEV)
        self.nut_sig = torch.zeros(0, DIM, device=DEV)
        # Meta
        self.nuts = []
        self.t = 0; self.births = 0; self.deaths = 0
        self.ves_on = True
        self.div_hist = []
        # Ablation flags (used by experiment.py)
        self._disable_sexual = False
        self._disable_predation = False
        self._disable_speciation = False
        self._disable_aging = False
        self._init()

    def _rebuild_nut_cache(self):
        """Change 17: Rebuild cached nutrient position/signature tensors."""
        if self.nuts:
            self.nut_pos = torch.tensor([[n.x, n.y] for n in self.nuts], device=DEV)
            self.nut_sig = torch.stack([n.sig for n in self.nuts])  # already on DEV
        else:
            self.nut_pos = torch.zeros(0, 2, device=DEV)
            self.nut_sig = torch.zeros(0, DIM, device=DEV)

    def add_nutrient(self, x, y):
        """Add a nutrient and rebuild cache."""
        self.nuts.append(Nut(x, y))
        ni = len(self.nuts) - 1
        if ni < MAX_NUTS:
            self.nut_energy[ni] = 100.0
        self._rebuild_nut_cache()

    def _init(self):
        self.calive[:] = False; self.valive[:] = False
        self.crecent[:] = 0
        self.t = self.births = self.deaths = 0
        self.div_hist = []
        self.nuts = [Nut(random.uniform(80,SW-80), random.uniform(80,TH-80)) for _ in range(NNUTS)]
        self.nut_energy[:MAX_NUTS] = 100.0
        self._rebuild_nut_cache()
        self._spawn(INIT_N, near_nuts=True)

    def _spawn(self, n, near_nuts=False):
        dead = (~self.calive).nonzero(as_tuple=True)[0][:n]
        m = len(dead)
        if m == 0: return
        self.calive[dead] = True
        if near_nuts and self.nuts:
            # Spawn near random nutrient sources (Gaussian scatter)
            nut_idx = torch.randint(len(self.nuts), (m,), device=DEV)
            centers = torch.tensor([[self.nuts[i].x, self.nuts[i].y] for i in nut_idx.cpu()], device=DEV)
            self.cpos[dead] = (centers + torch.randn(m, 2, device=DEV) * 60) % SWt
        else:
            self.cpos[dead] = torch.rand(m, 2, device=DEV) * SWt
        self.cvel[dead] = 0
        self.cstate[dead] = torch.randn(m, DIM, device=DEV) * .1
        self.cgenome[dead] = torch.randn(m, GDIM, device=DEV) * .5
        self.cenergy[dead] = INIT_E
        self.cage[dead] = 0; self.cgen[dead] = 0; self.cflash[dead] = 0
        self.cprev1[dead] = self.cpos[dead].clone()
        self.cprev2[dead] = self.cpos[dead].clone()
        self.cprev3[dead] = self.cpos[dead].clone()
        self.crecent[dead] = 0

    def step(self):
        self.t += 1
        idx = self.calive.nonzero(as_tuple=True)[0]
        N = len(idx)
        if N == 0: self._spawn(20); return

        # ── Decay recent absorption counter (exponential) ──
        self.crecent[idx] *= 0.95

        # ── Vesicle physics (vectorized) ──
        va = self.valive
        if va.any():
            self.vprev[va] = self.vpos[va].clone()  # store previous position
            self.vpos[va] += self.vvel[va] + torch.randn(va.sum(), 2, device=DEV) * .2
            self.vpos[va] %= SWt
            self.vvel[va] *= .994
            self.vlife[va] -= 1
            self.valive &= (self.vlife > 0)

        # ── Change 1 (Bug Fix): Gate BOTH absorption AND emission on ves_on ──

        # ── Change 2 (Bug Fix): Vesicle absorption — nearest-cell-per-vesicle ──
        if self.ves_on:
            va_idx = self.valive.nonzero(as_tuple=True)[0]
            if len(va_idx) > 0 and N > 0:
                # For each alive vesicle, find nearest cell
                vc_d = wrapped_dist(self.vpos[va_idx], self.cpos[idx])  # (M, N)
                min_d, min_ci = vc_d.min(dim=1)  # (M,) — nearest cell for each vesicle
                absorb = min_d < ARAD  # (M,) bool — which vesicles are absorbed
                if absorb.any():
                    a_ves = va_idx[absorb]
                    a_cell_local = min_ci[absorb]  # indices into idx
                    a_cells = idx[a_cell_local]

                    # Change 8: Nonlinear vesicle exchange
                    content = self.vcont[a_ves]
                    state = self.cstate[a_cells]
                    recent = self.crecent[a_cells]

                    # Write mask: only write to active channels
                    write_mask = (content.abs() > 0.3).float()
                    # Blend factor: boost if recent absorptions > 3
                    blend = torch.where(recent > 3, torch.tensor(0.5, device=DEV), torch.tensor(0.3, device=DEV))
                    blend = blend.unsqueeze(1)  # (n_abs, 1)
                    self.cstate[a_cells] = state * (1 - blend * write_mask) + content * blend * write_mask

                    # Track absorptions
                    self.crecent[a_cells] += 1.0
                    self.cflash[a_cells] = 5.
                    self.valive[a_ves] = False

        # ── Neighbor finding ──
        p = self.cpos[idx]                             # (N, 2)
        s = self.cstate[idx]                           # (N, DIM)
        g = self.cgenome[idx]                          # (N, GDIM)
        dd = wrapped_dist(p, p)                         # (N, N)
        dd.fill_diagonal_(1e9)
        k = min(K_N, N - 1)

        # Change 5: Build attention mask — True = masked position
        # Sequence: [self(1), neighbors(K_N), vesicle_tokens(VES_TOKENS)]
        attn_mask = torch.zeros(N, SL, dtype=torch.bool, device=DEV)
        # Self token is never masked (position 0)

        if k == 0:
            ns = torch.zeros(N, K_N, DIM, device=DEV)
            attn_mask[:, 1:1+K_N] = True  # all neighbor slots masked
        else:
            _, nidx = dd.topk(k, largest=False)         # (N, k)
            ns = s[nidx]                                # (N, k, DIM)

            # Change 6: Relational features for neighbors
            # Compute relative positions, distances, relative velocities
            vel = self.cvel[idx]  # (N, 2)
            neigh_pos = p[nidx]    # (N, k, 2)
            neigh_vel = vel[nidx]  # (N, k, 2)

            # Wrapped relative position
            rel_pos = wrapped_diff(p, p)  # (N, N, 2)
            # Gather neighbor relative positions
            nidx_exp = nidx.unsqueeze(-1).expand(-1, -1, 2)
            rel_xy = torch.gather(rel_pos, 1, nidx_exp)  # (N, k, 2)
            # Distances
            neigh_d = torch.gather(dd, 1, nidx)  # (N, k)
            # Relative velocities
            rel_vel = vel.unsqueeze(1) - neigh_vel  # (N, k, 2)
            # Normalized relational features: [rel_x/100, rel_y/100, dist/200, rel_vx/5, rel_vy/5]
            rel_feats = torch.cat([
                rel_xy[:, :, 0:1] / 100.0,
                rel_xy[:, :, 1:2] / 100.0,
                neigh_d.unsqueeze(-1) / 200.0,
                rel_vel[:, :, 0:1] / 5.0,
                rel_vel[:, :, 1:2] / 5.0,
            ], dim=-1)  # (N, k, 5)

            # Project and add to neighbor tokens
            rel_embed = self.brain.rel_proj(rel_feats)  # (N, k, DIM)
            ns = ns + rel_embed

            if k < K_N:
                ns = F.pad(ns, (0, 0, 0, K_N - k))
                # Mask padded neighbor positions
                attn_mask[:, 1+k:1+K_N] = True

        # ── Change 10: Vesicle tokens — nearby vesicles as attention tokens ──
        ves_tokens = torch.zeros(N, VES_TOKENS, DIM, device=DEV)
        attn_mask[:, 1+K_N:] = True  # default: all vesicle token slots masked

        va_idx = self.valive.nonzero(as_tuple=True)[0]
        if len(va_idx) > 0 and N > 0:
            ves_sense_rad = ARAD * 2  # 52px
            cv_d = wrapped_dist(p, self.vpos[va_idx])  # (N, M)
            # For each cell, find up to 3 nearest vesicles within range
            n_ves = len(va_idx)
            max_k_ves = min(VES_TOKENS, n_ves)
            if max_k_ves > 0:
                topk_d, topk_vi = cv_d.topk(max_k_ves, largest=False, dim=1)  # (N, max_k_ves)
                in_range = topk_d < ves_sense_rad  # (N, max_k_ves)
                # Gather vesicle contents
                ves_contents = self.vcont[va_idx[topk_vi]]  # (N, max_k_ves, DIM)
                # Write to ves_tokens and unmask where in range
                for vi in range(max_k_ves):
                    cell_mask = in_range[:, vi]  # (N,) bool
                    if cell_mask.any():
                        ves_tokens[cell_mask, vi] = ves_contents[cell_mask, vi]
                        attn_mask[cell_mask, 1 + K_N + vi] = False  # unmask

        # ── Brain inference (batched, single forward) ──
        act = self.brain(s, g, ns, ves_tokens=ves_tokens, mask=attn_mask)  # (N, ADIM)

        dxy = torch.tanh(act[:, :2]) * 2.5
        emit_p = torch.sigmoid(act[:, 2])
        alpha = torch.sigmoid(act[:, 3]) * .25
        vc_out = act[:, 4:]                             # (N, DIM)

        # ── Movement ──
        # Shift previous positions for motion trails
        self.cprev3[idx] = self.cprev2[idx].clone()
        self.cprev2[idx] = self.cprev1[idx].clone()
        self.cprev1[idx] = self.cpos[idx].clone()
        new_vel = dxy * .6 + self.cvel[idx] * .4
        self.cpos[idx] = (p + new_vel) % SWt
        self.cvel[idx] = new_vel

        # ── State update ──
        # Change 3 (Bug Fix): Do state update BEFORE nutrient interaction
        self.cstate[idx] = (1 - alpha.unsqueeze(1)) * s + alpha.unsqueeze(1) * vc_out

        # ── Energy ──
        speed = new_vel.norm(dim=1)
        self.cenergy[idx] += PASSIVE_REGEN - DECAY - MCOST * speed
        self.cage[idx] += 1
        self.cflash[idx] = (self.cflash[idx] - 1).clamp(min=0)

        # ── Change 4: Aging pressure (faster — threshold 1200, doubled rate) ──
        if not self._disable_aging:
            ages = self.cage[idx].float()  # (N,)
            extra_drain = ((ages - 1200).clamp(min=0) / 3000.0) * 0.08
            self.cenergy[idx] -= extra_drain

        # ── Change 4: Crowding penalty ──
        # dd already computed with diagonal = 1e9
        if N > 1:
            nearest_d, _ = dd.min(dim=1)  # (N,)
            crowded = nearest_d < 15.0
            if crowded.any():
                self.cenergy[idx[crowded]] -= 0.005

        # ── Speciation pressure & Predation (genome-based interactions) ──
        if N > 1 and k > 0:
            # Genome cosine similarity between each cell and its k nearest neighbors
            g_norm = F.normalize(g, dim=1)                         # (N, GDIM)
            neigh_g = g[nidx]                                      # (N, k, GDIM)
            neigh_g_norm = F.normalize(neigh_g, dim=2)             # (N, k, GDIM)
            gsim = (g_norm.unsqueeze(1) * neigh_g_norm).sum(dim=2) # (N, k) cosine sim

            # Speciation: if avg similarity to neighbors > 0.8 → small energy drain
            if not self._disable_speciation:
                avg_gsim = gsim.mean(dim=1)                            # (N,)
                too_similar = avg_gsim > 0.8
                if too_similar.any():
                    self.cenergy[idx[too_similar]] -= 0.003

            # Predation: strong negative similarity (< -0.5) → energy transfer
            # Only check nearest neighbor (column 0) for efficiency, 5% chance
            if not self._disable_predation:
                nn_sim = gsim[:, 0]                                    # (N,)
                pred_mask = (nn_sim < -0.5) & (torch.rand(N, device=DEV) < 0.05)
                if pred_mask.any():
                    predator_idx = idx[pred_mask]
                    prey_local = nidx[pred_mask, 0]                    # local indices into idx
                    prey_idx = idx[prey_local]
                    self.cenergy[predator_idx] += 0.1
                    self.cenergy[prey_idx] -= 0.15

        # ── Change 3 (Bug Fix): Nutrient interaction using expressed phenotype ──
        # Use self.cstate[idx] (already updated above) for cosine similarity
        new_p = self.cpos[idx]
        expressed = self.cstate[idx]  # (N, DIM) — expressed phenotype after state update

        # Change 9: Nutrient energy regen
        n_nuts = len(self.nuts)
        if n_nuts > 0:
            self.nut_energy[:n_nuts] = (self.nut_energy[:n_nuts] + 0.02).clamp(max=100.0)

        # Change 17: Use cached nutrient tensors
        if n_nuts > 0 and N > 0:
            # Vectorized nutrient interaction using cached tensors
            # nut_pos: (n_nuts, 2), nut_sig: (n_nuts, DIM)
            for ni in range(n_nuts):
                np_ = self.nut_pos[ni]  # (2,)
                nut_radius = self.nuts[ni].radius
                d = (new_p - np_.unsqueeze(0)).abs()
                d = torch.min(d, SWt - d)
                nd = d.norm(dim=1)  # (N,)
                mask = nd < nut_radius
                if mask.any():
                    # Change 3: Use expressed phenotype for cosine similarity
                    sim = F.cosine_similarity(expressed[mask], self.nut_sig[ni].unsqueeze(0))
                    # Base gain always positive; sim bonus on top (gentle selection)
                    gain = NGAIN * (0.6 + 0.4 * sim) * (1 - nd[mask] / nut_radius).clamp(min=0)

                    # Nutrient competition: divide gain by number of feeders
                    n_feeders = mask.sum().item()
                    if n_feeders > 1:
                        gain = gain / n_feeders

                    # Change 9: Nutrient depletion modulates gain
                    ne = self.nut_energy[ni].item()
                    if ne < 10.0:
                        gain = gain * (ne / 10.0)

                    self.cenergy[idx[mask]] += gain

                    # Change 9: Deplete nutrient energy (faster depletion rate: 0.03)
                    total_gain = gain.sum().item()
                    self.nut_energy[ni] = (self.nut_energy[ni] - total_gain * 0.03).clamp(min=0)

        # ── Emit vesicles (vectorized) ──
        # Change 1 (Bug Fix): Gate emission on ves_on
        if self.ves_on:
            # Suppress emission when starving (energy < 40)
            emit_gate = (self.cenergy[idx] > 40).float()
            emit_roll = torch.rand(N, device=DEV) < (emit_p * emit_gate)
            if emit_roll.any():
                e_idx = idx[emit_roll]
                n_emit = emit_roll.sum().item()
                dead_v = (~self.valive).nonzero(as_tuple=True)[0]
                n_emit = min(n_emit, len(dead_v))
                if n_emit > 0:
                    e_idx = e_idx[:n_emit]
                    dv = dead_v[:n_emit]
                    ang = torch.rand(n_emit, device=DEV) * (2 * math.pi)
                    self.vprev[dv] = self.cpos[e_idx].clone()  # init prev to spawn pos
                    self.vpos[dv] = self.cpos[e_idx]
                    self.vvel[dv] = torch.stack([ang.cos(), ang.sin()], -1) * VSPD
                    self.vcont[dv] = vc_out[emit_roll][:n_emit]
                    self.vlife[dv] = VLIFE
                    self.valive[dv] = True
                    self.cenergy[e_idx] -= VCOST

        # ── Sexual recombination (crossover reproduction) ──
        if N > 1 and k > 0 and not self._disable_sexual:
            # Eligible: energy > DIV_E * 0.7, nearest neighbor < 20px
            energy_ok = self.cenergy[idx] > DIV_E * 0.7           # (N,)
            nn_dist = dd.min(dim=1).values                         # (N,)
            close_enough = nn_dist < 20.0                          # (N,)
            nn_idx_local = dd.argmin(dim=1)                        # (N,) local index of nearest

            # Genome cosine similarity with nearest neighbor
            g_norm_sex = F.normalize(g, dim=1)                     # (N, GDIM)
            nn_g = g[nn_idx_local]                                 # (N, GDIM)
            nn_g_norm = F.normalize(nn_g, dim=1)                   # (N, GDIM)
            sex_sim = (g_norm_sex * nn_g_norm).sum(dim=1)          # (N,)

            # Both parents need enough energy; moderate similarity 0.3-0.7
            nn_energy_ok = self.cenergy[idx[nn_idx_local]] > DIV_E * 0.7
            sex_eligible = (energy_ok & nn_energy_ok & close_enough
                            & (sex_sim > 0.3) & (sex_sim < 0.7))
            # 2% chance per eligible pair per step
            sex_roll = torch.rand(N, device=DEV) < 0.02
            sex_mask = sex_eligible & sex_roll

            if sex_mask.any():
                parent_a = idx[sex_mask]
                parent_b = idx[nn_idx_local[sex_mask]]
                n_sex = len(parent_a)
                dead_c_sex = (~self.calive).nonzero(as_tuple=True)[0]
                n_sex = min(n_sex, len(dead_c_sex))
                if n_sex > 0:
                    parent_a = parent_a[:n_sex]
                    parent_b = parent_b[:n_sex]
                    child_slots = dead_c_sex[:n_sex]
                    # Per-gene coin flip crossover + mutation
                    coin = torch.rand(n_sex, GDIM, device=DEV) < 0.5
                    child_genome = torch.where(coin,
                                               self.cgenome[parent_a],
                                               self.cgenome[parent_b])
                    child_genome += torch.randn(n_sex, GDIM, device=DEV) * MUT
                    # Child state = average of parents
                    child_state = (self.cstate[parent_a] + self.cstate[parent_b]) * 0.5
                    # Spawn child near midpoint
                    child_pos = ((self.cpos[parent_a] + self.cpos[parent_b]) * 0.5
                                 + torch.randn(n_sex, 2, device=DEV) * 10) % SWt
                    # Cost: 30% energy each parent
                    self.cenergy[parent_a] *= 0.7
                    self.cenergy[parent_b] *= 0.7
                    # Initialize child
                    self.calive[child_slots] = True
                    self.cpos[child_slots] = child_pos
                    self.cvel[child_slots] = 0
                    self.cstate[child_slots] = child_state
                    self.cgenome[child_slots] = child_genome
                    self.cenergy[child_slots] = INIT_E * 0.8
                    self.cage[child_slots] = 0
                    self.cgen[child_slots] = (torch.max(self.cgen[parent_a],
                                                         self.cgen[parent_b]) + 1)
                    self.cflash[child_slots] = 8.
                    self.crecent[child_slots] = 0
                    self.cprev1[child_slots] = child_pos.clone()
                    self.cprev2[child_slots] = child_pos.clone()
                    self.cprev3[child_slots] = child_pos.clone()
                    self.births += n_sex

        # ── Division (vectorized) ──
        div_mask = self.cenergy[idx] > DIV_E
        if div_mask.any():
            d_idx = idx[div_mask]
            dead_c = (~self.calive).nonzero(as_tuple=True)[0]
            nd = min(len(d_idx), len(dead_c))
            if nd > 0:
                d_idx = d_idx[:nd]; dc = dead_c[:nd]
                self.calive[dc] = True
                self.cpos[dc] = (self.cpos[d_idx] + torch.randn(nd, 2, device=DEV) * 15) % SWt
                self.cvel[dc] = 0
                self.cstate[dc] = self.cstate[d_idx].clone()
                self.cgenome[dc] = self.cgenome[d_idx] + torch.randn(nd, GDIM, device=DEV) * MUT
                self.cenergy[dc] = self.cenergy[d_idx] * .45
                self.cenergy[d_idx] *= .45
                self.cage[dc] = 0
                self.cgen[dc] = self.cgen[d_idx] + 1
                self.cflash[d_idx] = 8.
                self.crecent[dc] = 0
                self.births += nd

        # ── Death ──
        death_mask = self.cenergy[idx] <= 0
        if death_mask.any():
            self.calive[idx[death_mask]] = False
            self.deaths += death_mask.sum().item()

        # Min pop
        n_alive = self.calive.sum().item()
        if n_alive < 10:
            self._spawn(10, near_nuts=True)

        # ── Cluster assignment ──
        if self.nuts:
            alive_idx = self.calive.nonzero(as_tuple=True)[0]
            if len(alive_idx) > 0:
                ap = self.cpos[alive_idx]
                cd = wrapped_dist(ap, self.nut_pos)
                self.cclust[alive_idx] = cd.argmin(1)

        # Diversity
        if self.t % 6 == 0:
            ai = self.calive.nonzero(as_tuple=True)[0]
            if len(ai) > 1:
                self.div_hist.append(self.cstate[ai].var(0).sum().item())
                if len(self.div_hist) > 400:
                    self.div_hist = self.div_hist[-400:]

        # MPS sync: flush lazy ops to prevent GPU memory buildup
        if DEV.type == "mps" and self.t % 4 == 0:
            torch.mps.synchronize()


# ━━ Renderer (Enhanced) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class Ren:
    def __init__(self):
        pygame.init()
        self.scr = pygame.display.set_mode((TW, TH))
        pygame.display.set_caption("ALife v7 — Transformer Vesicle Life")
        self.clk = pygame.time.Clock()
        self.f11 = pygame.font.SysFont("Menlo", 11)
        self.f13 = pygame.font.SysFont("Menlo", 13)
        self.f15 = pygame.font.SysFont("Menlo", 15)
        self.trail = pygame.Surface((SW, TH))
        self.trail.fill(BG)

        # [15] Preallocated surfaces
        self.fade_surf = pygame.Surface((SW, TH))
        self.fade_surf.fill(BG)
        self.fade_surf.set_alpha(14)
        self.cell_surf = pygame.Surface((SW, TH), pygame.SRCALPHA)
        self.nut_surf = pygame.Surface((SW, TH), pygame.SRCALPHA)

        # [14] Panel toggle
        self.show_panel = True

        # [12] Absorption ripples
        self.ripples = []

        # [13] Nutrient cloud sprites
        self.nut_sprites = {}
        self._prev_nut_ids = None

        # Track previous flash state for ripple detection
        self._prev_flash = None

        # Composition surface for zoom
        self._comp_surf = pygame.Surface((SW, TH))

        # ── Camera / Zoom ──
        self.zoom_level = 1.0   # 0.5 .. 3.0
        self.cam_offset = [0.0, 0.0]  # pan offset in world coords
        self._mid_drag = False
        self._mid_prev = None

        # ── Recording ──
        self.recording = False
        self.rec_frame = 0

        # ── Fullscreen ──
        self._fullscreen = False
        self._win_size = (TW, TH)

        # ── Stats overlay ──
        self.show_stats = False
        self._step_ms = 0.0
        self._draw_ms = 0.0

    def _make_nutrient_sprite(self, nut):
        """Create a cloudy sprite using overlapping translucent circles with glow."""
        size = int(nut.radius * 2.8)
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        cx, cy = size // 2, size // 2
        col = nut.bright
        rng = random.Random(int(nut.x * 100 + nut.y))
        # Outer glow — large diffuse halo
        for gr in range(3):
            glow_r = int(nut.radius * (1.1 + gr * 0.15))
            glow_a = max(3, 12 - gr * 3)
            pygame.draw.circle(surf, (*col, glow_a), (cx, cy), glow_r)
        # 9 layers of offset translucent circles for cloud effect
        for layer in range(9):
            ox = rng.randint(-int(nut.radius * 0.18), int(nut.radius * 0.18))
            oy = rng.randint(-int(nut.radius * 0.18), int(nut.radius * 0.18))
            scale = 0.4 + layer * 0.08
            r = int(nut.radius * scale)
            a = max(6, 45 - layer * 4)
            pygame.draw.circle(surf, (*col, a), (cx + ox, cy + oy), r)
        # Bright core
        pygame.draw.circle(surf, (*col, 70), (cx, cy), int(nut.radius * 0.3))
        pygame.draw.circle(surf, (255, 255, 255, 45), (cx, cy), int(nut.radius * 0.12))
        return surf

    def _ensure_nut_sprites(self, w):
        """Regenerate nutrient sprites only when the set of nutrients changes."""
        nut_ids = tuple((int(n.x), int(n.y)) for n in w.nuts)
        if nut_ids != self._prev_nut_ids:
            self.nut_sprites = {}
            for i, nut in enumerate(w.nuts):
                self.nut_sprites[i] = self._make_nutrient_sprite(nut)
            self._prev_nut_ids = nut_ids

    def draw(self, w, paused, speed):
        t0 = time.perf_counter()
        if self.show_panel:
            self._sim(w, sim_w=SW)
            self._panel(w)
        else:
            self._sim(w, sim_w=TW)
        self._hud(w, paused, speed)
        self._draw_ms = (time.perf_counter() - t0) * 1000.0

        # ── Recording: save frame before flip ──
        if self.recording:
            rec_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")
            os.makedirs(rec_dir, exist_ok=True)
            fname = os.path.join(rec_dir, f"frame_{self.rec_frame:06d}.png")
            pygame.image.save(self.scr, fname)
            self.rec_frame += 1

        pygame.display.flip()

    def _sim(self, w, sim_w=SW):
        # Fade trail — reuse preallocated surface
        self.trail.blit(self.fade_surf, (0, 0))

        # [16] Batch tensor reads to numpy
        ai = w.calive.nonzero(as_tuple=True)[0]
        N = len(ai)
        if N > 0:
            pos_np = w.cpos[ai].cpu().numpy()
            state_np = w.cstate[ai].cpu().numpy()
            energy_np = w.cenergy[ai].cpu().numpy()
            vel_np = w.cvel[ai].cpu().numpy()
            age_np = w.cage[ai].long().cpu().numpy()
            flash_np = w.cflash[ai].cpu().numpy()
            genome_np = w.cgenome[ai].cpu().numpy()
            prev1_np = w.cprev1[ai].cpu().numpy()
            prev2_np = w.cprev2[ai].cpu().numpy()
            prev3_np = w.cprev3[ai].cpu().numpy()
        else:
            pos_np = state_np = energy_np = vel_np = age_np = flash_np = genome_np = None
            prev1_np = prev2_np = prev3_np = None

        # Same for vesicles
        va = w.valive.nonzero(as_tuple=True)[0]
        nv = len(va)
        if nv > 0:
            vpos_np = w.vpos[va].cpu().numpy()
            vvel_np = w.vvel[va].cpu().numpy()
            vcont_np = w.vcont[va].cpu().numpy()
            vlife_np = w.vlife[va].cpu().numpy()
            vprev_np = w.vprev[va].cpu().numpy()
        else:
            vpos_np = vvel_np = vcont_np = vlife_np = vprev_np = None

        # ── Vesicle trails — softer wider glow + point light [12] ──
        if nv > 0:
            for vi in range(nv):
                t = vlife_np[vi] / VLIFE
                x, y = int(vpos_np[vi, 0]), int(vpos_np[vi, 1])
                col = pheno_rgb_np(vcont_np[vi, :4], .5 + .4 * t)
                vx, vy = vvel_np[vi, 0], vvel_np[vi, 1]
                speed = math.sqrt(vx*vx + vy*vy)
                if speed > 0.1:
                    tail_x = x - int(vx * 3)
                    tail_y = y - int(vy * 3)
                    # Wider softer trail glow underneath
                    soft_col = (col[0] // 3, col[1] // 3, col[2] // 3)
                    pygame.draw.line(self.trail, soft_col, (tail_x, tail_y), (x, y),
                                     max(3, int(5 * t)))
                    # Draw streak from tail to head
                    pygame.draw.line(self.trail, col, (tail_x, tail_y), (x, y),
                                     max(1, int(2 * t)))
                    # Extended trail from previous position
                    px, py = int(vprev_np[vi, 0]), int(vprev_np[vi, 1])
                    d_prev = abs(px - x) + abs(py - y)
                    if d_prev < 100:
                        dim_c = (col[0] // 2, col[1] // 2, col[2] // 2)
                        pygame.draw.line(self.trail, dim_c, (px, py), (tail_x, tail_y),
                                         max(1, int(1.5 * t)))
                    # Bright head dot
                    bright_col = (min(255, col[0] + 80),
                                  min(255, col[1] + 80),
                                  min(255, col[2] + 80))
                    pygame.draw.circle(self.trail, bright_col, (x, y),
                                       max(2, int(2.5 * t)))
                    # Tiny white point light at vesicle head
                    pygame.draw.circle(self.trail, (255, 255, 255), (x, y), 1)
                else:
                    pygame.draw.circle(self.trail, col, (x, y),
                                       max(1, int(2 * t + .5)))

        # ── Cell motion trails — fading comet tails on trail surface ──
        if N > 0:
            for ci in range(N):
                x, y = int(pos_np[ci, 0]), int(pos_np[ci, 1])
                e = energy_np[ci]
                vx, vy = vel_np[ci, 0], vel_np[ci, 1]
                spd = math.sqrt(vx*vx + vy*vy)
                col = pheno_rgb_np(state_np[ci, :4], .15 + .2 * min(1., e / DIV_E))
                # Always draw subtle cell glow
                pygame.draw.circle(self.trail, col, (x, y), 10)
                # Motion trail: fading comet tail from previous positions
                if spd > 0.5:
                    trail_a = min(1.0, spd * 0.3)
                    p1x, p1y = int(prev1_np[ci, 0]), int(prev1_np[ci, 1])
                    p2x, p2y = int(prev2_np[ci, 0]), int(prev2_np[ci, 1])
                    p3x, p3y = int(prev3_np[ci, 0]), int(prev3_np[ci, 1])
                    t_w = max(1, int(3 * min(1., e / DIV_E)))
                    if abs(p1x - x) + abs(p1y - y) < 100:
                        c1 = (int(col[0] * 0.7 * trail_a),
                              int(col[1] * 0.7 * trail_a),
                              int(col[2] * 0.7 * trail_a))
                        pygame.draw.line(self.trail, c1, (p1x, p1y), (x, y), t_w)
                    if abs(p2x - p1x) + abs(p2y - p1y) < 100:
                        c2 = (int(col[0] * 0.4 * trail_a),
                              int(col[1] * 0.4 * trail_a),
                              int(col[2] * 0.4 * trail_a))
                        pygame.draw.line(self.trail, c2, (p2x, p2y), (p1x, p1y), max(1, t_w - 1))
                    if abs(p3x - p2x) + abs(p3y - p2y) < 100:
                        c3 = (int(col[0] * 0.2 * trail_a),
                              int(col[1] * 0.2 * trail_a),
                              int(col[2] * 0.2 * trail_a))
                        pygame.draw.line(self.trail, c3, (p3x, p3y), (p2x, p2y), max(1, t_w - 2))

        # Compose trail into comp surface
        self._comp_surf.blit(self.trail, (0, 0))

        # ── Nutrient zones — cloudy sprites [13] ──
        self._ensure_nut_sprites(w)
        self.nut_surf.fill((0, 0, 0, 0))
        for ni, nut in enumerate(w.nuts):
            sprite = self.nut_sprites.get(ni)
            if sprite is None:
                continue
            size = sprite.get_width()
            nx, ny = int(nut.x), int(nut.y)
            # Pulse effect
            pulse = 0.92 + 0.08 * math.sin(w.t * 0.025 + nut.x * 0.01)
            if abs(pulse - 1.0) > 0.01:
                scaled_size = int(size * pulse)
                scaled = pygame.transform.smoothscale(sprite, (scaled_size, scaled_size))
                self.nut_surf.blit(scaled, (nx - scaled_size // 2, ny - scaled_size // 2))
            else:
                self.nut_surf.blit(sprite, (nx - size // 2, ny - size // 2))
            # Bright center dot
            c = nut.bright
            pygame.draw.circle(self.nut_surf, (*c, 120), (nx, ny), 8)
            pygame.draw.circle(self.nut_surf, (255, 255, 255, 50), (nx, ny), 3)
            # Outer ring
            rr = int(nut.radius * pulse)
            pygame.draw.circle(self.nut_surf, (*c, 22), (nx, ny), rr, 1)
        self._comp_surf.blit(self.nut_surf, (0, 0))

        # ── Absorption ripples [12] — detect new flashes ──
        if N > 0:
            if self._prev_flash is not None and len(self._prev_flash) == N:
                for ci in range(N):
                    # Flash just appeared (was 0 or less, now > 0)
                    if flash_np[ci] > 0 and self._prev_flash[ci] <= 0:
                        x, y = int(pos_np[ci, 0]), int(pos_np[ci, 1])
                        col = pheno_rgb_np(state_np[ci, :4], 0.8)
                        self.ripples.append([x, y, 5.0, 200.0, col])
            self._prev_flash = flash_np.copy()
        else:
            self._prev_flash = None

        # Update and draw ripples on SRCALPHA surface (so alpha actually works)
        ripple_surf = self.cell_surf  # reuse, will be cleared below for cells
        new_ripples = []
        for rip in self.ripples:
            rx, ry, rr, ra, rc = rip
            if ra > 0:
                a = max(0, min(255, int(ra)))
                pygame.draw.circle(ripple_surf, (*rc, a), (int(rx), int(ry)), int(rr), 1)
                if rr > 8:
                    a2 = max(0, min(255, int(ra * 0.5)))
                    pygame.draw.circle(ripple_surf, (*rc, a2), (int(rx), int(ry)), int(rr - 4), 1)
                rip[2] = rr + 2.0
                rip[3] = ra - 12.0
                if rip[3] > 0:
                    new_ripples.append(rip)
        self.ripples = new_ripples
        if self.ripples:
            self._comp_surf.blit(ripple_surf, (0, 0))
            ripple_surf.fill((0, 0, 0, 0))

        # ── Cells — transparent lens / fluid membrane ──
        self.cell_surf.fill((0, 0, 0, 0))
        if N > 0:
            # Dynamic LOD: fewer points and layers when many cells
            if N > 400:
                n_pts = 12; n_layers = 3  # fast mode
            elif N > 200:
                n_pts = 16; n_layers = 4
            else:
                n_pts = 20; n_layers = 5  # full quality
            tau = math.tau
            for ci in range(N):
                x, y = int(pos_np[ci, 0]), int(pos_np[ci, 1])
                e = energy_np[ci]
                vx, vy = vel_np[ci, 0], vel_np[ci, 1]
                spd = math.sqrt(vx*vx + vy*vy)
                fl = flash_np[ci] / 8.0
                age = age_np[ci]
                t = min(1., e / DIV_E)

                idx_val = ai[ci].item()
                breath = math.sin(age * 0.06 + idx_val * 1.3) * 0.10
                r_base = 7 + 10 * t
                r = max(5, int(r_base * (1 + breath)))

                bri = .5 + .4 * t
                col = pheno_rgb_np(state_np[ci, :4], bri)

                # ── Fluid deformation: stretch along velocity ──
                if spd > 0.3:
                    dx_n, dy_n = vx / spd, vy / spd
                    stretch = min(0.25, spd * 0.08)
                else:
                    dx_n, dy_n = 0., 0.
                    stretch = 0.

                # ── Generate membrane polygon with fluid deformation ──
                def _membrane_pts(radius, noise_amp):
                    pts = []
                    for j in range(n_pts):
                        angle = j * tau / n_pts
                        ca, sa = math.cos(angle), math.sin(angle)
                        # Organic noise from genome + state + time
                        g_idx = j % GDIM
                        s_idx = (j * 3) % DIM
                        noise = (math.sin(angle * 3 + genome_np[ci, g_idx] * 5) * 0.12 +
                                 math.sin(angle * 5 + state_np[ci, s_idx] * 3 + age * 0.002) * 0.06 +
                                 math.sin(angle * 7 + idx_val * 0.5 + age * 0.004) * 0.04)
                        # Fluid stretch along velocity direction
                        dir_align = ca * dx_n + sa * dy_n  # dot product
                        fluid_stretch = 1.0 + stretch * (0.6 * dir_align + 0.4 * dir_align**3)
                        vr = radius * (1 + noise * noise_amp) * fluid_stretch
                        pts.append((x + int(ca * vr), y + int(sa * vr)))
                    return pts

                body_pts = _membrane_pts(r, 1.0)

                # ── Lens layers (LOD-aware, deep transparency ~30% reduced alphas) ──
                if n_layers >= 5:
                    # Full: haze + mid + inner + core + membrane
                    haze_pts = _membrane_pts(r + 6, 0.5)
                    pygame.draw.polygon(self.cell_surf, (*col, int(5 + fl * 10)), haze_pts)
                    mid_pts = _membrane_pts(r + 1, 0.8)
                    pygame.draw.polygon(self.cell_surf, (*col, int(14 + 10 * t)), mid_pts)
                    inner_pts = _membrane_pts(r * 0.7, 0.6)
                    pygame.draw.polygon(self.cell_surf, (*col, int(24 + 18 * t + fl * 14)), inner_pts)
                elif n_layers >= 4:
                    # No haze, mid + inner
                    mid_pts = _membrane_pts(r + 1, 0.8)
                    pygame.draw.polygon(self.cell_surf, (*col, int(15 + 12 * t)), mid_pts)
                    inner_pts = _membrane_pts(r * 0.7, 0.6)
                    pygame.draw.polygon(self.cell_surf, (*col, int(28 + 18 * t + fl * 14)), inner_pts)
                else:
                    # Minimal: just body fill
                    pygame.draw.polygon(self.cell_surf, (*col, int(24 + 21 * t + fl * 14)), body_pts)

                # Core (always drawn — the "focal point", reduced alpha)
                core_col = (min(255, col[0]+30), min(255, col[1]+30), min(255, col[2]+30))
                pygame.draw.circle(self.cell_surf, (*core_col, int(35 + 21 * t)),
                                   (x, y), max(2, int(r * 0.35)))

                # ── Membrane outline — thickness varies with energy (surface tension) ──
                # Iridescence: membrane color shifts with angle + time
                iri_phase = w.t * 0.03 + idx_val * 0.7
                iri_r = min(255, col[0] + 60 + int(fl * 80) + int(15 * math.sin(iri_phase)))
                iri_g = min(255, col[1] + 60 + int(fl * 80) + int(15 * math.sin(iri_phase + 2.09)))
                iri_b = min(255, col[2] + 60 + int(fl * 80) + int(15 * math.sin(iri_phase + 4.19)))
                mc = (max(0, iri_r), max(0, iri_g), max(0, iri_b))
                # Membrane thickness: 1 for dying cells, 2-3 for healthy
                mem_thick = max(1, min(3, int(1 + 2 * t)))
                pygame.draw.polygon(self.cell_surf, (*mc, int(70 + 56 * t)), body_pts, mem_thick)

                # ── Specular highlight — more dramatic shift with velocity ──
                spec_angle = math.atan2(dy_n, dx_n) if spd > 0.3 else (age * 0.01 + idx_val)
                # More dramatic offset: radius scales with speed
                spec_r = r * (0.3 + min(0.3, spd * 0.1))
                spec_x = x + int(math.cos(spec_angle + 0.8) * spec_r)
                spec_y = y + int(math.sin(spec_angle + 0.8) * spec_r)
                spec_pulse = math.sin(age * 0.1 + idx_val * 0.7) * 0.2 + 0.8
                spec_a = int((45 + 60 * spec_pulse + fl * 35) * t)
                spec_sz = max(2, int(r * 0.2))
                pygame.draw.circle(self.cell_surf, (255, 255, 255, min(180, spec_a)),
                                   (spec_x, spec_y), spec_sz)

                # ── Lens shadow — dark circle opposite to specular ──
                shadow_x = x - int(math.cos(spec_angle + 0.8) * spec_r * 0.8)
                shadow_y = y - int(math.sin(spec_angle + 0.8) * spec_r * 0.8)
                shadow_sz = max(2, int(r * 0.25))
                pygame.draw.circle(self.cell_surf, (0, 0, 0, int(10 + 8 * t)),
                                   (shadow_x, shadow_y), shadow_sz)

                if n_layers >= 4:
                    spec2_x = x - int(math.cos(spec_angle + 0.3) * spec_r * 0.6)
                    spec2_y = y - int(math.sin(spec_angle + 0.3) * spec_r * 0.6)
                    pygame.draw.circle(self.cell_surf, (255, 255, 240, min(85, spec_a // 2)),
                                       (spec2_x, spec2_y), max(1, spec_sz - 1))

                # ── Division-ready: pulsing outer ring ──
                if e > DIV_E * .85:
                    da = int(30 + 25 * math.sin(age * 0.15))
                    div_pts = _membrane_pts(r + 4, 0.6)
                    pygame.draw.polygon(self.cell_surf, (255, 255, 200, da), div_pts, 1)

        # ── Neighbor connection lines — faint synaptic network ──
        if N > 1:
            conn_surf = pygame.Surface((SW, TH), pygame.SRCALPHA)
            # Vectorized: for each cell, draw line to its 3 nearest neighbors
            pos_t = w.cpos[ai]  # (N, 2) tensor
            dd_t = wrapped_dist(pos_t, pos_t)  # (N, N) tensor
            dd_t.fill_diagonal_(1e9)
            k_conn = min(3, N - 1)
            _, nn_idx = dd_t.topk(k_conn, largest=False)  # (N, k_conn)
            nn_d = dd_t.gather(1, nn_idx).cpu().numpy()  # (N, k_conn)
            nn_idx_np = nn_idx.cpu().numpy()
            for ci in range(min(N, 350)):
                cx0, cy0 = int(pos_np[ci, 0]), int(pos_np[ci, 1])
                col_ci = pheno_rgb_np(state_np[ci, :4], 0.3)
                for ki in range(k_conn):
                    d = nn_d[ci, ki]
                    if d > 45 or d < 5:
                        continue
                    cj = nn_idx_np[ci, ki]
                    cx1, cy1 = int(pos_np[cj, 0]), int(pos_np[cj, 1])
                    # Skip wrapped-around lines (screen distance)
                    if abs(cx0 - cx1) > 60 or abs(cy0 - cy1) > 60:
                        continue
                    fade = max(4, int(16 * (1 - d / 45)))
                    pygame.draw.line(conn_surf, (*col_ci, fade),
                                     (cx0, cy0), (cx1, cy1), 1)
            self._comp_surf.blit(conn_surf, (0, 0))

        self._comp_surf.blit(self.cell_surf, (0, 0))

        # ── Zoom + Pan: scale comp surface and blit to screen ──
        z = self.zoom_level
        if abs(z - 1.0) < 0.01 and abs(self.cam_offset[0]) < 1 and abs(self.cam_offset[1]) < 1:
            # No zoom — direct blit (fast path)
            self.scr.blit(self._comp_surf, (0, 0))
        else:
            # Viewport in world coords: what region of the SW x TH world is visible
            vw = SW / z
            vh = TH / z
            # Center of view in world coords
            cx = SW / 2.0 + self.cam_offset[0]
            cy = TH / 2.0 + self.cam_offset[1]
            src_x = int(cx - vw / 2)
            src_y = int(cy - vh / 2)
            src_w = int(vw)
            src_h = int(vh)
            # Clamp source rect to surface bounds
            src_x = max(0, min(src_x, SW - 1))
            src_y = max(0, min(src_y, TH - 1))
            src_w = min(src_w, SW - src_x)
            src_h = min(src_h, TH - src_y)
            if src_w > 0 and src_h > 0:
                sub = self._comp_surf.subsurface((src_x, src_y, src_w, src_h))
                scaled = pygame.transform.scale(sub, (sim_w, TH))
                self.scr.blit(scaled, (0, 0))
            else:
                self.scr.blit(self._comp_surf, (0, 0))

    def _panel(self, w):
        px = SW
        pygame.draw.rect(self.scr, PBG, (px, 0, PW, TH))
        pygame.draw.line(self.scr, (25, 25, 45), (px, 0), (px, TH), 1)
        self.scr.blit(self.f15.render("GENOME MAP", True, (140,140,170)), (px+10, 8))

        bx = px + 8
        seg = 10  # pixels per barcode segment (wide + readable)
        gw = GDIM * seg   # genome barcode width 160
        pw2 = 16 * seg     # pheno barcode width 160
        gap = 8

        self.scr.blit(self.f11.render("DNA", True, (120,80,80)), (bx, 28))
        self.scr.blit(self.f11.render("PHENO", True, (80,80,120)), (bx+gw+gap, 28))

        ai = w.calive.nonzero(as_tuple=True)[0]
        n = len(ai)
        if n == 0: return

        # Sort by cluster, then genome sum
        clusters = w.cclust[ai]
        gsums = w.cgenome[ai].sum(1)
        sort_key = clusters.float() * 1e6 + gsums
        _, order = sort_key.sort()
        sorted_idx = ai[order]

        top = 42; bot = TH - 130
        avail = bot - top
        rh = max(2, min(6, avail // max(1, n)))

        # Backbone
        pygame.draw.line(self.scr, (30,30,48), (bx-2, top), (bx-2, top+rh*min(n, avail//max(1,rh))), 1)
        end_x = bx + gw + gap + pw2 + 2
        pygame.draw.line(self.scr, (30,30,48), (end_x, top), (end_x, top+rh*min(n, avail//max(1,rh))), 1)

        prev_cl = -1
        y = top
        for ci in range(min(n, avail // max(1, rh))):
            i = sorted_idx[ci].item()
            cl = w.cclust[i].item()
            if cl != prev_cl and prev_cl >= 0:
                pygame.draw.line(self.scr, (80,80,110), (bx-2, y), (end_x+2, y), 1)
                # Cluster color tag
                if cl < len(w.nuts):
                    nc = w.nuts[cl].bright
                    pygame.draw.rect(self.scr, nc, (end_x+4, y, 4, max(2, rh*3)))
            prev_cl = cl

            gvals = torch.tanh(w.cgenome[i]).cpu().tolist()
            pvals = torch.tanh(w.cstate[i, :16]).cpu().tolist()
            for d in range(GDIM):
                pygame.draw.rect(self.scr, dim_col(gvals[d]), (bx+d*seg, y, seg-1, max(1,rh-1)))
            for d in range(16):
                pygame.draw.rect(self.scr, dim_col(pvals[d]), (bx+gw+gap+d*seg, y, seg-1, max(1,rh-1)))
            y += rh

        # Cluster legend
        ly = min(y + 8, bot + 5)
        for ni, nut in enumerate(w.nuts):
            if ly + 13 > TH - 60: break
            cnt = (w.cclust[w.calive] == ni).sum().item()
            pygame.draw.rect(self.scr, nut.bright, (bx, ly, 8, 8))
            self.scr.blit(self.f11.render(f"N{ni}: {cnt}", True, (110,110,140)), (bx+14, ly-1))
            ly += 14

        # Diversity graph
        self._divgraph(w, px+10, TH-105, PW-20, 50)

    def _divgraph(self, w, gx, gy, gw, gh):
        self.scr.blit(self.f11.render("phenotype diversity", True, (90,90,120)), (gx, gy-13))
        pygame.draw.rect(self.scr, (18,18,30), (gx, gy, gw, gh))
        pygame.draw.rect(self.scr, (35,35,55), (gx, gy, gw, gh), 1)
        if len(w.div_hist) < 2: return
        vals = w.div_hist[-400:]
        mx = max(vals) or 1
        pts = [(gx + int(i/(len(vals)-1)*gw), gy+gh-int(v/mx*(gh-4))-2) for i, v in enumerate(vals)]
        col = (80,220,120) if w.ves_on else (220,80,80)
        pygame.draw.lines(self.scr, col, False, pts, 2)

    def _hud(self, w, paused, speed):
        vc = (80,220,120) if w.ves_on else (220,80,80)
        vl = "ON" if w.ves_on else "OFF"
        n = w.calive.sum().item()
        nv = w.valive.sum().item()
        y = 8
        zoom_str = f"  zoom={self.zoom_level:.1f}x" if abs(self.zoom_level - 1.0) > 0.01 else ""
        self.scr.blit(self.f11.render(
            f"t={w.t}  cells={n}  ves={nv}  x{speed}{zoom_str}",
            True, (120,120,150)), (10, y)); y += 15
        if n > 0:
            ai = w.calive.nonzero(as_tuple=True)[0]
            ae = w.cenergy[ai].mean().item()
            mg = w.cgen[ai].max().item()
            ag = w.cgen[ai].float().mean().item()
            self.scr.blit(self.f11.render(f"gen={ag:.1f}/{mg}  E={ae:.0f}  +{w.births}/-{w.deaths}",
                                          True, (120,120,150)), (10, y)); y += 15
        self.scr.blit(self.f13.render(f"VESICLE: {vl}", True, vc), (10, y)); y += 18

        # ── Speed display (prominent) ──
        spd_col = (255, 200, 80) if speed > 5 else (120, 200, 120)
        self.scr.blit(self.f13.render(f"SPEED: x{speed}", True, spd_col), (10, y)); y += 18

        # ── Recording indicator ──
        if self.recording:
            rec_col = (255, 40, 40) if (w.t // 15) % 2 == 0 else (180, 20, 20)
            pygame.draw.circle(self.scr, rec_col, (10 + 5, y + 6), 5)
            self.scr.blit(self.f13.render(f" REC  frame {self.rec_frame}", True, (255, 60, 60)),
                          (22, y)); y += 18

        # ── Stats overlay ──
        if self.show_stats:
            fps_val = self.clk.get_fps()
            mem_str = ""
            if _TM:
                cur, peak = _tm.get_traced_memory()
                mem_str = f"  mem={cur/1e6:.1f}MB (peak {peak/1e6:.1f}MB)"
            stats_lines = [
                f"FPS: {fps_val:.1f}  draw: {self._draw_ms:.1f}ms  step: {self._step_ms:.1f}ms{mem_str}",
            ]
            # Nutrient energy levels
            if w.nuts:
                ne_vals = [f"{w.nut_energy[i].item():.0f}" for i in range(len(w.nuts))]
                stats_lines.append(f"nut energy: [{', '.join(ne_vals)}]")
            for line in stats_lines:
                self.scr.blit(self.f11.render(line, True, (180, 180, 220)), (10, y)); y += 14

        if paused:
            sw_actual = SW if self.show_panel else TW
            self.scr.blit(self.f15.render("|| PAUSED", True, (255,200,100)), (sw_actual//2-50, 10))
        self.scr.blit(self.f11.render(
            "SPC:pause V:vesicle D:panel R:reset +/-:speed Z/X:zoom S:stats F5:rec F12:snap F11:full ESC:quit",
            True, (50,50,70)), (10, TH-18))


# ━━ Main ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    w = World()
    r = Ren()
    paused = False
    speed = 1

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT: pygame.quit(); return
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q): pygame.quit(); return
                elif ev.key == pygame.K_SPACE: paused = not paused
                elif ev.key == pygame.K_v: w.ves_on = not w.ves_on
                elif ev.key == pygame.K_d: r.show_panel = not r.show_panel  # [14]
                elif ev.key == pygame.K_r:
                    w = World()
                    r.trail.fill(BG)
                    r.ripples = []
                    r._prev_flash = None
                    r._prev_nut_ids = None
                    r.nut_sprites = {}
                    r.zoom_level = 1.0
                    r.cam_offset = [0.0, 0.0]
                elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS): speed = min(20, speed+1)
                elif ev.key == pygame.K_MINUS: speed = max(1, speed-1)
                # ── Zoom keys ──
                elif ev.key == pygame.K_z:
                    r.zoom_level = min(3.0, r.zoom_level * 1.15)
                elif ev.key == pygame.K_x:
                    r.zoom_level = max(0.5, r.zoom_level / 1.15)
                    # Clamp cam_offset so we don't go out of bounds
                    max_ox = max(0, SW / 2 * (1 - 1 / r.zoom_level))
                    max_oy = max(0, TH / 2 * (1 - 1 / r.zoom_level))
                    r.cam_offset[0] = max(-max_ox, min(max_ox, r.cam_offset[0]))
                    r.cam_offset[1] = max(-max_oy, min(max_oy, r.cam_offset[1]))
                # ── Stats overlay ──
                elif ev.key == pygame.K_s:
                    r.show_stats = not r.show_stats
                # ── Recording (F5) ──
                elif ev.key == pygame.K_F5:
                    r.recording = not r.recording
                    if r.recording:
                        r.rec_frame = 0
                        print("[REC] Recording started -> recordings/")
                    else:
                        print(f"[REC] Recording stopped. {r.rec_frame} frames saved.")
                        print("Convert to video:")
                        print("  ffmpeg -framerate 60 -i recordings/frame_%06d.png "
                              "-c:v libx264 -pix_fmt yuv420p output.mp4")
                # ── Screenshot (F12) ──
                elif ev.key == pygame.K_F12:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    fname = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         f"screenshot_{ts}.png")
                    pygame.image.save(r.scr, fname)
                    print(f"[SNAP] Screenshot saved: {fname}")
                # ── Fullscreen (F11) ──
                elif ev.key == pygame.K_F11:
                    r._fullscreen = not r._fullscreen
                    if r._fullscreen:
                        r.scr = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                    else:
                        r.scr = pygame.display.set_mode(r._win_size)

            # ── Scroll wheel zoom ──
            elif ev.type == pygame.MOUSEWHEEL:
                if ev.y > 0:
                    r.zoom_level = min(3.0, r.zoom_level * 1.1)
                elif ev.y < 0:
                    r.zoom_level = max(0.5, r.zoom_level / 1.1)
                    max_ox = max(0, SW / 2 * (1 - 1 / r.zoom_level))
                    max_oy = max(0, TH / 2 * (1 - 1 / r.zoom_level))
                    r.cam_offset[0] = max(-max_ox, min(max_ox, r.cam_offset[0]))
                    r.cam_offset[1] = max(-max_oy, min(max_oy, r.cam_offset[1]))

            # ── Middle-click pan ──
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button == 2:  # middle click
                    r._mid_drag = True
                    r._mid_prev = ev.pos
                elif ev.button == 1:  # left click — add nutrient
                    mx, my = ev.pos
                    sim_w = SW if r.show_panel else TW
                    if mx < sim_w:
                        # Convert screen coords to world coords (account for zoom+pan)
                        z = r.zoom_level
                        vw = SW / z
                        vh = TH / z
                        cx = SW / 2.0 + r.cam_offset[0]
                        cy = TH / 2.0 + r.cam_offset[1]
                        wx = cx - vw / 2 + (mx / sim_w) * vw
                        wy = cy - vh / 2 + (my / TH) * vh
                        wx = max(0, min(SW - 1, wx))
                        wy = max(0, min(TH - 1, wy))
                        w.add_nutrient(wx, wy)
                        r.nut_sprites = {}
            elif ev.type == pygame.MOUSEBUTTONUP:
                if ev.button == 2:
                    r._mid_drag = False
                    r._mid_prev = None
            elif ev.type == pygame.MOUSEMOTION:
                if r._mid_drag and r._mid_prev is not None:
                    dx = ev.pos[0] - r._mid_prev[0]
                    dy = ev.pos[1] - r._mid_prev[1]
                    r._mid_prev = ev.pos
                    # Pan: move offset in world coords (inverted, scaled by zoom)
                    z = r.zoom_level
                    r.cam_offset[0] -= dx / z
                    r.cam_offset[1] -= dy / z
                    # Clamp
                    max_ox = max(0, SW / 2 * (1 - 1 / z))
                    max_oy = max(0, TH / 2 * (1 - 1 / z))
                    r.cam_offset[0] = max(-max_ox, min(max_ox, r.cam_offset[0]))
                    r.cam_offset[1] = max(-max_oy, min(max_oy, r.cam_offset[1]))

        if not paused:
            t0 = time.perf_counter()
            for _ in range(speed):
                w.step()
            r._step_ms = (time.perf_counter() - t0) * 1000.0

        r.draw(w, paused, speed)
        r.clk.tick(FPS)

if __name__ == "__main__":
    main()
