"""
ALife v6: Rewritten Simulation Core — Transformer Vesicle System
================================================================
Upgraded brain (relational features, genome Q/K modulation, vesicle attention tokens),
nonlinear vesicle exchange, nutrient depletion, aging/crowding pressure,
attention masking, bug fixes (V-toggle, absorption uniqueness, nutrient reward timing).

Renderer code (class Ren) is preserved exactly from v5.

Controls: SPC:pause  V:vesicle  R:reset  +/-:speed  click:nutrient  ESC:quit
"""

import math, random
import pygame, torch, torch.nn as nn, torch.nn.functional as F

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
BG = (4, 4, 12); PBG = (8, 8, 18); FADE = 14

SWt = torch.tensor([float(SW), float(TH)])


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
    v = torch.tanh(p[:4]).tolist() if len(p) >= 4 else [0]*4
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
        self.x, self.y, self.sig, self.radius = x, y, torch.randn(DIM), NRAD
    @property
    def color(self): return pheno_rgb(self.sig, .45)
    @property
    def bright(self): return pheno_rgb(self.sig, .8)


# ━━ World (fully tensorized, upgraded) ━━━━━━━━━━━━━━━━━━━━━
class World:
    def __init__(self):
        self.brain = Brain()
        # Cell pool
        self.cpos = torch.zeros(MAX_C, 2)
        self.cvel = torch.zeros(MAX_C, 2)
        self.cstate = torch.zeros(MAX_C, DIM)
        self.cgenome = torch.zeros(MAX_C, GDIM)
        self.cenergy = torch.zeros(MAX_C)
        self.cage = torch.zeros(MAX_C, dtype=torch.long)
        self.cgen = torch.zeros(MAX_C, dtype=torch.long)
        self.calive = torch.zeros(MAX_C, dtype=torch.bool)
        self.cclust = torch.zeros(MAX_C, dtype=torch.long)
        self.cflash = torch.zeros(MAX_C)
        # Change 8: Recent absorption counter (exponential decay)
        self.crecent = torch.zeros(MAX_C)
        # Vesicle pool
        self.vpos = torch.zeros(MAX_V, 2)
        self.vprev = torch.zeros(MAX_V, 2)  # previous vesicle position for visual use
        self.vvel = torch.zeros(MAX_V, 2)
        self.vcont = torch.zeros(MAX_V, DIM)
        self.vlife = torch.zeros(MAX_V)
        self.valive = torch.zeros(MAX_V, dtype=torch.bool)
        # Change 9: Nutrient energy
        self.nut_energy = torch.full((MAX_NUTS,), 100.0)
        # Change 17: Cached nutrient tensors
        self.nut_pos = torch.zeros(0, 2)
        self.nut_sig = torch.zeros(0, DIM)
        # Meta
        self.nuts = []
        self.t = 0; self.births = 0; self.deaths = 0
        self.ves_on = True
        self.div_hist = []
        self._init()

    def _rebuild_nut_cache(self):
        """Change 17: Rebuild cached nutrient position/signature tensors."""
        if self.nuts:
            self.nut_pos = torch.tensor([[n.x, n.y] for n in self.nuts])
            self.nut_sig = torch.stack([n.sig for n in self.nuts])
        else:
            self.nut_pos = torch.zeros(0, 2)
            self.nut_sig = torch.zeros(0, DIM)

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
            nut_idx = torch.randint(len(self.nuts), (m,))
            centers = torch.tensor([[self.nuts[i].x, self.nuts[i].y] for i in nut_idx])
            self.cpos[dead] = (centers + torch.randn(m, 2) * 60) % SWt
        else:
            self.cpos[dead] = torch.rand(m, 2) * SWt
        self.cvel[dead] = 0
        self.cstate[dead] = torch.randn(m, DIM) * .1
        self.cgenome[dead] = torch.randn(m, GDIM) * .5
        self.cenergy[dead] = INIT_E
        self.cage[dead] = 0; self.cgen[dead] = 0; self.cflash[dead] = 0
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
            self.vpos[va] += self.vvel[va] + torch.randn(va.sum(), 2) * .2
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
                    blend = torch.where(recent > 3, torch.tensor(0.5), torch.tensor(0.3))
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
        attn_mask = torch.zeros(N, SL, dtype=torch.bool)
        # Self token is never masked (position 0)

        if k == 0:
            ns = torch.zeros(N, K_N, DIM)
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
        ves_tokens = torch.zeros(N, VES_TOKENS, DIM)
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

        # ── Change 4: Aging pressure ──
        ages = self.cage[idx].float()  # (N,)
        extra_drain = ((ages - 2500).clamp(min=0) / 5000.0) * 0.05
        self.cenergy[idx] -= extra_drain

        # ── Change 4: Crowding penalty ──
        # dd already computed with diagonal = 1e9
        if N > 1:
            nearest_d, _ = dd.min(dim=1)  # (N,)
            crowded = nearest_d < 15.0
            if crowded.any():
                self.cenergy[idx[crowded]] -= 0.005

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

                    # Change 9: Nutrient depletion modulates gain
                    ne = self.nut_energy[ni].item()
                    if ne < 10.0:
                        gain = gain * (ne / 10.0)

                    self.cenergy[idx[mask]] += gain

                    # Change 9: Deplete nutrient energy
                    total_gain = gain.sum().item()
                    self.nut_energy[ni] = (self.nut_energy[ni] - total_gain * 0.01).clamp(min=0)

        # ── Emit vesicles (vectorized) ──
        # Change 1 (Bug Fix): Gate emission on ves_on
        if self.ves_on:
            # Suppress emission when starving (energy < 40)
            emit_gate = (self.cenergy[idx] > 40).float()
            emit_roll = torch.rand(N) < (emit_p * emit_gate)
            if emit_roll.any():
                e_idx = idx[emit_roll]
                n_emit = emit_roll.sum().item()
                dead_v = (~self.valive).nonzero(as_tuple=True)[0]
                n_emit = min(n_emit, len(dead_v))
                if n_emit > 0:
                    e_idx = e_idx[:n_emit]
                    dv = dead_v[:n_emit]
                    ang = torch.rand(n_emit) * (2 * math.pi)
                    self.vprev[dv] = self.cpos[e_idx].clone()  # init prev to spawn pos
                    self.vpos[dv] = self.cpos[e_idx]
                    self.vvel[dv] = torch.stack([ang.cos(), ang.sin()], -1) * VSPD
                    self.vcont[dv] = vc_out[emit_roll][:n_emit]
                    self.vlife[dv] = VLIFE
                    self.valive[dv] = True
                    self.cenergy[e_idx] -= VCOST

        # ── Division (vectorized) ──
        div_mask = self.cenergy[idx] > DIV_E
        if div_mask.any():
            d_idx = idx[div_mask]
            dead_c = (~self.calive).nonzero(as_tuple=True)[0]
            nd = min(len(d_idx), len(dead_c))
            if nd > 0:
                d_idx = d_idx[:nd]; dc = dead_c[:nd]
                self.calive[dc] = True
                self.cpos[dc] = (self.cpos[d_idx] + torch.randn(nd, 2) * 15) % SWt
                self.cvel[dc] = 0
                self.cstate[dc] = self.cstate[d_idx].clone()
                self.cgenome[dc] = self.cgenome[d_idx] + torch.randn(nd, GDIM) * MUT
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


# ━━ Renderer (Enhanced) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class Ren:
    def __init__(self):
        pygame.init()
        self.scr = pygame.display.set_mode((TW, TH))
        pygame.display.set_caption("ALife v5 — Enhanced Organic Renderer")
        self.clk = pygame.time.Clock()
        self.f11 = pygame.font.SysFont("Menlo", 11)
        self.f13 = pygame.font.SysFont("Menlo", 13)
        self.f15 = pygame.font.SysFont("Menlo", 15)
        self.trail = pygame.Surface((SW, TH))
        self.trail.fill(BG)

        # [15] Preallocated surfaces
        self.fade_surf = pygame.Surface((SW, TH))
        self.fade_surf.fill(BG)
        self.fade_surf.set_alpha(18)
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

    def _make_nutrient_sprite(self, nut):
        """Create a cloudy sprite using overlapping translucent circles."""
        size = int(nut.radius * 2.4)
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        cx, cy = size // 2, size // 2
        col = nut.bright
        # 7 layers of offset translucent circles for cloud effect
        rng = random.Random(int(nut.x * 100 + nut.y))
        for layer in range(7):
            ox = rng.randint(-int(nut.radius * 0.15), int(nut.radius * 0.15))
            oy = rng.randint(-int(nut.radius * 0.15), int(nut.radius * 0.15))
            scale = 0.5 + layer * 0.1
            r = int(nut.radius * scale)
            a = max(5, 35 - layer * 4)
            pygame.draw.circle(surf, (*col, a), (cx + ox, cy + oy), r)
        # Bright core
        pygame.draw.circle(surf, (*col, 55), (cx, cy), int(nut.radius * 0.25))
        pygame.draw.circle(surf, (255, 255, 255, 30), (cx, cy), int(nut.radius * 0.1))
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
        if self.show_panel:
            self._sim(w, sim_w=SW)
            self._panel(w)
        else:
            self._sim(w, sim_w=TW)
        self._hud(w, paused, speed)
        pygame.display.flip()

    def _sim(self, w, sim_w=SW):
        # Fade trail — reuse preallocated surface
        self.trail.blit(self.fade_surf, (0, 0))

        # [16] Batch tensor reads to numpy
        ai = w.calive.nonzero(as_tuple=True)[0]
        N = len(ai)
        if N > 0:
            pos_np = w.cpos[ai].numpy()
            state_np = w.cstate[ai].numpy()
            energy_np = w.cenergy[ai].numpy()
            vel_np = w.cvel[ai].numpy()
            age_np = w.cage[ai].long().numpy()
            flash_np = w.cflash[ai].numpy()
            genome_np = w.cgenome[ai].numpy()
        else:
            pos_np = state_np = energy_np = vel_np = age_np = flash_np = genome_np = None

        # Same for vesicles
        va = w.valive.nonzero(as_tuple=True)[0]
        nv = len(va)
        if nv > 0:
            vpos_np = w.vpos[va].numpy()
            vvel_np = w.vvel[va].numpy()
            vcont_np = w.vcont[va].numpy()
            vlife_np = w.vlife[va].numpy()
        else:
            vpos_np = vvel_np = vcont_np = vlife_np = None

        # ── Vesicle trails — tapered streaks [12] ──
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
                    # Draw streak from tail to head
                    pygame.draw.line(self.trail, col, (tail_x, tail_y), (x, y),
                                     max(1, int(2 * t)))
                    # Bright head dot
                    bright_col = (min(255, col[0] + 60),
                                  min(255, col[1] + 60),
                                  min(255, col[2] + 60))
                    pygame.draw.circle(self.trail, bright_col, (x, y),
                                       max(1, int(1.5 * t)))
                else:
                    pygame.draw.circle(self.trail, col, (x, y),
                                       max(1, int(2 * t + .5)))

        # Cell glow on trail — subtle
        if N > 0:
            for ci in range(N):
                x, y = int(pos_np[ci, 0]), int(pos_np[ci, 1])
                e = energy_np[ci]
                col = pheno_rgb_np(state_np[ci, :4], .15 + .2 * min(1., e / DIV_E))
                pygame.draw.circle(self.trail, col, (x, y), 10)

        self.scr.blit(self.trail, (0, 0))

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
        self.scr.blit(self.nut_surf, (0, 0))

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
            self.scr.blit(ripple_surf, (0, 0))
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

                # ── Lens layers (LOD-aware) ──
                if n_layers >= 5:
                    # Full: haze + mid + inner + core + membrane
                    haze_pts = _membrane_pts(r + 6, 0.5)
                    pygame.draw.polygon(self.cell_surf, (*col, int(8 + fl * 15)), haze_pts)
                    mid_pts = _membrane_pts(r + 1, 0.8)
                    pygame.draw.polygon(self.cell_surf, (*col, int(20 + 15 * t)), mid_pts)
                    inner_pts = _membrane_pts(r * 0.7, 0.6)
                    pygame.draw.polygon(self.cell_surf, (*col, int(35 + 25 * t + fl * 20)), inner_pts)
                elif n_layers >= 4:
                    # No haze, mid + inner
                    mid_pts = _membrane_pts(r + 1, 0.8)
                    pygame.draw.polygon(self.cell_surf, (*col, int(22 + 18 * t)), mid_pts)
                    inner_pts = _membrane_pts(r * 0.7, 0.6)
                    pygame.draw.polygon(self.cell_surf, (*col, int(40 + 25 * t + fl * 20)), inner_pts)
                else:
                    # Minimal: just body fill
                    pygame.draw.polygon(self.cell_surf, (*col, int(35 + 30 * t + fl * 20)), body_pts)

                # Core (always drawn — the "focal point")
                core_col = (min(255, col[0]+30), min(255, col[1]+30), min(255, col[2]+30))
                pygame.draw.circle(self.cell_surf, (*core_col, int(50 + 30 * t)),
                                   (x, y), max(2, int(r * 0.35)))

                # Membrane outline (always drawn)
                mc = (min(255, col[0] + 60 + int(fl * 80)),
                      min(255, col[1] + 60 + int(fl * 80)),
                      min(255, col[2] + 60 + int(fl * 80)))
                pygame.draw.polygon(self.cell_surf, (*mc, int(100 + 80 * t)), body_pts, 1)

                # Specular highlight (skip secondary in fast mode)
                spec_angle = math.atan2(dy_n, dx_n) if spd > 0.3 else (age * 0.01 + idx_val)
                spec_r = r * 0.3
                spec_x = x + int(math.cos(spec_angle + 0.8) * spec_r)
                spec_y = y + int(math.sin(spec_angle + 0.8) * spec_r)
                spec_pulse = math.sin(age * 0.1 + idx_val * 0.7) * 0.2 + 0.8
                spec_a = int((60 + 80 * spec_pulse + fl * 50) * t)
                spec_sz = max(2, int(r * 0.2))
                pygame.draw.circle(self.cell_surf, (255, 255, 255, min(200, spec_a)),
                                   (spec_x, spec_y), spec_sz)
                if n_layers >= 4:
                    spec2_x = x - int(math.cos(spec_angle + 0.3) * spec_r * 0.6)
                    spec2_y = y - int(math.sin(spec_angle + 0.3) * spec_r * 0.6)
                    pygame.draw.circle(self.cell_surf, (255, 255, 240, min(120, spec_a // 2)),
                                       (spec2_x, spec2_y), max(1, spec_sz - 1))

                # ── Division-ready: pulsing outer ring ──
                if e > DIV_E * .85:
                    da = int(30 + 25 * math.sin(age * 0.15))
                    div_pts = _membrane_pts(r + 4, 0.6)
                    pygame.draw.polygon(self.cell_surf, (255, 255, 200, da), div_pts, 1)

        self.scr.blit(self.cell_surf, (0, 0))

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

            gvals = torch.tanh(w.cgenome[i]).tolist()
            pvals = torch.tanh(w.cstate[i, :16]).tolist()
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
        self.scr.blit(self.f11.render(f"t={w.t}  cells={n}  ves={nv}  x{speed}", True, (120,120,150)), (10, y)); y += 15
        if n > 0:
            ai = w.calive.nonzero(as_tuple=True)[0]
            ae = w.cenergy[ai].mean().item()
            mg = w.cgen[ai].max().item()
            ag = w.cgen[ai].float().mean().item()
            self.scr.blit(self.f11.render(f"gen={ag:.1f}/{mg}  E={ae:.0f}  +{w.births}/-{w.deaths}",
                                          True, (120,120,150)), (10, y)); y += 15
        self.scr.blit(self.f13.render(f"VESICLE: {vl}", True, vc), (10, y))
        if paused:
            sw_actual = SW if self.show_panel else TW
            self.scr.blit(self.f15.render("|| PAUSED", True, (255,200,100)), (sw_actual//2-50, 10))
        self.scr.blit(self.f11.render(
            "SPC:pause  V:vesicle  D:panel  R:reset  +/-:speed  click:nutrient  ESC:quit",
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
                elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS): speed = min(5, speed+1)
                elif ev.key == pygame.K_MINUS: speed = max(1, speed-1)
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                mx, my = ev.pos
                sim_w = SW if r.show_panel else TW
                if mx < sim_w:
                    w.add_nutrient(mx, my)
                    r.nut_sprites = {}  # force sprite regeneration

        if not paused:
            for _ in range(speed):
                w.step()

        r.draw(w, paused, speed)
        r.clk.tick(FPS)

if __name__ == "__main__":
    main()
