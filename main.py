"""
ALife v5: Fully-Tensorized Transformer Vesicle System
=====================================================
500+ cells, ALL data in PyTorch tensors, zero Python loops in simulation.
Brain = shared single multi-head attention layer (4 heads) over K=6 neighbors.
Vesicle exchange = cultural transfer (state blending).
Genome (16-dim DNA) mutates on division; State (32-dim phenotype) changes each step.

DNA panel: dual barcodes — DNA (inherited) vs PHENO (expressed).
  V-toggle: PHENO converges (ON) / diverges (OFF). DNA unchanged.

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
ADIM = 2 + 1 + 1 + DIM  # dx,dy,emit,alpha,vesicle_content

# ━━ Sim ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MAX_C = 700; MAX_V = 3000; INIT_N = 300
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


# ━━ Brain: single multi-head attention layer ━━━━━━━━━━━━━━
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

    @torch.no_grad()
    def forward(self, s, g, ns):
        """s:(N,DIM) g:(N,GDIM) ns:(N,K,DIM) → (N,ADIM)"""
        x = self.ln(s + self.gp(g))
        N, Kn, D = ns.shape
        Q = self.Wq(x).view(N, 1, NH, self.hd).permute(0,2,1,3)
        Kk = self.Wk(ns).view(N, Kn, NH, self.hd).permute(0,2,1,3)
        Vv = self.Wv(ns).view(N, Kn, NH, self.hd).permute(0,2,1,3)
        attn = (Q @ Kk.transpose(-2,-1)) * (self.hd**-.5)
        ctx = (attn.softmax(-1) @ Vv).permute(0,2,1,3).reshape(N, D)
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


# ━━ World (fully tensorized) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
        # Vesicle pool
        self.vpos = torch.zeros(MAX_V, 2)
        self.vvel = torch.zeros(MAX_V, 2)
        self.vcont = torch.zeros(MAX_V, DIM)
        self.vlife = torch.zeros(MAX_V)
        self.valive = torch.zeros(MAX_V, dtype=torch.bool)
        # Meta
        self.nuts = []
        self.t = 0; self.births = 0; self.deaths = 0
        self.ves_on = True
        self.div_hist = []
        self._init()

    def _init(self):
        self.calive[:] = False; self.valive[:] = False
        self.t = self.births = self.deaths = 0
        self.div_hist = []
        self.nuts = [Nut(random.uniform(80,SW-80), random.uniform(80,TH-80)) for _ in range(NNUTS)]
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

    def step(self):
        self.t += 1
        idx = self.calive.nonzero(as_tuple=True)[0]
        N = len(idx)
        if N == 0: self._spawn(20); return

        # ── Vesicle physics (vectorized) ──
        va = self.valive
        if va.any():
            self.vpos[va] += self.vvel[va] + torch.randn(va.sum(), 2) * .2
            self.vpos[va] %= SWt
            self.vvel[va] *= .994
            self.vlife[va] -= 1
            self.valive &= (self.vlife > 0)

        # ── Vesicle absorption (vectorized) ──
        if self.ves_on:
            va_idx = self.valive.nonzero(as_tuple=True)[0]
            if len(va_idx) > 0 and N > 0:
                cv_d = wrapped_dist(self.cpos[idx], self.vpos[va_idx])  # (N, M)
                # For each cell, nearest vesicle
                min_d, min_vi = cv_d.min(dim=1)        # (N,)
                absorb = min_d < ARAD                   # (N,) bool
                if absorb.any():
                    a_cells = idx[absorb]
                    a_ves = va_idx[min_vi[absorb]]
                    # Cultural transfer: blend vesicle → state
                    self.cstate[a_cells] = .7 * self.cstate[a_cells] + .3 * self.vcont[a_ves]
                    self.cflash[a_cells] = 5.
                    self.valive[a_ves] = False

        # ── Neighbor finding ──
        p = self.cpos[idx]                             # (N, 2)
        s = self.cstate[idx]                           # (N, DIM)
        g = self.cgenome[idx]                          # (N, GDIM)
        dd = wrapped_dist(p, p)                         # (N, N)
        dd.fill_diagonal_(1e9)
        k = min(K_N, N - 1)
        if k == 0:
            ns = torch.zeros(N, K_N, DIM)
        else:
            _, nidx = dd.topk(k, largest=False)         # (N, k)
            ns = s[nidx]                                # (N, k, DIM)
            if k < K_N:
                ns = F.pad(ns, (0, 0, 0, K_N - k))

        # ── Brain inference (batched, single forward) ──
        act = self.brain(s, g, ns)                      # (N, ADIM)

        dxy = torch.tanh(act[:, :2]) * 2.5
        emit_p = torch.sigmoid(act[:, 2])
        alpha = torch.sigmoid(act[:, 3]) * .25
        vc_out = act[:, 4:]                             # (N, DIM)

        # ── Movement ──
        new_vel = dxy * .6 + self.cvel[idx] * .4
        self.cpos[idx] = (p + new_vel) % SWt
        self.cvel[idx] = new_vel

        # ── State update ──
        self.cstate[idx] = (1 - alpha.unsqueeze(1)) * s + alpha.unsqueeze(1) * vc_out

        # ── Energy ──
        speed = new_vel.norm(dim=1)
        self.cenergy[idx] += PASSIVE_REGEN - DECAY - MCOST * speed
        self.cage[idx] += 1
        self.cflash[idx] = (self.cflash[idx] - 1).clamp(min=0)

        # ── Nutrient interaction ──
        new_p = self.cpos[idx]
        for nut in self.nuts:
            np_ = torch.tensor([nut.x, nut.y])
            d = (new_p - np_).abs()
            d = torch.min(d, SWt - d)
            nd = d.norm(dim=1)                          # (N,)
            mask = nd < nut.radius
            if mask.any():
                sim = F.cosine_similarity(vc_out[mask], nut.sig.unsqueeze(0))
                # Base gain always positive; sim bonus on top (gentle selection)
                gain = NGAIN * (0.6 + 0.4 * sim) * (1 - nd[mask] / nut.radius).clamp(min=0)
                self.cenergy[idx[mask]] += gain

        # ── Emit vesicles (vectorized) ──
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
                nps = torch.tensor([[n.x, n.y] for n in self.nuts])
                cd = wrapped_dist(ap, nps)
                self.cclust[alive_idx] = cd.argmin(1)

        # Diversity
        if self.t % 6 == 0:
            ai = self.calive.nonzero(as_tuple=True)[0]
            if len(ai) > 1:
                self.div_hist.append(self.cstate[ai].var(0).sum().item())
                if len(self.div_hist) > 400:
                    self.div_hist = self.div_hist[-400:]


# ━━ Renderer ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class Ren:
    def __init__(self):
        pygame.init()
        self.scr = pygame.display.set_mode((TW, TH))
        pygame.display.set_caption("ALife v5 — Dense Vectorized Transformer Vesicles")
        self.clk = pygame.time.Clock()
        self.f11 = pygame.font.SysFont("Menlo", 11)
        self.f13 = pygame.font.SysFont("Menlo", 13)
        self.f15 = pygame.font.SysFont("Menlo", 15)
        self.trail = pygame.Surface((SW, TH))
        self.trail.fill(BG)

    def draw(self, w, paused, speed):
        self._sim(w)
        self._panel(w)
        self._hud(w, paused, speed)
        pygame.display.flip()

    def _sim(self, w):
        # Fade trail — moderate, keep background clean for transparency
        f = pygame.Surface((SW, TH)); f.fill(BG); f.set_alpha(18)
        self.trail.blit(f, (0, 0))

        # Vesicle trails — delicate streaks
        va = w.valive.nonzero(as_tuple=True)[0]
        for i in va:
            t = w.vlife[i].item() / VLIFE
            x, y = int(w.vpos[i, 0].item()), int(w.vpos[i, 1].item())
            col = pheno_rgb(w.vcont[i], .5 + .4 * t)
            pygame.draw.circle(self.trail, col, (x, y), max(1, int(2 * t + .5)))

        # Cell glow on trail — subtle, restrained (clean background = transparency)
        ai = w.calive.nonzero(as_tuple=True)[0]
        for i in ai:
            x, y = int(w.cpos[i, 0].item()), int(w.cpos[i, 1].item())
            e = w.cenergy[i].item()
            col = pheno_rgb(w.cstate[i], .15 + .2 * min(1., e / DIV_E))
            pygame.draw.circle(self.trail, col, (x, y), 10)

        self.scr.blit(self.trail, (0, 0))

        # ── Nutrient zones — soft translucent pools ──
        ns = pygame.Surface((SW, TH), pygame.SRCALPHA)
        for nut in w.nuts:
            c = nut.bright
            nx, ny = int(nut.x), int(nut.y)
            pulse = 0.92 + 0.08 * math.sin(w.t * 0.025 + nut.x * 0.01)
            rr = int(nut.radius * pulse)
            for m, a in [(1.1, 12), (.85, 25), (.6, 40), (.35, 60), (.15, 85)]:
                pygame.draw.circle(ns, (*c, a), (nx, ny), int(rr * m))
            pygame.draw.circle(ns, (*c, 120), (nx, ny), 8)
            pygame.draw.circle(ns, (255, 255, 255, 50), (nx, ny), 3)
            pygame.draw.circle(ns, (*c, 22), (nx, ny), rr, 1)
        self.scr.blit(ns, (0, 0))

        # ── Cells — translucent, breathing, elastic ──
        cs = pygame.Surface((SW, TH), pygame.SRCALPHA)
        for i in ai:
            x, y = int(w.cpos[i, 0].item()), int(w.cpos[i, 1].item())
            e = w.cenergy[i].item()
            vx, vy = w.cvel[i, 0].item(), w.cvel[i, 1].item()
            fl = w.cflash[i].item() / 8
            age = w.cage[i].item()
            t = min(1., e / DIV_E)

            # ── Breathing: radius pulses per-cell rhythm ──
            breath = math.sin(age * 0.08 + i * 1.3) * 0.12
            r_base = 6 + 9 * t
            r = max(4, int(r_base * (1 + breath)))

            bri = .55 + .4 * t
            col = pheno_rgb(w.cstate[i], bri)

            # ── Elastic squish: stretch along movement ──
            spd = math.sqrt(vx*vx + vy*vy)
            stretch_pts = []
            if spd > 0.4:
                dx, dy = vx / spd, vy / spd
                stretch = min(0.3, spd * 0.1)  # up to 30% elongation
                # Leading and trailing blobs for squish illusion
                lx = x + int(dx * r * stretch)
                ly = y + int(dy * r * stretch)
                tx = x - int(dx * r * stretch * 0.5)
                ty = y - int(dy * r * stretch * 0.5)
                stretch_pts = [(lx, ly, int(r * 0.85)), (tx, ty, int(r * 0.7))]

            # ── Layer 1: wide diffuse glow (translucent) ──
            pygame.draw.circle(cs, (*col, int(12 + fl * 25)), (x, y), r + 8)

            # ── Layer 2: outer body (very translucent) ──
            pygame.draw.circle(cs, (*col, int(30 + 15 * t)), (x, y), r + 2)

            # ── Layer 3: squish blobs (elastic motion) ──
            for sx, sy, sr in stretch_pts:
                pygame.draw.circle(cs, (*col, 35), (sx, sy), sr)

            # ── Layer 4: inner body (semi-translucent — can see through) ──
            pygame.draw.circle(cs, (*col, int(55 + 35 * t + fl * 30)), (x, y), r - 1)

            # ── Layer 5: membrane — the sharpest element ──
            mc = (min(255, col[0] + 50 + int(fl * 80)),
                  min(255, col[1] + 50 + int(fl * 80)),
                  min(255, col[2] + 50 + int(fl * 80)))
            pygame.draw.circle(cs, (*mc, int(140 + 60 * t)), (x, y), r, 1)

            # ── Layer 6: floating nucleus (bobs inside cell) ──
            nox = math.sin(age * 0.05 + i * 2.1) * (r * 0.2)
            noy = math.cos(age * 0.07 + i * 1.7) * (r * 0.2)
            nx_n = x + int(nox)
            ny_n = y + int(noy)
            npulse = math.sin(age * 0.12 + i * 0.7) * 0.3 + 0.7
            na = int(80 + 100 * npulse + fl * 60)
            nr = max(2, int(r * .25))
            pygame.draw.circle(cs, (255, 255, 255, min(220, na)), (nx_n, ny_n), nr)
            # Nucleus inner glow
            pygame.draw.circle(cs, (255, 255, 240, min(180, na - 20)), (nx_n, ny_n), max(1, nr - 1))

            # ── Division-ready: soft pulsing halo ──
            if e > DIV_E * .85:
                da = int(40 + 35 * math.sin(age * 0.15))
                pygame.draw.circle(cs, (255, 255, 200, da), (x, y), r + 5, 1)

        self.scr.blit(cs, (0, 0))

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
            self.scr.blit(self.f15.render("|| PAUSED", True, (255,200,100)), (SW//2-50, 10))
        self.scr.blit(self.f11.render(
            "SPC:pause  V:vesicle  R:reset  +/-:speed  click:nutrient  ESC:quit",
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
                elif ev.key == pygame.K_r: w = World(); r.trail.fill(BG)
                elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS): speed = min(5, speed+1)
                elif ev.key == pygame.K_MINUS: speed = max(1, speed-1)
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                mx, my = ev.pos
                if mx < SW: w.nuts.append(Nut(mx, my))

        if not paused:
            for _ in range(speed):
                w.step()

        r.draw(w, paused, speed)
        r.clk.tick(FPS)

if __name__ == "__main__":
    main()
