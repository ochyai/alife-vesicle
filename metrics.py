"""
MetricsCollector — per-tick and periodic metrics from World for paper analysis.
"""

import torch, torch.nn.functional as F, json, os, time
from collections import defaultdict


class MetricsCollector:
    """Collect per-tick and periodic metrics from World for paper analysis."""

    def __init__(self, world, interval=10):
        """
        world: World instance
        interval: collect detailed metrics every N ticks
        """
        self.w = world
        self.interval = interval
        self.data = defaultdict(list)  # metric_name -> list of (tick, value)
        self._start_time = time.time()

    def collect(self):
        """Call every tick. Collects basic metrics every tick, detailed metrics every `interval` ticks."""
        w = self.w
        t = w.t
        idx = w.calive.nonzero(as_tuple=True)[0]
        N = len(idx)

        # === Every tick: population dynamics ===
        self.data['pop_count'].append((t, N))
        self.data['births_cum'].append((t, w.births))
        self.data['deaths_cum'].append((t, w.deaths))

        if N < 2:
            return

        # === Every interval: detailed metrics ===
        if t % self.interval != 0:
            return

        # --- Energy ---
        energy = w.cenergy[idx]
        self.data['energy_mean'].append((t, energy.mean().item()))
        self.data['energy_std'].append((t, energy.std().item()))

        # --- Age ---
        ages = w.cage[idx].float()
        self.data['age_mean'].append((t, ages.mean().item()))
        self.data['age_max'].append((t, ages.max().item()))

        # --- Generation ---
        gens = w.cgen[idx].float()
        self.data['gen_mean'].append((t, gens.mean().item()))
        self.data['gen_max'].append((t, gens.max().item()))

        # --- Genetic diversity (genome variance + pairwise cosine distance) ---
        genomes = w.cgenome[idx]  # (N, GDIM)
        genome_var = genomes.var(0).sum().item()
        self.data['genome_variance'].append((t, genome_var))

        # Pairwise cosine distance (sample if N > 200 for speed)
        if N > 200:
            sample_idx = torch.randperm(N, device=idx.device)[:200]
            g_sample = genomes[sample_idx]
        else:
            sample_idx = None
            g_sample = genomes
        g_norm = F.normalize(g_sample, dim=1)
        cos_sim = (g_norm @ g_norm.T)
        # Exclude diagonal
        mask = ~torch.eye(len(g_sample), dtype=torch.bool, device=cos_sim.device)
        mean_cos = cos_sim[mask].mean().item()
        self.data['genome_cos_sim_mean'].append((t, mean_cos))

        # --- Phenotypic diversity (state variance) ---
        states = w.cstate[idx]
        pheno_var = states.var(0).sum().item()
        self.data['pheno_variance'].append((t, pheno_var))

        # Phenotype pairwise cosine
        if N > 200:
            s_sample = states[sample_idx]
        else:
            s_sample = states
        s_norm = F.normalize(s_sample, dim=1)
        ps_sim = (s_norm @ s_norm.T)
        ps_mask = ~torch.eye(len(s_sample), dtype=torch.bool, device=ps_sim.device)
        self.data['pheno_cos_sim_mean'].append((t, ps_sim[ps_mask].mean().item()))

        # --- Speciation: count genetic clusters ---
        # Count nutrient clusters with >= 5 cells
        if hasattr(w, 'cclust'):
            clusters = w.cclust[idx]
            unique_clusters = clusters.unique()
            active_clusters = sum(1 for c in unique_clusters if (clusters == c).sum().item() >= 5)
            self.data['active_clusters'].append((t, active_clusters))

        # --- Vesicle metrics ---
        n_ves = w.valive.sum().item()
        self.data['vesicle_count'].append((t, n_ves))

        if n_ves > 0:
            va_idx = w.valive.nonzero(as_tuple=True)[0]
            ves_content = w.vcont[va_idx]
            # Content entropy approximation: variance of content vectors
            ves_var = ves_content.var(0).sum().item()
            self.data['vesicle_content_var'].append((t, ves_var))
            # Mean content magnitude
            ves_mag = ves_content.norm(dim=1).mean().item()
            self.data['vesicle_content_mag'].append((t, ves_mag))

        # --- Nutrient energy ---
        if w.nuts:
            n_nuts = len(w.nuts)
            nut_e = w.nut_energy[:n_nuts]
            self.data['nutrient_energy_mean'].append((t, nut_e.mean().item()))
            self.data['nutrient_energy_min'].append((t, nut_e.min().item()))

        # --- Spatial metrics ---
        # Mean nearest-neighbor distance
        from main import wrapped_dist
        pos = w.cpos[idx]
        dd = wrapped_dist(pos, pos)
        dd.fill_diagonal_(1e9)
        nn_dist = dd.min(dim=1).values
        self.data['nn_dist_mean'].append((t, nn_dist.mean().item()))
        self.data['nn_dist_std'].append((t, nn_dist.std().item()))

        # --- Attention analysis (expensive, do every 100 ticks) ---
        if t % 100 == 0 and N > 10:
            self._collect_attention(idx, N)

    def _collect_attention(self, idx, N):
        """Analyze attention patterns of the brain."""
        w = self.w
        t = w.t
        from main import wrapped_dist, wrapped_diff, K_N, VES_TOKENS, SL, DIM, GDIM, ARAD, SWt, DEV, NH

        # Reconstruct attention computation (mirrors World.step logic)
        p = w.cpos[idx]
        s = w.cstate[idx]
        g = w.cgenome[idx]
        dd = wrapped_dist(p, p)
        dd.fill_diagonal_(1e9)
        k = min(K_N, N - 1)

        if k == 0:
            return

        _, nidx = dd.topk(k, largest=False)
        ns = s[nidx]  # (N, k, DIM)

        # Relational features
        vel = w.cvel[idx]
        rel_pos = wrapped_diff(p, p)  # (N, N, 2)
        nidx_exp = nidx.unsqueeze(-1).expand(-1, -1, 2)
        rel_xy = torch.gather(rel_pos, 1, nidx_exp)  # (N, k, 2)
        neigh_d = torch.gather(dd, 1, nidx)  # (N, k)
        neigh_vel = vel[nidx]  # (N, k, 2)
        rel_vel = vel.unsqueeze(1) - neigh_vel  # (N, k, 2)
        rel_feats = torch.cat([
            rel_xy[:, :, 0:1] / 100.0,
            rel_xy[:, :, 1:2] / 100.0,
            neigh_d.unsqueeze(-1) / 200.0,
            rel_vel[:, :, 0:1] / 5.0,
            rel_vel[:, :, 1:2] / 5.0,
        ], dim=-1)  # (N, k, 5)
        rel_embed = w.brain.rel_proj(rel_feats)  # (N, k, DIM)
        ns = ns + rel_embed

        if k < K_N:
            ns = F.pad(ns, (0, 0, 0, K_N - k))

        # Build attention mask
        attn_mask = torch.zeros(N, SL, dtype=torch.bool, device=DEV)
        if k < K_N:
            attn_mask[:, 1 + k:1 + K_N] = True
        attn_mask[:, 1 + K_N:] = True  # mask vesicle tokens (not reconstructed here)

        # Forward through brain components to get attention weights
        brain = w.brain
        hd = brain.hd  # DIM // NH = 8

        x = brain.ln(s + brain.gp(g))  # (N, DIM)
        self_tok = x.unsqueeze(1)  # (N, 1, DIM)
        ves_tokens = torch.zeros(N, VES_TOKENS, DIM, device=DEV)
        seq = torch.cat([self_tok, ns, ves_tokens], dim=1)  # (N, SL, DIM)

        # Q from self token only
        Q = brain.Wq(x).view(N, 1, NH, hd).permute(0, 2, 1, 3)  # (N, NH, 1, hd)
        # K from full sequence
        Kk = brain.Wk(seq).view(N, SL, NH, hd).permute(0, 2, 1, 3)  # (N, NH, SL, hd)

        # Genome-based Q/K scaling
        scales = torch.sigmoid(brain.genome_qk(g))  # (N, NH*2)
        q_scale = scales[:, :NH].view(N, NH, 1, 1)
        k_scale = scales[:, NH:].view(N, NH, 1, 1)
        Q = Q * q_scale
        Kk = Kk * k_scale

        # Compute attention scores
        attn = (Q @ Kk.transpose(-2, -1)) * (hd ** -0.5)  # (N, NH, 1, SL)

        # Apply mask
        mask_expanded = attn_mask.unsqueeze(1).unsqueeze(2)  # (N, 1, 1, SL)
        attn = attn.masked_fill(mask_expanded, -1e9)

        attn_weights = attn.softmax(-1)  # (N, NH, 1, SL)

        # Average attention over all cells: squeeze the query dim
        avg_attn = attn_weights.squeeze(2).mean(0)  # (NH, SL)

        # Store per-head attention distribution
        for h in range(NH):
            self_attn = avg_attn[h, 0].item()
            neigh_attn = avg_attn[h, 1:1 + k].sum().item()
            ves_attn = avg_attn[h, 1 + K_N:].sum().item()
            self.data[f'attn_h{h}_self'].append((t, self_attn))
            self.data[f'attn_h{h}_neigh'].append((t, neigh_attn))
            self.data[f'attn_h{h}_ves'].append((t, ves_attn))

    def save(self, path):
        """Save all metrics to JSON file."""
        out = {}
        for k, v in self.data.items():
            out[k] = v  # list of (tick, value) tuples
        out['_meta'] = {
            'total_ticks': self.w.t,
            'wall_time_sec': time.time() - self._start_time,
        }
        with open(path, 'w') as f:
            json.dump(out, f)
        print(f"[Metrics] Saved {len(self.data)} metrics ({sum(len(v) for v in self.data.values())} data points) to {path}")

    @staticmethod
    def load(path):
        """Load metrics from JSON file."""
        with open(path) as f:
            return json.load(f)
