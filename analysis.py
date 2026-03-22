#!/usr/bin/env python3
"""
Analysis and figure generation for ALife vesicle paper.
Usage:
    python analysis.py results/          # analyze all results in directory
    python analysis.py results/ --paper  # generate paper-ready figures
"""
import json, os, sys, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
from collections import defaultdict

# Paper style
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
})

# Condition colors (colorblind-friendly)
COLORS = {
    'baseline': '#2196F3',       # blue
    'no_vesicle': '#F44336',     # red
    'no_genome_mod': '#FF9800',  # orange
    'no_sexual': '#9C27B0',      # purple
    'no_predation': '#4CAF50',   # green
    'no_speciation_pressure': '#795548',  # brown
    'no_aging': '#607D8B',       # gray-blue
    'minimal': '#E91E63',        # pink
}

LABELS = {
    'baseline': 'Full System',
    'no_vesicle': 'No Vesicle',
    'no_genome_mod': 'No Genome Mod',
    'no_sexual': 'No Sexual',
    'no_predation': 'No Predation',
    'no_speciation_pressure': 'No Speciation',
    'no_aging': 'No Aging',
    'minimal': 'Minimal',
}


# Mapping from experiment.py record keys to analysis metric names
RECORD_KEY_MAP = {
    'n_cells': 'pop_count',
    'births': 'births_cum',
    'deaths': 'deaths_cum',
    'mean_energy': 'energy_mean',
    'max_energy': 'energy_max',
    'min_energy': 'energy_min',
    'mean_age': 'age_mean',
    'max_gen': 'gen_max',
    'mean_gen': 'gen_mean',
    'pheno_diversity': 'pheno_variance',
    'genome_diversity': 'genome_variance',
    'n_vesicles': 'vesicle_count',
}


def _convert_records_to_dict(records):
    """Convert experiment.py record-array format to {metric: [[tick,val],...]} format."""
    out = defaultdict(list)
    for rec in records:
        t = rec.get('step', 0)
        for rec_key, metric_key in RECORD_KEY_MAP.items():
            if rec_key in rec:
                out[metric_key].append((t, rec[rec_key]))
    return dict(out)


def load_results(results_dir):
    """Load all experiment results from directory.
    Returns: dict of {condition_name: [list of metric dicts per seed]}
    Supports both formats:
      - experiment.py: list of record dicts [{step, n_cells, ...}, ...]
      - metrics.py:    dict of {metric_name: [[tick, value], ...]}
    """
    results = defaultdict(list)
    for fpath in sorted(glob.glob(os.path.join(results_dir, '*_seed*.json'))):
        fname = os.path.basename(fpath)
        if '_summary' in fname:
            continue
        parts = fname.rsplit('_seed', 1)
        if len(parts) != 2:
            continue
        cond_name = parts[0]
        with open(fpath) as f:
            data = json.load(f)
        # Auto-detect format
        if isinstance(data, list):
            data = _convert_records_to_dict(data)
        results[cond_name].append(data)
    return dict(results)


def extract_timeseries(data_list, metric_name):
    """Extract timeseries from list of metric dicts.
    Returns: (ticks, mean_values, std_values) arrays
    """
    all_series = []
    for data in data_list:
        if metric_name not in data:
            continue
        series = data[metric_name]  # list of [tick, value] or (tick, value)
        all_series.append({t: v for t, v in series})

    if not all_series:
        return None, None, None

    # Get common ticks
    all_ticks = sorted(set().union(*[s.keys() for s in all_series]))

    # Build aligned arrays
    values = []
    for s in all_series:
        row = [s.get(t, np.nan) for t in all_ticks]
        values.append(row)

    values = np.array(values)
    ticks = np.array(all_ticks)

    with np.errstate(all='ignore'):
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)

    return ticks, mean, std


def smooth(y, window=50):
    """Moving average smoothing."""
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode='same')


def plot_metric_comparison(results, metric_name, title, ylabel,
                           out_path, conditions=None, smooth_window=50,
                           figsize=(8, 4)):
    """Plot one metric across multiple conditions with mean+-std bands."""
    fig, ax = plt.subplots(figsize=figsize)

    if conditions is None:
        conditions = list(results.keys())

    for cond in conditions:
        if cond not in results:
            continue
        ticks, mean, std = extract_timeseries(results[cond], metric_name)
        if ticks is None:
            continue

        color = COLORS.get(cond, '#999999')
        label = LABELS.get(cond, cond)

        if smooth_window > 1:
            mean_s = smooth(mean, smooth_window)
            std_s = smooth(std, smooth_window)
        else:
            mean_s, std_s = mean, std

        ax.plot(ticks, mean_s, color=color, label=label, linewidth=1.2)
        ax.fill_between(ticks, mean_s - std_s, mean_s + std_s,
                        color=color, alpha=0.15)

    ax.set_xlabel('Simulation Step')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_population_dynamics(results, out_path, conditions=None):
    """2-panel: population count + birth/death rates."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    if conditions is None:
        conditions = list(results.keys())

    for cond in conditions:
        if cond not in results:
            continue
        color = COLORS.get(cond, '#999999')
        label = LABELS.get(cond, cond)

        ticks, mean, std = extract_timeseries(results[cond], 'pop_count')
        if ticks is None:
            continue
        mean_s = smooth(mean, 100)
        ax1.plot(ticks, mean_s, color=color, label=label, linewidth=1)
        ax1.fill_between(ticks, smooth(mean - std, 100), smooth(mean + std, 100),
                         color=color, alpha=0.1)

    ax1.set_xlabel('Step')
    ax1.set_ylabel('Population')
    ax1.set_title('Population Dynamics')
    ax1.legend(loc='best', fontsize=7, framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # Birth/death cumulative
    for cond in conditions:
        if cond not in results:
            continue
        color = COLORS.get(cond, '#999999')
        label = LABELS.get(cond, cond)

        ticks_b, births, _ = extract_timeseries(results[cond], 'births_cum')
        ticks_d, deaths, _ = extract_timeseries(results[cond], 'deaths_cum')
        if ticks_b is not None:
            ax2.plot(ticks_b, smooth(births, 100), color=color, label=f'{label} births',
                     linewidth=1, linestyle='-')
        if ticks_d is not None:
            ax2.plot(ticks_d, smooth(deaths, 100), color=color, label=f'{label} deaths',
                     linewidth=1, linestyle='--')

    ax2.set_xlabel('Step')
    ax2.set_ylabel('Cumulative Count')
    ax2.set_title('Births & Deaths')
    ax2.legend(loc='best', fontsize=6, framealpha=0.9, ncol=2)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_diversity_panel(results, out_path, conditions=None):
    """3-panel: genome diversity, phenotype diversity, cosine similarity."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    if conditions is None:
        conditions = list(results.keys())

    metrics = [
        ('genome_variance', 'Genome Variance', 'Genetic Diversity'),
        ('pheno_variance', 'Phenotype Variance', 'Phenotypic Diversity'),
        ('genome_cos_sim_mean', 'Mean Cosine Similarity', 'Genome Similarity'),
    ]

    for ax, (metric, ylabel, title) in zip(axes, metrics):
        for cond in conditions:
            if cond not in results:
                continue
            ticks, mean, std = extract_timeseries(results[cond], metric)
            if ticks is None:
                continue
            color = COLORS.get(cond, '#999999')
            label = LABELS.get(cond, cond)
            mean_s = smooth(mean, 20)
            ax.plot(ticks, mean_s, color=color, label=label, linewidth=1)
            ax.fill_between(ticks, smooth(mean - std, 20), smooth(mean + std, 20),
                           color=color, alpha=0.1)
        ax.set_xlabel('Step')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc='best', fontsize=7, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_attention_patterns(results, out_path):
    """Plot attention distribution across heads for baseline."""
    if 'baseline' not in results:
        print("  Skipping attention plot (no baseline data)")
        return

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

    for h in range(4):
        ax = axes[h]
        for target, style in [('self', '-'), ('neigh', '--'), ('ves', ':')]:
            metric = f'attn_h{h}_{target}'
            ticks, mean, std = extract_timeseries(results['baseline'], metric)
            if ticks is None:
                continue
            ax.plot(ticks, mean, linestyle=style, label=target, linewidth=1.2)
            if std is not None:
                ax.fill_between(ticks, mean - std, mean + std, alpha=0.15)
        ax.set_title(f'Head {h}')
        ax.set_xlabel('Step')
        ax.set_ylabel('Attention Weight')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    fig.suptitle('Attention Distribution per Head (Baseline)', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_vesicle_analysis(results, out_path, conditions=None):
    """2-panel: vesicle count + content variance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    if conditions is None:
        conditions = [c for c in results if c != 'no_vesicle' and c != 'minimal']

    for cond in conditions:
        if cond not in results:
            continue
        color = COLORS.get(cond, '#999999')
        label = LABELS.get(cond, cond)

        ticks, mean, std = extract_timeseries(results[cond], 'vesicle_count')
        if ticks is not None:
            ax1.plot(ticks, smooth(mean, 20), color=color, label=label, linewidth=1)

        ticks, mean, std = extract_timeseries(results[cond], 'vesicle_content_var')
        if ticks is not None:
            ax2.plot(ticks, smooth(mean, 20), color=color, label=label, linewidth=1)

    ax1.set_xlabel('Step')
    ax1.set_ylabel('Active Vesicles')
    ax1.set_title('Vesicle Population')
    ax1.legend(loc='best', fontsize=7)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Step')
    ax2.set_ylabel('Content Variance')
    ax2.set_title('Vesicle Content Diversity')
    ax2.legend(loc='best', fontsize=7)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def generate_summary_table(results, out_path):
    """Generate LaTeX summary table of final metrics."""
    rows = []
    for cond in ['baseline', 'no_vesicle', 'no_genome_mod', 'no_sexual',
                 'no_predation', 'no_speciation_pressure', 'no_aging', 'minimal']:
        if cond not in results:
            continue
        label = LABELS.get(cond, cond)

        # Get final values (last 10% of simulation)
        def final_mean(metric):
            ticks, mean, std = extract_timeseries(results[cond], metric)
            if ticks is None or len(mean) == 0:
                return np.nan, np.nan
            n = max(1, len(mean) // 10)
            return np.nanmean(mean[-n:]), np.nanstd(mean[-n:])

        pop_m, pop_s = final_mean('pop_count')
        gvar_m, gvar_s = final_mean('genome_variance')
        pvar_m, pvar_s = final_mean('pheno_variance')
        gsim_m, gsim_s = final_mean('genome_cos_sim_mean')
        ves_m, ves_s = final_mean('vesicle_count')

        rows.append({
            'Condition': label,
            'Pop': f'{pop_m:.0f}±{pop_s:.0f}',
            'Genome Var': f'{gvar_m:.2f}±{gvar_s:.2f}',
            'Pheno Var': f'{pvar_m:.2f}±{pvar_s:.2f}',
            'Genome Sim': f'{gsim_m:.3f}±{gsim_s:.3f}',
            'Vesicles': f'{ves_m:.0f}±{ves_s:.0f}',
        })

    # Write as LaTeX table
    with open(out_path, 'w') as f:
        f.write('\\begin{table}[h]\n')
        f.write('\\centering\n')
        f.write('\\caption{Ablation study results (mean±std over last 10\\% of simulation)}\n')
        f.write('\\label{tab:ablation}\n')
        cols = list(rows[0].keys())
        f.write('\\begin{tabular}{' + 'l' + 'r' * (len(cols) - 1) + '}\n')
        f.write('\\toprule\n')
        f.write(' & '.join(cols) + ' \\\\\n')
        f.write('\\midrule\n')
        for row in rows:
            f.write(' & '.join(str(row[c]) for c in cols) + ' \\\\\n')
        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')
        f.write('\\end{table}\n')

    print(f"  Saved: {out_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analysis.py <results_dir> [--paper]")
        sys.exit(1)

    results_dir = sys.argv[1]
    paper_mode = '--paper' in sys.argv

    print(f"Loading results from {results_dir}...")
    results = load_results(results_dir)

    if not results:
        print("No results found!")
        sys.exit(1)

    print(f"Found {len(results)} conditions: {list(results.keys())}")
    for cond, data_list in results.items():
        print(f"  {cond}: {len(data_list)} seeds")

    # Output directory for figures
    fig_dir = os.path.join(results_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    # Key comparison: baseline vs no_vesicle vs minimal
    key_conditions = ['baseline', 'no_vesicle', 'minimal']
    all_conditions = list(results.keys())

    print("\nGenerating figures...")

    # Figure 1: Population dynamics
    plot_population_dynamics(results, os.path.join(fig_dir, 'fig1_population.png'),
                           all_conditions)

    # Figure 2: Diversity panel
    plot_diversity_panel(results, os.path.join(fig_dir, 'fig2_diversity.png'),
                        all_conditions)

    # Figure 3: Vesicle communication
    plot_vesicle_analysis(results, os.path.join(fig_dir, 'fig3_vesicles.png'))

    # Figure 4: Attention patterns
    plot_attention_patterns(results, os.path.join(fig_dir, 'fig4_attention.png'))

    # Figure 5: Energy dynamics
    plot_metric_comparison(results, 'energy_mean',
                          'Mean Cell Energy', 'Energy',
                          os.path.join(fig_dir, 'fig5_energy.png'),
                          all_conditions)

    # Figure 6: Spatial organization
    plot_metric_comparison(results, 'nn_dist_mean',
                          'Mean Nearest-Neighbor Distance', 'Distance (px)',
                          os.path.join(fig_dir, 'fig6_spatial.png'),
                          all_conditions)

    # Figure 7: Generation progression
    plot_metric_comparison(results, 'gen_max',
                          'Maximum Generation', 'Generation',
                          os.path.join(fig_dir, 'fig7_generations.png'),
                          all_conditions, smooth_window=20)

    # Figure 8: Nutrient competition
    plot_metric_comparison(results, 'nutrient_energy_mean',
                          'Mean Nutrient Energy', 'Energy',
                          os.path.join(fig_dir, 'fig8_nutrients.png'),
                          all_conditions, smooth_window=20)

    # Summary table
    generate_summary_table(results, os.path.join(fig_dir, 'table_ablation.tex'))

    print(f"\nAll figures saved to {fig_dir}/")


if __name__ == '__main__':
    main()
