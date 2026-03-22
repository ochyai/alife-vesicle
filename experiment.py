#!/usr/bin/env python3
"""
Headless experiment runner for ALife vesicle ablation studies.
Usage:
    python experiment.py --condition baseline --steps 50000 --seed 42 --out results/
    python experiment.py --run-all --steps 50000 --seeds 5 --out results/
"""
import sys, os, argparse, json, time, random

# Force CPU for reproducible experiments (must be set before main import)
# Also suppress pygame welcome message for headless operation
if '--cpu' not in sys.argv:
    sys.argv.append('--cpu')
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
os.environ['SDL_VIDEODRIVER'] = 'dummy'  # headless pygame

import torch


def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


# Define ablation conditions
CONDITIONS = {
    'baseline': {
        'desc': 'Full system (vesicle + genome modulation + sexual + predation)',
        'patches': {},
    },
    'no_vesicle': {
        'desc': 'Vesicle communication disabled',
        'patches': {'ves_on': False},
    },
    'no_genome_mod': {
        'desc': 'Genome modulation of attention disabled (Q/K scaling fixed to 0.5)',
        'patches': {'disable_genome_mod': True},
    },
    'no_sexual': {
        'desc': 'Sexual recombination disabled (asexual only)',
        'patches': {'disable_sexual': True},
    },
    'no_predation': {
        'desc': 'Predation disabled',
        'patches': {'disable_predation': True},
    },
    'no_speciation_pressure': {
        'desc': 'Speciation pressure (similarity drain) disabled',
        'patches': {'disable_speciation': True},
    },
    'no_aging': {
        'desc': 'Aging pressure disabled',
        'patches': {'disable_aging': True},
    },
    'minimal': {
        'desc': 'No vesicle, no sexual, no predation (minimal baseline)',
        'patches': {'ves_on': False, 'disable_sexual': True, 'disable_predation': True},
    },
}


def apply_patches(world, patches):
    """Apply ablation patches to a World instance."""
    if 'ves_on' in patches:
        world.ves_on = patches['ves_on']

    if patches.get('disable_genome_mod'):
        # Fix genome Q/K scaling to constant 0.5 (neutral)
        # sigmoid(0) = 0.5, so zero out weights and biases
        with torch.no_grad():
            world.brain.genome_qk.weight.zero_()
            world.brain.genome_qk.bias.fill_(0.0)

    if patches.get('disable_sexual'):
        world._disable_sexual = True

    if patches.get('disable_predation'):
        world._disable_predation = True

    if patches.get('disable_speciation'):
        world._disable_speciation = True

    if patches.get('disable_aging'):
        world._disable_aging = True


class MetricsCollector:
    """Lightweight metrics collector for headless experiments."""

    def __init__(self, world, interval=10):
        self.world = world
        self.interval = interval
        self.records = []

    def collect(self):
        w = self.world
        if w.t % self.interval != 0:
            return
        idx = w.calive.nonzero(as_tuple=True)[0]
        n = len(idx)
        record = {
            'step': w.t,
            'n_cells': n,
            'births': w.births,
            'deaths': w.deaths,
        }
        if n > 0:
            record['mean_energy'] = w.cenergy[idx].mean().item()
            record['max_energy'] = w.cenergy[idx].max().item()
            record['min_energy'] = w.cenergy[idx].min().item()
            record['mean_age'] = w.cage[idx].float().mean().item()
            record['max_gen'] = w.cgen[idx].max().item()
            record['mean_gen'] = w.cgen[idx].float().mean().item()
            # Phenotype diversity (variance of state vectors)
            if n > 1:
                record['pheno_diversity'] = w.cstate[idx].var(0).sum().item()
                # Genome diversity
                record['genome_diversity'] = w.cgenome[idx].var(0).sum().item()
            else:
                record['pheno_diversity'] = 0.0
                record['genome_diversity'] = 0.0
            # Vesicle count
            record['n_vesicles'] = w.valive.sum().item()
        else:
            record['mean_energy'] = 0.0
            record['max_energy'] = 0.0
            record['min_energy'] = 0.0
            record['mean_age'] = 0.0
            record['max_gen'] = 0
            record['mean_gen'] = 0.0
            record['pheno_diversity'] = 0.0
            record['genome_diversity'] = 0.0
            record['n_vesicles'] = 0

        self.records.append(record)

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.records, f, indent=2)


def run_experiment(condition_name, steps, seed, out_dir):
    """Run a single experiment condition."""
    import main

    setup_seed(seed)

    cond = CONDITIONS[condition_name]
    print(f"\n{'='*60}")
    print(f"Condition: {condition_name} (seed={seed})")
    print(f"  {cond['desc']}")
    print(f"  Steps: {steps}")
    print(f"{'='*60}")

    w = main.World()
    apply_patches(w, cond['patches'])

    mc = MetricsCollector(w, interval=10)

    t0 = time.time()
    for i in range(steps):
        w.step()
        mc.collect()
        if i % 5000 == 0 and i > 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (steps - i) / rate
            n = w.calive.sum().item()
            print(f"  [{condition_name}] step {i}/{steps} cells={n} "
                  f"births={w.births} deaths={w.deaths} "
                  f"({rate:.0f} steps/s, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s ({steps/elapsed:.0f} steps/s)")

    # Save
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{condition_name}_seed{seed}.json"
    fpath = os.path.join(out_dir, fname)
    mc.save(fpath)

    # Also save summary
    summary = {
        'condition': condition_name,
        'desc': cond['desc'],
        'seed': seed,
        'steps': steps,
        'wall_time': elapsed,
        'final_cells': w.calive.sum().item(),
        'total_births': w.births,
        'total_deaths': w.deaths,
        'final_gen_max': w.cgen[w.calive].max().item() if w.calive.any() else 0,
    }
    summary_path = os.path.join(out_dir, f"{condition_name}_seed{seed}_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description='ALife Vesicle Ablation Experiments')
    parser.add_argument('--condition', type=str, choices=list(CONDITIONS.keys()),
                        help='Single condition to run')
    parser.add_argument('--run-all', action='store_true',
                        help='Run all conditions')
    parser.add_argument('--steps', type=int, default=50000,
                        help='Simulation steps per run (default: 50000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (for single run)')
    parser.add_argument('--seeds', type=int, default=5,
                        help='Number of seeds (for --run-all)')
    parser.add_argument('--out', type=str, default='results/',
                        help='Output directory')
    parser.add_argument('--list', action='store_true',
                        help='List all conditions')
    args, _ = parser.parse_known_args()

    if args.list:
        print("Available ablation conditions:")
        for name, cond in CONDITIONS.items():
            print(f"  {name:25s} {cond['desc']}")
        return

    if args.condition:
        run_experiment(args.condition, args.steps, args.seed, args.out)
    elif args.run_all:
        all_summaries = []
        for seed in range(args.seeds):
            for cond_name in CONDITIONS:
                summary = run_experiment(cond_name, args.steps, seed, args.out)
                all_summaries.append(summary)
        # Save combined summary
        combined_path = os.path.join(args.out, 'all_summaries.json')
        with open(combined_path, 'w') as f:
            json.dump(all_summaries, f, indent=2)
        print(f"\nAll experiments complete. Summaries: {combined_path}")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
