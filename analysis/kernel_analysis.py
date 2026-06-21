"""
Kernel Learning Effectiveness Analysis
=======================================
Analyses the saved kernel states from ARFLD experiments to answer:

  L1 – Performance:   Is learnable > fixed?  (from acc logs)
  L2 – Behaviour:     Do kernel weights actually learn? (trajectories, entropy)
  L3 – Causality:     Do kernel configs correlate with data distributions?
  L4 – Mechanism:     Which bandwidth regimes matter for which non-IID settings?

Usage:
    python analysis/kernel_analysis.py --exp_dir experiments/M3D/<exp_id>

Output:
    - Printed summary to stdout
    - Figures saved to <exp_dir>/kernel_analysis/
"""

import os
import sys
import argparse
import glob
import numpy as np
import torch
import json
from collections import defaultdict

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_kernel_states(kernel_dir):
    """Load all kernel state files from a directory, sorted by epoch.

    Returns:
        dict: client_id -> { 'weights': list of (epoch, ndarray),
                             'bandwidths': list of (epoch, ndarray)  }
    """
    files = sorted(glob.glob(os.path.join(kernel_dir, 'exp*_epoch*_kernel.pt')))
    if not files:
        print('[ERROR] No kernel state files found in %s' % kernel_dir)
        return None

    # Structure: client_states[client_id][metric] = [(epoch, value), ...]
    client_states = defaultdict(lambda: {'weights': [], 'bandwidths': []})

    for f in files:
        basename = os.path.basename(f)
        # Parse epoch: exp0_epoch005_kernel.pt
        epoch_str = basename.split('epoch')[1].split('_')[0]
        epoch = int(epoch_str)

        state = torch.load(f, map_location='cpu')
        for cid_str, cstate in state.items():
            cid = int(cid_str)
            if cstate['weights'] is not None:
                client_states[cid]['weights'].append((epoch, cstate['weights'].numpy()))
            if cstate['bandwidth_multipliers'] is not None:
                client_states[cid]['bandwidths'].append((epoch, cstate['bandwidth_multipliers'].numpy()))

    return dict(client_states)


def load_client_stats(stats_path):
    """Load per-client data statistics."""
    if not os.path.exists(stats_path):
        print('[WARN] Client stats file not found: %s' % stats_path)
        return None
    return torch.load(stats_path, map_location='cpu')


def load_acc_log(log_path):
    """Parse acc_mean and acc_std from log.txt."""
    if not os.path.exists(log_path):
        return None
    with open(log_path, 'r') as f:
        lines = f.readlines()
    # Last 3 lines should contain acc_mean, acc_std
    acc_mean, acc_std = None, None
    for line in reversed(lines):
        line = line.strip()
        if line and acc_mean is None:
            try:
                vals = np.array(eval(line))
                if vals.ndim == 1:
                    acc_mean = vals
            except Exception:
                pass
    # Simpler: find lines like "[0.xx 0.xx ...]"
    arrays_found = []
    for line in lines:
        line = line.strip()
        if line.startswith('[') and line.endswith(']'):
            try:
                arr = np.array(eval(line))
                if arr.ndim == 1:
                    arrays_found.append(arr)
            except Exception:
                pass
    if len(arrays_found) >= 2:
        acc_mean = arrays_found[-2]
        acc_std = arrays_found[-1]
    return acc_mean, acc_std


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def analyse_weight_trajectories(client_states, save_dir=None):
    """L2: Show kernel weight trajectories over epochs for each client.

    This answers: "Are the weights actually changing in a structured way,
    or just jittering randomly?"
    """
    print('\n' + '=' * 70)
    print('L2 — KERNEL WEIGHT TRAJECTORIES')
    print('=' * 70)

    n_clients = len(client_states)
    n_kernels = None

    for cid in sorted(client_states.keys()):
        entries = client_states[cid]['weights']
        if not entries:
            continue
        epochs = [e for e, _ in entries]
        weights = np.stack([w for _, w in entries])  # (T, K)

        if n_kernels is None:
            n_kernels = weights.shape[1]

        # Convergence check: std of weights in last 30% of epochs vs first 30%
        split = max(1, len(epochs) // 3)
        early_std = weights[:split].std(axis=0).mean()
        late_std = weights[-split:].std(axis=0).mean()
        stability_ratio = late_std / (early_std + 1e-8)

        # Weight range: did weights diversify from uniform?
        initial_entropy = -np.sum(weights[0] * np.log(weights[0] + 1e-8))
        final_entropy = -np.sum(weights[-1] * np.log(weights[-1] + 1e-8))
        max_entropy = np.log(n_kernels)
        final_weight_range = weights[-1].max() - weights[-1].min()

        print(f'  Client {cid:2d}: '
              f'init_ent={initial_entropy:.3f} → final_ent={final_entropy:.3f} '
              f'(max={max_entropy:.3f}) | '
              f'stability_ratio={stability_ratio:.3f} | '
              f'final_range={final_weight_range:.3f} | '
              f'final_weights={np.round(weights[-1], 3)}')

    # Overall summary
    all_final_weights = []
    all_initial_entropies = []
    all_final_entropies = []
    for cid in client_states:
        entries = client_states[cid]['weights']
        if not entries:
            continue
        w = entries[-1][1]
        all_final_weights.append(w)
        all_initial_entropies.append(
            -np.sum(entries[0][1] * np.log(entries[0][1] + 1e-8)))
        all_final_entropies.append(
            -np.sum(w * np.log(w + 1e-8)))

    if all_final_weights:
        all_final = np.stack(all_final_weights)
        cross_client_std = all_final.std(axis=0).mean()
        print(f'\n  Cross-client weight std: {cross_client_std:.4f} '
              f'(higher → clients differ more)')
        print(f'  Mean entropy: init={np.mean(all_initial_entropies):.4f} '
              f'→ final={np.mean(all_final_entropies):.4f} '
              f'(max={max_entropy:.4f})')
        if np.mean(all_final_entropies) > 0.8 * max_entropy:
            print('  ✓ Entropy is high → regularisation is preventing collapse')
        else:
            print('  ✗ Entropy is low → kernel may be collapsing, increase ent_weight')

    return {
        'cross_client_std': cross_client_std if all_final_weights else None,
        'mean_final_entropy': np.mean(all_final_entropies) if all_final_entropies else None,
    }


def analyse_bandwidth_evolution(client_states, save_dir=None):
    """L2: Show bandwidth multiplier trajectories.

    This answers: "Are bandwidths converging to stable values,
    and are they different from the initial geometric progression?"
    """
    print('\n' + '=' * 70)
    print('L2 — BANDWIDTH MULTIPLIER EVOLUTION')
    print('=' * 70)

    for cid in sorted(client_states.keys()):
        entries = client_states[cid]['bandwidths']
        if not entries:
            continue
        epochs = [e for e, _ in entries]
        bws = np.stack([b for _, b in entries])  # (T, K)

        init_bw = bws[0]
        final_bw = bws[-1]
        drift = np.abs(final_bw - init_bw).mean()

        print(f'  Client {cid:2d}: init_bw={np.round(init_bw, 3)}')
        print(f'            final_bw={np.round(final_bw, 3)} | drift={drift:.4f}')

    # Overall
    all_drifts = []
    for cid in client_states:
        entries = client_states[cid]['bandwidths']
        if not entries:
            continue
        bws = np.stack([b for _, b in entries])
        drift = np.abs(bws[-1] - bws[0]).mean()
        all_drifts.append(drift)

    if all_drifts:
        print(f'\n  Mean bandwidth drift: {np.mean(all_drifts):.4f}')
        if np.mean(all_drifts) > 0.1:
            print('  ✓ Bandwidths are adapting (non-trivial drift detected)')
        else:
            print('  ~ Bandwidths barely moved — consider increasing kernel_lr_scale')

    return {'mean_bandwidth_drift': np.mean(all_drifts) if all_drifts else None}


def analyse_kernel_data_correlation(client_states, client_stats, save_dir=None):
    """L3: Correlate final kernel weights with client data statistics.

    This answers: "Do clients with different data distributions
    learn different kernel configurations?"
    """
    print('\n' + '=' * 70)
    print('L3 — KERNEL–DATA CORRELATION')
    print('=' * 70)

    if client_stats is None:
        print('  [SKIP] No client statistics available')
        return None

    # Build feature matrix
    client_ids = sorted(set(client_states.keys()) & set(int(k) for k in client_stats.keys()))
    if len(client_ids) < 3:
        print('  [SKIP] Need ≥3 clients for correlation (got %d)' % len(client_ids))
        return None

    X_data = []  # data statistics per client
    X_kernel = []  # final kernel weights per client
    for cid in client_ids:
        entries = client_states[cid]['weights']
        if not entries:
            continue
        stats = client_stats[str(cid)]
        X_data.append([
            stats['num_classes'],
            stats['num_samples'],
            stats['label_entropy'],
            stats['label_entropy'] / np.log(max(stats['num_classes'], 2)),  # normalised
        ])
        X_kernel.append(entries[-1][1])  # final weights

    X_data = np.array(X_data)       # (C, D_features)
    X_kernel = np.stack(X_kernel)   # (C, K)

    # Pearson correlation between each data feature and each kernel weight
    print('  Feature → Kernel weight correlations (Pearson r):')
    feature_names = ['n_classes', 'n_samples', 'label_entropy', 'norm_entropy']
    for fi, fname in enumerate(feature_names):
        correlations = []
        for ki in range(X_kernel.shape[1]):
            if X_data[:, fi].std() > 1e-8:
                r = np.corrcoef(X_data[:, fi], X_kernel[:, ki])[0, 1]
            else:
                r = 0.0
            correlations.append(r)
        print(f'    {fname:16s}: {np.round(correlations, 3)}  '
              f'(max_abs={max(np.abs(correlations)):.3f})')

    # CCA-like: can kernel weights be predicted from data stats?
    # Simple linear regression R²
    from numpy.linalg import lstsq
    X_aug = np.concatenate([X_data, np.ones((X_data.shape[0], 1))], axis=1)
    r2_scores = []
    for ki in range(X_kernel.shape[1]):
        y = X_kernel[:, ki]
        try:
            w, residuals, rank, sv = lstsq(X_aug, y, rcond=None)
            y_pred = X_aug @ w
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-8)
            r2_scores.append(max(0, r2))
        except Exception:
            r2_scores.append(0)

    print(f'\n  Predictability of kernel weights from data stats (R²): '
          f'{np.round(r2_scores, 3)}')
    print(f'  Mean R² = {np.mean(r2_scores):.3f}')
    if np.mean(r2_scores) > 0.3:
        print('  ✓ Kernel weights are partially predictable from data → '
              'kernel adapts to data distribution')
    else:
        print('  ~ Low predictability — kernel may be learning task-specific '
              'features not captured by simple data stats')

    return {'mean_r2': np.mean(r2_scores)}


def analyse_regularisation_effect(client_states, save_dir=None):
    """L2 supplementary: Show entropy stays high throughout training.

    If entropy drops to 0, the learnable multi-kernel has collapsed to a
    single kernel — proving the entropy regularisation is necessary.
    """
    print('\n' + '=' * 70)
    print('L2 — ENTROPY REGULARISATION CHECK')
    print('=' * 70)

    max_ent = None
    for cid in sorted(client_states.keys()):
        entries = client_states[cid]['weights']
        if not entries:
            continue
        if max_ent is None:
            max_ent = np.log(entries[0][1].shape[0])

        for epoch, w in entries:
            ent = -np.sum(w * np.log(w + 1e-8))
            if ent < 0.1 * max_ent:
                print(f'  ✗ Client {cid} epoch {epoch:03d}: entropy={ent:.4f} '
                      f'(near collapse — increase ent_weight)')
                break
        else:
            final_ent = -np.sum(entries[-1][1] * np.log(entries[-1][1] + 1e-8))
            print(f'  ✓ Client {cid}: final entropy={final_ent:.3f} / '
                  f'max={max_ent:.3f} ({100*final_ent/max_ent:.0f}%)')

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Kernel Learning Analysis')
    parser.add_argument('--exp_dir', type=str, required=True,
                        help='Path to experiment directory '
                             '(e.g., experiments/M3D/dev_ConvNet_CIFAR10_...)')
    parser.add_argument('--exp_id', type=int, default=0,
                        help='Experiment index (default: 0)')
    parser.add_argument('--save_plots', action='store_true', default=True,
                        help='Save analysis figures')
    args = parser.parse_args()

    exp_dir = args.exp_dir
    kernel_dir = os.path.join(exp_dir, 'kernel_states')
    log_path = os.path.join(exp_dir, 'log.txt')

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print('Loading kernel states from: %s' % kernel_dir)
    client_states = load_kernel_states(kernel_dir)
    if client_states is None:
        sys.exit(1)

    client_stats = load_client_stats(
        os.path.join(kernel_dir, 'exp%d_client_stats.pt' % args.exp_id))
    acc = load_acc_log(log_path)

    # ------------------------------------------------------------------
    # Summary header
    # ------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('KERNEL LEARNING ANALYSIS — %s' % os.path.basename(exp_dir))
    print('=' * 70)
    print(f'  Clients: {len(client_states)}')
    n_epochs = max(len(v['weights']) for v in client_states.values())
    print(f'  Epochs tracked: {n_epochs}')

    if client_stats:
        entropies = [s['label_entropy'] for s in client_stats.values()]
        print(f'  Label entropy range: [{min(entropies):.3f}, {max(entropies):.3f}] '
              f'(lower = more skewed non-IID)')

    if acc is not None and acc[0] is not None:
        print(f'  Final test acc (mean): {acc[0][-1]:.2%}' if len(acc[0]) > 0
              else '  (could not parse acc)')
    print()

    # ------------------------------------------------------------------
    # Run analyses
    # ------------------------------------------------------------------
    results = {}

    results['weight_trajectories'] = analyse_weight_trajectories(
        client_states, exp_dir if args.save_plots else None)

    results['bandwidth_evolution'] = analyse_bandwidth_evolution(
        client_states, exp_dir if args.save_plots else None)

    results['regularisation'] = analyse_regularisation_effect(
        client_states, exp_dir if args.save_plots else None)

    results['data_correlation'] = analyse_kernel_data_correlation(
        client_states, client_stats, exp_dir if args.save_plots else None)

    # ------------------------------------------------------------------
    # Final verdict
    # ------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('VERDICT: Is the learnable kernel effective?')
    print('=' * 70)

    checks = []

    # Check 1: Weight diversity across clients
    if results['weight_trajectories'] and \
       results['weight_trajectories']['cross_client_std'] is not None:
        if results['weight_trajectories']['cross_client_std'] > 0.02:
            checks.append(('✓', 'Clients learn DIFFERENT kernel weights '
                                '(cross-client std > 0.02)'))
        else:
            checks.append(('?', 'All clients converged to similar weights '
                                '— may not need per-client learning'))

    # Check 2: Entropy maintenance
    if results['weight_trajectories'] and \
       results['weight_trajectories']['mean_final_entropy'] is not None:
        n_kernels = len(client_states[0]['weights'][0][1])
        max_ent = np.log(n_kernels)
        ratio = results['weight_trajectories']['mean_final_entropy'] / max_ent
        if ratio > 0.6:
            checks.append(('✓', 'Entropy regularisation working '
                                f'(final entropy = {ratio:.0%} of max)'))
        else:
            checks.append(('✗', 'Kernel collapsing to single kernel '
                                f'(entropy = {ratio:.0%} of max) — increase ent_weight'))

    # Check 3: Bandwidth adaptation
    if results['bandwidth_evolution'] and \
       results['bandwidth_evolution']['mean_bandwidth_drift'] is not None:
        if results['bandwidth_evolution']['mean_bandwidth_drift'] > 0.05:
            checks.append(('✓', 'Bandwidth multipliers are adapting (drift > 0.05)'))
        else:
            checks.append(('~', 'Bandwidths barely moved — may need higher '
                                'kernel_lr_scale or more epochs'))

    # Check 4: Data–kernel correlation
    if results['data_correlation'] and \
       results['data_correlation']['mean_r2'] is not None:
        if results['data_correlation']['mean_r2'] > 0.3:
            checks.append(('✓', f'Kernel weights predictable from data stats '
                                f'(R² = {results["data_correlation"]["mean_r2"]:.3f})'))
        else:
            checks.append(('~', f'Low data–kernel correlation '
                                f'(R² = {results["data_correlation"]["mean_r2"]:.3f}) '
                                f'— kernel may encode task info beyond label stats'))

    for symbol, msg in checks:
        print(f'  [{symbol}] {msg}')

    n_pass = sum(1 for s, _ in checks if s == '✓')
    n_warn = sum(1 for s, _ in checks if s == '~')
    n_fail = sum(1 for s, _ in checks if s == '✗')
    print(f'\n  Result: {n_pass} pass, {n_warn} uncertain, {n_fail} fail')

    if n_fail == 0 and n_pass >= 2:
        print('  → Evidence supports that the learnable kernel is effective.')
    elif n_fail > 0:
        print('  → Issues detected — tune hyperparameters and re-run.')
    else:
        print('  → Inconclusive — need more data/epochs to establish.')


if __name__ == '__main__':
    main()
