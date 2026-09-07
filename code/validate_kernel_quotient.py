"""Executed exploratory extension on the already retained benchmark cases.

This revision reuses the previously inspected holdout; it is not a new
prospective benchmark. Models and source data remain frozen. Temperature is
chosen using only the original validation images.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
from scipy.special import softmax
import joblib
from kernel_quotient import (output_partition, quotient_weights,
    maximum_replication_shift, aggregate_action_mass)
from morphology import METRICS

ROOT = Path(__file__).resolve().parents[1]
TEMPERATURES = [.005, .02, .1]
SEEDS = [101, 102, 103, 104, 105]


def evaluate(ds):
    data = {s: dict(np.load(ROOT/'results'/f'{ds}_{s}_bank.npz'))
            for s in ('val', 'test')}
    partitions = {s: [output_partition(b) for b in data[s]['outputs']]
                  for s in data}
    te, va = data['test'], data['val']
    n = len(te['scores'])
    primary = te['condition'] < 6
    choices = json.loads((ROOT/'results'/f'{ds}_selection.json').read_text())
    raw = []; saved = {}; runs = []; residual = 0.
    qcount = np.array([len(p[2]) for p in partitions['test']])
    saved['distinct_outputs'] = qcount
    saved['image_index'] = te['image_index']; saved['condition'] = te['condition']
    for seed in SEEDS:
        model = joblib.load(ROOT/'models'/f'{ds}_forest_{seed}.joblib')
        R = model.predict(te['features']); V = model.predict(va['features'])
        costs = []
        for tau in TEMPERATURES:
            p = np.stack([quotient_weights(r, part, tau)[0]
                          for r, part in zip(V, partitions['val'])])
            costs.append(float(np.sum(p * va['scores'][:, :, 0], axis=1).mean()))
        tau = TEMPERATURES[int(np.argmin(costs))]
        p = np.stack([quotient_weights(r, part, tau)[0]
                      for r, part in zip(R, partitions['test'])])
        old_tau = next(r['temperature'] for r in choices['runs'] if r['seed'] == seed)
        old_p = softmax(-R / old_tau, axis=1)
        metric = np.einsum('ij,ijk->ik', p, te['scores'])
        old_metric = np.einsum('ij,ijk->ik', old_p, te['scores'])
        shifts = np.array([maximum_replication_shift(pr, part)
                           for pr, part in zip(old_p, partitions['test'])])
        for method, values in [('Output-quotient kernel', metric),
                               ('Action-Gibbs kernel', old_metric)]:
            frame = pd.DataFrame(values, columns=METRICS)
            frame.insert(0, 'condition', te['condition'])
            frame.insert(0, 'image', te['image_index'])
            frame.insert(0, 'seed', seed); frame.insert(0, 'method', method)
            frame.insert(0, 'dataset', ds); raw.append(frame)
        saved[f'risks_{seed}'] = R.astype(np.float32)
        saved[f'probabilities_{seed}'] = p.astype(np.float32)
        saved[f'replication_tv_{seed}'] = shifts.astype(np.float32)
        # Verify every possible single-label replication on 32 fixed cases.
        # These are deterministic algebra checks, not extra test images.
        for i in np.linspace(0, n-1, 32, dtype=int):
            base_part = partitions['test'][i]
            base_law = aggregate_action_mass(p[i], base_part)
            for j in range(31):
                idx = np.r_[np.arange(31), np.repeat(j, 8)]
                rep_part = output_partition(te['outputs'][i, idx])
                lifted = quotient_weights(R[i, idx], rep_part, tau)[0]
                law = aggregate_action_mass(lifted, rep_part)
                residual = max(residual, float(np.max(abs(base_law-law))))
                assert np.max(abs(base_law-law)) < 1e-12
        runs.append({'seed': seed, 'validation_losses': costs, 'temperature': tau,
                     'primary_quotient_loss': float(metric[primary, 0].mean()),
                     'primary_action_loss': float(old_metric[primary, 0].mean()),
                     'mean_max_replication_tv': float(shifts[primary].mean()),
                     'max_replication_tv': float(shifts[primary].max())})
        print(ds, seed, runs[-1], flush=True)
    table = pd.concat(raw, ignore_index=True)
    table.to_csv(ROOT/'results'/f'{ds}_quotient_metrics.csv.gz', index=False, float_format='%.9g')
    np.savez_compressed(ROOT/'results'/f'{ds}_quotient_predictions.npz', **saved)
    record = {'dataset': ds, 'status': 'exploratory reuse of previously inspected test cases',
              'temperature_candidates': TEMPERATURES, 'runs': runs,
              'mean_distinct_outputs_primary': float(qcount[primary].mean()),
              'min_distinct_outputs_primary': int(qcount[primary].min()),
              'max_distinct_outputs_primary': int(qcount[primary].max()),
              'duplicate_bank_fraction_primary': float((qcount[primary] < 31).mean()),
              'replication_invariance_max_residual': residual,
              'replication_checks': 32 * 31 * len(SEEDS)}
    (ROOT/'results'/f'{ds}_quotient_validation.json').write_text(json.dumps(record, indent=2)+'\n')
    return record


def main():
    records = [evaluate(ds) for ds in ('shapes', 'mnist', 'fashion')]
    rows = []
    for r in records:
        runs = r['runs']
        rows.append({'dataset': r['dataset'], 'distinct_outputs': r['mean_distinct_outputs_primary'],
            'action_loss': np.mean([v['primary_action_loss'] for v in runs]),
            'quotient_loss': np.mean([v['primary_quotient_loss'] for v in runs]),
            'mean_max_replication_tv': np.mean([v['mean_max_replication_tv'] for v in runs]),
            'invariance_residual': r['replication_invariance_max_residual']})
    frame = pd.DataFrame(rows); frame.to_csv(ROOT/'results/quotient_summary.csv', index=False)
    lines = [r'\begin{table}[htbp]', r'\centering\small',
        r'\caption{Exploratory kernel-law analysis on the retained primary cases. Losses are exact categorical expectations averaged over five fitted models. $\bar M$ is the mean number of distinct output masks. TV stress is the mean, over cases and models, of the largest total-variation change caused by adding eight copies of one action with its fitted score fixed. No new holdout or significance claim is introduced.}\label{tab:quotient}',
        r'\begin{tabular}{lrrrr}\toprule',
        r'Dataset & $\bar M$ & Action Gibbs & Output quotient & TV stress\\\midrule']
    for r in rows:
        name={'shapes':'Shapes','mnist':'MNIST masks','fashion':'Fashion masks'}[r['dataset']]
        lines.append(f"{name} & {r['distinct_outputs']:.2f} & {r['action_loss']:.4f} & {r['quotient_loss']:.4f} & {r['mean_max_replication_tv']:.4f} \\\\")
    lines += [r'\bottomrule\end{tabular}',r'\end{table}']
    (ROOT/'results/generated_quotient.tex').write_text('\n'.join(lines)+'\n')
    print(frame.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
