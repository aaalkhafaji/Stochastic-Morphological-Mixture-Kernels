"""Deterministic Step-2 tests for end-to-end representation invariance.

These are construction checks, not benchmark evidence.  They deliberately
retrain the direct-output predictor after permuting and duplicating action
labels.  Passing them closes the implementation gap documented by the original
post-hoc quotient extension.
"""
import json
from pathlib import Path
import numpy as np

from morphology import bank, measure
from output_space_learning import (
    build_training_table, fit_direct_output_forest, direct_output_law)

ROOT = Path(__file__).resolve().parents[1]


def make_truth(i, shape=(16, 16)):
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    cy = 4 + (3 * i) % 8
    cx = 4 + (5 * i) % 8
    ry = 2 + (i % 4)
    rx = 2 + ((i // 2) % 4)
    y = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1
    if i % 3 == 0:
        y |= (abs(yy - (h - 1 - cy)) <= 1) & (abs(xx - cx) <= 3)
    if i % 5 == 0:
        y[cy:cy+1, max(0, cx-1):min(w, cx+2)] = False
    return y


def corrupt(y, i):
    rng = np.random.default_rng(87000 + i)
    u = rng.random(y.shape)
    return np.where(y, u >= 0.07, u < 0.05)


def representation(outputs, mode):
    m = len(outputs)
    if mode == 'base':
        idx = np.arange(m)
    elif mode == 'permuted':
        idx = np.random.default_rng(443).permutation(m)
    elif mode == 'duplicated':
        idx = np.r_[np.arange(m), np.repeat([0, 2, 7, 11], [5, 3, 4, 2])]
    elif mode == 'permuted_duplicated':
        raw = np.r_[np.arange(m), np.repeat([0, 2, 7, 11], [5, 3, 4, 2])]
        idx = raw[np.random.default_rng(991).permutation(len(raw))]
    else:
        raise ValueError(mode)
    return outputs[idx], idx


def prepare(n):
    xs, banks, costs = [], [], []
    for i in range(n):
        y = make_truth(i)
        x = corrupt(y, i)
        b = bank(x)
        c = np.array([measure(z, y)[0] for z in b])
        xs.append(x); banks.append(b); costs.append(c)
    return xs, banks, costs


def transform_dataset(banks, costs, mode):
    B, C = [], []
    for b, c in zip(banks, costs):
        br, idx = representation(b, mode)
        B.append(br); C.append(c[idx])
    return B, C


def main():
    train_x, train_b, train_c = prepare(30)
    test_x, test_b, _ = prepare(8)
    modes = ['base', 'permuted', 'duplicated', 'permuted_duplicated']

    tables = {}
    models = {}
    metadata = {}
    for mode in modes:
        B, C = transform_dataset(train_b, train_c, mode)
        X, y, image_index, output_rank = build_training_table(train_x, B, C)
        tables[mode] = (X, y, image_index, output_rank)
        model, meta = fit_direct_output_forest(
            train_x, B, C, seed=2026, min_samples_leaf=2, n_estimators=48)
        models[mode] = model; metadata[mode] = meta

    base = tables['base']
    table_residual = 0.0
    for mode in modes[1:]:
        t = tables[mode]
        assert all(a.shape == b.shape for a, b in zip(base, t))
        for a, b in zip(base, t):
            if np.issubdtype(a.dtype, np.floating):
                table_residual = max(table_residual, float(np.max(np.abs(a-b))))
            assert np.array_equal(a, b), f'canonical table changed under {mode}'

    max_cost_residual = 0.0
    max_law_residual = 0.0
    checks = 0
    for i, (x, b) in enumerate(zip(test_x, test_b)):
        base_law = direct_output_law(models['base'], x, b, temperature=0.02)
        for mode in modes[1:]:
            br, _ = representation(b, mode)
            law = direct_output_law(models[mode], x, br, temperature=0.02)
            # Canonical output ordering is exact-byte ordering, so these arrays align.
            cr = float(np.max(np.abs(base_law['predicted_costs'] - law['predicted_costs'])))
            qr = float(np.max(np.abs(base_law['output_masses'] - law['output_masses'])))
            max_cost_residual = max(max_cost_residual, cr)
            max_law_residual = max(max_law_residual, qr)
            assert cr <= 1e-12, (i, mode, cr)
            assert qr <= 1e-12, (i, mode, qr)
            checks += 1

    record = {
        'status': 'PASS',
        'purpose': 'Step-2 deterministic construction test; not benchmark evidence',
        'training_images': len(train_x),
        'test_images': len(test_x),
        'representations': modes,
        'feature_dimension': int(base[0].shape[1]),
        'canonical_training_rows': int(base[0].shape[0]),
        'retraining_comparisons': checks,
        'max_canonical_table_residual': table_residual,
        'max_predicted_cost_residual': max_cost_residual,
        'max_output_law_residual': max_law_residual,
        'required_tolerance': 1e-12,
        'model_metadata': {k: {kk: vv for kk, vv in v.items()
                                if kk not in ('image_index', 'output_rank')}
                           for k, v in metadata.items()},
    }
    path = ROOT / 'results' / 'step2_end_to_end_invariance.json'
    path.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
