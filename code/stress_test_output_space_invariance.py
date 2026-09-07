"""Step-3 end-to-end representation-invariance stress suite.

This is a deterministic construction/verification campaign, not benchmark
performance evidence. It deliberately retrains the direct-output scalar
regressor after aggressive output-preserving changes of the action
representation and verifies the complete fitted pipeline.

Gate: every exact-output-preserving representation must agree with the base
representation to <= 1e-12 in output probability mass after retraining.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.special import softmax

from morphology import bank, betti, measure, NAMES
from output_space_learning import (
    build_training_table,
    direct_output_law,
    fit_direct_output_forest,
    unique_output_rows,
)
from kernel_quotient import output_partition

ROOT = Path(__file__).resolve().parents[1]
TOL = 1e-12
DUPLICATE_COUNTS = (1, 2, 4, 8, 16, 32, 64)
DUPLICATE_ACTIONS = (0, 2, 7, 11, 30)
PERMUTATION_SEEDS = (443, 991, 1777, 2718)
FIT_SEEDS = (2026, 314159, 8675309)
SHAPES = ((12, 12), (16, 16), (20, 24), (28, 28))
TEMPERATURES = (0.005, 0.02, 0.1)


def make_truth(i: int, shape: Tuple[int, int], scenario_seed: int = 7301) -> np.ndarray:
    """Deterministic heterogeneous binary truth mask on an arbitrary grid."""
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    rng = np.random.default_rng(scenario_seed + 10007 * i + 31 * h + w)
    y = np.zeros(shape, dtype=bool)
    n_obj = 1 + (i % 3)
    for r in range(n_obj):
        cy = int(rng.integers(max(2, h // 5), max(3, h - h // 5)))
        cx = int(rng.integers(max(2, w // 5), max(3, w - w // 5)))
        ry = int(rng.integers(1, max(2, h // 5 + 1)))
        rx = int(rng.integers(1, max(2, w // 5 + 1)))
        y |= ((yy - cy) / max(ry, 1)) ** 2 + ((xx - cx) / max(rx, 1)) ** 2 <= 1
    if i % 4 == 0:
        row = int(rng.integers(1, h - 1))
        c0 = int(rng.integers(0, max(1, w // 2)))
        c1 = min(w, c0 + max(2, w // 3))
        y[max(0, row - 1):min(h, row + 1), c0:c1] = True
    if i % 5 == 0 and y.any():
        coords = np.argwhere(y)
        rr, cc = coords[len(coords) // 2]
        y[max(0, rr - 1):min(h, rr + 2), max(0, cc - 1):min(w, cc + 2)] = False
    return y


def corrupt(y: np.ndarray, i: int, scenario_seed: int = 7301) -> np.ndarray:
    rng = np.random.default_rng(scenario_seed + 88001 + 7919 * i + y.shape[0] * 101 + y.shape[1])
    p_fn = 0.04 + 0.01 * (i % 5)
    p_fp = 0.03 + 0.01 * ((i // 2) % 4)
    u = rng.random(y.shape)
    return np.where(y, u >= p_fn, u < p_fp)


def prepare(n: int, shape: Tuple[int, int], scenario_seed: int = 7301):
    xs, banks, costs = [], [], []
    for i in range(n):
        y = make_truth(i, shape, scenario_seed)
        x = corrupt(y, i, scenario_seed)
        b = bank(x)
        c = np.array([measure(z, y)[0] for z in b], dtype=float)
        xs.append(x); banks.append(b); costs.append(c)
    return xs, banks, costs


def _permute(idx: np.ndarray, seed: int) -> np.ndarray:
    return idx[np.random.default_rng(seed).permutation(len(idx))]


def transform_one(b: np.ndarray, c: np.ndarray, spec: Dict, image_index: int):
    """Apply one exact-output-preserving representation transformation."""
    m = len(b)
    base = np.arange(m, dtype=int)
    kind = spec['kind']
    if kind == 'base':
        idx = base
    elif kind == 'single_duplicate':
        j, k = spec['action'], spec['copies']
        idx = np.r_[base, np.repeat(j, k)]
    elif kind == 'permutation':
        idx = _permute(base, spec['perm_seed'])
    elif kind == 'permuted_duplicate':
        j, k = spec['action'], spec['copies']
        raw = np.r_[base, np.repeat(j, k)]
        idx = _permute(raw, spec['perm_seed'] + 1009 * image_index)
    elif kind == 'multi_duplicate':
        k = spec['copies']
        idx = np.r_[base, *[np.repeat(j, k) for j in DUPLICATE_ACTIONS]]
    elif kind == 'weighted_global':
        # Unequal label multiplicities, fixed globally across images.
        idx = np.concatenate([np.repeat(j, 1 + (j % 5)) for j in base])
        idx = _permute(idx, 6203 + image_index)
    elif kind == 'class_refinement':
        # More general dataset-level output-preserving refinement: repeat one
        # canonical representative of each realized distinct class according
        # to a deterministic function of the output bytes. This is not claimed
        # to be a new global morphological operator bank; it is a direct stress
        # test of the theorem's Q_x-preserving representation premise.
        cap = spec['cap']
        rows = unique_output_rows(np.zeros(b.shape[1:], dtype=bool), b)
        extra = []
        for rep in rows.representatives:
            key = np.packbits(b[rep].astype(np.uint8), bitorder='little').tobytes()
            h = int.from_bytes(hashlib.sha256(key).digest()[:4], 'little')
            extra.extend([int(rep)] * (1 + h % cap))
        idx = np.r_[base, np.asarray(extra, dtype=int)]
        idx = _permute(idx, spec['perm_seed'] + 7919 * image_index)
    else:
        raise ValueError(kind)
    return b[idx], c[idx], idx


def transform_dataset(banks, costs, spec):
    B, C = [], []
    for i, (b, c) in enumerate(zip(banks, costs)):
        br, cr, _ = transform_one(b, c, spec, i)
        B.append(br); C.append(cr)
    return B, C


def representation_specs() -> List[Dict]:
    specs = [{'name': 'base', 'kind': 'base'}]
    for j in DUPLICATE_ACTIONS:
        for k in DUPLICATE_COUNTS:
            specs.append({'name': f'dup_{j}_{k}', 'kind': 'single_duplicate',
                          'action': j, 'copies': k})
    for s in PERMUTATION_SEEDS:
        specs.append({'name': f'perm_{s}', 'kind': 'permutation', 'perm_seed': s})
    for j in DUPLICATE_ACTIONS:
        for s in PERMUTATION_SEEDS[:2]:
            specs.append({'name': f'permdup_{j}_64_{s}', 'kind': 'permuted_duplicate',
                          'action': j, 'copies': 64, 'perm_seed': s})
    for k in (1, 8, 64):
        specs.append({'name': f'multidup_{k}', 'kind': 'multi_duplicate', 'copies': k})
    specs.append({'name': 'weighted_global', 'kind': 'weighted_global'})
    for cap, s in ((2, 191), (4, 313), (8, 571)):
        specs.append({'name': f'classref_cap{cap}', 'kind': 'class_refinement',
                      'cap': cap, 'perm_seed': s})
    return specs


def table_residual(base, other):
    if any(a.shape != b.shape for a, b in zip(base, other)):
        return float('inf'), False
    r = 0.0
    for a, b in zip(base, other):
        if np.issubdtype(a.dtype, np.floating) and a.size:
            r = max(r, float(np.max(np.abs(a - b))))
        if not np.array_equal(a, b):
            return max(r, 1.0), False
    return r, True


def canonical_outputs(outputs):
    reps, _, _ = output_partition(outputs)
    return outputs[reps]


def masses_from_costs(costs, temperature, reference=None):
    costs = np.asarray(costs, dtype=float)
    if reference is None:
        return softmax(-(costs - costs.min()) / temperature)
    nu = np.asarray(reference, dtype=float)
    nu = nu / nu.sum()
    return softmax(np.log(nu) - (costs - costs.min()) / temperature)


def output_reference(canon: np.ndarray) -> np.ndarray:
    """Strictly positive nonuniform reference depending only on outputs."""
    vals = []
    for z in canon:
        b = betti(z)
        area = z.mean()
        vals.append(np.exp(-0.15 * b.sum()) * (0.8 + 0.4 * area))
    vals = np.asarray(vals, dtype=float)
    return vals / vals.sum()


def observables(canon: np.ndarray) -> np.ndarray:
    rows = []
    for z in canon:
        b = betti(z)
        boundary = (np.count_nonzero(z[1:] != z[:-1]) +
                    np.count_nonzero(z[:, 1:] != z[:, :-1])) / z.size
        rows.append([z.mean(), float(b[0]), float(b[1]), boundary])
    return np.asarray(rows, dtype=float)


def decode_indices(masses: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    cdf = np.cumsum(masses)
    cdf[-1] = 1.0
    return np.searchsorted(cdf, uniforms, side='right')


def max_abs(a, b):
    a = np.asarray(a); b = np.asarray(b)
    if a.shape != b.shape:
        return float('inf')
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def canonical_signature(outputs, costs=None):
    reps, inverse, counts = output_partition(outputs)
    canon = outputs[reps]
    cc = None if costs is None else np.asarray(costs, dtype=float)[reps]
    return canon, cc, counts


def assert_same_signature(base_b, base_c, rep_b, rep_c):
    bo, bc, _ = canonical_signature(base_b, base_c)
    ro, rc, _ = canonical_signature(rep_b, rep_c)
    if not np.array_equal(bo, ro):
        return False, float('inf')
    residual = max_abs(bc, rc)
    return residual <= TOL, residual


def main():
    specs = representation_specs()
    global_max = {
        'canonical_output_cost_signature': 0.0,
        'predicted_cost': 0.0,
        'output_law': 0.0,
        'expected_observable': 0.0,
        'sample_decode_disagreement': 0.0,
        'map_decode_disagreement': 0.0,
        'direct_api_output_law': 0.0,
    }
    detail_rows = []
    signature_checks = 0
    retraining_comparisons = 0
    law_comparisons = 0
    direct_api_checks = 0
    max_action_count = len(NAMES)

    # Stratified retraining subset: every promised duplicate count, every chosen
    # action identity (at the hardest 64-copy level), all permutation seeds,
    # combined permutation+duplication, multi-action duplication, unequal global
    # multiplicities, and general Q_x-preserving class refinements.
    retrain_names = set()
    retrain_names.update(f'dup_0_{k}' for k in DUPLICATE_COUNTS)
    retrain_names.update(f'dup_{j}_64' for j in DUPLICATE_ACTIONS)
    retrain_names.update(f'perm_{s}' for s in PERMUTATION_SEEDS)
    retrain_names.update(f'permdup_7_64_{s}' for s in PERMUTATION_SEEDS[:2])
    retrain_names.update(f'multidup_{k}' for k in (1, 8, 64))
    retrain_names.add('weighted_global')
    retrain_names.update(('classref_cap2', 'classref_cap4', 'classref_cap8'))
    retrain_specs = [sp for sp in specs[1:] if sp['name'] in retrain_names]

    for shape_id, shape in enumerate(SHAPES):
        train_x, train_b, train_c = prepare(14, shape, scenario_seed=7301 + shape_id * 1000)
        test_x, test_b, _ = prepare(4, shape, scenario_seed=17301 + shape_id * 1000)

        # Exhaustive signature checks over ALL representation specifications.
        for spec in specs[1:]:
            for image_i, (b, c) in enumerate(zip(train_b, train_c)):
                br, cr, _ = transform_one(b, c, spec, image_i)
                max_action_count = max(max_action_count, len(br))
                ok, r = assert_same_signature(b, c, br, cr)
                global_max['canonical_output_cost_signature'] = max(
                    global_max['canonical_output_cost_signature'], r)
                signature_checks += 1
                if not ok:
                    raise AssertionError((shape, spec['name'], image_i, 'signature', r))

        # Cache canonical output features/observables ONCE for inference.
        test_cache = []
        for test_i, (x, b) in enumerate(zip(test_x, test_b)):
            rows = unique_output_rows(x, b)
            canon = b[rows.representatives]
            test_cache.append({
                'x': x,
                'b': b,
                'pair_features': rows.pair_features,
                'canon': canon,
                'obs': observables(canon),
                'ref': output_reference(canon),
            })

        for fit_seed in FIT_SEEDS:
            base_model, _ = fit_direct_output_forest(
                train_x, train_b, train_c, seed=fit_seed,
                min_samples_leaf=2, n_estimators=8)
            base_cost_cache = [np.asarray(base_model.predict(tc['pair_features']), dtype=float)
                               for tc in test_cache]

            for spec in retrain_specs:
                B, C = transform_dataset(train_b, train_c, spec)
                model, _ = fit_direct_output_forest(
                    train_x, B, C, seed=fit_seed,
                    min_samples_leaf=2, n_estimators=8)
                row_max = {
                    'predicted_cost': 0.0,
                    'output_law': 0.0,
                    'expected_observable': 0.0,
                    'sample_decode_disagreement': 0.0,
                    'map_decode_disagreement': 0.0,
                }

                for test_i, tc in enumerate(test_cache):
                    x, b, canon = tc['x'], tc['b'], tc['canon']
                    dummy_c = np.zeros(len(b), dtype=float)
                    br, _, _ = transform_one(b, dummy_c, spec, test_i + 1000)
                    rep_canon, _, _ = canonical_signature(br)
                    if not np.array_equal(canon, rep_canon):
                        raise AssertionError((shape, fit_seed, spec['name'], test_i,
                                              'inference_signature'))

                    # Since phi(x,z) is output-only and canonical z is identical,
                    # the exact same cached pair-feature matrix is the appropriate
                    # input to both independently retrained models.
                    base_costs = base_cost_cache[test_i]
                    rep_costs = np.asarray(model.predict(tc['pair_features']), dtype=float)
                    cr = max_abs(base_costs, rep_costs)
                    row_max['predicted_cost'] = max(row_max['predicted_cost'], cr)
                    uniforms = np.random.default_rng(
                        900000 + shape_id * 10000 + fit_seed % 997 + test_i
                    ).random(128)

                    for temp in TEMPERATURES:
                        for reference in (None, tc['ref']):
                            bm = masses_from_costs(base_costs, temp, reference)
                            rm = masses_from_costs(rep_costs, temp, reference)
                            qr = max_abs(bm, rm)
                            er = max_abs(bm @ tc['obs'], rm @ tc['obs'])
                            map_dis = float(np.argmax(bm) != np.argmax(rm))
                            sample_dis = float(np.mean(
                                decode_indices(bm, uniforms) != decode_indices(rm, uniforms)))
                            row_max['output_law'] = max(row_max['output_law'], qr)
                            row_max['expected_observable'] = max(
                                row_max['expected_observable'], er)
                            row_max['map_decode_disagreement'] = max(
                                row_max['map_decode_disagreement'], map_dis)
                            row_max['sample_decode_disagreement'] = max(
                                row_max['sample_decode_disagreement'], sample_dis)
                            law_comparisons += 1
                            if cr > TOL or qr > TOL or er > TOL or map_dis or sample_dis:
                                raise AssertionError({
                                    'shape': shape, 'fit_seed': fit_seed,
                                    'representation': spec['name'], 'test_i': test_i,
                                    'temperature': temp,
                                    'cost_residual': cr, 'law_residual': qr,
                                    'observable_residual': er,
                                    'map_disagreement': map_dis,
                                    'sample_disagreement': sample_dis,
                                })

                    # Direct public API check for one temperature/reference on
                    # one case per representation/shape/seed. This deliberately
                    # recomputes canonical pair features from the refined bank.
                    if test_i == 0:
                        base_api = direct_output_law(base_model, x, b, temperature=0.02)
                        rep_api = direct_output_law(model, x, br, temperature=0.02)
                        ar = max_abs(base_api['output_masses'], rep_api['output_masses'])
                        global_max['direct_api_output_law'] = max(
                            global_max['direct_api_output_law'], ar)
                        direct_api_checks += 1
                        if ar > TOL:
                            raise AssertionError((shape, fit_seed, spec['name'], 'direct_api', ar))

                retraining_comparisons += 1
                for k in row_max:
                    global_max[k] = max(global_max[k], row_max[k])
                detail_rows.append({
                    'shape': f'{shape[0]}x{shape[1]}',
                    'fit_seed': fit_seed,
                    'representation': spec['name'],
                    'kind': spec['kind'],
                    'max_predicted_cost_residual': row_max['predicted_cost'],
                    'max_output_law_residual': row_max['output_law'],
                    'max_expected_observable_residual': row_max['expected_observable'],
                    'max_sample_decode_disagreement': row_max['sample_decode_disagreement'],
                    'max_map_decode_disagreement': row_max['map_decode_disagreement'],
                })

    detail_path = ROOT / 'results' / 'step3_invariance_stress_detail.csv'
    with detail_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader(); writer.writerows(detail_rows)

    summary = {
        'status': 'PASS',
        'purpose': 'Step-3 deterministic retraining stress test; not benchmark evidence',
        'required_tolerance': TOL,
        'image_shapes': [list(s) for s in SHAPES],
        'fit_seeds': list(FIT_SEEDS),
        'duplicate_copy_counts': list(DUPLICATE_COUNTS),
        'duplicated_action_indices': list(DUPLICATE_ACTIONS),
        'duplicated_action_names': [NAMES[j] for j in DUPLICATE_ACTIONS],
        'permutation_seeds': list(PERMUTATION_SEEDS),
        'temperatures': list(TEMPERATURES),
        'reference_laws': ['uniform', 'output-dependent strictly positive'],
        'representation_count_including_base': len(specs),
        'all_representation_signature_checks': signature_checks,
        'retraining_representation_count_per_shape_seed': len(retrain_specs),
        'nonbase_retraining_comparisons': retraining_comparisons,
        'law_comparisons': law_comparisons,
        'direct_output_law_api_checks': direct_api_checks,
        'sample_draws_per_law': 128,
        'training_images_per_shape': 14,
        'test_images_per_shape': 4,
        'trees_per_retraining_fit': 8,
        'maximum_action_coordinates_after_refinement': int(max_action_count),
        'global_maxima': global_max,
        'detail_csv': str(detail_path.relative_to(ROOT)),
        'interpretation': (
            'All tested output-preserving representation changes retained the exact '
            'canonical output/cost signature. Across the stratified retraining matrix, '
            'matched deterministic fits retained predicted costs, output masses, '
            'expected observables, MAP outputs, fixed-uniform sampled outputs, and the '
            'public direct_output_law API to the stated tolerance.'
        ),
    }
    out = ROOT / 'results' / 'step3_invariance_stress_summary.json'
    out.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
