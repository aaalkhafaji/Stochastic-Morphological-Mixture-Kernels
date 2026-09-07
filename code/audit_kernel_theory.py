"""Independent finite checks for the expanded kernel theory.

Exact combinatorial enumeration supplements the written proofs. It is not
formal verification or external mathematical peer review.
"""
from pathlib import Path
import json
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog
from scipy.special import softmax
from morphology import betti
from kernel_quotient import output_partition, quotient_weights, aggregate_action_mass

ROOT = Path(__file__).resolve().parents[1]


def cc_bits(bits, h, w, connectivity):
    full = (1 << (h*w)) - 1
    left = sum(1 << (i*w) for i in range(h))
    right = left << (w-1)
    count = 0
    while bits:
        frontier = bits & -bits
        bits ^= frontier
        count += 1
        while frontier:
            horizontal = ((frontier & ~left) >> 1) | ((frontier & ~right) << 1)
            adjacent = horizontal | (frontier >> w) | (frontier << w)
            if connectivity == 8:
                adjacent |= (horizontal >> w) | (horizontal << w)
            frontier = adjacent & bits & full
            bits &= ~frontier
    return count


def graph_betti(bits, h, w):
    padded = 0
    for i in range(h):
        row = (bits >> (i*w)) & ((1 << w)-1)
        padded |= row << ((i+1)*(w+2)+1)
    background = ((1 << ((h+2)*(w+2)))-1) ^ padded
    return (cc_bits(bits, h, w, 8), cc_bits(background, h+2, w+2, 4)-1)


def touched_components(vertices, p, connectivity):
    remaining = set(vertices); touched = 0
    neighbors = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)
                 if (i or j) and (connectivity == 8 or abs(i)+abs(j) == 1)]
    while remaining:
        component = {remaining.pop()}; stack = list(component)
        while stack:
            u = stack.pop()
            near = {(u[0]+d[0], u[1]+d[1]) for d in neighbors} & remaining
            remaining -= near; component |= near; stack.extend(near)
        touched += any((p[0]+d[0], p[1]+d[1]) in component for d in neighbors)
    return touched


def exact_topology_checks():
    count = 0; scipy_matches = 0; max_changes = np.zeros(2, dtype=int)
    for h, w in [(1, 1), (1, 4), (4, 1), (2, 3), (3, 3), (4, 4)]:
        vals = np.array([graph_betti(i, h, w) for i in range(1 << (h*w))], dtype=int)
        for bits, value in enumerate(vals):
            mask = np.array([(bits >> j) & 1 for j in range(h*w)], bool).reshape(h, w)
            assert np.array_equal(value, betti(mask))
            scipy_matches += 1
        states = np.arange(len(vals))
        for j in range(h*w):
            changes = abs(vals[states ^ (1 << j)] - vals)
            assert np.all(changes <= 3)
            max_changes = np.maximum(max_changes, changes.max(0)); count += len(vals)
        print('Exact graph enumeration:', h, w, len(vals), flush=True)
    # Exact local identities, with global component incidences and exterior frame.
    identities = 0; h = w = 3
    domain = {(i, j) for i in range(h) for j in range(w)}
    extended = {(i, j) for i in range(-1, h+1) for j in range(-1, w+1)}
    for bits in range(512):
        F = {(i, j) for i, j in domain if (bits >> (i*w+j)) & 1}
        for p in domain-F:
            new = bits | (1 << (p[0]*w+p[1]))
            c8 = touched_components(F, p, 8)
            c4 = touched_components(extended-(F | {p}), p, 4)
            delta = np.array(graph_betti(new, h, w))-np.array(graph_betti(bits, h, w))
            assert np.array_equal(delta, [1-c8, c4-1]); identities += 1
    # Both constants are attained, including the background split into four holes.
    x = np.zeros((3, 3), bool); x[::2, ::2] = True
    y = x.copy(); y[1, 1] = True
    assert betti(y)[0]-betti(x)[0] == -3
    x = np.ones((5, 5), bool)
    x[2, 1:4] = False; x[1:4, 2] = False
    y = x.copy(); y[2, 2] = True
    assert tuple(betti(x)) == (1, 1) and tuple(betti(y)) == (1, 4)
    return {'single_pixel_comparisons': count, 'independent_graph_vs_scipy_masks': scipy_matches,
            'exact_local_identity_cases': identities, 'largest_changes': max_changes.tolist(),
            'sharpness_foreground_and_holes': True}


def transport_checks(rng):
    states = np.array([[(i >> k) & 1 for k in range(3)] for i in range(8)])
    cost = np.count_nonzero(states[:, None] != states[None, :], axis=2).astype(float)
    n = len(states)
    A = np.r_[np.kron(np.eye(n), np.ones((1, n))),
              np.kron(np.ones((1, n)), np.eye(n))]
    def distance(p, q):
        result = linprog(cost.ravel(), A_eq=A, b_eq=np.r_[p, q], bounds=(0, None), method='highs')
        assert result.success
        return float(result.fun)
    K = rng.dirichlet(np.ones(n)*2, size=n)
    L = rng.dirichlet(np.ones(n)*2, size=n)
    def coefficient(M):
        return max(distance(M[i], M[j])/cost[i, j] for i in range(n) for j in range(i))
    a, b = coefficient(K), coefficient(L)
    composed = coefficient(K @ L)
    assert composed <= a*b+1e-10
    # Binary 1x3 Betti counts are still governed by the fixed 8/4 convention.
    obs = np.array([graph_betti(i, 1, 3) for i in range(n)])
    residual = 0.
    for _ in range(32):
        p, q = rng.dirichlet(np.ones(n), size=2)
        d = distance(p, q)
        assert distance(p @ K, q @ K) <= a*d+1e-10
        assert np.max(abs(p @ obs-q @ obs)) <= 3*d+1e-10
        # Same-index coupling bound for state-dependent mixtures of finite maps.
        maps = rng.integers(0, n, size=(5, n)); x, y = rng.choice(n, size=2, replace=False)
        u, v = rng.dirichlet(np.ones(5), size=2)
        ku = np.bincount(maps[:, x], weights=u, minlength=n)
        kv = np.bincount(maps[:, y], weights=v, minlength=n)
        bound = np.minimum(u, v) @ cost[maps[:, x], maps[:, y]]+3*.5*abs(u-v).sum()
        actual = distance(ku, kv); assert actual <= bound+1e-10
        residual = max(residual, actual-bound)
    return {'transport_pairs_checked': 32, 'coefficient_K': a, 'coefficient_L': b,
            'coefficient_KL': composed, 'coefficient_product_bound': a*b,
            'same_index_bound_max_violation': max(0., residual)}


def entropy_and_risk_checks(rng):
    kl_error = 0.; replication_error = 0.; regret_violation = 0.
    for _ in range(2000):
        m = int(rng.integers(2, 20)); outputs = rng.integers(0, 2, size=(m, 3), dtype=np.uint8)
        part = output_partition(outputs); representatives, inverse, counts = part; qn = len(counts)
        truth = rng.random(qn); error = rng.uniform(0, .1)
        estimate = truth[inverse]+rng.uniform(-error, error, size=m)
        tau = rng.uniform(.001, .3)
        lifted, q, scores = quotient_weights(estimate, part, tau)
        gap = q @ truth-truth.min(); bound = 2*error+tau*np.log(qn)
        assert gap <= bound+1e-12; regret_violation = max(regret_violation, gap-bound)
        # Full KL chain rule, using arbitrary positive action priors and masses.
        p = rng.dirichlet(np.ones(m)); omega = rng.dirichlet(np.ones(m))
        nu = aggregate_action_mass(omega, part); pq = aggregate_action_mass(p, part)
        direct = np.sum(p*np.log(p/omega)); coarse = np.sum(pq*np.log(pq/nu))
        conditional = np.sum(p*np.log((p/pq[inverse])/(omega/nu[inverse])))
        kl_error = max(kl_error, abs(direct-coarse-conditional)); assert abs(direct-coarse-conditional) < 1e-12
        j = int(rng.integers(m)); extra = 8; idx = np.r_[np.arange(m), np.repeat(j, extra)]
        rep_part = output_partition(outputs[idx]); rp = p[idx]/p[idx].sum()
        observed = .5*abs(aggregate_action_mass(rp, rep_part)-pq).sum()
        expected = extra*p[j]*(1-pq[inverse[j]])/(1+extra*p[j])
        replication_error = max(replication_error, abs(observed-expected))
        assert abs(observed-expected) < 1e-12 and observed <= .5+1e-12
        nq = quotient_weights(estimate[idx], rep_part, tau)[1]
        assert np.max(abs(q-nq)) < 1e-12
    # Sharp replication bound at p=1/4 and eight additional copies.
    sharp = Fraction(8)*Fraction(1, 4)*Fraction(3, 4)/(1+8*Fraction(1, 4))
    assert sharp == Fraction(1, 2)
    return {'trials': 2000, 'kl_chain_rule_max_residual': kl_error,
            'replication_formula_max_residual': replication_error,
            'quotient_regret_max_violation': max(0., regret_violation),
            'sharp_replication_tv_exact': str(sharp)}


def bayes_and_gradient_checks(rng):
    prior = rng.dirichlet(np.ones(4)); H = rng.dirichlet(np.ones(5), size=4)
    nu = prior @ H; reverse = (prior[:, None]*H).T/nu[:, None]
    assert np.max(abs(nu @ reverse-prior)) < 1e-14
    P, Q = rng.dirichlet(np.ones(5), size=2)
    error = .5*np.minimum(P, Q).sum()
    best = .5*np.maximum(P, Q).sum()
    assert abs(error-(1-best)) < 1e-14
    # Complete shared-parameter gradient of finite categorical kernel risks.
    feature = rng.normal(size=(7, 4)); W = rng.normal(size=(4, 5)); losses = rng.random((7, 5))
    def fun(W):
        p = softmax(feature @ W, axis=1)
        return np.sum(p*losses)/len(feature)
    p = softmax(feature @ W, axis=1); means = np.sum(p*losses, axis=1, keepdims=True)
    analytical = feature.T @ (p*(losses-means))/len(feature)
    numerical = np.empty_like(W); step = 1e-6
    for idx in np.ndindex(W.shape):
        e = np.zeros_like(W); e[idx] = step
        numerical[idx] = (fun(W+e)-fun(W-e))/(2*step)
    residual = float(np.max(abs(numerical-analytical))); assert residual < 1e-8
    return {'bayes_marginal_residual': float(np.max(abs(nu @ reverse-prior))),
            'two_state_testing_identity_residual': abs(error-(1-best)),
            'gradient_max_residual': residual}


def main():
    rng = np.random.default_rng(620260906)
    result = {'topology': exact_topology_checks(), 'entropy_risk': entropy_and_risk_checks(rng),
              'transport': transport_checks(rng), 'bayes_gradient': bayes_and_gradient_checks(rng),
              'scope': 'Finite and numerical checks; not formal verification or external proof review.'}
    (ROOT/'results/kernel_theory_checks.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
