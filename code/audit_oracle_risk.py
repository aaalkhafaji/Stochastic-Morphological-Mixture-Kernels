"""Finite numerical checks for the Step-4 oracle/excess-risk decomposition.

These checks support, but do not replace, the manuscript proof. They sample finite
conditional-risk problems, arbitrary positive output references, approximate
costs, and bank refinements, then verify the pointwise/global inequalities.
"""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def gibbs(estimate, nu, tau):
    logw = np.log(nu) - np.asarray(estimate, float) / float(tau)
    logw -= np.max(logw)
    w = np.exp(logw)
    return w / w.sum()


def main():
    rng = np.random.default_rng(40720260907)
    batches = 2500
    total_inputs = 0
    max_pointwise_violation = 0.0
    max_uniform_violation = 0.0
    max_global_violation = 0.0
    max_decomposition_residual = 0.0
    max_hard_violation = 0.0
    max_monotonicity_violation = 0.0
    min_selection_excess = np.inf

    for _ in range(batches):
        nx = int(rng.integers(3, 14))
        eta = rng.dirichlet(np.ones(nx))
        R = RB = Rstar = 0.0
        rhs_global = 0.0
        for px in eta:
            nfull = int(rng.integers(4, 24))
            full_cost = rng.random(nfull)
            m = int(rng.integers(1, nfull + 1))
            bank_idx = np.sort(rng.choice(nfull, size=m, replace=False))
            r = full_cost[bank_idx]
            rho_star = float(full_cost.min())
            rho_bank = float(r.min())
            approx = rho_bank - rho_star

            err_radius = float(rng.uniform(0.0, 0.25))
            estimate = r + rng.uniform(-err_radius, err_radius, size=m)
            eQ = float(np.max(np.abs(estimate - r)))
            tau = float(np.exp(rng.uniform(np.log(1e-4), np.log(0.5))))
            nu = rng.dirichlet(rng.uniform(0.25, 3.0, size=m))
            q = gibbs(estimate, nu, tau)
            actual = float(q @ r)
            zstar = int(np.argmin(r))
            point_bound = 2 * eQ + tau * np.log(1.0 / nu[zstar])
            point_excess = actual - rho_bank
            max_pointwise_violation = max(max_pointwise_violation, point_excess - point_bound)
            min_selection_excess = min(min_selection_excess, point_excess)
            assert point_excess >= -2e-14
            assert point_excess <= point_bound + 2e-12

            # Uniform-reference specialization.
            unif = np.full(m, 1.0 / m)
            qu = gibbs(estimate, unif, tau)
            uniform_excess = float(qu @ r) - rho_bank
            uniform_bound = 2 * eQ + tau * np.log(m)
            max_uniform_violation = max(max_uniform_violation, uniform_excess - uniform_bound)
            assert uniform_excess <= uniform_bound + 2e-12

            # Hard output selector.
            hard = int(np.argmin(estimate))
            hard_excess = float(r[hard] - rho_star)
            hard_bound = approx + 2 * eQ
            max_hard_violation = max(max_hard_violation, hard_excess - hard_bound)
            assert hard_excess <= hard_bound + 2e-12

            # Bank enlargement can only lower approximation error.
            remaining = np.setdiff1d(np.arange(nfull), bank_idx, assume_unique=True)
            if len(remaining):
                add = rng.choice(remaining, size=int(rng.integers(1, len(remaining) + 1)), replace=False)
                enlarged = np.r_[bank_idx, add]
                approx2 = float(full_cost[enlarged].min() - rho_star)
                max_monotonicity_violation = max(max_monotonicity_violation, approx2 - approx)
                assert approx2 <= approx + 2e-14

            R += float(px) * actual
            RB += float(px) * rho_bank
            Rstar += float(px) * rho_star
            rhs_global += float(px) * (approx + point_bound)
            total_inputs += 1

        residual = abs((R - Rstar) - ((RB - Rstar) + (R - RB)))
        max_decomposition_residual = max(max_decomposition_residual, residual)
        global_excess = R - Rstar
        max_global_violation = max(max_global_violation, global_excess - rhs_global)
        assert RB >= Rstar - 2e-14
        assert R >= RB - 2e-14
        assert residual <= 2e-14
        assert global_excess <= rhs_global + 3e-12

    # Near-sharpness of coefficient 2 for hard selection.
    e = 0.2
    eps = 1e-9
    r = np.array([0.0, 2 * e - eps])
    estimate = np.array([e, e - eps])
    assert np.max(np.abs(estimate - r)) <= e + 1e-15
    hard = int(np.argmin(estimate))
    regret = float(r[hard] - r.min())
    sharp_ratio = regret / (2 * e)
    assert sharp_ratio > 0.99999999

    result = {
        'status': 'PASS',
        'random_batches': batches,
        'conditional_inputs_checked': total_inputs,
        'max_pointwise_oracle_bound_violation': max(0.0, float(max_pointwise_violation)),
        'max_uniform_reference_bound_violation': max(0.0, float(max_uniform_violation)),
        'max_global_oracle_bound_violation': max(0.0, float(max_global_violation)),
        'max_exact_decomposition_residual': float(max_decomposition_residual),
        'max_hard_selector_bound_violation': max(0.0, float(max_hard_violation)),
        'max_bank_enlargement_monotonicity_violation': max(0.0, float(max_monotonicity_violation)),
        'minimum_selection_excess_seen': float(min_selection_excess),
        'hard_selector_2e_near_sharp_ratio': float(sharp_ratio),
        'scope': 'Finite numerical sanity checks of the written theorem; not formal proof or independent benchmark evidence.'
    }
    out = ROOT / 'results' / 'step4_oracle_risk_checks.json'
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
