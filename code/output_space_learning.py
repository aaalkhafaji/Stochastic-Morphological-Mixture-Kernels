"""Direct learning on distinct morphological outputs.

This module implements the Step-2 representation-invariant learner.  Unlike
``kernel_quotient.py``, which collapses predictions produced in action
coordinates, the functions here remove duplicate output masks *before* fitting.
The scalar cost regressor receives only phi(x, z), where z is a distinct
candidate output.  Consequently, action permutations and exact duplicate labels
leave the canonical training table unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
from scipy.special import softmax
from sklearn.ensemble import ExtraTreesRegressor

from kernel_quotient import output_partition
from morphology import pair_features


@dataclass(frozen=True)
class UniqueOutputRows:
    """Canonical distinct-output representation for one observed image."""

    representatives: np.ndarray
    inverse: np.ndarray
    counts: np.ndarray
    pair_features: np.ndarray
    costs: Optional[np.ndarray] = None


def unique_output_rows(x, outputs, costs=None, consistency_tol=1e-12):
    """Collapse an action bank to one canonical row per distinct output.

    Parameters
    ----------
    x : (H, W) bool array
        Observed mask.
    outputs : (m, H, W) array
        Candidate masks.  Action order and multiplicity are arbitrary.
    costs : optional (m,) array
        Target loss of each candidate.  Equal masks must have equal costs up to
        ``consistency_tol`` because the target loss is assumed to depend on the
        realized output rather than its action label.

    Returns
    -------
    UniqueOutputRows
        Rows are ordered by the exact output-byte key used by
        :func:`kernel_quotient.output_partition`, not by action identity.
    """
    x = np.asarray(x, dtype=bool)
    outputs = np.asarray(outputs, dtype=bool)
    if outputs.ndim != x.ndim + 1 or outputs.shape[1:] != x.shape:
        raise ValueError("outputs must have shape (m, *x.shape)")
    representatives, inverse, counts = output_partition(outputs)
    phi = np.stack([pair_features(x, outputs[j]) for j in representatives])

    unique_costs = None
    if costs is not None:
        costs = np.asarray(costs, dtype=float)
        if costs.shape != (len(outputs),) or not np.isfinite(costs).all():
            raise ValueError("one finite scalar cost is required per action")
        unique_costs = costs[representatives].copy()
        for a in range(len(counts)):
            vals = costs[inverse == a]
            if np.max(np.abs(vals - vals[0])) > consistency_tol:
                raise ValueError(
                    "equal output masks have inconsistent target costs; "
                    "the direct-output target must be a function of (output, target)"
                )
            unique_costs[a] = vals[0]

    return UniqueOutputRows(
        representatives=np.asarray(representatives, dtype=np.int64),
        inverse=np.asarray(inverse, dtype=np.int64),
        counts=np.asarray(counts, dtype=np.int64),
        pair_features=np.asarray(phi, dtype=float),
        costs=unique_costs,
    )


def build_training_table(inputs: Sequence[np.ndarray],
                         output_banks: Sequence[np.ndarray],
                         cost_banks: Sequence[np.ndarray]):
    """Build the canonical scalar-regression table over distinct outputs.

    The input-image order is retained, while rows within each image use the
    canonical exact-byte output order.  Action labels never enter the table.
    """
    if not (len(inputs) == len(output_banks) == len(cost_banks)):
        raise ValueError("inputs, output_banks, and cost_banks must align")
    X, y, image_index, output_rank = [], [], [], []
    for i, (x, outputs, costs) in enumerate(zip(inputs, output_banks, cost_banks)):
        rows = unique_output_rows(x, outputs, costs)
        X.append(rows.pair_features)
        y.append(rows.costs)
        image_index.extend([i] * len(rows.representatives))
        output_rank.extend(range(len(rows.representatives)))
    if not X:
        raise ValueError("at least one training image is required")
    return (np.vstack(X), np.concatenate(y),
            np.asarray(image_index, dtype=np.int64),
            np.asarray(output_rank, dtype=np.int64))


def fit_direct_output_forest(inputs, output_banks, cost_banks, seed=101,
                             min_samples_leaf=8, n_estimators=96):
    """Fit the scalar direct-output ExtraTrees cost predictor.

    With fixed library version, fixed seed, fixed fitting options, and the same
    canonical table, this call is deterministic.  The theoretical invariance
    statement treats the fitting routine abstractly as a deterministic
    functional of that table.
    """
    X, y, image_index, output_rank = build_training_table(
        inputs, output_banks, cost_banks)
    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        max_features=1.0,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
        n_jobs=1,
    )
    model.fit(X, y)
    return model, {
        "n_images": int(len(inputs)),
        "n_distinct_rows": int(len(y)),
        "feature_dimension": int(X.shape[1]),
        "seed": int(seed),
        "min_samples_leaf": int(min_samples_leaf),
        "n_estimators": int(n_estimators),
        "image_index": image_index,
        "output_rank": output_rank,
    }


def direct_output_law(model, x, outputs, temperature, reference=None):
    """Predict distinct-output costs and return Gibbs output/action laws."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    rows = unique_output_rows(x, outputs)
    costs = np.asarray(model.predict(rows.pair_features), dtype=float)
    if costs.shape != (len(rows.representatives),) or not np.isfinite(costs).all():
        raise ValueError("the fitted model must return one finite cost per output")

    if reference is None:
        masses = softmax(-(costs - costs.min()) / temperature)
    else:
        nu = np.asarray(reference, dtype=float)
        if nu.shape != costs.shape or np.any(nu <= 0) or not np.isfinite(nu).all():
            raise ValueError("reference must be finite, strictly positive, and aligned")
        nu = nu / nu.sum()
        logw = np.log(nu) - (costs - costs.min()) / temperature
        masses = softmax(logw)

    lifted = masses[rows.inverse] / rows.counts[rows.inverse]
    return {
        "representatives": rows.representatives,
        "inverse": rows.inverse,
        "counts": rows.counts,
        "pair_features": rows.pair_features,
        "predicted_costs": costs,
        "output_masses": masses,
        "action_lift": lifted,
    }


def expected_output_loss(law, action_costs):
    """Exact expected cost of a distinct-output law using action-aligned costs."""
    action_costs = np.asarray(action_costs, dtype=float)
    inverse = np.asarray(law["inverse"], dtype=np.int64)
    masses = np.asarray(law["output_masses"], dtype=float)
    class_cost = np.empty(len(masses), dtype=float)
    for a in range(len(masses)):
        vals = action_costs[inverse == a]
        if not len(vals):
            raise ValueError("empty output class")
        class_cost[a] = vals[0]
    return float(np.dot(masses, class_cost))
