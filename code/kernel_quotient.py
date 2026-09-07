"""Finite morphological kernels on distinct output masks.

This module implements the archived post-hoc quotient baseline: the input bank
and its fitted action-risk predictions are frozen.  Replication invariance here
concerns copying an existing (output, risk) pair, not retraining.  The Step-2
module ``output_space_learning.py`` removes duplicates before fitting and covers
retraining under output-preserving representation changes.
"""
import numpy as np
from scipy.special import softmax


def output_partition(outputs):
    """Return representatives, action-to-output-class map, and class sizes.

    Exact array bytes determine equality. All output arrays must have a common
    shape and dtype. Packed binary masks have canonical zero padding bits.
    """
    a = np.ascontiguousarray(outputs)
    if a.ndim < 2 or len(a) == 0:
        raise ValueError('A nonempty bank of equally shaped outputs is required.')
    rows = a.reshape(len(a), -1)
    keys = rows.view(np.dtype((np.void, rows.shape[1] * rows.dtype.itemsize))).ravel()
    _, representatives, inverse, counts = np.unique(
        keys, return_index=True, return_inverse=True, return_counts=True)
    return representatives, inverse, counts


def quotient_weights(action_risks, partition, temperature):
    """Gibbs law on distinct outputs, uniformly lifted to action labels.

    The output score is the minimum fitted risk among labels for that output.
    It is unchanged by copying an existing (output, risk) pair. Its pointwise
    error is at most the maximum individual action-risk prediction error.
    """
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive')
    representatives, inverse, counts = partition
    risks = np.asarray(action_risks, dtype=float)
    if risks.shape != inverse.shape or not np.isfinite(risks).all():
        raise ValueError('One finite risk is required for every action.')
    output_risks = np.full(len(counts), np.inf)
    np.minimum.at(output_risks, inverse, risks)
    masses = softmax(-(output_risks - output_risks.min()) / temperature)
    lifted = masses[inverse] / counts[inverse]
    return lifted, masses, output_risks


def aggregate_action_mass(probabilities, partition):
    return np.bincount(partition[1], weights=probabilities,
                       minlength=len(partition[2]))


def maximum_replication_shift(probabilities, partition, extra_copies=8):
    """Largest row-law TV shift from replicating any one action label.

    This is a label-only stress test with fixed logits, with no target access.
    Each action is considered separately; the maximum is reported.
    """
    p = np.asarray(probabilities)
    q = aggregate_action_mass(p, partition)
    shifts = extra_copies * p * (1 - q[partition[1]]) / (1 + extra_copies * p)
    return float(np.max(shifts))
