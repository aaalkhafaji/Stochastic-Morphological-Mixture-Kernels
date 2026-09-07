"""Finite binary morphology, topology, features, and exact categorical gradients."""
import numpy as np
from scipy import ndimage as ndi
from scipy.special import softmax

FG = np.ones((3, 3), dtype=bool)
BG = ndi.generate_binary_structure(2, 1)
METRICS = ['loss', 'mse', 'dice', 'iou', 'beta0_error', 'beta1_error', 'topology_exact']

def footprint(kind, size):
    if kind == 'square': return np.ones((size, size), bool)
    if kind == 'diamond':
        a = np.arange(-size, size + 1)
        return abs(a[:, None]) + abs(a[None, :]) <= size
    if kind == 'horizontal': return np.ones((1, size), bool)
    if kind == 'vertical': return np.ones((size, 1), bool)
    raise ValueError(kind)

def library():
    specs = [('identity', None, 0)]
    specs += [('O', 'square', n) for n in (2, 3, 4, 5)]
    shapes = [('square', 3), ('square', 5), ('diamond', 1), ('diamond', 2),
              ('horizontal', 3), ('vertical', 3)]
    specs += [(op, kind, size) for kind, size in shapes for op in ('C', 'OC', 'CO')]
    specs += [('O', kind, size) for kind, size in shapes[2:]]
    specs += [('area', 'connected', t) for t in (3,8,16,32)]
    return specs

SPECS = library()
NAMES = [f'{op}_{kind}{size}' if kind else op for op, kind, size in SPECS]

def transform(x, spec):
    """Operations on Z^2 with zero extension, cropping after the full pipeline.

    OC means opening first, closing second. origin=0 is fixed also for even sizes.
    Ten-pixel padding exceeds the propagation radius of all listed pipelines.
    """
    op, kind, size = spec
    if op == 'identity': return x.astype(bool).copy()
    if op == 'area': return area_filter(x,size)
    a = np.pad(x.astype(bool), 10)
    f = footprint(kind, size)
    for letter in op:
        fun = ndi.binary_opening if letter == 'O' else ndi.binary_closing
        a = fun(a, structure=f, iterations=1, origin=0, border_value=0)
    return a[10:-10, 10:-10]

def area_filter(x, threshold):
    """Remove small 8-connected foreground components, then fill small 4-holes.

    Strict area cutoff: size < threshold. Exterior background is never filled.
    """
    lab,n=ndi.label(x,structure=FG)
    counts=np.bincount(lab.ravel());keep=counts>=threshold;keep[0]=False
    a=keep[lab]
    lab,n=ndi.label(~np.pad(a,1),structure=BG)
    counts=np.bincount(lab.ravel());fill=counts<threshold;fill[0]=False;fill[lab[0,0]]=False
    return a | fill[lab][1:-1,1:-1]

def betti(x):
    """8-connected foreground components; bounded 4-connected background regions."""
    b0 = ndi.label(x, structure=FG)[1]
    b1 = ndi.label(~np.pad(x.astype(bool), 1), structure=BG)[1] - 1
    return np.array([b0, b1], dtype=np.int64)

def measure(y, truth, true_betti=None):
    y, truth = np.asarray(y, bool), np.asarray(truth, bool)
    a, b = int(y.sum()), int(truth.sum())
    inter = int(np.logical_and(y, truth).sum())
    dice = 2 * inter / (a + b) if a + b else 1.
    union = a + b - inter
    iou = inter / union if union else 1.
    mse = np.not_equal(y, truth).mean()
    tb = betti(truth) if true_betti is None else true_betti
    errors = abs(betti(y) - tb)
    topo = np.minimum(errors / (1 + tb), 1.)
    loss = .4 * (1 - dice) + .2 * mse + .2 * topo.sum()
    return np.array([loss, mse, dice, iou, *errors, float(not errors.any())])

def bank(x):
    return np.stack([transform(x, s) for s in SPECS])

def input_features(x):
    """Five action-label-free summaries of an observed binary mask."""
    x = np.asarray(x, bool)
    f = [x.mean()]
    for k in (3, 5):
        avg = ndi.uniform_filter(x.astype(float), size=k, mode='constant')
        f.extend([avg.std(), np.mean((avg >= .5) != x)])
    return np.asarray(f, dtype=float)


def candidate_features(x, y):
    """Five summaries of one candidate output, depending only on (x, y)."""
    x, y = np.asarray(x, bool), np.asarray(y, bool)
    b = betti(y)
    boundary = np.count_nonzero(y[1:] != y[:-1]) + np.count_nonzero(y[:, 1:] != y[:, :-1])
    return np.asarray([y.mean(), np.log1p(b[0]), np.log1p(b[1]),
                       (y != x).mean(), boundary / y.size], dtype=float)


def pair_features(x, y):
    """Ten-dimensional direct output-space feature map phi(x, y).

    The vector contains five input summaries followed by five summaries of the
    candidate output. It contains no action identity, class multiplicity, or
    other bank-label information, which is essential for representation
    invariance after deduplication.
    """
    return np.r_[input_features(x), candidate_features(x, y)]


def features(x, outputs):
    """Legacy 5+5m action-coordinate feature vector used by the baseline."""
    f = list(input_features(x))
    for y in outputs:
        f.extend(candidate_features(x, y))
    return np.asarray(f)

def patches(x, size=5):
    r = size // 2
    p = np.pad(x.astype(float), r)
    return np.lib.stride_tricks.sliding_window_view(p, (size, size)).reshape(-1, size * size)

def soft_transform(x, size, temperature, order):
    """Normalized log-sum-exp dilation and its dual erosion; smooth control."""
    a = np.pad(x.astype(float), 10)
    for letter in order:
        sequence = ('e', 'd') if letter == 'O' else ('d', 'e')
        for basic in sequence:
            if basic == 'd':
                a = 1 + np.log(ndi.uniform_filter(np.exp(temperature * (a - 1)), size=size, mode='constant', cval=np.exp(-temperature))) / temperature
            else:
                a = -np.log(ndi.uniform_filter(np.exp(-temperature * a), size=size, mode='constant', cval=1.)) / temperature
    return a[10:-10, 10:-10] >= .5

def softmax_objective(w, X, costs, penalty=1e-4):
    W = w.reshape(X.shape[1], costs.shape[1])
    p = softmax(X @ W, axis=1)
    r = np.sum(p * costs, axis=1, keepdims=True)
    value = r.mean() + .5 * penalty * np.sum(W * W)
    grad = X.T @ (p * (costs - r)) / len(X) + penalty * W
    return float(value), grad.ravel()
