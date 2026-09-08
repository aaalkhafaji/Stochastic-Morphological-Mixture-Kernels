"""Vectorized reference implementation of the SoftMorph2 product-logic 2D operators.

This module does NOT vendor SoftMorph source code.  It independently implements the
product-fuzzy formulas described by and present in the public SoftMorph2 2D code.
Step 8 downloads the authors' public implementation at run time and verifies this
reference numerically before any benchmark result is accepted.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi


def _neighbors(a: np.ndarray, outside: float) -> list[np.ndarray]:
    a=np.asarray(a,dtype=np.float32)
    if a.ndim==2: a=a[None,...]
    if a.ndim!=3: raise ValueError("expected [N,H,W] or [H,W]")
    p=np.pad(a,((0,0),(1,1),(1,1)),mode='constant',constant_values=float(outside))
    h,w=a.shape[-2:]
    # Same order as SoftMorph2: N, NE, E, SE, S, SW, W, NW, center.
    return [
        p[:,0:h,1:w+1], p[:,0:h,2:w+2], p[:,1:h+1,2:w+2],
        p[:,2:h+2,2:w+2], p[:,2:h+2,1:w+1], p[:,2:h+2,0:w],
        p[:,1:h+1,0:w], p[:,0:h,0:w], p[:,1:h+1,1:w+1],
    ]


def dilation(a, iterations=1, connectivity=4):
    a=np.asarray(a,dtype=np.float32)
    squeeze=(a.ndim==2)
    if squeeze:a=a[None,...]
    for _ in range(int(iterations)):
        n=_neighbors(a,0.0)
        idx=(0,2,4,6,8) if int(connectivity)==4 else range(9)
        prod=np.ones_like(a,dtype=np.float32)
        for j in idx: prod *= (1.0-n[j])
        a=1.0-prod
    return a[0] if squeeze else a


def erosion(a, iterations=1, connectivity=4):
    a=np.asarray(a,dtype=np.float32)
    squeeze=(a.ndim==2)
    if squeeze:a=a[None,...]
    for _ in range(int(iterations)):
        n=_neighbors(a,1.0)
        idx=(0,2,4,6,8) if int(connectivity)==4 else range(9)
        prod=np.ones_like(a,dtype=np.float32)
        for j in idx: prod *= n[j]
        # Matches SoftMorph2 SoftErosion.forward: im = im * output.
        a=a*prod
    return a[0] if squeeze else a


def opening(a, iterations=1, connectivity=4):
    return dilation(erosion(a,iterations,connectivity),iterations,connectivity)


def closing(a, iterations=1, connectivity=4):
    return erosion(dilation(a,iterations,connectivity),iterations,connectivity)

OPS={"erosion":erosion,"dilation":dilation,"opening":opening,"closing":closing}


def relax_binary(x: np.ndarray, sigma: float) -> np.ndarray:
    x=np.asarray(x,dtype=np.float32)
    if float(sigma)<=0: return x
    if x.ndim==2:
        return ndi.gaussian_filter(x,float(sigma),mode='constant',cval=0.0).astype(np.float32)
    return np.stack([ndi.gaussian_filter(z,float(sigma),mode='constant',cval=0.0) for z in x]).astype(np.float32)


def apply(a, operation: str, connectivity: int, iterations: int, sigma: float=0.0, threshold: float=.5):
    p=relax_binary(a,sigma)
    q=OPS[operation](p,iterations,connectivity)
    return np.asarray(q>=float(threshold),dtype=bool)
