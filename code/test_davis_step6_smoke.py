#!/usr/bin/env python3
"""Offline Step-6 smoke test; no DAVIS result is claimed."""
import json, tempfile
from pathlib import Path
import numpy as np
from PIL import Image

from run_davis_step6 import (
    PRIMARY_SOURCE_METHODS, boundary_f08, letterbox_bool, extract_case,
    split_train_sequences, discover_source_methods, png_map
)

rng=np.random.default_rng(20260907)
# Geometry and candidate bookkeeping at target scale.
y=np.zeros((83,137),bool); yy,xx=np.ogrid[:83,:137]
y[((yy-42)/25)**2+((xx-68)/43)**2<1]=1
x=y.copy(); x[35:40,70:79]=0; x[20:24,105:111]=1
ys=letterbox_bool(y,128); xs=letterbox_bool(x,128)
e=extract_case(xs,ys)
assert e['scores'].shape==(31,8)
assert e['action_features'].shape==(160,)
assert e['direct_features'].shape==(31,10)
assert 1 <= int(e['n_distinct']) <= 31
assert 0 <= boundary_f08(xs,ys) <= 1

# Deterministic sequence splitting.
seqs=[f'seq{i:02d}' for i in range(30)]
a=split_train_sequences(seqs,.2); b=split_train_sequences(seqs,.2)
assert a==b and len(a['train'])==24 and len(a['tune'])==6
assert set(a['train']).isdisjoint(a['tune'])

# Official-method directory discovery contract and PNG matching.
with tempfile.TemporaryDirectory() as td:
    root=Path(td)
    for method in PRIMARY_SOURCE_METHODS:
        d=root/method/'seq00'; d.mkdir(parents=True)
        Image.fromarray((ys*255).astype('uint8')).save(d/'00000.png')
    found=discover_source_methods(root,PRIMARY_SOURCE_METHODS)
    assert set(found)==set(PRIMARY_SOURCE_METHODS)
    assert set(png_map(found['NLC']/'seq00'))=={'00000'}

print(json.dumps({
  'status':'PASS', 'target_resolution':[128,128], 'n_actions':31,
  'n_distinct':int(e['n_distinct']), 'boundary_f08':float(boundary_f08(xs,ys)),
  'train_sequences':len(a['train']), 'tune_sequences':len(a['tune']),
  'source_methods':PRIMARY_SOURCE_METHODS
}, indent=2))
