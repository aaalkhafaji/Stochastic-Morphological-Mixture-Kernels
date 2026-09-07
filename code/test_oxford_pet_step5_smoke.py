#!/usr/bin/env python3
"""Offline smoke tests for the Step-5 Oxford campaign code.

Does not claim Oxford benchmark evidence. It validates preprocessing, corruption,
cache row construction, and representation-invariant candidate bookkeeping using
synthetic masks at the target 128x128 resolution.
"""
from pathlib import Path
import json, tempfile
import numpy as np
from PIL import Image

from run_oxford_pet_step5 import letterbox_bool, corrupt, extract_case, CONDITIONS

rng=np.random.default_rng(20260907)
checks=[]
for h,w in [(57,91),(211,133),(128,128),(90,240)]:
    y=np.zeros((h,w),bool)
    yy,xx=np.ogrid[:h,:w]
    y[((yy-h*.5)/(h*.3))**2+((xx-w*.5)/(w*.35))**2<1]=1
    z=letterbox_bool(y,128)
    assert z.shape==(128,128) and z.dtype==bool
    for ci,(_,d,a) in enumerate(CONDITIONS[:2]):
        x=corrupt(z,d,a,rng)
        e=extract_case(x,z)
        assert e['scores'].shape[0]==31
        assert e['action_features'].shape==(160,)
        assert e['direct_features'].shape==(31,10)
        assert 1 <= int(e['n_distinct']) <= 31
        reps=e['direct_reps'][:int(e['n_distinct'])]
        assert np.all(reps>=0)
        checks.append({'source_shape':[h,w],'condition':ci,'n_distinct':int(e['n_distinct']),
                       'input_loss':float(e['scores'][0,0]),'oracle_loss':float(e['scores'][:,0].min())})
print(json.dumps({'status':'PASS','checks':checks},indent=2))
