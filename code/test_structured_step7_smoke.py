#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_structured_step7 import PRIMARY_CONDITIONS, structured_corrupt, component_dropout
from morphology import betti

y=np.zeros((64,64),bool); y[12:52,16:48]=1; y[25:32,26:37]=0
outs={}
for i,n in enumerate(PRIMARY_CONDITIONS):
    x=structured_corrupt(y,n,np.random.default_rng(100+i))
    assert x.shape==y.shape and x.dtype==bool
    assert np.any(x!=y), n
    outs[n]=x
assert np.all(outs["boundary_expand2"] | ~y)
assert np.all(~outs["boundary_contract2"] | y)
assert np.all(outs["clustered_fp03"] | ~y)
assert np.all(~outs["clustered_fn08"] | y)
assert np.all(~outs["holes2_r4"] | y)
assert np.all(~outs["fragment_cut2"] | y)
assert np.all(outs["bridge_spur12"] | ~y)
assert np.all(~outs["occlusion25"] | y)
# Natural component dropout only when eligible.
z=np.zeros((32,32),bool); z[2:10,2:10]=1; z[20:30,20:30]=1
q=component_dropout(z); assert q is not None and betti(q)[0]==1
assert component_dropout(y) is None
print({"status":"PASS","conditions":PRIMARY_CONDITIONS})
