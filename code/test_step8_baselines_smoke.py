#!/usr/bin/env python3
import json, tempfile
from pathlib import Path
import numpy as np
from softmorph_reference import apply
from run_baselines_step8 import encode3, fit_woperator_counts, woperator_predict

def main():
    rng=np.random.default_rng(8)
    ys=[]; xs=[]
    for i in range(8):
        y=np.zeros((24,24),bool); y[4+i%3:18,5:19]=True
        x=y.copy(); x ^= (rng.random(x.shape)<.05)
        ys.append(y); xs.append(x)
    X=np.stack(xs); Y=np.stack(ys)
    pred=apply(X,"opening",4,1,.75,.5)
    assert pred.shape==X.shape and pred.dtype==bool
    fg,tot=fit_woperator_counts(X,Y)
    q=woperator_predict(X,fg,tot,10.0,.5)
    assert q.shape==X.shape and q.dtype==bool
    assert np.all((encode3(X)>=0)&(encode3(X)<512))
    out={"status":"PASS","n_masks":len(X),"softmorph_shape":list(pred.shape),"w_patterns_seen":int((tot>0).sum())}
    Path("results").mkdir(exist_ok=True)
    Path("results/step8_baselines_smoke.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out,indent=2))
if __name__=="__main__": main()
