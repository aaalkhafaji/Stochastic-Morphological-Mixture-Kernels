#!/usr/bin/env python3
"""Cross-check local product-logic reference against public SoftMorph2 code."""
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np
import torch
from softmorph_reference import OPS


def load_module(path):
    spec=importlib.util.spec_from_file_location("softmorph2_external",str(path))
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("source"); ap.add_argument("--out")
    args=ap.parse_args(); mod=load_module(Path(args.source))
    rng=np.random.default_rng(20260907)
    x=rng.uniform(.03,.97,size=(3,13,17)).astype(np.float32)
    xt=torch.from_numpy(x[:,None])
    classes={"erosion":mod.SoftErosion,"dilation":mod.SoftDilation,
             "opening":mod.SoftOpening,"closing":mod.SoftClosing}
    worst=0.0; checks=[]
    with torch.no_grad():
        for op,cls in classes.items():
            obj=cls()
            for conn in (4,8):
                for it in (1,2):
                    if op in ("erosion","dilation"):
                        y=obj(xt.clone(),iterations=it,connectivity=conn,method="product")
                    else:
                        y=obj(xt.clone(),iterations=it,dilation_connectivity=conn,
                              erosion_connectivity=conn,method="product")
                    got=y.detach().cpu().numpy()[:,0]
                    ref=OPS[op](x,it,conn)
                    err=float(np.max(np.abs(got-ref))); worst=max(worst,err)
                    checks.append({"operation":op,"connectivity":conn,"iterations":it,"max_abs":err})
    result={"checks":checks,"max_abs_discrepancy":worst,"tolerance":2e-6,
            "status":"PASS" if worst<=2e-6 else "FAIL",
            "torch_version":torch.__version__}
    print(json.dumps(result,indent=2))
    if args.out: Path(args.out).write_text(json.dumps(result,indent=2)+"\n")
    if result["status"]!="PASS": raise SystemExit(1)
if __name__=="__main__": main()
