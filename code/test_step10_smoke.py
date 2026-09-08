#!/usr/bin/env python3
import numpy as np, pandas as pd
from pathlib import Path
import tempfile
from run_davis_step6 import read_sequence_list
from run_step10_generalization import reliability_table, adaptive_ece, selective_curve, extra_specs, synthetic_mask, apply_spec

df=pd.DataFrame({'case':range(6),'confidence':[.95,.8,.7,.4,.3,.1],'top_correct':[1,1,0,1,0,0],
                 'map_loss':[.1,.2,.4,.3,.5,.8],'map_regret':[0,.1,.3,.1,.4,.7],'expected_regret':[.02,.12,.25,.15,.35,.6]})
tab,ece=reliability_table(df,5)
assert 0<=ece<=1 and 0<=adaptive_ece(df,3)<=1
sel=selective_curve(df); assert len(sel)==5 and sel.iloc[0].n_cases==6
assert len(extra_specs())==32
x=synthetic_mask(32); y=apply_spec(x,('O','square',7)); assert y.shape==x.shape
print({'status':'PASS','ece':ece,'n_extra_specs':len(extra_specs())})

# Regression: original DAVIS-2016 split files contain frame paths, not one sequence per row.
with tempfile.TemporaryDirectory() as td:
    p=Path(td)/"val.txt"
    p.write_text(
        "/JPEGImages/480p/bear/00000.jpg /Annotations/480p/bear/00000.png\n"
        "/JPEGImages/480p/bear/00001.jpg /Annotations/480p/bear/00001.png\n"
        "/JPEGImages/480p/blackswan/00000.jpg /Annotations/480p/blackswan/00000.png\n"
    )
    assert read_sequence_list(p)==["bear","blackswan"]
