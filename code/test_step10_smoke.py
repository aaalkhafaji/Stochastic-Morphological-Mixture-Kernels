#!/usr/bin/env python3
import numpy as np, pandas as pd
from run_step10_generalization import reliability_table, adaptive_ece, selective_curve, extra_specs, synthetic_mask, apply_spec

df=pd.DataFrame({'case':range(6),'confidence':[.95,.8,.7,.4,.3,.1],'top_correct':[1,1,0,1,0,0],
                 'map_loss':[.1,.2,.4,.3,.5,.8],'map_regret':[0,.1,.3,.1,.4,.7],'expected_regret':[.02,.12,.25,.15,.35,.6]})
tab,ece=reliability_table(df,5)
assert 0<=ece<=1 and 0<=adaptive_ece(df,3)<=1
sel=selective_curve(df); assert len(sel)==5 and sel.iloc[0].n_cases==6
assert len(extra_specs())==32
x=synthetic_mask(32); y=apply_spec(x,('O','square',7)); assert y.shape==x.shape
print({'status':'PASS','ece':ece,'n_extra_specs':len(extra_specs())})
