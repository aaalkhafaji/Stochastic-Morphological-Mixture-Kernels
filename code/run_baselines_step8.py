#!/usr/bin/env python3
"""Step 8: modern and fair morphology/post-processing baselines.

Frozen protocol (before Step-8 outcomes)
----------------------------------------
Primary fit/tune source is the successful Step-5 Oxford-IIIT Pet artifact.
No RGB image is used.  New baselines see the same binary corrupted masks and the
same clean targets as the manuscript selectors.

Baselines added here:
1) condition-aware fixed 31-action morphology (strong same-bank control; IID only),
2) validation-selected SoftMorph2 product-logic post-processing, evaluated through
   an independently vectorized implementation that MUST first agree numerically
   with the public authors' SoftMorph_2D.py,
3) regularized empirical 3x3 translation-invariant W-operator learned from the
   same Oxford train masks.

The SoftMorph/W-operator choices are tuned on Oxford validation only, then frozen.
They are evaluated both on the Oxford IID held-out test and on the Step-7 nine-
family structured OOD test, with no retuning on structured errors.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, time, tempfile, zipfile
from pathlib import Path
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ.setdefault(k,"1")
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from morphology import METRICS, measure, betti, NAMES
from run_oxford_pet_step5 import CONDITIONS, corrupt, case_rng
from run_structured_step7 import (load_step5_artifact, PRIMARY_CONDITIONS,
                                  structured_corrupt, rng_for, measure7)
from softmorph_reference import apply as soft_apply

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL_SEED=20260907
STEP5_EXPECTED_SHA256="889e7df9af279974d46595444bf768c627602d188ca8c93d0f65b30739f16e96"
W_ALPHAS=[1.0,10.0,100.0]
W_THRESHOLDS=[.4,.5,.6]
SOFT_CONFIGS=[
    {"operation":op,"connectivity":conn,"iterations":it,"sigma":sig,"threshold":.5}
    for op in ("erosion","dilation","opening","closing")
    for conn in (4,8) for it in (1,2) for sig in (0.0,.75)
]
NEW_METHODS=["SoftMorph2 validation-selected","Empirical W3 operator"]


def sha256(path,chunk=1<<20):
    h=hashlib.sha256();
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(chunk),b''):h.update(b)
    return h.hexdigest()


def encode3(x: np.ndarray) -> np.ndarray:
    """Encode each zero-padded 3x3 binary neighborhood as an integer in [0,511]."""
    x=np.asarray(x,bool); squeeze=x.ndim==2
    if squeeze:x=x[None]
    weights=(1<<np.arange(9,dtype=np.uint16)).reshape(3,3)
    out=np.empty(x.shape,dtype=np.uint16)
    for i,a in enumerate(x):
        out[i]=ndi.correlate(a.astype(np.uint16),weights,mode='constant',cval=0)
    return out[0] if squeeze else out


def fit_woperator_counts(X: np.ndarray,Y: np.ndarray):
    fg=np.zeros(512,np.int64); tot=np.zeros(512,np.int64)
    for x,y in zip(X,Y):
        c=encode3(x).ravel(); yy=np.asarray(y,bool).ravel().astype(np.int64)
        tot += np.bincount(c,minlength=512)
        fg += np.bincount(c,weights=yy,minlength=512).astype(np.int64)
    return fg,tot


def woperator_predict(X,fg,tot,alpha,threshold):
    X=np.asarray(X,bool); squeeze=X.ndim==2
    if squeeze:X=X[None]
    out=np.empty_like(X,bool)
    for i,x in enumerate(X):
        c=encode3(x)
        center=x.astype(float)
        # Identity-centered beta-like backoff for rare/unseen neighborhoods.
        p=(fg[c] + float(alpha)*center)/(tot[c] + float(alpha))
        out[i]=p>=float(threshold)
    return out[0] if squeeze else out


def protocol_lists(protocol: pd.DataFrame):
    return {s:protocol.loc[protocol.analysis_split==s].reset_index(drop=True) for s in ("train","val","test")}


def reconstruct_split_inputs(cache, rows: pd.DataFrame, split: str):
    n=len(cache["condition"]); X=np.empty((n,128,128),bool); Y=np.empty_like(X)
    code={"train":0,"val":1,"test":2}[split]
    for qi in range(n):
        i=int(cache["image_index"][qi]); c=int(cache["condition"][qi]); truth=cache["clean_masks"][i].astype(bool)
        rec=rows.iloc[i]; _,d,a=CONDITIONS[c]
        X[qi]=corrupt(truth,d,a,case_rng(code,str(rec.image_id),c)); Y[qi]=truth
    return X,Y


def metrics_rows(preds, truths, meta, method, boundary=False):
    rows=[]; tb_cache={}
    for i,(p,y) in enumerate(zip(preds,truths)):
        im=int(meta.iloc[i].image)
        key=(im,id(y))
        tb=betti(y)
        m=measure7(p,y,tb) if boundary else measure(p,y,tb)
        names=METRICS+(["boundary_f"] if boundary else [])
        r={"method":method,**{c:meta.iloc[i][c] for c in meta.columns if c!="method"}}
        r.update({k:float(v) for k,v in zip(names,m)}); rows.append(r)
    return rows


def avg_existing_step5(path):
    d=pd.read_csv(path)
    keep=["Input","Validation best","Action-coordinate selector","Direct-output selector","Oracle (diagnostic)"]
    d=d[d.method.isin(keep)].copy()
    group=["dataset","method","image","class_id","species","condition"]
    metrics=[c for c in METRICS if c in d.columns]
    return d.groupby(group,as_index=False)[metrics].mean()


def condition_aware_fixed(val,test,protocol_test):
    best={}
    for c in range(len(CONDITIONS)):
        idx=np.flatnonzero(val["condition"]==c)
        best[c]=int(np.argmin(val["scores"][idx,:,0].mean(axis=0)))
    out=[]
    for qi in range(len(test["condition"])):
        c=int(test["condition"][qi]); a=best[c]; im=int(test["image_index"][qi]); rec=protocol_test.iloc[im]
        r={"dataset":"oxford_pet","method":"Condition-aware fixed morphology","image":im,
           "class_id":int(rec.class_id),"species":int(rec.species),"condition":CONDITIONS[c][0]}
        r.update({k:float(v) for k,v in zip(METRICS,test["scores"][qi,a])}); out.append(r)
    return pd.DataFrame(out),{CONDITIONS[c][0]:{"action_index":a,"action":NAMES[a]} for c,a in best.items()}


def mean_loss(pred,Y):
    vals=[]
    for p,y in zip(pred,Y): vals.append(measure(p,y)[0])
    return float(np.mean(vals))


def select_woperator(Xtr,Ytr,Xv,Yv):
    fg,tot=fit_woperator_counts(Xtr,Ytr); trials=[]
    for a in W_ALPHAS:
        for t in W_THRESHOLDS:
            p=woperator_predict(Xv,fg,tot,a,t); loss=mean_loss(p,Yv)
            trials.append({"alpha":a,"threshold":t,"validation_loss":loss})
    best=min(trials,key=lambda z:z["validation_loss"])
    return fg,tot,best,trials


def select_softmorph(Xv,Yv,batch=64):
    trials=[]
    for ci,cfg in enumerate(SOFT_CONFIGS):
        total=0.; n=0; t0=time.perf_counter()
        for s in range(0,len(Xv),batch):
            q=soft_apply(Xv[s:s+batch],**cfg)
            for p,y in zip(q,Yv[s:s+batch]): total += measure(p,y)[0]; n+=1
        trials.append({**cfg,"validation_loss":total/n,"seconds":time.perf_counter()-t0})
        print(f"SoftMorph candidate {ci+1}/{len(SOFT_CONFIGS)} loss={total/n:.6f} {cfg}",flush=True)
    return min(trials,key=lambda z:z["validation_loss"]),trials


def selected_soft_predictions(X,cfg,batch=64):
    out=np.empty_like(X,bool)
    for s in range(0,len(X),batch): out[s:s+batch]=soft_apply(X[s:s+batch],**{k:cfg[k] for k in ("operation","connectivity","iterations","sigma","threshold")})
    return out


def step5_meta(cache,proto):
    rows=[]
    for qi in range(len(cache["condition"])):
        im=int(cache["image_index"][qi]); c=int(cache["condition"][qi]); rec=proto.iloc[im]
        rows.append({"dataset":"oxford_pet","image":im,"class_id":int(rec.class_id),"species":int(rec.species),"condition":CONDITIONS[c][0]})
    return pd.DataFrame(rows)


def make_ood_cases(clean,proto):
    X=[];Y=[];rows=[]
    for im,yu in enumerate(clean):
        y=yu.astype(bool); rec=proto.iloc[im]
        for ci,name in enumerate(PRIMARY_CONDITIONS):
            X.append(structured_corrupt(y,name,rng_for(im,ci)));Y.append(y)
            rows.append({"dataset":"oxford_pet_structured","image":im,"class_id":int(rec.class_id),"species":int(rec.species),"condition":name})
    return np.asarray(X,bool),np.asarray(Y,bool),pd.DataFrame(rows)


def summary(df,dataset):
    metrics=[c for c in METRICS+['boundary_f'] if c in df.columns]
    d=df[df.dataset==dataset]
    # average conditions within clean mask first
    g=d.groupby(["method","image"],as_index=False)[metrics].mean()
    rows=[]
    for m,sub in g.groupby("method"):
        for metric in metrics:
            rows.append({"dataset":dataset,"method":m,"metric":metric,"n_images":len(sub),"mean":float(sub[metric].mean())})
    return pd.DataFrame(rows)


def clustered_ci(df,dataset,n_boot=5000):
    metrics=[c for c in ("loss","dice","iou","topology_exact") if c in df.columns]
    d=df[df.dataset==dataset]; rows=[]
    for method,sub0 in d.groupby("method"):
        g=sub0.groupby("image",as_index=False)[metrics].mean(); n=len(g)
        for metric in metrics:
            vals=g[metric].to_numpy(float); rng=np.random.default_rng(PROTOCOL_SEED+int(hashlib.sha256((dataset+method+metric).encode()).hexdigest()[:8],16)%100000)
            boots=np.array([vals[rng.integers(0,n,n)].mean() for _ in range(n_boot)])
            lo,hi=np.quantile(boots,[.025,.975]); rows.append({"dataset":dataset,"method":method,"metric":metric,"mean":vals.mean(),"ci95_low":lo,"ci95_high":hi,"n_images":n,"bootstrap_resamples":n_boot})
    return pd.DataFrame(rows)


def paired(df,dataset,a,b,n_boot=5000,family=8):
    d=df[df.dataset==dataset]; A=d[d.method==a].groupby('image').loss.mean(); B=d[d.method==b].groupby('image').loss.mean(); common=A.index.intersection(B.index); z=(A.loc[common]-B.loc[common]).to_numpy(float); n=len(z)
    rng=np.random.default_rng(PROTOCOL_SEED+int(hashlib.sha256((dataset+a+b).encode()).hexdigest()[:8],16)%100000)
    boots=np.array([z[rng.integers(0,n,n)].mean() for _ in range(n_boot)])
    alpha=.05/family; lo,hi=np.quantile(boots,[alpha/2,1-alpha/2])
    return {"dataset":dataset,"a":a,"b":b,"difference":"a-b","estimate":float(z.mean()),"family_size":family,"familywise_confidence":1-.05,"low":float(lo),"high":float(hi),"n_images":n,"bootstrap_resamples":n_boot}


def latex_table(summary_df,out):
    methods=["Input","Validation best","Condition-aware fixed morphology","Action-coordinate selector","Direct-output selector","SoftMorph2 validation-selected","Empirical W3 operator"]
    lines=[r"\\begin{table}[t]",r"\\centering",r"\\caption{Step-8 fair-baseline comparison on Oxford-IIIT Pet. SoftMorph2 and the empirical $3\\times3$ W-operator are tuned on validation data only.}",r"\\label{tab:step8baselines}",r"\\small",r"\\begin{tabular}{lcccc}",r"\\toprule",r"Method & Loss $\\downarrow$ & Dice $\\uparrow$ & IoU $\\uparrow$ & Topology exact $\\uparrow$\\\\",r"\\midrule"]
    s=summary_df[summary_df.dataset=='oxford_pet'].pivot(index='method',columns='metric',values='mean')
    for m in methods:
        if m not in s.index: continue
        lines.append(f"{m} & {s.loc[m,'loss']:.4f} & {s.loc[m,'dice']:.4f} & {s.loc[m,'iou']:.4f} & {s.loc[m,'topology_exact']:.4f}\\\\")
    lines += [r"\\bottomrule",r"\\end{tabular}",r"\\end{table}"]
    Path(out).write_text("\n".join(lines)+"\n")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--step5-artifact',required=True); ap.add_argument('--step7-dir',default=str(ROOT/'results/step7_structured'))
    ap.add_argument('--softmorph-equivalence-json'); ap.add_argument('--softmorph-commit',default='unrecorded')
    ap.add_argument('--outdir',default=str(ROOT/'results/step8_baselines')); ap.add_argument('--n-boot',type=int,default=5000)
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args(); out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        caches,selection,protocol=load_step5_artifact(Path(args.step5_artifact),Path(td))
    proto=protocol_lists(protocol)
    if args.smoke:
        # deterministic tiny prefix; software check only
        for s in caches:
            n=min(24,len(caches[s]['condition']));
            for k,v in list(caches[s].items()):
                if isinstance(v,np.ndarray) and len(v.shape)>0 and v.shape[0]==len(caches[s]['condition']): caches[s][k]=v[:n]
            # clean mask arrays need corresponding prefix images
            max_i=int(caches[s]['image_index'].max())+1; caches[s]['clean_masks']=caches[s]['clean_masks'][:max_i]; proto[s]=proto[s].iloc[:max_i].reset_index(drop=True)
    Xtr,Ytr=reconstruct_split_inputs(caches['train'],proto['train'],'train'); Xv,Yv=reconstruct_split_inputs(caches['val'],proto['val'],'val')
    fg,tot,wbest,wtrials=select_woperator(Xtr,Ytr,Xv,Yv)
    sbest,strials=select_softmorph(Xv,Yv,batch=16 if args.smoke else 64)
    selection8={"protocol_seed":PROTOCOL_SEED,"step5_sha256":sha256(args.step5_artifact),"woperator":{"best":wbest,"trials":wtrials,"patterns_seen":int((tot>0).sum())},"softmorph":{"best":sbest,"trials":strials,"external_commit":args.softmorph_commit},"softmorph_equivalence_json":args.softmorph_equivalence_json,"no_step8_test_tuning":True}
    (out/'step8_selection.json').write_text(json.dumps(selection8,indent=2)+"\n")

    # IID held-out test.
    Xt,Yt=reconstruct_split_inputs(caches['test'],proto['test'],'test'); meta=step5_meta(caches['test'],proto['test'])
    sw=selected_soft_predictions(Xt,sbest,batch=16 if args.smoke else 64); ww=woperator_predict(Xt,fg,tot,wbest['alpha'],wbest['threshold'])
    new=pd.DataFrame(metrics_rows(sw,Yt,meta,'SoftMorph2 validation-selected'))
    new=pd.concat([new,pd.DataFrame(metrics_rows(ww,Yt,meta,'Empirical W3 operator'))],ignore_index=True)
    with zipfile.ZipFile(args.step5_artifact) as z:
        z.extract('step5_oxford_pet/oxford_pet_test_metrics.csv.gz',out)
    existing=avg_existing_step5(out/'step5_oxford_pet/oxford_pet_test_metrics.csv.gz')
    cond,cond_sel=condition_aware_fixed(caches['val'],caches['test'],proto['test'])
    iid=pd.concat([existing,cond,new],ignore_index=True,sort=False); iid.to_csv(out/'step8_iid_case_metrics.csv.gz',index=False,compression='gzip',float_format='%.9g')
    del Xt,Yt,sw,ww,new,existing,cond

    # Structured OOD test; no retuning.
    Xo,Yo,metao=make_ood_cases(caches['test']['clean_masks'],proto['test'])
    so=selected_soft_predictions(Xo,sbest,batch=16 if args.smoke else 64); wo=woperator_predict(Xo,fg,tot,wbest['alpha'],wbest['threshold'])
    nrows=pd.DataFrame(metrics_rows(so,Yo,metao,'SoftMorph2 validation-selected',boundary=True))
    nrows=pd.concat([nrows,pd.DataFrame(metrics_rows(wo,Yo,metao,'Empirical W3 operator',boundary=True))],ignore_index=True)
    old=pd.read_csv(Path(args.step7_dir)/'step7_case_metrics.csv.gz')
    old.insert(0,'dataset','oxford_pet_structured')
    ood=pd.concat([old,nrows],ignore_index=True,sort=False); ood.to_csv(out/'step8_ood_case_metrics.csv.gz',index=False,compression='gzip',float_format='%.9g')

    summ=pd.concat([summary(iid,'oxford_pet'),summary(ood,'oxford_pet_structured')],ignore_index=True); summ.to_csv(out/'step8_summary.csv',index=False,float_format='%.9g')
    cis=pd.concat([clustered_ci(iid,'oxford_pet',args.n_boot),clustered_ci(ood,'oxford_pet_structured',args.n_boot)],ignore_index=True); cis.to_csv(out/'step8_clustered_ci.csv',index=False,float_format='%.9g')
    comparisons=[]
    pairs=[('Direct-output selector','SoftMorph2 validation-selected'),('Action-coordinate selector','SoftMorph2 validation-selected'),('Direct-output selector','Empirical W3 operator'),('Action-coordinate selector','Empirical W3 operator')]
    for ds in ('oxford_pet','oxford_pet_structured'):
        for a,b in pairs: comparisons.append(paired(iid if ds=='oxford_pet' else ood,ds,a,b,args.n_boot,8))
    (out/'step8_paired_comparisons.json').write_text(json.dumps(comparisons,indent=2)+"\n")
    (out/'step8_condition_aware_selection.json').write_text(json.dumps(cond_sel,indent=2)+"\n")
    latex_table(summ,out/'generated_step8_baselines.tex')
    provenance={"status":"SMOKE" if args.smoke else "COMPLETE","primary_dataset":"Oxford-IIIT Pet","resolution":128,"step5_artifact_sha256":sha256(args.step5_artifact),"step7_directory":str(args.step7_dir),"softmorph_public_repository":"https://github.com/lisaGUZZI/SoftMorph2","softmorph_publication_doi":"10.1016/j.media.2026.104284","softmorph_commit":args.softmorph_commit,"softmorph_source_redistributed":False,"woperator":"regularized empirical 3x3 translation-invariant binary W-operator","training":"Step-5 Oxford IID train only","selection":"Step-5 Oxford validation only","structured_ood_retuning":False,"iid_test_cases":int(len(iid[iid.method=='Input'])),"structured_test_cases":int(len(ood[ood.method=='Input'])),"bootstrap_resamples":args.n_boot,"condition_aware_fixed_iid_only":True}
    (out/'step8_provenance.json').write_text(json.dumps(provenance,indent=2)+"\n")
    # Freeze result bytes.
    files=sorted(p for p in out.iterdir() if p.is_file() and p.name!='SHA256SUMS')
    (out/'SHA256SUMS').write_text(''.join(f"{sha256(p)}  {p.name}\n" for p in files))
    print(json.dumps({"status":provenance['status'],"softmorph_best":sbest,"woperator_best":wbest,"iid_cases":provenance['iid_test_cases'],"ood_cases":provenance['structured_test_cases']},indent=2))
if __name__=='__main__': main()
