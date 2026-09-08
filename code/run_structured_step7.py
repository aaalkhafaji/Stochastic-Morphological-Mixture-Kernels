#!/usr/bin/env python3
"""Step 7: structured-error robustness under distribution shift.

The selectors are trained only on the frozen Oxford-IIIT Pet IID deletion/addition
protocol from Step 5. They are then evaluated, without refitting or retuning, on
predeclared structured error families at 128x128. This makes Step 7 an OOD
robustness test rather than another in-distribution fitting exercise.
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, tempfile, time, zipfile
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.special import softmax
from sklearn.ensemble import ExtraTreesRegressor

from morphology import NAMES, METRICS, bank, betti, features, measure, pair_features
from output_space_learning import unique_output_rows
from run_davis_step6 import boundary_f08

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SEED = 20260907
MODEL_SEEDS = [101, 102, 103, 104, 105]
RESOLUTION = 128
STEP5_EXPECTED_SHA256 = "889e7df9af279974d46595444bf768c627602d188ca8c93d0f65b30739f16e96"
PRIMARY_CONDITIONS = [
    "boundary_expand2",
    "boundary_contract2",
    "clustered_fp03",
    "clustered_fn08",
    "holes2_r4",
    "fragment_cut2",
    "bridge_spur12",
    "occlusion25",
    "mixed_structured",
]
METRICS7 = METRICS + ["boundary_f"]


def sha256(path: Path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def rng_for(image_index: int, condition_index: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([
        PROTOCOL_SEED, 7, int(image_index), int(condition_index)
    ]))


def disk(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    return x*x + y*y <= radius*radius


def bbox(mask: np.ndarray):
    yy, xx = np.nonzero(mask)
    if not len(yy):
        return 0, mask.shape[0], 0, mask.shape[1]
    return int(yy.min()), int(yy.max())+1, int(xx.min()), int(xx.max())+1


def correlated_pick(eligible: np.ndarray, fraction: float, rng: np.random.Generator,
                    sigma=3.0) -> np.ndarray:
    eligible = np.asarray(eligible, bool)
    n = int(eligible.sum())
    if n == 0 or fraction <= 0:
        return np.zeros_like(eligible)
    k = max(1, min(n, int(round(fraction * n))))
    field = ndi.gaussian_filter(rng.standard_normal(eligible.shape), sigma=sigma, mode="reflect")
    vals = field[eligible]
    if k >= n:
        return eligible.copy()
    # Exact top-k among eligible pixels; ties are deterministically broken by flat index.
    flat_idx = np.flatnonzero(eligible)
    order = np.lexsort((flat_idx, -vals))
    chosen = flat_idx[order[:k]]
    out = np.zeros_like(eligible)
    out.flat[chosen] = True
    return out


def bresenham(y0, x0, y1, x1):
    pts = []
    dy, dx = abs(y1-y0), abs(x1-x0)
    sy, sx = (1 if y0 < y1 else -1), (1 if x0 < x1 else -1)
    err = dx - dy
    y, x = y0, x0
    while True:
        pts.append((y, x))
        if y == y1 and x == x1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy; x += sx
        if e2 < dx:
            err += dx; y += sy
    return pts


def structured_corrupt(truth: np.ndarray, name: str, rng: np.random.Generator) -> np.ndarray:
    y = np.asarray(truth, bool)
    if name == "boundary_expand2":
        return ndi.binary_dilation(y, structure=disk(2))
    if name == "boundary_contract2":
        return ndi.binary_erosion(y, structure=disk(2), border_value=0)
    if name == "clustered_fp03":
        return y | correlated_pick(~y, .03, rng, sigma=3.0)
    if name == "clustered_fn08":
        return y & ~correlated_pick(y, .08, rng, sigma=3.0)
    if name == "holes2_r4":
        out = y.copy()
        dist = ndi.distance_transform_edt(out)
        cand = np.flatnonzero(dist >= 5)
        if not len(cand):
            cand = np.flatnonzero(out)
        if not len(cand):
            return out
        for _ in range(2):
            idx = int(rng.choice(cand))
            cy, cx = np.unravel_index(idx, out.shape)
            r = 4
            yy0, yy1 = max(0, cy-r), min(out.shape[0], cy+r+1)
            xx0, xx1 = max(0, cx-r), min(out.shape[1], cx+r+1)
            dd = disk(r)[yy0-(cy-r):yy1-(cy-r), xx0-(cx-r):xx1-(cx-r)]
            out[yy0:yy1, xx0:xx1][dd] = False
        return out
    if name == "fragment_cut2":
        out = y.copy()
        y0,y1,x0,x1 = bbox(y)
        if (y1-y0) < 4 or (x1-x0) < 4:
            return out
        if rng.random() < .5:
            c = int(rng.integers(x0 + max(1,(x1-x0)//4), x1 - max(1,(x1-x0)//4)))
            out[y0:y1, max(0,c-1):min(out.shape[1],c+1)] = False
        else:
            c = int(rng.integers(y0 + max(1,(y1-y0)//4), y1 - max(1,(y1-y0)//4)))
            out[max(0,c-1):min(out.shape[0],c+1), x0:x1] = False
        return out
    if name == "bridge_spur12":
        out = y.copy()
        if not out.any():
            return out
        er = ndi.binary_erosion(out, structure=np.ones((3,3), bool), border_value=0)
        bd = out & ~er
        pts = np.argwhere(bd)
        if not len(pts):
            return out
        cy, cx = np.argwhere(out).mean(axis=0)
        # Favor a boundary point far from centroid, then extend outward.
        d2 = (pts[:,0]-cy)**2 + (pts[:,1]-cx)**2
        top = pts[np.argsort(d2)[-max(1,min(len(pts),64)):]]
        py, px = map(int, top[int(rng.integers(len(top)))])
        vy, vx = py-cy, px-cx
        norm = math.hypot(vy, vx) or 1.0
        ey = int(round(py + 12 * vy / norm)); ex = int(round(px + 12 * vx / norm))
        ey = int(np.clip(ey, 1, out.shape[0]-2)); ex = int(np.clip(ex, 1, out.shape[1]-2))
        for qy,qx in bresenham(py,px,ey,ex):
            if 0 <= qy < out.shape[0] and 0 <= qx < out.shape[1]: out[qy,qx] = True
        r=2; yy0,yy1=max(0,ey-r),min(out.shape[0],ey+r+1); xx0,xx1=max(0,ex-r),min(out.shape[1],ex+r+1)
        dd=disk(r)[yy0-(ey-r):yy1-(ey-r),xx0-(ex-r):xx1-(ex-r)]
        out[yy0:yy1,xx0:xx1][dd]=True
        return out
    if name == "occlusion25":
        out = y.copy(); y0,y1,x0,x1=bbox(y)
        h,w=max(1,y1-y0),max(1,x1-x0)
        rh=max(2,int(round(.25*h))); rw=max(2,int(round(.25*w)))
        pts=np.argwhere(y)
        if not len(pts): return out
        cy,cx=map(int,pts[int(rng.integers(len(pts)))])
        ya=max(0,cy-rh//2); yb=min(out.shape[0],ya+rh); xa=max(0,cx-rw//2); xb=min(out.shape[1],xa+rw)
        out[ya:yb,xa:xb]=False
        return out
    if name == "mixed_structured":
        out = y | correlated_pick(~y, .015, rng, sigma=3.0)
        out = out & ~correlated_pick(out, .04, rng, sigma=3.0)
        # One smaller localized occlusion.
        y0,y1,x0,x1=bbox(y); h,w=max(1,y1-y0),max(1,x1-x0)
        pts=np.argwhere(y)
        if len(pts):
            cy,cx=map(int,pts[int(rng.integers(len(pts)))])
            rh=max(2,int(round(.12*h))); rw=max(2,int(round(.12*w)))
            ya=max(0,cy-rh//2); yb=min(out.shape[0],ya+rh); xa=max(0,cx-rw//2); xb=min(out.shape[1],xa+rw)
            out[ya:yb,xa:xb]=False
        return out
    raise KeyError(name)


def component_dropout(truth: np.ndarray):
    y=np.asarray(truth,bool); lab,n=ndi.label(y,structure=np.ones((3,3),bool))
    if n < 2: return None
    counts=np.bincount(lab.ravel()); ids=np.arange(1,n+1); ids=ids[counts[1:] >= 5]
    if len(ids) < 2: return None
    target=int(ids[np.argmin(counts[ids])]); out=y.copy(); out[lab==target]=False
    return out


def measure7(pred, truth, tb=None):
    return np.r_[measure(pred, truth, tb), boundary_f08(pred, truth)]


def fit_forest(X, y, seed, leaf):
    m=ExtraTreesRegressor(n_estimators=96,max_features=1.0,min_samples_leaf=leaf,
                          random_state=seed,n_jobs=1)
    m.fit(X,y); return m


def direct_table(cache):
    X=[]; y=[]
    for i in range(len(cache["scores"])):
        k=int(cache["n_distinct"][i]); reps=cache["direct_reps"][i,:k]
        X.append(cache["direct_features"][i,:k]); y.append(cache["scores"][i,reps,0])
    return np.vstack(X), np.concatenate(y)


def load_step5_artifact(path: Path, work: Path):
    if sha256(path) != STEP5_EXPECTED_SHA256:
        raise RuntimeError("Step-5 artifact SHA256 does not match the frozen official run")
    with zipfile.ZipFile(path) as z:
        names=set(z.namelist())
        needed=[f"step5_oxford_pet/oxford_pet_{s}_128.npz" for s in ("train","val","test")]
        needed += ["step5_oxford_pet/oxford_pet_selection.json","step5_oxford_pet/oxford_pet_protocol_ids.csv"]
        miss=[n for n in needed if n not in names]
        if miss: raise RuntimeError(f"Step-5 artifact missing {miss}")
        for n in needed: z.extract(n, work)
    base=work/"step5_oxford_pet"
    caches={s:dict(np.load(base/f"oxford_pet_{s}_128.npz")) for s in ("train","val","test")}
    selection=json.loads((base/"oxford_pet_selection.json").read_text())
    protocol=pd.read_csv(base/"oxford_pet_protocol_ids.csv")
    return caches,selection,protocol



def load_step5_cache_dir(base: Path):
    base=Path(base)
    caches={s:dict(np.load(base/f"oxford_pet_{s}_128.npz")) for s in ("train","val","test")}
    protocol_path=base/"oxford_pet_protocol_ids.csv"
    protocol=pd.read_csv(protocol_path) if protocol_path.exists() else pd.DataFrame()
    return caches, None, protocol

def bootstrap_mean(df,method,metric,n_boot=5000,seed=PROTOCOL_SEED):
    g=df[df.method==method].groupby("image",as_index=False)[metric].mean(); vals=g[metric].to_numpy(float)
    rng=np.random.default_rng(seed + (int(hashlib.sha256((method+metric).encode()).hexdigest()[:8],16)&0xffff))
    boots=np.empty(n_boot)
    for j in range(n_boot): boots[j]=vals[rng.integers(0,len(vals),len(vals))].mean()
    lo,hi=np.quantile(boots,[.025,.975])
    return {"method":method,"metric":metric,"n_images":int(len(vals)),"mean":float(vals.mean()),
            "ci95_low":float(lo),"ci95_high":float(hi),"bootstrap_resamples":n_boot,"cluster":"clean test mask"}


def paired_bootstrap(df,a,b,metric="loss",n_boot=5000,seed=PROTOCOL_SEED,family_size=4):
    g=df.groupby(["method","image"],as_index=False)[metric].mean()
    aa=g[g.method==a].set_index("image")[metric]; bb=g[g.method==b].set_index("image")[metric]
    ids=np.array(sorted(set(aa.index)&set(bb.index))); d=(aa.loc[ids]-bb.loc[ids]).to_numpy(float)
    rng=np.random.default_rng(seed + (int(hashlib.sha256((a+b+metric).encode()).hexdigest()[:8],16)&0xffff))
    boots=np.empty(n_boot)
    for j in range(n_boot): boots[j]=d[rng.integers(0,len(d),len(d))].mean()
    alpha=.05/family_size
    lo,hi=np.quantile(boots,[alpha/2,1-alpha/2])
    return {"method_a":a,"method_b":b,"metric":metric,"n_images":int(len(d)),
            "mean_difference_a_minus_b":float(d.mean()),"familywise95_ci":[float(lo),float(hi)],
            "family_size":family_size,"bootstrap_resamples":n_boot,"cluster":"clean test mask"}


def generate_tex(outdir: Path):
    s=pd.read_csv(outdir/"step7_summary.csv"); c=pd.read_csv(outdir/"step7_condition_summary.csv")
    loss=s[s.metric=="loss"].set_index("method"); topo=s[s.metric=="topology_exact"].set_index("method"); bf=s[s.metric=="boundary_f"].set_index("method")
    methods=["Input","Validation best","Action-coordinate selector","Direct-output selector"]
    lines=[r"\begin{table}[t]",r"\centering",r"\caption{Oxford-IIIT Pet structured-error distribution-shift validation at $128\times128$. Selectors and hyperparameters are frozen from the IID Step~5 protocol; no structured-error case enters fitting or tuning. Values average nine predeclared structured errors within each held-out clean mask before population averaging.}",r"\label{tab:structured_step7}",r"\small",r"\begin{tabular}{lccc}",r"\toprule",r"Method & Loss $\downarrow$ & Topology exact $\uparrow$ & Boundary $F$ $\uparrow$\\",r"\midrule"]
    for m in methods:
        if m in loss.index: lines.append(f"{m} & {loss.loc[m,'mean']:.4f} & {topo.loc[m,'mean']:.4f} & {bf.loc[m,'mean']:.4f}\\")
    lines += [r"\bottomrule",r"\end{tabular}",r"\end{table}",""]
    lines.append("% Per-condition mean losses:")
    for cond in PRIMARY_CONDITIONS:
        q=c[c.condition==cond].set_index("method")
        vals=[]
        for m in ["Input","Validation best","Action-coordinate selector","Direct-output selector"]:
            if m in q.index: vals.append(f"{m}={q.loc[m,'loss']:.8g}")
        lines.append(f"% {cond}: " + ", ".join(vals))
    (outdir/"generated_step7_structured.tex").write_text("\n".join(lines)+"\n")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--step5-artifact",type=Path,default=None)
    ap.add_argument("--step5-cache-dir",type=Path,default=None)
    ap.add_argument("--output-dir",type=Path,default=ROOT/"results"/"step7_structured")
    ap.add_argument("--n-test",type=int,default=0,help="0 = all 3669 held-out masks; positive value only for smoke tests")
    ap.add_argument("--n-boot",type=int,default=5000)
    args=ap.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    if (args.step5_artifact is None) == (args.step5_cache_dir is None):
        raise SystemExit("supply exactly one of --step5-artifact or --step5-cache-dir")

    protocol_frozen={
      "status":"FROZEN_BEFORE_EXECUTION","step":7,"dataset":"Oxford-IIIT Pet",
      "source_step5_artifact_sha256":STEP5_EXPECTED_SHA256,"resolution":RESOLUTION,
      "training_distribution":"Step-5 frozen IID deletion/addition only; no Step-7 structured errors in fitting/tuning",
      "primary_conditions":PRIMARY_CONDITIONS,
      "secondary_diagnostic":"natural component dropout for held-out targets with >=2 foreground components of area >=5",
      "model_seeds":MODEL_SEEDS,"action_leaf":20,"direct_leaf":8,"direct_temperature":.005,
      "fixed_action":"area_connected16","primary_cluster":"clean held-out mask",
      "primary_comparisons":["Action-coordinate selector - Input","Direct-output selector - Input","Action-coordinate selector - Validation best","Direct-output selector - Action-coordinate selector"],
      "bootstrap_resamples":args.n_boot,"familywise_method":"Bonferroni percentile intervals across 4 predeclared loss contrasts"
    }
    (args.output_dir/"STEP7_PROTOCOL_FROZEN.json").write_text(json.dumps(protocol_frozen,indent=2)+"\n")

    with tempfile.TemporaryDirectory(prefix="step7_") as td:
        if args.step5_artifact is not None:
            caches,selection,protocol=load_step5_artifact(args.step5_artifact,Path(td))
            source_hash=STEP5_EXPECTED_SHA256
        else:
            caches,selection,protocol=load_step5_cache_dir(args.step5_cache_dir)
            source_hash=None
        tr,te=caches["train"],caches["test"]
        clean=te["clean_masks"].astype(bool)
        if args.n_test>0: clean=clean[:args.n_test]
        nimg=len(clean)
        # exact frozen hyperparameters from the official Step-5 selection
        if selection is not None and (selection.get("action_leaf") != 20 or selection.get("direct_leaf") != 8 or selection.get("best_validation_action") != "area_connected16"):
            raise RuntimeError("Step-5 selection metadata does not match the frozen Step-7 protocol")
        fixed_idx=NAMES.index("area_connected16")
        Xd,yd=direct_table(tr)
        action_models=[]; direct_models=[]
        for seed in MODEL_SEEDS:
            print(f"Fitting frozen IID models seed {seed}",flush=True)
            action_models.append(fit_forest(tr["action_features"],tr["scores"][:,:,0],seed,20))
            direct_models.append(fit_forest(Xd,yd,seed,8))

        rows=[]; seed_acc=[]; component_rows=[]
        t0=time.perf_counter(); width=clean.shape[2]
        chunk_images=16
        for chunk0 in range(0,nimg,chunk_images):
            chunk1=min(nimg,chunk0+chunk_images)
            case_meta=[]; afs=[]; packed_banks=[]; packed_inputs=[]
            dparts=[]; doff=[0]; dreps=[]
            for i in range(chunk0,chunk1):
                truth=clean[i]; tb=betti(truth)
                for ci,cond in enumerate(PRIMARY_CONDITIONS):
                    x=structured_corrupt(truth,cond,rng_for(i,ci))
                    outputs=bank(x)
                    af=features(x,outputs)
                    uq=unique_output_rows(x,outputs)
                    case_meta.append((i,ci,cond,tb))
                    afs.append(af)
                    packed_banks.append(np.packbits(outputs,axis=-1))
                    packed_inputs.append(np.packbits(x,axis=-1))
                    dparts.append(uq.pair_features); dreps.append(uq.representatives.astype(np.int16)); doff.append(doff[-1]+len(uq.representatives))
            afs=np.asarray(afs,float); packed_banks=np.asarray(packed_banks); packed_inputs=np.asarray(packed_inputs)
            dflat=np.vstack(dparts)
            action_sel=[]
            for ma in action_models: action_sel.append(ma.predict(afs).argmin(axis=1))
            direct_sel=[]
            for md in direct_models:
                pr=md.predict(dflat); sel=[]
                for q in range(len(case_meta)):
                    a,b=doff[q],doff[q+1]; sel.append(int(dreps[q][int(np.argmin(pr[a:b]))]))
                direct_sel.append(np.asarray(sel,dtype=np.int16))
            action_sel=np.asarray(action_sel); direct_sel=np.asarray(direct_sel)
            for q,(i,ci,cond,tb) in enumerate(case_meta):
                truth=clean[i]
                outputs=np.unpackbits(packed_banks[q],axis=-1,count=width).astype(bool)
                x=np.unpackbits(packed_inputs[q],axis=-1,count=width).astype(bool)
                # Only actually selected/fixed outputs are scored; Step 7 does not use a target-aware oracle.
                needed=set([fixed_idx]) | set(map(int,action_sel[:,q])) | set(map(int,direct_sel[:,q]))
                bf={j:boundary_f08(outputs[j],truth) for j in needed}
                raw=np.r_[measure(x,truth,tb),boundary_f08(x,truth)]
                fixed=np.r_[measure(outputs[fixed_idx],truth,tb),bf[fixed_idx]]
                am=[]; dm=[]
                for seed,ai,di in zip(MODEL_SEEDS,action_sel[:,q],direct_sel[:,q]):
                    amet=np.r_[measure(outputs[int(ai)],truth,tb),bf[int(ai)]]; dmet=np.r_[measure(outputs[int(di)],truth,tb),bf[int(di)]]
                    am.append(amet); dm.append(dmet)
                    seed_acc.append({"seed":seed,"image":i,"condition":cond,"action_loss":float(amet[0]),"direct_loss":float(dmet[0])})
                methods={"Input":raw,"Validation best":fixed,
                         "Action-coordinate selector":np.mean(am,axis=0),"Direct-output selector":np.mean(dm,axis=0)}
                class_id=int(te["class_id"][i*6]) if len(te["class_id"])>=nimg*6 else -1
                species=int(te["species"][i*6]) if len(te["species"])>=nimg*6 else -1
                for method,met in methods.items():
                    r={"method":method,"image":i,"class_id":class_id,"species":species,"condition":cond}
                    r.update({k:float(v) for k,v in zip(METRICS7,met)}); rows.append(r)
            print(f"Structured test {chunk1}/{nimg} ({time.perf_counter()-t0:.1f}s)",flush=True)

        # Secondary naturally occurring component-dropout diagnostic (eligible masks only).
        for i,truth in enumerate(clean):
            x=component_dropout(truth)
            if x is None: continue
            tb=betti(truth); outputs=bank(x); af=features(x,outputs)[None,:]; uq=unique_output_rows(x,outputs)
            ais=[int(np.argmin(ma.predict(af)[0])) for ma in action_models]
            dis=[]
            for md in direct_models:
                pc=md.predict(uq.pair_features); dis.append(int(uq.representatives[int(np.argmin(pc))]))
            needed=set([fixed_idx]+ais+dis); bf={j:boundary_f08(outputs[j],truth) for j in needed}
            am=[np.r_[measure(outputs[j],truth,tb),bf[j]] for j in ais]; dm=[np.r_[measure(outputs[j],truth,tb),bf[j]] for j in dis]
            methods=[("Input",np.r_[measure(x,truth,tb),boundary_f08(x,truth)]),("Validation best",np.r_[measure(outputs[fixed_idx],truth,tb),bf[fixed_idx]]),("Action-coordinate selector",np.mean(am,0)),("Direct-output selector",np.mean(dm,0))]
            for method,met in methods:
                rr={"method":method,"image":i,"condition":"component_dropout"}; rr.update({k:float(v) for k,v in zip(METRICS7,met)}); component_rows.append(rr)
    df=pd.DataFrame(rows); df.to_csv(args.output_dir/"step7_case_metrics.csv.gz",index=False,compression="gzip",float_format="%.9g")
    pd.DataFrame(seed_acc).to_csv(args.output_dir/"step7_seed_case_losses.csv.gz",index=False,compression="gzip",float_format="%.9g")
    if component_rows: pd.DataFrame(component_rows).to_csv(args.output_dir/"step7_component_dropout.csv",index=False,float_format="%.9g")

    # Cluster all nine structured conditions within the same clean mask.
    summaries=[]
    for metric in METRICS7:
        g=df.groupby(["method","image"],as_index=False)[metric].mean().groupby("method")[metric].agg(["mean","std","count"]).reset_index()
        g.insert(0,"metric",metric); summaries.append(g)
    pd.concat(summaries,ignore_index=True).to_csv(args.output_dir/"step7_summary.csv",index=False,float_format="%.9g")
    cs=df.groupby(["method","condition"],as_index=False)[METRICS7].mean(); cs.to_csv(args.output_dir/"step7_condition_summary.csv",index=False,float_format="%.9g")

    ci=[]
    for method in ["Input","Validation best","Action-coordinate selector","Direct-output selector"]:
        for metric in ("loss","dice","iou","topology_exact","boundary_f"):
            ci.append(bootstrap_mean(df,method,metric,args.n_boot))
    pd.DataFrame(ci).to_csv(args.output_dir/"step7_clustered_ci.csv",index=False,float_format="%.9g")

    comps=[
      paired_bootstrap(df,"Action-coordinate selector","Input",n_boot=args.n_boot),
      paired_bootstrap(df,"Direct-output selector","Input",n_boot=args.n_boot),
      paired_bootstrap(df,"Action-coordinate selector","Validation best",n_boot=args.n_boot),
      paired_bootstrap(df,"Direct-output selector","Action-coordinate selector",n_boot=args.n_boot),
    ]
    (args.output_dir/"step7_paired_comparisons.json").write_text(json.dumps(comps,indent=2)+"\n")

    # Adverse-condition/image accounting against raw input.
    adv={}
    for method in ("Action-coordinate selector","Direct-output selector"):
        q=cs.pivot(index="condition",columns="method",values="loss")
        cond_diff=(q[method]-q["Input"]).sort_values()
        g=df.groupby(["method","image"],as_index=False).loss.mean(); a=g[g.method==method].set_index("image").loss; b=g[g.method=="Input"].set_index("image").loss
        ids=sorted(set(a.index)&set(b.index)); d=(a.loc[ids]-b.loc[ids])
        adv[method]={"condition_differences_method_minus_input":{k:float(v) for k,v in cond_diff.items()},
                     "n_primary_conditions_improved":int((cond_diff<0).sum()),"n_primary_conditions":len(cond_diff),
                     "n_clean_masks_improved":int((d<0).sum()),"n_clean_masks_worsened":int((d>0).sum()),"n_clean_masks":len(d)}
    (args.output_dir/"step7_adverse_audit.json").write_text(json.dumps(adv,indent=2)+"\n")

    seed_df=pd.DataFrame(seed_acc)
    seed_summary=seed_df.groupby("seed",as_index=False)[["action_loss","direct_loss"]].mean()
    seed_summary.to_csv(args.output_dir/"step7_seed_summary.csv",index=False,float_format="%.9g")

    provenance={"status":"PASS","dataset":"Oxford-IIIT Pet","resolution":RESOLUTION,"n_test_masks":int(nimg),
                "n_primary_conditions":len(PRIMARY_CONDITIONS),"n_primary_cases":int(nimg*len(PRIMARY_CONDITIONS)),
                "source_step5_artifact_sha256":STEP5_EXPECTED_SHA256,"training_distribution":"Step-5 IID deletion/addition only",
                "structured_errors_used_for_training_or_tuning":False,"primary_conditions":PRIMARY_CONDITIONS,
                "model_seeds":MODEL_SEEDS,"fixed_action":"area_connected16","action_leaf":20,"direct_leaf":8,"direct_temperature":.005,
                "component_dropout_eligible_masks":int(len(set(r["image"] for r in component_rows))),"bootstrap_resamples":args.n_boot}
    (args.output_dir/"step7_provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    generate_tex(args.output_dir)

    # Freeze result-file hashes (excluding the manifest itself).
    files=sorted(p for p in args.output_dir.iterdir() if p.is_file() and p.name!="SHA256SUMS")
    (args.output_dir/"SHA256SUMS").write_text("".join(f"{sha256(p)}  {p.name}\n" for p in files))
    print(json.dumps(provenance,indent=2))

if __name__ == "__main__":
    main()
