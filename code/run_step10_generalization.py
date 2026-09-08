#!/usr/bin/env python3
"""Step 10: generalization, calibration, selective risk, and computational scaling.

Primary questions
-----------------
1. How well do Oxford-trained selectors transfer, without refitting, to genuine DAVIS errors?
2. Is the direct-output Gibbs law calibrated as an uncertainty distribution?
3. Does confidence support useful selective prediction / abstention?
4. How do candidate generation and representation-invariant scoring scale with image and bank size?

The Step-5 Oxford artifact is frozen by SHA-256. DAVIS is used only as a held-out
cross-domain target; no DAVIS case enters fitting, hyperparameter selection, or
calibration tuning. The direct-output temperature is the already-frozen Step-5
risk-selected temperature for each fitted seed.
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, time, zipfile
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.special import softmax
from sklearn.ensemble import ExtraTreesRegressor

from morphology import SPECS, NAMES, METRICS, betti, measure, features, pair_features, transform, footprint, area_filter
from run_davis_step6 import (
    PRIMARY_SOURCE_METHODS, ensure_davis, locate_split_file, read_sequence_list,
    discover_source_methods, enumerate_cases, build_cache,
)

ROOT = Path(__file__).resolve().parents[1]
STEP5_SHA256 = "889e7df9af279974d46595444bf768c627602d188ca8c93d0f65b30739f16e96"
MODEL_SEEDS = [101, 102, 103, 104, 105]
RESOLUTION = 128
COVERAGES = [1.0, .8, .6, .4, .2]
EPS = 1e-15


def sha256(path: Path, chunk=1 << 20) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def load_step5_artifact(path: Path, work: Path):
    if sha256(path) != STEP5_SHA256:
        raise RuntimeError("Step-5 artifact SHA256 mismatch")
    needed=[f"step5_oxford_pet/oxford_pet_{s}_128.npz" for s in ("train","val","test")]
    needed += ["step5_oxford_pet/oxford_pet_selection.json", "step5_oxford_pet/oxford_pet_protocol_ids.csv"]
    work.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path) as z:
        names=set(z.namelist()); miss=[n for n in needed if n not in names]
        if miss: raise RuntimeError(f"Step-5 artifact missing {miss}")
        for n in needed: z.extract(n, work)
    base=work/'step5_oxford_pet'
    caches={s:dict(np.load(base/f"oxford_pet_{s}_128.npz")) for s in ("train","val","test")}
    selection=json.loads((base/'oxford_pet_selection.json').read_text())
    protocol=pd.read_csv(base/'oxford_pet_protocol_ids.csv')
    return caches,selection,protocol


def fit_forest(X,y,seed,leaf):
    m=ExtraTreesRegressor(n_estimators=96,max_features=1.0,min_samples_leaf=leaf,
                          random_state=seed,n_jobs=1)
    m.fit(X,y); return m


def direct_table(cache):
    X=[]; y=[]
    for i in range(len(cache['scores'])):
        k=int(cache['n_distinct'][i]); reps=cache['direct_reps'][i,:k]
        X.append(cache['direct_features'][i,:k]); y.append(cache['scores'][i,reps,0])
    return np.vstack(X),np.concatenate(y)


def direct_predict(model,cache):
    return [model.predict(cache['direct_features'][i,:int(cache['n_distinct'][i])])
            for i in range(len(cache['scores']))]


def direct_selected(cache,preds):
    out=np.empty((len(preds),cache['scores'].shape[2]),float)
    for i,pc in enumerate(preds):
        reps=cache['direct_reps'][i,:len(pc)]
        out[i]=cache['scores'][i,reps[int(np.argmin(pc))]]
    return out


def action_selected(cache,pred):
    idx=pred.argmin(axis=1)
    return cache['scores'][np.arange(len(idx)),idx]


def ensemble_direct_laws(cache, models, taus):
    """Average the five frozen direct-output Gibbs laws casewise."""
    all_pred=[direct_predict(m,cache) for m in models]
    laws=[]
    for i in range(len(cache['scores'])):
        k=int(cache['n_distinct'][i]); q=np.zeros(k,float)
        for pred,tau in zip(all_pred,taus):
            pc=pred[i]; q += softmax(-(pc-pc.min())/float(tau))
        q /= len(models)
        laws.append(q)
    return laws,all_pred


def calibration_rows(cache,laws,dataset,group_labels=None,condition_labels=None):
    rows=[]
    for i,q in enumerate(laws):
        k=len(q); reps=cache['direct_reps'][i,:k]; costs=cache['scores'][i,reps,0].astype(float)
        cmin=float(costs.min()); optimal=np.isclose(costs,cmin,rtol=0,atol=1e-12)
        target=optimal.astype(float)/optimal.sum()
        top=int(np.argmax(q)); conf=float(q[top]); correct=int(optimal[top]); qopt=float(q[optimal].sum())
        entropy=float(-(q*np.log(np.maximum(q,EPS))).sum())
        hnorm=entropy/math.log(k) if k>1 else 0.0
        row={
            'dataset':dataset,'case':i,'n_distinct':k,'confidence':conf,'top_correct':correct,
            'optimal_set_mass':qopt,'nll_optimal_set':float(-math.log(max(qopt,EPS))),
            'brier':float(np.sum((q-target)**2)),'entropy_norm':hnorm,
            'map_loss':float(costs[top]),'oracle_loss':cmin,'map_regret':float(costs[top]-cmin),
            'expected_loss':float(q@costs),'expected_regret':float(q@costs-cmin),
        }
        if group_labels is not None: row['group']=str(group_labels[i])
        if condition_labels is not None: row['condition']=str(condition_labels[i])
        rows.append(row)
    return pd.DataFrame(rows)


def reliability_table(df,nbins=10):
    edges=np.linspace(0,1,nbins+1); rows=[]
    for j in range(nbins):
        lo,hi=edges[j],edges[j+1]
        mask=(df.confidence>=lo)&(df.confidence<(hi if j<nbins-1 else hi+1e-12))
        g=df[mask]
        rows.append({'bin':j,'low':lo,'high':hi,'count':len(g),
                     'mean_confidence':float(g.confidence.mean()) if len(g) else np.nan,
                     'accuracy':float(g.top_correct.mean()) if len(g) else np.nan})
    tab=pd.DataFrame(rows)
    used=tab[tab['count']>0]
    ece=float(np.sum((used['count']/len(df))*np.abs(used['accuracy']-used['mean_confidence'])))
    return tab,ece


def adaptive_ece(df,nbins=10):
    d=df.sort_values('confidence').reset_index(drop=True); parts=np.array_split(d,nbins)
    return float(sum(len(g)/len(d)*abs(g.top_correct.mean()-g.confidence.mean()) for g in parts if len(g)))


def calibration_summary(df):
    _,ece=reliability_table(df)
    return {
        'n_cases':int(len(df)), 'top_choice_accuracy':float(df.top_correct.mean()),
        'mean_confidence':float(df.confidence.mean()), 'ece10':ece,
        'adaptive_ece10':adaptive_ece(df), 'mean_optimal_set_mass':float(df.optimal_set_mass.mean()),
        'mean_nll_optimal_set':float(df.nll_optimal_set.mean()), 'mean_brier':float(df.brier.mean()),
        'mean_entropy_norm':float(df.entropy_norm.mean()), 'mean_map_regret':float(df.map_regret.mean()),
        'mean_expected_regret':float(df.expected_regret.mean()),
    }


def selective_curve(df):
    d=df.sort_values(['confidence','case'],ascending=[False,True]).reset_index(drop=True); rows=[]
    for cov in COVERAGES:
        n=max(1,int(math.ceil(cov*len(d)))); g=d.iloc[:n]
        rows.append({'coverage':cov,'n_cases':n,'confidence_cutoff':float(g.confidence.min()),
                     'top_accuracy':float(g.top_correct.mean()),'map_loss':float(g.map_loss.mean()),
                     'map_regret':float(g.map_regret.mean()),'expected_regret':float(g.expected_regret.mean())})
    return pd.DataFrame(rows)


def aggregate_metrics(metrics,labels,method,dataset):
    f=pd.DataFrame(metrics,columns=list(METRICS)+(['boundary_f08'] if metrics.shape[1]==8 else []))
    f.insert(0,'method',method); f.insert(0,'dataset',dataset)
    for key,val in labels.items(): f[key]=val
    return f


def oxford_train_models(caches,selection):
    tr=caches['train']; Xd,yd=direct_table(tr)
    am=[]; dm=[]; taus=[]
    run_tau={int(r['seed']):float(r['temperature']) for r in selection['runs']}
    for seed in MODEL_SEEDS:
        am.append(fit_forest(tr['action_features'],tr['scores'][:,:,0],seed,int(selection['action_leaf'])))
        dm.append(fit_forest(Xd,yd,seed,int(selection['direct_leaf'])))
        taus.append(run_tau[seed])
    return am,dm,taus


def evaluate_cross_domain(dcache, amodels, dmodels, taus, ox_selection, seq_names, method_names):
    rows=[]; n=len(dcache['scores'])
    labels={'sequence':[seq_names[int(x)] for x in dcache['sequence_index']],
            'base_method':[method_names[int(x)] for x in dcache['base_method_index']]}
    rows.append(aggregate_metrics(dcache['scores'][:,0],labels,'Input segmentation','DAVIS2016 Oxford-transfer'))
    ba=int(ox_selection['best_validation_action_index'])
    rows.append(aggregate_metrics(dcache['scores'][:,ba],labels,'Oxford validation-best fixed','DAVIS2016 Oxford-transfer'))
    for seed,m in zip(MODEL_SEEDS,amodels):
        rows.append(aggregate_metrics(action_selected(dcache,m.predict(dcache['action_features'])),labels,
                                      f'Oxford-trained action seed {seed}','DAVIS2016 Oxford-transfer'))
    laws,_=ensemble_direct_laws(dcache,dmodels,taus)
    direct_map=np.empty((n,dcache['scores'].shape[2]),float); direct_exp=np.empty_like(direct_map)
    for i,q in enumerate(laws):
        reps=dcache['direct_reps'][i,:len(q)]; direct_map[i]=dcache['scores'][i,reps[int(np.argmax(q))]]
        direct_exp[i]=q@dcache['scores'][i,reps]
    rows.append(aggregate_metrics(direct_map,labels,'Oxford-trained direct MAP ensemble','DAVIS2016 Oxford-transfer'))
    rows.append(aggregate_metrics(direct_exp,labels,'Oxford-trained direct Gibbs ensemble','DAVIS2016 Oxford-transfer'))
    df=pd.concat(rows,ignore_index=True)
    # collapse action seeds to an ensemble performance summary by sequence via seed-averaged method family
    df['method_family']=df['method'].where(~df['method'].str.startswith('Oxford-trained action seed'),'Oxford-trained action selector')
    return df,laws


def extra_specs():
    shapes=[('square',7),('diamond',3),('horizontal',5),('vertical',5),
            ('square',9),('diamond',4),('horizontal',7),('vertical',7)]
    return [(op,k,s) for k,s in shapes for op in ('O','C','OC','CO')]


def apply_spec(x,spec):
    op,kind,size=spec
    if op=='identity': return x.astype(bool).copy()
    if op=='area': return area_filter(x,size)
    pad=max(10,2*int(size)+4); a=np.pad(x.astype(bool),pad); f=footprint(kind,size)
    for letter in op:
        fun=ndi.binary_opening if letter=='O' else ndi.binary_closing
        a=fun(a,structure=f,iterations=1,origin=0,border_value=0)
    return a[pad:-pad,pad:-pad]


def synthetic_mask(n):
    yy,xx=np.ogrid[:n,:n]; cy,cx=.5*(n-1),.5*(n-1)
    y=((yy-cy)/(0.31*n))**2+((xx-cx)/(0.22*n))**2 <= 1
    y |= (((yy-.35*n)/(0.08*n))**2+((xx-.68*n)/(0.07*n))**2 <= 1)
    y &= ~((((yy-.50*n)/(0.055*n))**2+((xx-.48*n)/(0.045*n))**2) <= 1)
    return y


def scaling_benchmark(direct_model,outdir,repeats=3):
    specs=list(SPECS)+extra_specs(); assert len(specs)==63
    rows=[]
    for n in (64,128,256,480):
        x=synthetic_mask(n)
        for m in (7,15,31,63):
            use=specs[:m]; bank_times=[]; score_times=[]
            for _ in range(repeats):
                t=time.perf_counter(); outs=np.stack([apply_spec(x,s) for s in use]); bank_times.append(time.perf_counter()-t)
                t=time.perf_counter(); phi=np.stack([pair_features(x,y) for y in outs]); _=direct_model.predict(phi); score_times.append(time.perf_counter()-t)
            rows.append({'resolution':n,'pixels':n*n,'bank_size':m,
                         'bank_ms_median':1000*float(np.median(bank_times)),
                         'direct_score_ms_median':1000*float(np.median(score_times)),
                         'total_ms_median':1000*float(np.median(np.array(bank_times)+np.array(score_times))),
                         'candidate_tensor_mib':float(m*n*n/(1024**2)),
                         'pair_feature_kib':float(m*10*8/1024)})
            print(f"scale n={n} m={m} total={rows[-1]['total_ms_median']:.1f}ms",flush=True)
    df=pd.DataFrame(rows); df.to_csv(outdir/'step10_scaling.csv',index=False,float_format='%.9g')
    # empirical slopes at each bank size against pixel count, and at n=128 against bank size
    slopes=[]
    for m,g in df.groupby('bank_size'):
        slope=float(np.polyfit(np.log(g.pixels),np.log(g.total_ms_median),1)[0]); slopes.append({'axis':'pixels','fixed':int(m),'slope':slope})
    g=df[df.resolution==128]; slopes.append({'axis':'bank_size','fixed':128,'slope':float(np.polyfit(np.log(g.bank_size),np.log(g.total_ms_median),1)[0])})
    pd.DataFrame(slopes).to_csv(outdir/'step10_scaling_slopes.csv',index=False,float_format='%.9g')
    return df,slopes


def make_figures(outdir):
    import matplotlib.pyplot as plt
    # reliability
    rel=pd.read_csv(outdir/'step10_reliability.csv')
    for dataset in rel.dataset.unique():
        g=rel[(rel.dataset==dataset)&(rel['count']>0)]
        plt.figure(figsize=(5.4,4.2)); plt.plot([0,1],[0,1],'--'); plt.plot(g.mean_confidence,g.accuracy,'o-')
        plt.xlabel('Mean confidence'); plt.ylabel('Optimal-output accuracy'); plt.title(dataset); plt.tight_layout()
        stem='step10_reliability_'+('iid' if 'Oxford' in dataset else 'davis'); plt.savefig(outdir/f'{stem}.png',dpi=180); plt.savefig(outdir/f'{stem}.pdf'); plt.close()
    sel=pd.read_csv(outdir/'step10_selective_risk.csv')
    plt.figure(figsize=(5.4,4.2))
    for dataset,g in sel.groupby('dataset'): plt.plot(g.coverage,g.map_regret,'o-',label=dataset)
    plt.xlabel('Coverage'); plt.ylabel('Mean MAP regret'); plt.legend(); plt.tight_layout(); plt.savefig(outdir/'step10_selective_risk.png',dpi=180); plt.savefig(outdir/'step10_selective_risk.pdf'); plt.close()
    sc=pd.read_csv(outdir/'step10_scaling.csv')
    plt.figure(figsize=(5.4,4.2))
    for m,g in sc.groupby('bank_size'): plt.plot(g.pixels,g.total_ms_median,'o-',label=f'm={m}')
    plt.xscale('log'); plt.yscale('log'); plt.xlabel('Pixels'); plt.ylabel('Median total time (ms)'); plt.legend(); plt.tight_layout(); plt.savefig(outdir/'step10_scaling.png',dpi=180); plt.savefig(outdir/'step10_scaling.pdf'); plt.close()


def freeze(outdir):
    files=sorted(p for p in outdir.iterdir() if p.is_file() and p.name!='SHA256SUMS')
    (outdir/'SHA256SUMS').write_text(''.join(f"{sha256(p)}  {p.name}\n" for p in files))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--step5-artifact',type=Path,required=True)
    ap.add_argument('--davis-root',type=Path,default=ROOT/'data'/'davis2016_step10')
    ap.add_argument('--download-davis',action='store_true')
    ap.add_argument('--output-dir',type=Path,default=ROOT/'results'/'step10_generalization')
    ap.add_argument('--n-davis-cases',type=int,default=0,help='smoke-test truncation only; 0=all official held-out cases')
    ap.add_argument('--scaling-repeats',type=int,default=3)
    args=ap.parse_args(); out=args.output_dir; out.mkdir(parents=True,exist_ok=True)

    work=out/'_step5'; caches,selection,protocol=load_step5_artifact(args.step5_artifact,work)
    amodels,dmodels,taus=oxford_train_models(caches,selection)

    # Oxford IID calibration of the already-frozen Gibbs laws.
    ilaws,_=ensemble_direct_laws(caches['test'],dmodels,taus)
    cond_names=['balanced04','balanced10','salt04','salt10','pepper04','pepper10']
    iid_cal=calibration_rows(caches['test'],ilaws,'Oxford IID',group_labels=caches['test']['image_index'],
                             condition_labels=[cond_names[int(c)] for c in caches['test']['condition']])
    iid_cal.to_csv(out/'step10_iid_calibration_cases.csv.gz',index=False,float_format='%.9g')

    # DAVIS held-out genuine errors: Oxford-trained transfer, no fitting/tuning.
    data_archive=args.davis_root/'DAVIS-data.zip'; results_archive=args.davis_root/'DAVIS-results.zip'
    _,ann480,imagesets480,result480=ensure_davis(args.davis_root,data_archive,results_archive,args.download_davis)
    test_seq=sorted(read_sequence_list(locate_split_file(imagesets480,'val')))
    method_dirs=discover_source_methods(result480,PRIMARY_SOURCE_METHODS)
    cases=enumerate_cases(ann480,method_dirs,test_seq)
    if args.n_davis_cases>0: cases=cases[:args.n_davis_cases]
    seq_names=sorted(set(c['sequence'] for c in cases)); seq_index={s:i for i,s in enumerate(seq_names)}
    method_names=list(PRIMARY_SOURCE_METHODS); method_index={m:i for i,m in enumerate(method_names)}
    cache_path=out/'davis2016_transfer_test_128.npz'
    if not cache_path.exists(): build_cache('transfer_test',cases,out,RESOLUTION,seq_index,method_index)
    dcache=dict(np.load(cache_path))
    transfer,dlaws=evaluate_cross_domain(dcache,amodels,dmodels,taus,selection,seq_names,method_names)
    transfer.to_csv(out/'step10_davis_transfer_metrics.csv.gz',index=False,float_format='%.9g')
    dcal=calibration_rows(dcache,dlaws,'DAVIS Oxford-transfer',
                          group_labels=[seq_names[int(x)] for x in dcache['sequence_index']],
                          condition_labels=[method_names[int(x)] for x in dcache['base_method_index']])
    dcal.to_csv(out/'step10_davis_calibration_cases.csv.gz',index=False,float_format='%.9g')

    # Calibration/reliability/selective-risk summaries.
    relrows=[]; sumrows=[]; selrows=[]
    for df,name in ((iid_cal,'Oxford IID'),(dcal,'DAVIS Oxford-transfer')):
        tab,ece=reliability_table(df); tab.insert(0,'dataset',name); relrows.append(tab)
        s=calibration_summary(df); s['dataset']=name; sumrows.append(s)
        sc=selective_curve(df); sc.insert(0,'dataset',name); selrows.append(sc)
    pd.concat(relrows).to_csv(out/'step10_reliability.csv',index=False,float_format='%.9g')
    pd.DataFrame(sumrows).to_csv(out/'step10_calibration_summary.csv',index=False,float_format='%.9g')
    pd.concat(selrows).to_csv(out/'step10_selective_risk.csv',index=False,float_format='%.9g')

    # Cross-domain summary, sequence-weighted for DAVIS.
    t=transfer.copy()
    t['method_family']=t['method'].where(~t['method'].str.startswith('Oxford-trained action seed'),'Oxford-trained action selector')
    byseq=t.groupby(['method_family','sequence'],as_index=False)[['loss','dice','iou','topology_exact','boundary_f08']].mean()
    ts=byseq.groupby('method_family',as_index=False)[['loss','dice','iou','topology_exact','boundary_f08']].mean()
    ts.to_csv(out/'step10_davis_transfer_summary.csv',index=False,float_format='%.9g')

    # Scaling uses one frozen Oxford direct model and does not claim accuracy for expanded banks.
    scaling_benchmark(dmodels[0],out,args.scaling_repeats)
    make_figures(out)

    provenance={
        'status':'COMPLETE','step5_artifact_sha256':sha256(args.step5_artifact),
        'training_domain':'Oxford-IIIT Pet Step-5 IID deletion/addition only',
        'davis_fitting_or_tuning':False,'davis_source_methods':method_names,'davis_test_sequences':len(seq_names),
        'davis_transfer_cases':int(len(dcache['scores'])),'iid_calibration_cases':int(len(iid_cal)),
        'model_seeds':MODEL_SEEDS,'frozen_temperatures':taus,'confidence_definition':'maximum ensemble distinct-output Gibbs mass',
        'correctness_definition':'MAP output belongs to true minimum-loss distinct-output set',
        'calibration_bins':10,'selective_coverages':COVERAGES,
        'scaling_sizes':[64,128,256,480],'scaling_bank_sizes':[7,15,31,63],
        'expanded_bank_accuracy_claim':False,
    }
    (out/'step10_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    freeze(out)
    # remove extracted Step5 payload from artifact directory after all computations
    import shutil
    shutil.rmtree(work,ignore_errors=True)
    print(json.dumps({'status':'PASS','output_dir':str(out),'iid_cases':len(iid_cal),'davis_cases':len(dcal)},indent=2))

if __name__=='__main__': main()
