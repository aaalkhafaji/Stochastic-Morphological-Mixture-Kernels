"""One command from frozen clean data to every raw test result. CPU only.

Train/validation/test separation is at the clean-image level. Test outcomes never
enter selection or fitting. Repeated corruptions remain clustered by clean image.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import argparse, gzip, json, platform, time, warnings
import numpy as np
import pandas as pd
import scipy
from scipy import ndimage as ndi
from scipy.optimize import minimize
from scipy.special import softmax
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
import joblib
from morphology import (bank, betti, features, measure, patches, soft_transform,
    softmax_objective, area_filter, NAMES, SPECS, METRICS)

ROOT=Path(__file__).resolve().parents[1]
CONDITIONS=[('balanced04',.04,.04),('balanced10',.10,.10),
    ('salt04',.01,.04),('salt10',.01,.10),('pepper04',.04,.01),('pepper10',.10,.01),
    ('shift20',.20,.20),('clean',0.,0.)]
SEEDS=[101,102,103,104,105]
CLASSICAL=['median3','median5','gaussian06','gaussian10', 'area3','area8','area16','area32']
SOFT=[(3,t,op) for t in (1.,3.,10.) for op in ('OC','CO')]

def corrupt(y, d, a, rng):
    u=rng.random(y.shape)
    return np.where(y, u>=d, u<a)

def classical(x):
    out=[ndi.median_filter(x.astype(np.uint8),size=k,mode='constant')>0 for k in (3,5)]
    out += [ndi.gaussian_filter(x.astype(float),sigma=s,mode='constant')>=.5 for s in (.6,1.)]
    for threshold in (3,8,16,32):
        lab,n=ndi.label(x,structure=np.ones((3,3)))
        counts=np.bincount(lab.ravel()); keep=counts>=threshold; keep[0]=False
        a=keep[lab]
        b=~np.pad(a,1)
        lab,n=ndi.label(b,structure=ndi.generate_binary_structure(2,1))
        counts=np.bincount(lab.ravel()); fill=counts<threshold;fill[0]=False;fill[lab[0,0]]=False
        out.append(a | fill[lab][1:-1,1:-1])
    return out

def extract(ds):
    clean=np.load(ROOT/'data'/f'{ds}_clean.npz')
    did=['shapes','mnist','fashion'].index(ds)
    timing={}
    for sidx,split in enumerate(('train','val','test')):
        path=ROOT/'results'/f'{ds}_{split}_bank.npz'
        if path.exists(): continue
        truth=clean[split]; h,w=truth.shape[1:]
        included=np.arange(len(truth))
        if split=='test' and 'test_excluded' in clean:
            included=np.setdiff1d(included,clean['test_excluded'])
        ii=np.repeat(included,8) if split=='test' else np.arange(len(truth))
        cc=np.tile(np.arange(8),len(included)) if split=='test' else np.arange(len(truth))%6
        n=len(ii)
        scores=np.empty((n,len(NAMES),len(METRICS)),np.float32)
        fs=[]; noisy=[]; packed=[]; bs=np.empty((n,len(CLASSICAL),len(METRICS)),np.float32)
        ss=np.empty((n,len(SOFT),len(METRICS)),np.float32) if split!='train' else np.empty((0,))
        t=time.perf_counter()
        for q,(i,c) in enumerate(zip(ii,cc)):
            rng=np.random.default_rng(np.random.SeedSequence([20260905,did,sidx,int(i),int(c)]))
            x=corrupt(truth[i],*CONDITIONS[c][1:],rng)
            b=bank(x); tb=betti(truth[i])
            scores[q]=[measure(y,truth[i],tb) for y in b]
            fs.append(features(x,b));noisy.append(np.packbits(x,axis=None));packed.append(np.packbits(b,axis=-1))
            bs[q]=[measure(y,truth[i],tb) for y in classical(x)]
            if split!='train':ss[q]=[measure(soft_transform(x,*s),truth[i],tb) for s in SOFT]
            if q and q%1000==0:print(ds,split,q,'/',n,flush=True)
        timing[split]=time.perf_counter()-t
        np.savez_compressed(path, scores=scores, features=np.asarray(fs,dtype=np.float32),
            noisy=np.asarray(noisy), outputs=np.asarray(packed), image_index=ii, condition=cc,
            classical=bs, soft=ss, shape=np.array([h,w]))
        print(ds,split,'extracted',n,'in',round(timing[split],2),'s',flush=True)
    (ROOT/'results'/f'{ds}_extraction_timing.json').write_text(json.dumps(timing,indent=2)+'\n')

def fit_forest(X,L,seed,leaf):
    m=ExtraTreesRegressor(n_estimators=96,max_features=1.,min_samples_leaf=leaf,
                         random_state=seed,n_jobs=1)
    m.fit(X,L);return m

def train_evaluate(ds):
    # Materialize compressed arrays once, rather than decompressing on each lookup.
    a={s:dict(np.load(ROOT/'results'/f'{ds}_{s}_bank.npz')) for s in ('train','val','test')}
    clean=dict(np.load(ROOT/'data'/f'{ds}_clean.npz'))
    tr,va,te=[a[s] for s in ('train','val','test')]
    X=tr['features'];V=va['features'];Z=te['features']
    L=tr['scores'][:,:,0];VL=va['scores'][:,:,0];TL=te['scores'][:,:,0]
    ni=len(te['scores']); ix=np.arange(ni); nv=np.arange(len(V))
    chosen={'dataset':ds,'operator_names':NAMES,'seeds':SEEDS,'forest_leaf_candidates':[]}
    t=time.perf_counter()
    for leaf in (2,8,20):
        m=fit_forest(X,L,99,leaf); pred=m.predict(V).argmin(1)
        chosen['forest_leaf_candidates'].append([leaf,float(VL[nv,pred].mean())])
    leaf=min(chosen['forest_leaf_candidates'],key=lambda z:z[1])[0]
    chosen['leaf']=leaf;chosen['tuning_seconds']=time.perf_counter()-t
    best=int(VL.mean(0).argmin());training_best=int(L.mean(0).argmin())
    classical_best=int(va['classical'][:,:,0].mean(0).argmin())
    soft_best=int(va['soft'][:,:,0].mean(0).argmin())
    chosen.update(best_deterministic=NAMES[best], training_best=NAMES[training_best],
                  classical_best=CLASSICAL[classical_best],soft_best=SOFT[soft_best])
    rows=[]
    def add(method,seed,metrics):
        f=pd.DataFrame(metrics,columns=METRICS)
        f.insert(0,'condition',[CONDITIONS[c][0] for c in te['condition']])
        f.insert(0,'image',te['image_index']); f.insert(0,'seed',seed)
        f.insert(0,'method',method);f.insert(0,'dataset',ds);rows.append(f)
    add('Input',-1,te['scores'][:,0])
    add('Validation best',-1,te['scores'][:,best])
    add('Training best',-1,te['scores'][:,training_best])
    add('Uniform sampled',-1,te['scores'].mean(1))
    add('Classical best',-1,te['classical'][:,classical_best])
    add('Soft morphology',-1,te['soft'][:,soft_best])
    add('Oracle (diagnostic)',-1,te['scores'][ix,TL.argmin(1)])
    for j,n in enumerate(CLASSICAL): add(n,-1,te['classical'][:,j])
    for j in (1,2,3):add(NAMES[j],-1,te['scores'][:,j])
    mean=X.mean(0);std=X.std(0);std[std<1e-6]=1
    XF=np.c_[np.ones(len(X)),(X-mean)/std]
    VF=np.c_[np.ones(len(V)),(V-mean)/std]
    ZF=np.c_[np.ones(len(Z)),(Z-mean)/std]
    predictions={};chosen['runs']=[]
    for seed in SEEDS:
        t=time.perf_counter();model=fit_forest(X,L,seed,leaf)
        R=model.predict(Z);RV=model.predict(V)
        selected=R.argmin(1)
        add('Conditional selector',seed,te['scores'][ix,selected])
        temps=[.005,.02,.1]
        temp=min(temps,key=lambda tau:np.sum(softmax(-RV/tau,axis=1)*VL,axis=1).mean())
        p=softmax(-R/temp,axis=1)
        add('Conditional sampled',seed,np.einsum('ij,ijk->ik',p,te['scores']))
        predictions[f'actions_{seed}']=selected.astype(np.int16)
        predictions[f'probabilities_{seed}']=p.astype(np.float32)
        fit_sec=time.perf_counter()-t
        joblib.dump(model,ROOT/'models'/f'{ds}_forest_{seed}.joblib',compress=3)
        # Ablation: remove both topology terms, keeping relative overlap/pixel weights.
        noL=(2*(1-tr['scores'][:,:,2])+tr['scores'][:,:,1])/3
        no=fit_forest(X,noL,seed,leaf)
        noselect=no.predict(Z).argmin(1)
        add('No topology',seed,te['scores'][ix,noselect])
        # Library ablation: the exact original three square openings (sizes 2,3,4).
        # Keep all features fixed to isolate the permitted output actions.
        small=fit_forest(X,L[:,1:4],seed,leaf)
        smselect=small.predict(Z).argmin(1)+1
        add('Three openings',seed,te['scores'][ix,smselect])
        # Exact finite-sum score-gradient control. No MC estimator needed for 31 actions.
        rng=np.random.default_rng(seed)
        init=rng.normal(0,.01,(XF.shape[1],len(NAMES)))
        opt=minimize(softmax_objective,init.ravel(),args=(XF,L,1e-4),jac=True,
                     method='L-BFGS-B',options={'maxiter':400,'ftol':1e-10,'gtol':1e-6})
        gp=softmax(ZF@opt.x.reshape(XF.shape[1],len(NAMES)),axis=1)
        add('Exact softmax gate',seed,te['scores'][ix,gp.argmax(1)])
        add('Exact gate sampled',seed,np.einsum('ij,ijk->ik',gp,te['scores']))
        predictions[f'gate_weights_{seed}']=opt.x.reshape(XF.shape[1],len(NAMES))
        # Explicit weighted mean decoder, evaluated as a distinct deterministic estimator.
        out=te['outputs'];w=int(te['shape'][1]); metrics=[]
        for q in range(ni):
            ybank=np.unpackbits(out[q],axis=-1,count=w).astype(bool)
            y=np.einsum('j,jhw->hw',p[q],ybank)>=.5
            metrics.append(measure(y,clean['test'][te['image_index'][q]]))
        add('Conditional mean mask',seed,np.asarray(metrics))
        chosen['runs'].append({'seed':seed,'temperature':temp,'fit_predict_seconds':fit_sec,
             'mean_action_entropy':float((-p*np.log(p+1e-300)).sum(1).mean()),
             'gate_success':bool(opt.success),'gate_iterations':int(opt.nit),
             'gate_gradient_inf':float(np.max(abs(opt.jac))), 'gate_objective':float(opt.fun)})
        print(ds,'completed estimator seed',seed,flush=True)
    # Supervised differentiable local filter: logistic regression on 5x5 neighborhoods.
    # Every training image contributes 64 uniformly selected pixels, no test supervision.
    patchX=[];patchY=[]
    for i in range(len(X)):
        h,w=map(int,tr['shape']);x=np.unpackbits(tr['noisy'][i],count=h*w).reshape(h,w)
        rng=np.random.default_rng(np.random.SeedSequence([781, i]))
        sel=rng.choice(h*w,size=64,replace=False)
        patchX.append(patches(x)[sel]);patchY.append(clean['train'][i].ravel()[sel])
    patchX=np.concatenate(patchX);patchY=np.concatenate(patchY)
    logistic=LogisticRegression(C=1.,solver='lbfgs',max_iter=400,tol=1e-8,random_state=0).fit(patchX,patchY)
    vm=[]
    for i in range(len(V)):
        h,w=map(int,va['shape']);x=np.unpackbits(va['noisy'][i],count=h*w).reshape(h,w)
        vm.append(logistic.predict_proba(patches(x))[:,1].reshape(h,w))
    thresholds=[.25,.35,.45,.5,.55,.65,.75]
    vals=[np.mean([measure(p>=thr,clean['val'][i])[0] for i,p in enumerate(vm)]) for thr in thresholds]
    threshold=thresholds[int(np.argmin(vals))]; chosen['logistic_threshold']=threshold
    chosen['logistic_iterations']=logistic.n_iter_.tolist()
    mm=[]
    for q in range(ni):
        h,w=map(int,te['shape']);x=np.unpackbits(te['noisy'][q],count=h*w).reshape(h,w)
        p=logistic.predict_proba(patches(x))[:,1].reshape(h,w)
        mm.append(measure(p>=threshold,clean['test'][te['image_index'][q]]))
    add('Learned local filter',-1,np.asarray(mm))
    joblib.dump(logistic,ROOT/'models'/f'{ds}_logistic.joblib',compress=3)
    predictions['feature_mean']=mean;predictions['feature_std']=std
    np.savez_compressed(ROOT/'results'/f'{ds}_predictions.npz',**predictions)
    pd.concat(rows,ignore_index=True).to_csv(ROOT/'results'/f'{ds}_test_metrics.csv.gz',index=False,float_format='%.9g')
    (ROOT/'results'/f'{ds}_selection.json').write_text(json.dumps(chosen,indent=2)+'\n')
    print(ds,'all fits and test evaluations finished',flush=True)

def main(stage,datasets):
    for ds in datasets:
        if stage in ('extract','all'):extract(ds)
        if stage in ('fit','all'):train_evaluate(ds)
    env={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
         'scikit-learn':sklearn.__version__,'platform':platform.platform(),
         'processor':platform.processor(),'conditions':CONDITIONS,'seeds':SEEDS,
         'blas_threads':1,'operator_names':NAMES}
    (ROOT/'results'/'environment.json').write_text(json.dumps(env,indent=2)+'\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['all','extract','fit'],default='all')
    ap.add_argument('--datasets',nargs='+',default=['shapes','mnist','fashion'])
    args=ap.parse_args();main(args.stage,args.datasets)
