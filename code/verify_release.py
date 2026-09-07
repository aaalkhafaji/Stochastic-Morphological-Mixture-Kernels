"""Release checks: split leakage, mask scores, model replay, seeded refits, timing."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import hashlib,json,time,platform
import numpy as np
import pandas as pd
from scipy.special import softmax
import joblib
from morphology import bank,features,measure,NAMES,area_filter
from run_campaign import fit_forest,corrupt,CONDITIONS

ROOT=Path(__file__).resolve().parents[1]

def main():
    report={};timings=[]
    for did,ds in enumerate(('shapes','mnist','fashion')):
        data=dict(np.load(ROOT/'data'/f'{ds}_clean.npz'))
        te=dict(np.load(ROOT/'results'/f'{ds}_test_bank.npz'))
        tr=dict(np.load(ROOT/'results'/f'{ds}_train_bank.npz'))
        pred=dict(np.load(ROOT/'results'/f'{ds}_predictions.npz'))
        selected=json.loads((ROOT/'results'/f'{ds}_selection.json').read_text())
        used=np.unique(te['image_index'])
        hashes={s:{np.packbits(x).tobytes() for x in data[s]} for s in ('train','val')}
        hashes['test']={np.packbits(data['test'][i]).tobytes() for i in used}
        overlap={a+'_'+b:len(hashes[a]&hashes[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
        assert not any(overlap.values())
        assert len(hashes['test'])==len(used)
        assert te['scores'].shape[1]==31 and te['features'].shape[1]==160
        assert len(te['image_index'])==8*len(used)
        assert np.all(np.bincount(te['condition'],minlength=8)==len(used))
        n=len(te['scores']);checks=np.linspace(0,n-1,32,dtype=int)
        residual=0.;h,w=map(int,te['shape'])
        for q in checks:
            i=int(te['image_index'][q]);c=int(te['condition'][q]);y=data['test'][i]
            x=np.unpackbits(te['noisy'][q],count=h*w).reshape(h,w).astype(bool)
            rng=np.random.default_rng(np.random.SeedSequence([20260905,did,2,i,c]))
            assert np.array_equal(x,corrupt(y,*CONDITIONS[c][1:],rng))
            fresh=bank(x);saved=np.unpackbits(te['outputs'][q],axis=-1,count=w).astype(bool)
            assert np.array_equal(fresh,saved)
            recalculated=np.array([measure(z,y) for z in fresh])
            residual=max(residual,float(np.max(abs(recalculated-te['scores'][q]))))
            assert residual<2e-6
            # Independent arithmetic for pixel/overlap columns, no metric helper.
            mse=np.mean(fresh!=y,axis=(1,2));inter=(fresh&y).sum(axis=(1,2));den=fresh.sum(axis=(1,2))+y.sum()
            dice=np.divide(2*inter,den,out=np.ones(31),where=den>0)
            assert np.allclose(mse,te['scores'][q,:,1],atol=1e-7)
            assert np.allclose(dice,te['scores'][q,:,2],atol=1e-7)
        replay=[]
        for seed in (101,102,103,104,105):
            model=joblib.load(ROOT/'models'/f'{ds}_forest_{seed}.joblib')
            r=model.predict(te['features']);actions=r.argmin(1)
            assert np.array_equal(actions,pred[f'actions_{seed}'])
            tau=next(z['temperature'] for z in selected['runs'] if z['seed']==seed)
            p=softmax(-r/tau,axis=1)
            assert np.max(abs(p-pred[f'probabilities_{seed}']))<1e-7
            replay.append(seed)
        # A fresh, deterministic refit of the entire training bank in each dataset.
        refit=fit_forest(tr['features'],tr['scores'][:,:,0],101,selected['leaf'])
        assert np.array_equal(refit.predict(te['features']).argmin(1),pred['actions_101'])
        # Validate raw CSV against retained metrics and actions for every test case.
        df=pd.read_csv(ROOT/'results'/f'{ds}_test_metrics.csv.gz')
        f=df[(df['method']=='Conditional selector')&(df.seed==101)]
        assert np.array_equal(f.image,te['image_index'])
        assert np.allclose(f['loss'],te['scores'][np.arange(n),pred['actions_101'],0],atol=1e-8)
        assert 173 not in f.image.to_numpy() if ds=='shapes' else True
        # End-to-end CPU timing on the first 100 primary cases, repeated three times.
        model=joblib.load(ROOT/'models'/f'{ds}_forest_101.joblib')
        qs=np.flatnonzero(te['condition']<6)[:100]
        inputs=[np.unpackbits(te['noisy'][q],count=h*w).reshape(h,w).astype(bool) for q in qs]
        model.predict(te['features'][:2])
        for repeat in range(3):
            t=time.perf_counter()
            for x in inputs:
                outputs=bank(x);f=features(x,outputs)
                jj=model.predict(f[None])[0].argmin();out=outputs[jj]
            elapsed=(time.perf_counter()-t)/len(inputs)*1000
            t=time.perf_counter()
            for x in inputs:out=area_filter(x,8)
            baseline=(time.perf_counter()-t)/len(inputs)*1000
            timings.append({'dataset':ds,'repeat':repeat,'conditional_ms_per_image':elapsed,'area8_ms_per_image':baseline,'batch_size':1,'cases':len(inputs)})
        report[ds]={'evaluated_test_images':len(used),'primary_cases':int((te['condition']<6).sum()),
            'diagnostic_cases':int((te['condition']>=6).sum()),'cross_split_mask_overlap':overlap,
            'independent_score_rechecks':len(checks)*31,'maximum_score_residual':residual,
            'models_replayed':replay,'seed101_fresh_refit':'all test actions identical',
            'forest_parameters':model.get_params()}
    report['hardware']={'platform':platform.platform(),'machine':platform.machine(),
        'logical_cpus_visible':os.cpu_count(),'processor':platform.processor(),
        'timing_note':'CPU-only, n_jobs=1, BLAS thread count 1. Shared-host wall times are environment-specific.'}
    (ROOT/'results'/'release_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    pd.DataFrame(timings).to_csv(ROOT/'results'/'runtime.csv',index=False)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
