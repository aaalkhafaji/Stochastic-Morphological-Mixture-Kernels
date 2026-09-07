"""Exercise the packaged bank, saved estimators, and output-space interface."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
from pathlib import Path
import json
import joblib
import numpy as np
from morphology import bank, features
from kernel_quotient import output_partition, quotient_weights, aggregate_action_mass
ROOT=Path(__file__).resolve().parents[1]
def main():
    report={}
    for ds in ['shapes','mnist','fashion']:
        b=dict(np.load(ROOT/'results'/f'{ds}_test_bank.npz'))
        saved=dict(np.load(ROOT/'results'/f'{ds}_predictions.npz'))
        h,w=map(int,b['shape']);primary=np.flatnonzero(b['condition']<6)
        cases=primary[[0,-1]];replayed=[];residual=0.
        for i in cases:
            x=np.unpackbits(b['noisy'][i],count=h*w).reshape(h,w).astype(bool)
            outputs=bank(x);expected=np.unpackbits(b['outputs'][i],axis=-1,count=w).astype(bool)
            assert np.array_equal(outputs,expected),(ds,int(i),'outputs')
            assert np.allclose(features(x,outputs),b['features'][i],atol=1e-6),(ds,int(i),'features')
        for seed in [101,102,103,104,105]:
            model=joblib.load(ROOT/'models'/f'{ds}_forest_{seed}.joblib')
            risks=model.predict(b['features'][cases])
            assert np.array_equal(risks.argmin(1),saved[f'actions_{seed}'][cases])
            replayed.append(seed)
            for row,i in enumerate(cases):
                outputs=b['outputs'][i];part=output_partition(outputs)
                p=quotient_weights(risks[row],part,.005)[0]
                base=aggregate_action_mass(p,part)
                copied_outputs=np.concatenate([outputs,np.repeat(outputs[:1],8,axis=0)])
                copied_risks=np.r_[risks[row],np.repeat(risks[row,0],8)]
                copied_part=output_partition(copied_outputs)
                copied_p=quotient_weights(copied_risks,copied_part,.005)[0]
                after=aggregate_action_mass(copied_p,copied_part)
                residual=max(residual,float(np.max(abs(base-after))))
                assert residual<1e-12
        report[ds]={'cases_reconstructed':list(map(int,cases)),'forest_seeds_replayed':replayed,'replication_residual':residual}
    report['scope']='Packaging smoke check, not a replacement for full release or theorem checks.'
    (ROOT/'results/package_smoke_check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
