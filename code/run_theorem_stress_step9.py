#!/usr/bin/env python3
"""Step 9: theorem-targeted numerical stress tests.

This script verifies numerical consequences of the paper's finite theorems. It is
not a performance benchmark and uses no held-out target to tune a method.
"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linprog

from morphology import SPECS, NAMES, transform, betti


def all_masks(h,w):
    n=h*w
    vals=np.arange(1<<n,dtype=np.uint64)[:,None]
    bits=((vals >> np.arange(n,dtype=np.uint64)) & 1).astype(bool)
    return bits.reshape(-1,h,w)


def mask_index(x):
    b=np.asarray(x,bool).ravel().astype(np.uint64)
    return int(np.dot(b, (np.uint64(1)<<np.arange(len(b),dtype=np.uint64))))


def hamming_cost(states):
    flat=states.reshape(len(states),-1).astype(np.int8)
    return np.count_nonzero(flat[:,None,:]!=flat[None,:,:],axis=2).astype(float)


def finite_ot(a,b,C):
    a=np.asarray(a,float); b=np.asarray(b,float)
    ia=np.flatnonzero(a>1e-15); ib=np.flatnonzero(b>1e-15)
    aa=a[ia]; bb=b[ib]; CC=C[np.ix_(ia,ib)]
    na,nb=len(ia),len(ib)
    # Variables row-major.
    Aeq=[]; beq=[]
    for i in range(na):
        row=np.zeros(na*nb); row[i*nb:(i+1)*nb]=1
        Aeq.append(row); beq.append(aa[i])
    for j in range(nb):
        row=np.zeros(na*nb); row[j::nb]=1
        Aeq.append(row); beq.append(bb[j])
    sol=linprog(CC.ravel(), A_eq=np.asarray(Aeq), b_eq=np.asarray(beq), bounds=(0,None), method='highs')
    if not sol.success: raise RuntimeError(sol.message)
    return float(sol.fun)


def replication_stress(outdir):
    rows=[]
    rng=np.random.default_rng(20260909)
    max_random_violation=0.0
    for r in [1,2,4,8,16,32,64,128,256]:
        bound=(math.sqrt(r+1)-1)/(math.sqrt(r+1)+1)
        pstar=1/(1+math.sqrt(r+1))
        exact=r*pstar*(1-pstar)/(1+r*pstar)
        rows.append({'extra_copies':r,'p_star':pstar,'sharp_shift':exact,'universal_bound':bound,
                     'sharpness_ratio':exact/bound if bound else 1.0,
                     'm_after_two_label_bank':2+r,'M_distinct':2,
                     'log_m':math.log(2+r),'log_M':math.log(2)})
        for _ in range(500):
            m=int(rng.integers(2,30)); p=rng.dirichlet(np.ones(m))
            # random partition into output classes, force label 0's class to be known
            cls=rng.integers(0,max(2,m//3),size=m)
            qj=p[cls==cls[0]].sum()
            shift=r*p[0]*(1-qj)/(1+r*p[0])
            max_random_violation=max(max_random_violation,shift-bound)
    df=pd.DataFrame(rows); df.to_csv(outdir/'step9_replication_cardinality.csv',index=False)
    return {'max_random_bound_violation':max_random_violation,'max_sharpness_residual':float(np.max(np.abs(df.sharp_shift-df.universal_bound)))}


def topology_stress(outdir):
    # Exhaustive 3x3 Hamming-one transitions for beta0; beta1 sharp 5x5 construction.
    states=all_masks(3,3)
    rec=[]; maxb=[0,0]; counts=[0,0]
    seen=set()
    for idx,x in enumerate(states):
        for u in range(9):
            y=x.copy().reshape(-1); y[u]=~y[u]; y=y.reshape(3,3)
            j=mask_index(y)
            pair=tuple(sorted((idx,j)))
            if pair in seen: continue
            seen.add(pair)
            bx,by=betti(x),betti(y)
            for k in [0,1]:
                d=abs(int(by[k])-int(bx[k])); maxb[k]=max(maxb[k],d); counts[k]+=int(d==3)
                rec.append({'grid':'3x3','pair_id':len(seen),'k':k,'d_h':1,'abs_delta':d,'ratio_to_3d':d/3})
    # Explicit beta0 sharpness.
    x=np.zeros((3,3),bool); x[0,0]=x[0,2]=x[2,0]=x[2,2]=1; y=x.copy(); y[1,1]=1
    b0sharp=abs(int(betti(y)[0])-int(betti(x)[0]))
    # Explicit beta1 sharpness from proof.
    z=np.ones((5,5),bool); z[2,2]=z[1,2]=z[3,2]=z[2,1]=z[2,3]=0
    w=z.copy(); w[2,2]=1
    b1sharp=abs(int(betti(w)[1])-int(betti(z)[1]))
    # Random multi-flip stress.
    rng=np.random.default_rng(909)
    max_violation=0.0; worst_ratio=0.0
    multi=[]
    for h,wid in [(5,5),(8,8),(12,12)]:
        for _ in range(5000):
            x=rng.random((h,wid))<rng.uniform(.1,.9); y=x.copy()
            d=int(rng.integers(1,min(12,h*wid)+1)); ids=rng.choice(h*wid,d,replace=False); y.flat[ids]=~y.flat[ids]
            bx,by=betti(x),betti(y)
            for k in [0,1]:
                delta=abs(int(by[k])-int(bx[k])); ratio=delta/(3*d)
                max_violation=max(max_violation,delta-3*d); worst_ratio=max(worst_ratio,ratio)
                multi.append({'grid':f'{h}x{wid}','d_h':d,'k':k,'abs_delta':delta,'ratio_to_3d':ratio})
    pd.DataFrame(rec).to_csv(outdir/'step9_topology_hamming1_3x3.csv',index=False)
    pd.DataFrame(multi).to_csv(outdir/'step9_topology_random_multiflip.csv',index=False)
    return {'exhaustive_3x3_pairs':len(seen),'beta0_max_abs_delta_3x3':maxb[0],'beta1_max_abs_delta_3x3':maxb[1],
            'beta0_sharp_count_3x3':counts[0],'beta1_sharp_count_3x3':counts[1],
            'explicit_beta0_sharp_delta':b0sharp,'explicit_beta1_sharp_delta':b1sharp,
            'random_multiflip_checks':len(multi),'max_bound_violation':max_violation,'max_random_ratio':worst_ratio}


def identifiability_stress(outdir):
    states=all_masks(3,3); n=len(states); m=len(SPECS)
    outs=np.empty((m,n),dtype=np.int32)
    for j,spec in enumerate(SPECS):
        for i,x in enumerate(states): outs[j,i]=mask_index(transform(x,spec))
    # Gram A^T A: number of inputs on which two maps agree.
    G=np.empty((m,m),float)
    for j in range(m):
        for k in range(m): G[j,k]=np.count_nonzero(outs[j]==outs[k])
    rank=int(np.linalg.matrix_rank(G,tol=1e-8))
    uniq=np.unique(outs,axis=0).shape[0]
    # Prefix diagnostics.
    pref=[]
    for q in range(1,m+1):
        r=int(np.linalg.matrix_rank(G[:q,:q],tol=1e-8)); u=np.unique(outs[:q],axis=0).shape[0]
        pref.append({'prefix_actions':q,'rank':r,'unique_maps':u,'identifiable':int(r==q)})
    pd.DataFrame(pref).to_csv(outdir/'step9_identifiability_prefix.csv',index=False)
    pd.DataFrame(G,index=NAMES,columns=NAMES).to_csv(outdir/'step9_identifiability_gram.csv')
    # Duplicate one action explicitly.
    Gdup=np.block([[G, G[:,[0]]],[G[[0],:], np.array([[n]],float)]])
    rankdup=int(np.linalg.matrix_rank(Gdup,tol=1e-8))
    # Null direction if rank deficient in original.
    evals,evecs=np.linalg.eigh(G)
    null_res=float(np.min(evals))
    return {'grid':'3x3','states':n,'actions':m,'global_incidence_rank':rank,'unique_deterministic_maps':int(uniq),
            'full_bank_identifiable':bool(rank==m),'duplicated_bank_actions':m+1,'duplicated_bank_rank':rankdup,
            'duplicate_creates_nonidentifiability':bool(rankdup<m+1),'min_gram_eigenvalue':null_res}


def build_kernel(states, specs, weights):
    n=len(states); K=np.zeros((n,n),float)
    for i,x in enumerate(states):
        for s,p in zip(specs,weights): K[i,mask_index(transform(x,s))]+=p
    return K


def bayes_reversal_stress(outdir):
    states=all_masks(3,3); n=len(states)
    # A lossy fixed mixture chosen from the paper's bank.
    specs=[SPECS[0], SPECS[2], SPECS[17], SPECS[25]]
    weights=np.array([.25,.25,.25,.25])
    H=build_kernel(states,specs,weights)
    mu=np.ones(n)/n; nu=mu@H
    rev=np.zeros((n,n),float)
    pos=nu>0
    rev[pos,:]=(mu[:,None]*H).T[pos,:]/nu[pos,None]
    rev[~pos,:]=mu
    marginal=nu@rev
    marginal_res=float(np.max(np.abs(marginal-mu)))
    left=H@rev
    left_res=float(np.max(np.abs(left-np.eye(n))))
    support=(H>0)
    overlaps=0; pair=[]; max_formula=0.0
    for a in range(n):
        for b in range(a+1,n):
            tv=.5*np.abs(H[a]-H[b]).sum(); err=.5*(1-tv)
            # direct equal-prior classification error
            err2=1-.5*np.maximum(H[a],H[b]).sum()
            max_formula=max(max_formula,abs(err-err2))
            ov=bool(np.any(support[a]&support[b])); overlaps+=ov
            pair.append({'x0':a,'x1':b,'tv':tv,'bayes_error':err,'support_overlap':int(ov)})
    pd.DataFrame(pair).to_csv(outdir/'step9_bayes_pairwise.csv',index=False)
    # Global Bayes error from uniform prior.
    global_err=1-float(np.max(mu[:,None]*H,axis=0).sum())
    return {'states':n,'kernel_actions':len(specs),'uniform_prior_marginal_recovery_max_abs':marginal_res,
            'HHreverse_identity_max_abs_residual':left_res,'row_pairs':len(pair),'row_pairs_with_support_overlap':overlaps,
            'pairwise_formula_max_abs_residual':max_formula,'global_optimal_classification_error':global_err,
            'exact_left_inverse_support_condition':bool(overlaps==0)}


def kappa_exact(K,C):
    n=len(K); k=0.0; arg=None
    for i in range(n):
        for j in range(i+1,n):
            d=C[i,j]
            if d<=0: continue
            w=finite_ot(K[i],K[j],C); ratio=w/d
            if ratio>k+1e-12: k=ratio; arg=(i,j,w,d)
    return k,arg


def transport_stress(outdir):
    states=all_masks(2,2); n=len(states); C=hamming_cost(states)
    identity=SPECS[0]; large_open=('O','square',5); dil=('C','square',3) # closing is extensive and can amplify boundary changes
    theta=.6
    Kc=build_kernel(states,[identity,large_open],[theta,1-theta])
    # Find a deterministic bank op with largest Hamming Lipschitz factor on 2x2.
    best=(0,None,None)
    for name,spec in zip(NAMES,SPECS):
        T=np.array([mask_index(transform(x,spec)) for x in states])
        L=0
        for i in range(n):
            for j in range(i+1,n):
                L=max(L,C[T[i],T[j]]/C[i,j])
        if L>best[0]: best=(L,name,spec)
    Kn=Kc.copy(); rows=[]; max_iter_violation=0.0
    beta=np.array([betti(x) for x in states])
    for step in range(1,9):
        kap,arg=kappa_exact(Kn,C)
        theoretical=theta**step
        # verify topology iterate inequality over all pairs/k
        max_top=0.0
        for i in range(n):
            for j in range(i+1,n):
                for kk in [0,1]:
                    lhs=abs(Kn[i]@beta[:,kk]-Kn[j]@beta[:,kk]); rhs=3*(theta**step)*C[i,j]
                    max_iter_violation=max(max_iter_violation,lhs-rhs)
                    if rhs>0: max_top=max(max_top,lhs/rhs)
        rows.append({'n':step,'kappa_exact':kap,'theta_power':theoretical,'abs_residual':abs(kap-theoretical),'max_topology_bound_ratio':max_top})
        Kn=Kn@Kc
    pd.DataFrame(rows).to_csv(outdir/'step9_transport_iterates.csv',index=False)
    # Deterministic expansion example exact kappa via row laws.
    spec=best[2]; Kexp=build_kernel(states,[spec],[1.0]); kap_exp,arg_exp=kappa_exact(Kexp,C)
    # Composition inequality for Kc with itself.
    kap1,_=kappa_exact(Kc,C); kap2,_=kappa_exact(Kc@Kc,C)
    return {'states':n,'contractive_kernel':'0.6 identity + 0.4 O_square5','kappa_K':kap1,'expected_kappa':theta,
            'kappa_K2':kap2,'kappa_product_bound':kap1**2,'max_iterate_kappa_residual':float(pd.DataFrame(rows).abs_residual.max()),
            'max_iterate_topology_bound_violation':max_iter_violation,'largest_deterministic_bank_lipschitz':float(best[0]),
            'largest_deterministic_bank_action':best[1],'exact_kappa_that_action':kap_exp}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output-dir',default='results/step9_theorem_stress')
    args=ap.parse_args(); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    t=time.time()
    sections={}
    sections['replication_cardinality']=replication_stress(out)
    sections['topology']=topology_stress(out)
    sections['identifiability']=identifiability_stress(out)
    sections['bayes_reversal']=bayes_reversal_stress(out)
    sections['transport_composition']=transport_stress(out)
    summary={'status':'PASS','seconds':time.time()-t,**sections}
    # Global gates
    assert sections['replication_cardinality']['max_random_bound_violation'] <= 1e-12
    assert sections['replication_cardinality']['max_sharpness_residual'] <= 1e-12
    assert sections['topology']['explicit_beta0_sharp_delta']==3 and sections['topology']['explicit_beta1_sharp_delta']==3
    assert sections['topology']['max_bound_violation'] <= 0
    assert sections['identifiability']['duplicate_creates_nonidentifiability']
    assert sections['bayes_reversal']['uniform_prior_marginal_recovery_max_abs'] < 1e-12
    assert sections['bayes_reversal']['pairwise_formula_max_abs_residual'] < 1e-12
    assert sections['transport_composition']['max_iterate_kappa_residual'] < 1e-10
    assert sections['transport_composition']['max_iterate_topology_bound_violation'] < 1e-10
    (out/'step9_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
