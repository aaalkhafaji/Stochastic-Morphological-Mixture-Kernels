"""Independent finite checks and counterexamples. These complement, not replace, proofs."""
from pathlib import Path
import itertools, json, math
import numpy as np
from scipy import ndimage as ndi
from scipy.linalg import expm
from scipy.optimize import check_grad
from scipy.special import softmax
from morphology import betti, transform, SPECS, NAMES, softmax_objective, measure

ROOT=Path(__file__).resolve().parents[1]

def main():
    rng=np.random.default_rng(20260905);report={}
    # Independently specified set erosion/dilation checks the library implementation.
    def erode(A,B):
        if not A:return set()
        return {x for x in range(-15,16) if all(x+b in A for b in B)}
    def dilate(A,B): return {a+b for a in A for b in B}
    nchecked=0
    for n in (2,3,4,5):
        B=set(range(-(n//2),n-n//2))
        for bits in itertools.product((0,1),repeat=7):
            A={i for i,b in enumerate(bits) if b}
            expected=dilate(erode(A,B),B)
            arr=np.zeros((1,7),bool);arr[0]=bits
            actual=transform(arr,('O','horizontal',n))[0]
            assert set(np.flatnonzero(actual))==expected.intersection(range(7))
            nchecked+=1
    report['set_definition_openings_checked']=nchecked
    # Exhaustive 3x3 state space and every single-pixel change (512*9 comparisons).
    masks=[np.array(z,dtype=bool).reshape(3,3) for z in itertools.product((0,1),repeat=9)]
    bs=np.array([betti(x) for x in masks]); maxdiff=np.zeros(2,int)
    for i,x in enumerate(masks):
        for k in range(9):
            y=x.copy(); y.flat[k]=~y.flat[k]
            d=abs(betti(y)-bs[i]);maxdiff=np.maximum(maxdiff,d)
            assert (d<=3).all()
    report['single_pixel_topology_checks']=len(masks)*9
    report['max_beta_changes']=maxdiff.tolist()
    ring=np.ones((3,3),bool); ring[1,1]=False
    assert tuple(betti(ring))==(1,1) and tuple(betti(np.ones((3,3),bool)))==(1,0)
    assert tuple(betti(np.eye(2,dtype=bool)))==(1,0)
    assert tuple(betti(np.zeros((3,3),bool)))==(0,0)
    report['topology_convention_cases']='passed'
    # Joint output-law identifiability, using all 512 inputs and all observed outcomes.
    rows=[]
    for x in masks:
        sig=[np.packbits(transform(x,s)).tobytes() for s in SPECS]
        for out in set(sig): rows.append([int(z==out) for z in sig])
    A=np.asarray(rows,dtype=float)
    singular=np.linalg.svd(A,compute_uv=False)
    report['identifiability_3x3']={'matrix_shape':list(A.shape),'rank':int(np.linalg.matrix_rank(A)),
        'actions':len(NAMES),'singular_values':singular.tolist()}
    # Four distinct maps on {0,1}, yet two different mixtures give the same kernel.
    K0=np.array([[1.,0.],[1.,0.]])
    K1=np.array([[0.,1.],[0.,1.]])
    KI=np.eye(2);KN=1-KI
    assert np.array_equal((K0+K1)/2,(KI+KN)/2)
    report['distinct_maps_nonidentifiable_counterexample']='passed'
    # Shared vs independent structuring elements yield different output laws.
    A0={0}; b0={0};b1={0,1}
    shared={};independent={}
    for b in (b0,b1):
        z=tuple(sorted(dilate(erode(A0,b),b)));shared[z]=shared.get(z,0)+.5
        for c in (b0,b1):
            z=tuple(sorted(dilate(erode(A0,b),c)));independent[z]=independent.get(z,0)+.25
    assert shared!=independent
    report['shared_parameter_law']={str(k):v for k,v in shared.items()}
    report['independent_parameter_law']={str(k):v for k,v in independent.items()}
    # Exact gradient and Monte Carlo variance identities.
    X=rng.normal(size=(13,4)); costs=rng.random((13,5)); w=rng.normal(size=20)
    def f(w):return softmax_objective(w,X,costs)[0]
    def g(w):return softmax_objective(w,X,costs)[1]
    error=check_grad(f,g,w,epsilon=1e-6)
    assert error<1e-6
    report['finite_difference_gradient_l2_error']=float(error)
    p=softmax(rng.normal(size=7)); losses=rng.random(7)
    exact=p*(losses-p@losses)
    scores=np.eye(7)-p
    raw=losses[:,None]*scores
    tracevar=float(np.sum(p[:,None]*(raw-exact)**2))
    assert np.max(abs(p@raw-exact))<1e-14
    report['gradient_variance']={'single_draw_trace':tracevar,
        'sample_mean_8_trace':tracevar/8,'sample_mean_64_trace':tracevar/64,
        'exact_enumeration_action_variance':0.}
    # Loss geometry, Gibbs regret, and risk-estimation perturbation bound.
    worst=-1.
    for _ in range(2000):
        r=rng.random(27);rh=np.clip(r+rng.normal(0,.03,27),0,1)
        eps=max(abs(r-rh));tau=float(rng.uniform(.001,.2));pi=softmax(-rh/tau)
        slack=pi@r-r.min()-2*eps-tau*np.log(len(r));worst=max(worst,slack)
        assert slack<=1e-12
        assert pi@r>=r.min()-1e-12
    report['risk_inequality_trials']=2000;report['max_gibbs_bound_violation']=max(0.,float(worst))
    # A C0 Markov semigroup has no Markov left inverse even when its matrix is invertible.
    t=.7;K=np.array([[1.,0.],[1-np.exp(-t),np.exp(-t)]])
    inv=np.linalg.inv(K);assert inv.min()<0
    report['C0_inverse_counterexample']={'kernel':K.tolist(),'algebraic_inverse':inv.tolist()}
    # Finite-state Poissonization and semigroup law, independent matrix computations.
    B=rng.random((5,5));B/=B.sum(1,keepdims=True);lam=1.7;G=lam*(B-np.eye(5))
    P=expm(t*G);series=np.zeros_like(B);power=np.eye(5)
    for n in range(40):
        series+=np.exp(-lam*t)*(lam*t)**n/math.factorial(n)*power;power=power@B
    residual=max(abs(P-series).max(),abs(expm(.2*G)@expm(.5*G)-P).max())
    assert residual<1e-13
    report['poisson_semigroup_max_residual']=float(residual)
    # Correct the original parabolic closing by evaluating its exact minimizer.
    residual=0.
    for a in (.2,1.,5.):
        for theta in (.1,1.,10.):
            x=np.linspace(-2,2,31);y=x*(1+a*theta)
            recovered=-a*y*y/(2*(1+a*theta))+(y-x)**2/(2*theta)
            residual=max(residual,float(abs(recovered+a*x*x/2).max()))
    assert residual<1e-11
    report['parabolic_closing_max_residual']=residual
    # Collision of empty and singleton states for the original openings.
    e=np.zeros((9,9),bool);s=e.copy();s[4,4]=True
    assert all(np.array_equal(transform(e,('O','square',k)),transform(s,('O','square',k))) for k in (2,3,4))
    report['original_library_information_loss_counterexample']='passed'
    (ROOT/'results'/'mathematical_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
