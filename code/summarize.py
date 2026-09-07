"""Generate every empirical table/plot from the saved raw outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import softmax
from morphology import NAMES, METRICS, measure, betti

ROOT=Path(__file__).resolve().parents[1]
DS=['shapes','mnist','fashion']
LABEL={'shapes':'Shapes','mnist':'MNIST masks','fashion':'Fashion masks'}
MAIN=['Input','Validation best','Classical best','Soft morphology','Learned local filter',
      'Conditional selector','Conditional sampled','Conditional mean mask','Exact softmax gate',
      'Uniform sampled','No topology','Three openings','Oracle (diagnostic)']
COLORS=['#263b53','#a7aeb8','#d2a648','#67998e','#4c78a8','#d7644b']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
    'axes.labelsize':10,'figure.dpi':150,'savefig.bbox':'tight','pdf.fonttype':42,
    'font.family':'DejaVu Sans'})

def intervals(v,alpha=.05,seed=41001):
    v=np.asarray(v);rng=np.random.default_rng(seed)
    draws=np.mean(v[rng.integers(0,len(v),(2000,len(v)))],axis=1)
    return np.quantile(draws,[alpha/2,1-alpha/2])

def escaped(s):return s.replace('_',r'\_').replace('&',r'\&')

def main():
    full=pd.concat([pd.read_csv(ROOT/'results'/f'{d}_test_metrics.csv.gz') for d in DS],ignore_index=True)
    primary=full[~full.condition.isin(['clean','shift20'])]
    clusters=primary.groupby(['dataset','method','image'])[METRICS].mean().reset_index()
    summary=[];paired=[];action=[]
    for (d,m),s in clusters.groupby(['dataset','method']):
        item={'dataset':d,'method':m,'n_images':len(s)}
        for metric in METRICS:
            v=s[metric].to_numpy();lo,hi=intervals(v)
            item.update({metric:float(v.mean()),metric+'_sd_images':float(v.std(ddof=1)),
                         metric+'_low':float(lo),metric+'_high':float(hi)})
        item['aggregate_psnr']=-10*np.log10(item['mse']) if item['mse']>0 else np.inf
        item['psnr_low']=-10*np.log10(item['mse_high']) if item['mse_high']>0 else np.inf
        item['psnr_high']=-10*np.log10(item['mse_low']) if item['mse_low']>0 else np.inf
        runs=primary[(primary.dataset==d)&(primary.method==m)].groupby('seed').loss.mean()
        item['run_loss_sd']=float(runs.std(ddof=1)) if len(runs)>1 else 0.
        summary.append(item)
    summary=pd.DataFrame(summary);summary.to_csv(ROOT/'results'/'summary.csv',index=False)
    for d in DS:
        s=clusters[clusters.dataset==d].pivot(index='image',columns='method',values='loss')
        for m in ['Validation best','Classical best']:
            v=(s[m]-s['Conditional selector']).to_numpy()
            lo,hi=intervals(v);simlo,simhi=intervals(v,alpha=.05/6)
            paired.append({'dataset':d,'control':m,'gain':v.mean(),'low':lo,'high':hi,
                'simultaneous_low':simlo,'simultaneous_high':simhi,'win_fraction':(v>0).mean()})
        b=np.load(ROOT/'results'/f'{d}_test_bank.npz');condition=b['condition'];ix=condition<6
        means=b['scores'][ix].mean(0)
        for j,n in enumerate(NAMES):action.append({'dataset':d,'action':n,**dict(zip(METRICS,means[j]))})
    paired=pd.DataFrame(paired);paired.to_csv(ROOT/'results'/'paired_comparisons.csv',index=False)
    action=pd.DataFrame(action);action.to_csv(ROOT/'results'/'all_candidate_metrics.csv',index=False)
    bycondition=full.groupby(['dataset','method','condition','image'])[METRICS].mean().reset_index().groupby(['dataset','method','condition'])[METRICS].mean()
    bycondition.to_csv(ROOT/'results'/'condition_metrics.csv')
    primary.groupby(['dataset','method','seed'])[METRICS].mean().to_csv(ROOT/'results'/'run_metrics.csv')
    # Main figure: all fixed controls and primary selector, common absolute scale.
    selected=['Validation best','Classical best','Soft morphology','Learned local filter','Conditional selector','Exact softmax gate']
    short=['Fixed bank','Classical','Smooth','Local learned','Conditional','Exact gate']
    fig,axes=plt.subplots(1,3,figsize=(11.6,3.5),sharey=True)
    for ax,d in zip(axes,DS):
        s=summary[summary.dataset==d].set_index('method').loc[selected]
        ax.bar(np.arange(len(s)),s.loss,color=COLORS,width=.72)
        ax.errorbar(np.arange(len(s)),s.loss,yerr=[s.loss-s.loss_low,s.loss_high-s.loss],fmt='none',ecolor='black',capsize=3,lw=.8)
        ax.set_xticks(np.arange(len(s)),short,rotation=38,ha='right');ax.set_title(LABEL[d])
        ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    axes[0].set_ylabel('Composite loss (lower is better)')
    fig.tight_layout();fig.savefig(ROOT/'figures'/'primary_comparison.pdf');fig.savefig(ROOT/'figures'/'primary_comparison.png');plt.close(fig)
    # The full candidate bank makes omitted-baseline selection visible.
    matrix=action.pivot(index='action',columns='dataset',values='loss').reindex(NAMES)[DS]
    fig,ax=plt.subplots(figsize=(6.2,10.5))
    im=ax.imshow(matrix,aspect='auto',cmap='viridis_r',vmin=0,vmax=.65)
    ax.set_xticks(range(3),[LABEL[d] for d in DS]);ax.set_yticks(range(len(NAMES)),NAMES,fontsize=8)
    for i in range(len(NAMES)):
        for j in range(3):
            v=matrix.iloc[i,j];ax.text(j,i,f'{v:.3f}',ha='center',va='center',fontsize=8,color='white' if v>.34 else 'black')
    fig.colorbar(im,ax=ax,shrink=.5,label='Composite loss')
    fig.tight_layout();fig.savefig(ROOT/'figures'/'candidate_bank.pdf');fig.savefig(ROOT/'figures'/'candidate_bank.png');plt.close(fig)
    # Noise and clean-input diagnostics. Means here also average model seeds.
    conditions=['balanced04','balanced10','salt04','salt10','pepper04','pepper10','shift20','clean']
    fig,axes=plt.subplots(1,3,figsize=(11.6,3.6),sharey=True)
    for ax,d in zip(axes,DS):
        z=bycondition.loc[d]
        for m,col,marker in [('Validation best','#5d7183','o'),('Conditional selector','#d7644b','s'),('Classical best','#b48d29','^')]:
            ax.plot(range(8),z.loc[m].reindex(conditions).loss,label=m,color=col,marker=marker,ms=4)
        ax.set_xticks(range(8),['B .04','B .10','S .04','S .10','P .04','P .10','B .20','Clean'],rotation=55,ha='right')
        ax.axvline(5.5,color='gray',ls='--',lw=.8);ax.set_title(LABEL[d]);ax.grid(alpha=.18)
    axes[0].set_ylabel('Composite loss');axes[1].legend(loc='upper center',bbox_to_anchor=(.5,1.32),ncol=3,fontsize=8)
    fig.tight_layout();fig.savefig(ROOT/'figures'/'noise_diagnostics.pdf');fig.savefig(ROOT/'figures'/'noise_diagnostics.png');plt.close(fig)
    # Deterministic qualitative rule: best and worst primary-condition gain, seed 101.
    fig,axes=plt.subplots(6,5,figsize=(10,12))
    qual=[]
    for row,d in enumerate(DS):
        b=np.load(ROOT/'results'/f'{d}_test_bank.npz');pred=np.load(ROOT/'results'/f'{d}_predictions.npz')
        clean=np.load(ROOT/'data'/f'{d}_clean.npz');sel=json.loads((ROOT/'results'/f'{d}_selection.json').read_text())
        j=NAMES.index(sel['best_deterministic']);actions=pred['actions_101'];idx=np.arange(len(actions))
        gain=b['scores'][:,j,0]-b['scores'][idx,actions,0]
        allowed=np.flatnonzero(b['condition']<6)
        cases=[allowed[np.argmax(gain[allowed])],allowed[np.argmin(gain[allowed])]]
        h,w=map(int,b['shape'])
        for k,q in enumerate(cases):
            r=2*row+k;i=int(b['image_index'][q]);truth=clean['test'][i]
            x=np.unpackbits(b['noisy'][q],count=h*w).reshape(h,w)
            outputs=np.unpackbits(b['outputs'][q],axis=-1,count=w).astype(bool)
            fixed=outputs[j];conditional=outputs[actions[q]]
            error=np.ones((h,w,3))*.97
            error[conditional & ~truth]=[.82,.18,.12];error[~conditional & truth]=[.12,.35,.8]
            for c,img in enumerate([truth,x,fixed,conditional,error]):
                axes[r,c].imshow(img,cmap='gray',vmin=0,vmax=1,interpolation='nearest');axes[r,c].set_xticks([]);axes[r,c].set_yticks([])
                for spine in axes[r,c].spines.values():spine.set_visible(False)
            axes[r,0].set_ylabel(LABEL[d]+('\nLargest gain' if k==0 else '\nLargest loss'),fontsize=9)
            axes[r,3].set_xlabel(f'Loss {b["scores"][q,actions[q],0]:.3f}; gain {gain[q]:+.3f}',fontsize=8)
            qual.append({'dataset':d,'row':r,'image':i,'case_index':int(q),'condition':int(b['condition'][q]),'selected_action':NAMES[actions[q]],'gain':float(gain[q])})
    for ax,title in zip(axes[0],['Clean target','Noisy input','Fixed bank action','Conditional action','Conditional error']):ax.set_title(title,fontsize=10)
    fig.tight_layout(h_pad=.4);fig.savefig(ROOT/'figures'/'qualitative_cases.pdf');fig.savefig(ROOT/'figures'/'qualitative_cases.png');plt.close(fig)
    (ROOT/'results'/'qualitative_case_ids.json').write_text(json.dumps(qual,indent=2)+'\n')
    # Actual score-gradient variance on 32 fixed MNIST training cases.
    b=np.load(ROOT/'results'/'mnist_train_bank.npz');pr=np.load(ROOT/'results'/'mnist_predictions.npz')
    x=(b['features'][:32]-pr['feature_mean'])/pr['feature_std'];x=np.c_[np.ones(len(x)),x]
    prob=softmax(x@pr['gate_weights_101'],axis=1);costs=b['scores'][:32,:,0]
    vr=[];rng=np.random.default_rng(9182)
    for baseline in ['zero','conditional mean']:
        for samples in [1,8,64]:
            theory=[];emp=[]
            for p,L in zip(prob,costs):
                g=p*(L-p@L);base=0. if baseline=='zero' else p@L
                draws=(L-base)[:,None]*(np.eye(len(p))-p)
                theory.append(np.sum(p[:,None]*(draws-g)**2)/samples)
                ids=rng.choice(len(p),size=(2000,samples),p=p)
                est=draws[ids].mean(1);emp.append(np.mean(np.sum((est-g)**2,axis=1)))
            vr.append({'baseline':baseline,'samples':samples,'theory_trace':np.mean(theory),'empirical_squared_error':np.mean(emp)})
    vr=pd.DataFrame(vr);vr.to_csv(ROOT/'results'/'gradient_variance.csv',index=False)
    fig,ax=plt.subplots(figsize=(6,3.5))
    for base,col in [('zero','#5d7183'),('conditional mean','#d7644b')]:
        z=vr[vr.baseline==base];ax.loglog(z.samples,z.theory_trace,color=col,label=base+' baseline: exact variance')
        ax.scatter(z.samples,z.empirical_squared_error,color=col,marker='x',s=45)
    ax.set_xlabel('Independent action draws');ax.set_ylabel('Logit-gradient squared error');ax.grid(alpha=.2,which='both')
    ax.legend(fontsize=8);ax.text(.98,.02,'Enumeration: zero action-sampling variance',ha='right',transform=ax.transAxes,fontsize=8)
    fig.tight_layout();fig.savefig(ROOT/'figures'/'gradient_variance.pdf');fig.savefig(ROOT/'figures'/'gradient_variance.png');plt.close(fig)
    # Dynamic LaTeX: every number below is sourced from the retained raw results.
    tex=[]
    for d in DS:
        s=summary[summary.dataset==d].set_index('method')
        tex += [r'\begin{table}[htbp]',r'\centering\small',
            r'\caption{'+LABEL[d]+r': primary conditions. Loss includes a 95\% image-cluster interval. PSNR is derived from aggregate binary MSE. Raw component and hole errors are reported separately.}',
            r'\begin{tabular}{lccrrr}',r'\toprule Method & Loss [95\% CI] & PSNR & Dice & $E_0$ & $E_1$\\\midrule']
        for m in MAIN:
            z=s.loc[m];label=m.replace('Oracle (diagnostic)','Target oracle')
            tex.append(f'{label} & {z.loss:.3f} [{z.loss_low:.3f}, {z.loss_high:.3f}] & {z.aggregate_psnr:.2f} & {z.dice:.3f} & {z.beta0_error:.2f} & {z.beta1_error:.2f}'+r'\\')
        tex += [r'\bottomrule\end{tabular}',r'\end{table}']
    tex += [r'\begin{table}[htbp]\centering\small',
        r'\caption{Paired gain in composite loss: control minus conditional selector. Positive values favor conditioning. Simultaneous intervals use Bonferroni-adjusted bootstrap percentiles for the six listed comparisons.}\label{tab:paired}',
        r'\begin{tabular}{llrr} \toprule Dataset & Control & Gain & Simultaneous interval\\\midrule']
    for _,r in paired.iterrows():
        tex.append(f'{LABEL[r.dataset]} & {r.control} & {r.gain:+.4f} & [{r.simultaneous_low:+.4f}, {r.simultaneous_high:+.4f}]'+r'\\')
    tex += [r'\bottomrule\end{tabular}\end{table}']
    for d in DS:
        s=summary[summary.dataset==d].set_index('method');c=s.loc['Conditional selector'];b=s.loc['Validation best'];cl=s.loc['Classical best']
        rows=paired[(paired.dataset==d)&(paired.control=='Validation best')].iloc[0]
        word='reduces' if rows.gain>0 else 'increases'
        tex.append(f'For {LABEL[d]}, conditional selection {word} mean composite loss relative to the validation-selected bank action by {abs(rows.gain):.4f}. '
          f'Its mean loss is {c.loss:.4f}, compared with {b.loss:.4f} for that fixed action and {cl.loss:.4f} for the selected classical control. '
          f'The standard deviation of the five conditional-selector run means is {c.run_loss_sd:.4f}. '
          f'The fraction with both Betti counts correct is {100*c.topology_exact:.1f}\\%, and mean IoU is {c.iou:.3f}.')
    tex.append('These comparisons measure the specified composite loss. A lower value does not imply an improvement in every individual metric, and a larger operator bank is not a new primitive morphological operation. All original square sizes and every additional action are retained in the complete candidate table and Figure~\\ref{fig:bank}.')
    tex.append('In particular, MNIST Dice decreases slightly under conditional selection, and the synthetic-shape hole error increases despite the composite-loss gain. Under stronger balanced noise on Fashion-MNIST, the conditional selector loses to the fixed area filter. These are material limitations of the fitted decision rule, rather than exceptions to be omitted from the evaluation.')
    # Runtime is measured separately to include full bank and feature evaluation.
    runtime_path=ROOT/'results'/'runtime.csv'
    if runtime_path.exists():
        rt=pd.read_csv(runtime_path).groupby('dataset')[['conditional_ms_per_image','area8_ms_per_image']].median()
        tex += [r'\begin{table}[htbp]\centering\small',
            r'\caption{Median of three end-to-end CPU timing runs, 100 primary cases per run, batch size one, milliseconds per image. Conditional timing includes all 31 filters and feature computation.}',
            r'\begin{tabular}{lrr}\toprule Dataset & Conditional & Fixed area-8\\\midrule']
        for d in DS:
            r=rt.loc[d];tex.append(f'{LABEL[d]} & {r.conditional_ms_per_image:.3f} & {r.area8_ms_per_image:.3f}'+r'\\')
        tex += [r'\bottomrule\end{tabular}\end{table}',
            'The implemented selector is substantially slower than the fixed filter because it evaluates the entire bank and all topological features. CPU measurements use one BLAS thread and one forest worker on the available shared host; they are environment-specific. No speed advantage is claimed.']
    tex += [r'\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{primary_comparison.pdf}',
        r'\caption{Primary test loss and 95\% image-cluster intervals for the main deployable controls. All methods use the same clean-image splits and corruption cases.}\label{fig:primary}\end{figure}',
        r'\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{noise_diagnostics.pdf}',
        r'\caption{Condition-specific losses. B, S, and P denote balanced, salt-dominant, and pepper-dominant noise. Conditions to the right of the dashed divider are diagnostics excluded from the primary averages.}\label{fig:noise}\end{figure}',
        r'\begin{figure}[p]\centering\includegraphics[width=\linewidth,height=.83\textheight,keepaspectratio]{qualitative_cases.pdf}',
        r'\caption{Actual final-test masks: largest gain and largest loss relative to the fixed bank action in each dataset, selected by a declared deterministic rule for seed 101. These are extreme illustrative cases, not representative random samples. Red denotes false foreground; blue denotes missed foreground. Case IDs and actions are supplied.}\label{fig:qualitative}\end{figure}',
        r'\begin{figure}[htbp]\centering\includegraphics[width=.8\linewidth]{gradient_variance.pdf}',
        r'\caption{Score-gradient variance on 32 fixed MNIST training cases using the fitted gate from seed 101. Lines are exact categorical variance divided by the draw count; crosses average 2,000 independently sampled estimates per case. Exact enumeration has zero action-sampling variance.}\label{fig:variance}\end{figure}',
        r'\begin{figure}[p]\centering\includegraphics[width=.8\linewidth,height=.85\textheight,keepaspectratio]{candidate_bank.pdf}',
        r'\caption{Mean loss of every fixed candidate on final primary test cases. These test values are reported for comparison; the deployed fixed action was chosen using validation data.}\label{fig:bank}\end{figure}']
    (ROOT/'results'/'generated_results.tex').write_text('\n\n'.join(tex)+'\n')
    print(summary[summary.method.isin(['Conditional selector','Validation best','Classical best','Oracle (diagnostic)'])][['dataset','method','loss','dice','beta0_error','beta1_error','run_loss_sd']].to_string(index=False))
    print(paired.to_string(index=False))

if __name__=='__main__':main()
