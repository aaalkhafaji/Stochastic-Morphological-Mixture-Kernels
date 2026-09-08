"""Create the four-page reader guide from retained experimental summaries."""
from pathlib import Path
import json, csv
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from matplotlib import get_data_path
ROOT=Path(__file__).resolve().parents[1]
FONT=Path(get_data_path())/'fonts/ttf'
pdfmetrics.registerFont(TTFont('DVS',str(FONT/'DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('DVS-B',str(FONT/'DejaVuSans-Bold.ttf')))
pdfmetrics.registerFontFamily('DVS',normal='DVS',bold='DVS-B',italic='DVS',boldItalic='DVS-B')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle('BodyS',fontName='DVS',fontSize=9.0,leading=12.6,spaceAfter=7,textColor=colors.HexColor('#182b3b')))
styles.add(ParagraphStyle('TitleS',fontName='DVS-B',fontSize=19,leading=24,spaceAfter=12,textColor=colors.HexColor('#173a53')))
styles.add(ParagraphStyle('HeadS',fontName='DVS-B',fontSize=11.5,leading=15,spaceBefore=9,spaceAfter=6,textColor=colors.HexColor('#173a53')))
styles.add(ParagraphStyle('CellS',fontName='DVS',fontSize=8.3,leading=11))
styles.add(ParagraphStyle('SmallS',fontName='DVS',fontSize=7.8,leading=10.5,spaceAfter=6))
styles.add(ParagraphStyle('CodeS',fontName='Courier',fontSize=8,leading=12,spaceAfter=8,backColor=colors.HexColor('#f2f5f7'),borderPadding=5))
story=[]
def p(t,style='BodyS'):story.append(Paragraph(t,styles[style]))
def head(t):p(t,'HeadS')
def table(rows,widths):
    tab=Table([[Paragraph(escape(str(c)),styles['CellS']) for c in row] for row in rows],colWidths=widths,hAlign='LEFT')
    tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e9eff3')),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#91a4b2')),('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#91a4b2')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
    story.append(tab);story.append(Spacer(1,8))
def page(title):story.append(PageBreak());p(title,'TitleS')
def footer(c,doc):
    c.saveState();c.setFont('DVS',7.6);c.setFillColor(colors.HexColor('#536b7c'))
    c.drawString(54,30,'Supplementary Material S1 | Stochastic Morphological pi-Mixture Kernels')
    c.drawRightString(558,30,str(doc.page));c.restoreState()
def main():
    summary=list(csv.DictReader((ROOT/'results/summary.csv').open()))
    quotient=list(csv.DictReader((ROOT/'results/quotient_summary.csv').open()))
    checks=json.loads((ROOT/'results/kernel_theory_checks.json').read_text())
    oracle=json.loads((ROOT/'results/step4_oracle_risk_checks.json').read_text())
    p('Supplementary Material S1','TitleS')
    p('<b>Stochastic Morphological π-Mixture Kernels: Identifiability, Topological Stability, and Output-Space Learning</b>')
    p('Adnan H. Abdulwahid · Ram C. Neupane','SmallS')
    p('Reproducibility guide | Strengthening branch Step 7 | Baseline archive v2.1.0 | Prepared 7 September 2026','SmallS')
    head('Purpose and scope')
    p('This archive supplies the complete computational record accompanying the manuscript: frozen inputs, fitted models, candidate outputs, primary and diagnostic results, figure and table generators, and numerical checks of the mathematical claims. The written proofs remain in the main manuscript.')
    p('The original campaign evaluates a fixed morphological bank on a held-out test set. The subsequent output-space construction reuses that inspected set and is explicitly exploratory. Steps 5 and 6 add independent public-data validations: Oxford-IIIT Pet supplies higher-resolution human annotations under frozen synthetic corruptions, while DAVIS 2016 supplies genuine algorithm-produced segmentation errors with no synthetic corruption. Step 7 then freezes the Step-5 IID-fitted selectors and probes nine structured Oxford error families without refitting or retuning.')
    head('Primary experiment at a glance')
    table([['Dataset','Mask size','Train / validation','Scored test'],['Shapes','48 × 48','1,000 / 300','499'],['MNIST','28 × 28','2,000 / 500','1,000'],['Fashion-MNIST','28 × 28','2,000 / 500','1,000']],[115,85,177,127])
    p('The 31-action bank is frozen before the final primary evaluation. Six primary corruption conditions produce 14,994 noisy cases; two additional conditions produce 4,998 diagnostic cases. Five fitted seeds (101-105) are retained for each dataset, together with three learned local-filter models.')
    head('Mean primary composite loss')
    rows=[['Dataset','Validation-best fixed filter','Conditional selector']]
    for ds in ['shapes','mnist','fashion']:
        vals={r['method']:float(r['loss']) for r in summary if r['dataset']==ds}
        rows.append([ds,f"{vals['Validation best']:.6f}",f"{vals['Conditional selector']:.6f}"])
    table(rows,[134,185,185])
    p('Lower is better. These are image-cluster means across primary corruptions and fitted seeds. The comparisons concern this fixed bank and study design, not all denoising methods. Full metric tables and paired intervals are in results/.','SmallS')
    page('Protocol and indexing')
    head('Frozen data and exclusions')
    p('MNIST and Fashion-MNIST retain selected original pixels, thresholded masks (intensity ≥128), and original partition row IDs. Training and validation come from official training partitions; evaluation uses official test partitions. Development-pilot IDs and identical binary masks are quarantined. The shapes file retains 500 prospective test masks; ID 173 duplicates validation and is excluded from scoring, leaving 499.')
    head('Corruption schedule')
    table([['Index / name','Deletion','Addition','Role'],['0 balanced04 / 1 balanced10','.04 / .10','.04 / .10','Primary'],['2 salt04 / 3 salt10','.01 / .01','.04 / .10','Primary'],['4 pepper04 / 5 pepper10','.04 / .10','.01 / .01','Primary'],['6 shift20 / 7 clean','.20 / 0','.20 / 0','Diagnostic']],[228,78,78,120])
    p('Training and validation use one corruption per image. Each final test image uses all eight conditions. The exact corruption law and seeded random-number construction are in run_campaign.py. Repeated corruptions of the same image are clustered for reporting.')
    head('Loss, topology, and uncertainty')
    p('The objective is 0.4(1-Dice) + 0.2(binary MSE) + 0.2 times each capped, target-normalized component and hole error. Separately reported Betti-error columns contain raw absolute errors. Foreground uses 8-connectivity; background uses 4-connectivity with a permanent exterior frame. β₀ counts foreground components; β₁ counts bounded background components.')
    p('Sampled-method metrics are exact expectations over the finite categorical law. Mean-mask methods are scored separately. Aggregate PSNR is -10 log₁₀(mean binary MSE). The bootstrap uses 2,000 clean-image cluster resamples; six paired control comparisons use Bonferroni-adjusted percentile intervals. These descriptive intervals do not certify performance on a new noise distribution.')
    head('Decode and join arrays correctly')
    p('Bank scores have axes case × action × metric. Metric order is loss, MSE, Dice, IoU, component error, hole error, exact topology. Action order is recorded in environment.json. Map image_index through the clean split IDs to recover a public source row. Primary CSVs use condition names; quotient CSVs use numeric condition indices.')
    p('Candidate masks are packed on the width axis in big bit order: unpack with count=W. Noisy masks are flattened before packing: unpack with count=H×W and then reshape. Detailed field names and commands are in docs/REPRODUCIBILITY.md.','SmallS')
    page('Reproduce the evidence')
    head('Environment and integrity')
    p('Use Python 3.12 and install the pinned requirements in a fresh virtual environment. No GPU or source-data download is needed for ordinary reproduction. The archive includes the clean masks and selected public pixels. The wrapper sets one thread for BLAS and OpenMP; fitted forests use n_jobs=1.')
    p('python -m pip install -r requirements.txt<br/>python code/check_package.py','CodeS')
    p('The checksum check verifies every frozen release file. Run it before making edits. The reproduction wrapper copies the package to a new directory and rejects an existing destination. All generated files and timings are written into that copy.')
    head('Three execution levels')
    p('python code/reproduce.py --level quick --workdir ../smm-quick<br/>python code/reproduce.py --level analysis --workdir ../smm-analysis<br/>python code/reproduce.py --level full --workdir ../smm-full','CodeS')
    table([['Level','Executed work'],['quick','Original mathematical checks; direct-output retraining invariance; Step-4 oracle-risk audit; regenerated masks/features for two cases per dataset; all 15 forests replayed on those cases.'],['analysis','Quick checks, Step-3 invariance stress suite, full saved-model replays, seed-101 fresh refits, all summaries/figures, expanded mathematical checks, quotient validation, and this guide.'],['full','Recompute all candidate banks from frozen clean masks, refit all models, then perform the analysis level.']],[78,426])
    head('Where each result comes from')
    p('summarize.py regenerates primary metrics, bootstrap intervals, ablations, diagnostic plots, qualitative examples, and gradient-variance diagnostics. validate_kernel_quotient.py regenerates the exploratory output-space table and saved probabilities. verify_release.py checks splits, reconstructed scores, all saved forest predictions, fresh seed-101 refits, and CPU timings.')
    p('audit_math.py, audit_kernel_theory.py, and audit_oracle_risk.py contain finite mathematical checks; test_output_space_invariance.py and stress_test_output_space_invariance.py verify direct output-space representation invariance; plot_topology_sharpness.py generates the explicit extremal masks. docs/RESULT_MAP.md links the manuscript evidence to exact filenames. models/ contains every fitted estimator. data/source_manifest.json preserves source archive URLs and hashes.')
    head('Expected agreement')
    p('Discrete actions and masks should agree under the pinned stack. Saved float32 arrays introduce rounding: replay probability tolerances are 10⁻⁷ and sampled score recomputation tolerances are 2×10⁻⁶. Timings, PDF metadata, and compressed-container bytes can differ. Compare scientific values rather than regeneration timestamps.')
    page('Kernel validation and limits')
    head('Exploratory output-space comparison')
    p('The output law uses the minimum predicted action risk within each exact-output class and a uniform reference over distinct outputs. Copying an existing frozen output/score pair preserves that law. Refitting after changing the feature representation is outside the invariance claim. Temperature is selected from .005, .02, and .1 using the original validation set; all 15 models select .005.')
    rows=[['Dataset','Mean distinct outputs','Action Gibbs loss','Output-space loss']]
    for r in quotient:rows.append([r['dataset'],f"{float(r['distinct_outputs']):.3f}",f"{float(r['action_loss']):.6f}",f"{float(r['quotient_loss']):.6f}"])
    table(rows,[105,143,128,128])
    p('The output-space loss is slightly higher on all three datasets. Its demonstrated benefit here is invariance to redundant action representation. There are 14,880 direct replication checks on retained cases; the largest residual is below 7×10⁻¹⁶. This extension is not an independent confirmatory benchmark.')
    p(f"Strengthening branch: direct output-cost learning makes the retrained pipeline invariant under output-preserving representation refinement. Step 3 records exact-zero discrepancies across 288 nonbase retraining and 6,912 output-law comparisons. Step 4 separates excess risk into bank approximation, estimation, and entropy/reference terms; {oracle['random_batches']:,} random finite global checks ({oracle['conditional_inputs_checked']:,} conditional inputs) show no positive bound violation and an exact-decomposition residual of {oracle['max_exact_decomposition_residual']:.2e}. These are mathematical/implementation checks, not new benchmark evidence.",'SmallS')
    head('Independent Oxford and DAVIS validations')
    p('Step 5 uses all 3,669 official Oxford-IIIT Pet test masks at 128×128 under six frozen corruption conditions (22,014 held-out cases). The action-coordinate selector has mean composite loss 0.0162 versus 0.0164 for the validation-best fixed morphology; the direct-output selector is adverse at 0.0201 under both predeclared paired comparisons.')
    p('Step 6 uses official DAVIS 2016 ground truth and pre-computed outputs from NLC, FST, SAL, TRC, MSG, and CVOS with no synthetic corruption. The held-out set contains 20 video sequences, 1,376 unique ground-truth frames, and 8,236 method-frame cases. Action-coordinate and direct-output selectors reduce sequence-mean composite loss from 0.3609 to 0.3071 and 0.3104, respectively; both improvements versus raw segmentations have family-wise 95% sequence-bootstrap intervals excluding zero. Region J and boundary F do not improve, while resized-target exact topology agreement rises from 0.1363 to 0.2339 for action-coordinate selection. Native/resized Betti pairs agree for only 26.24% of DAVIS held-out frames.')
    if (ROOT/'results/step7_structured/step7_summary.csv').exists():
        st=list(csv.DictReader((ROOT/'results/step7_structured/step7_summary.csv').open()))
        vals={}
        for r in st:
            if r['metric'] in ('loss','topology_exact','boundary_f'):
                vals.setdefault(r['method'],{})[r['metric']]=float(r['mean'])
        head('Structured-error distribution shift')
        rows=[['Method','Loss','Topology exact','Boundary F']]
        for m in ['Input','Validation best','Action-coordinate selector','Direct-output selector']:
            if m in vals:
                rows.append([m,f"{vals[m]['loss']:.4f}",f"{vals[m]['topology_exact']:.4f}",f"{vals[m]['boundary_f']:.4f}"])
        table(rows,[180,105,110,109])
        p('Step 7 evaluates all 3,669 held-out Oxford masks under nine predeclared structured errors after fitting only on the Step-5 IID deletion/addition distribution. No structured case enters fitting or tuning. Per-condition outcomes, family-wise clean-mask bootstrap contrasts, and adverse cases are frozen under results/step7_structured/.')
    head('Numerical support for the proofs')
    top=checks['topology']
    p(f"Independent graph calculations compare {top['independent_graph_vs_scipy_masks']:,} masks and {top['single_pixel_comparisons']:,} single-pixel changes, check {top['exact_local_identity_cases']:,} exact local component identities, and include separate constructions attaining both sharp sensitivity bounds. The suite also runs 2,000 entropy/risk/replication trials, 32 finite transport comparisons, Bayesian-reversal checks, and gradient comparisons.")
    p('The largest finite-difference gradient residual in the expanded suite is 8.2×10⁻¹¹. These finite and numerical checks complement the quantified written proofs; they are not a proof-assistant certificate or external mathematical review.')
    head('Limits to retain in any presentation')
    p('Individual metrics do not all improve: MNIST Dice and shapes hole error can worsen. Stronger Fashion-MNIST corruption gives a documented failure against the fixed area filter. Clean inputs can change. Evaluating a full 31-action bank is materially slower than using a single area filter. Oxford closes the low-resolution-only gap, DAVIS closes the synthetic-error-only gap, and Step 7 probes structured mask-error shift; none establishes RGB restoration, clinical performance, correlated real-sensor-noise robustness, or native-resolution topology preservation. DAVIS source segmentations are from the 2016 benchmark rather than contemporary architectures.')
    p('The smooth morphology control is not a full implementation of BiMoNN, DMNN, or SoftMorph. No superiority over those published architectures is claimed. The mathematical contraction statements require their stated coefficient conditions; the fitted model has not been shown to satisfy a global contraction coefficient below one.')
    head('Attribution and reuse')
    p('Dataset sources, original hashes, and the Fashion-MNIST license are supplied. LICENSE_NOTICE.md records the project rights status without asserting an unselected open-source license. CITATION.cff contains manuscript/software attribution without an invented DOI. The manuscript discloses generative AI assistance; the authors retain responsibility for its content.','SmallS')
    doc=SimpleDocTemplate(str(ROOT/'Supplementary_Guide.pdf'),pagesize=(612,792),leftMargin=54,rightMargin=54,topMargin=40,bottomMargin=40,title='Supplementary Material S1: Stochastic Morphological pi-Mixture Kernels',author='Adnan H. Abdulwahid; Ram C. Neupane')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print('Created Supplementary_Guide.pdf')
if __name__=='__main__':main()
