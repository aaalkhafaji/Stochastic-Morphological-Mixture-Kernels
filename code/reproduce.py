"""Run a documented reproduction level in a new work copy of the release."""
from pathlib import Path
import argparse, os, shutil, subprocess, sys, json
ROOT=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--level',choices=['quick','analysis','full'],default='quick')
    ap.add_argument('--workdir',type=Path,required=True)
    args=ap.parse_args();dest=args.workdir.expanduser().resolve()
    if dest.exists():raise SystemExit('Choose a new work directory; destination already exists: '+str(dest))
    if dest==ROOT or ROOT in dest.parents or dest in ROOT.parents:
        raise SystemExit('Choose a work directory outside the supplied release tree.')
    subprocess.run([sys.executable,str(ROOT/'code/check_package.py')],check=True)
    shutil.copytree(ROOT,dest,ignore=shutil.ignore_patterns('.git','.venv','venv','__pycache__','.ipynb_checkpoints','downloads','reproduction_runs'))
    env=os.environ.copy()
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:env[name]='1'
    commands=[]
    if args.level=='analysis':
        required=[dest/'results'/f'{ds}_test_bank.npz' for ds in ['shapes','mnist','fashion']]
        if not all(p.exists() for p in required):
            raise SystemExit('Analysis level needs the frozen Zenodo assets. Extract the Zenodo archive into this tree, or use --level full to regenerate them from the supplied clean inputs.')
    if args.level=='full':
        for p in (dest/'results').glob('*_bank.npz'):p.unlink()
        commands.append(['run_campaign.py','--stage','all'])
    commands += [['audit_math.py'], ['test_output_space_invariance.py'], ['audit_oracle_risk.py']]
    if args.level in ['analysis','full']:
        commands += [['stress_test_output_space_invariance.py'], ['smoke_check.py']]
        commands += [[name] for name in ['verify_release.py','summarize.py','audit_kernel_theory.py','validate_kernel_quotient.py','plot_topology_sharpness.py','build_supplement_guide.py']]
    for command in commands:
        print('Running: '+' '.join(command),flush=True)
        subprocess.run([sys.executable,str(dest/'code'/command[0]),*command[1:]],cwd=dest,env=env,check=True)
    # Refresh only the new work copy so subsequent operations have valid checksums.
    report={'level':args.level,'completed_commands':commands,'source_release_unchanged':True,
            'note':'A reproduction run; not additional independent test data.'}
    (dest/'results/reproduction_run.json').write_text(json.dumps(report,indent=2)+'\n')
    subprocess.run([sys.executable,str(dest/'code/check_package.py'),'--write-manifest'],cwd=dest,check=True)
    print('Completed reproduction in '+str(dest),flush=True)
if __name__=='__main__':main()
