"""Verify frozen release bytes, or explicitly refresh the manifest after edits."""
from pathlib import Path
import argparse, hashlib, json
ROOT=Path(__file__).resolve().parents[1]
SKIP={'.git','.venv','venv','__pycache__','.ipynb_checkpoints','downloads','reproduction_runs'}
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
def release_files():
    return [p for p in sorted(ROOT.rglob('*')) if p.is_file()
            and not any(x in SKIP for x in p.relative_to(ROOT).parts)
            and p.name!='SHA256SUMS' and not p.name.endswith(('.pyc','.log'))]
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--write-manifest',action='store_true',help='Explicitly accept current bytes as a new manifest')
    args=ap.parse_args();manifest=ROOT/'SHA256SUMS'
    if args.write_manifest:
        files=release_files()
        manifest.write_text(''.join(f'{sha(p)}  {p.relative_to(ROOT).as_posix()}\n' for p in files),encoding='utf-8')
        print(json.dumps({'manifest_written':len(files)},indent=2));return
    if not manifest.exists():raise SystemExit('SHA256SUMS is missing.')
    failures=[];names=set()
    for line in manifest.read_text().splitlines():
        digest,name=line.split('  ',1);p=(ROOT/name).resolve()
        if p==ROOT or ROOT not in p.parents:raise SystemExit('Invalid manifest path: '+name)
        if name in names:raise SystemExit('Duplicate manifest path: '+name)
        names.add(name)
        if not p.is_file() or sha(p)!=digest:failures.append(name)
    actual={p.relative_to(ROOT).as_posix() for p in release_files()}
    unlisted=sorted(actual-names)
    print(json.dumps({'files_checked':len(names),'mismatches_or_missing':failures,'unlisted_files':unlisted},indent=2))
    if failures or unlisted:raise SystemExit(1)
if __name__=='__main__':main()
