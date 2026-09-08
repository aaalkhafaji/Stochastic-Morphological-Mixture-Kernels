#!/usr/bin/env python3
import sys, zipfile, hashlib
from pathlib import Path
EXPECTED='889e7df9af279974d46595444bf768c627602d188ca8c93d0f65b30739f16e96'
p=Path(sys.argv[1]); h=hashlib.sha256(p.read_bytes()).hexdigest(); assert h==EXPECTED,(h,EXPECTED)
with zipfile.ZipFile(p) as z:
    names=set(z.namelist())
    for s in ('train','val','test'): assert f'step5_oxford_pet/oxford_pet_{s}_128.npz' in names
    assert 'step5_oxford_pet/oxford_pet_selection.json' in names
print({'status':'PASS','sha256':h})
