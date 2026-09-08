#!/usr/bin/env python3
"""Miniature end-to-end Step-6 integration test on synthetic directory data.
This validates software only; it is not DAVIS evidence.
"""
import json, tempfile
from pathlib import Path
import numpy as np
from PIL import Image

from run_davis_step6 import (
    PRIMARY_SOURCE_METHODS, split_train_sequences, enumerate_cases,
    build_cache, train_and_evaluate, create_result_tex
)

rng=np.random.default_rng(20260907)
with tempfile.TemporaryDirectory() as td:
    root=Path(td); ann=root/'Annotations'/'480p'; res=root/'Results'/'Segmentations'/'480p'; out=root/'out'
    train=[f'train{i:02d}' for i in range(8)]; test=[f'val{i:02d}' for i in range(4)]
    allseq=train+test
    for seqi,seq in enumerate(allseq):
        (ann/seq).mkdir(parents=True)
        for m in PRIMARY_SOURCE_METHODS: (res/m/seq).mkdir(parents=True,exist_ok=True)
        for fi in range(4):
            h,w=54,82; yy,xx=np.ogrid[:h,:w]
            cy=27+((seqi+fi)%5)-2; cx=41+((2*seqi+fi)%7)-3
            y=((yy-cy)/15)**2+((xx-cx)/24)**2<1
            Image.fromarray((y*255).astype('uint8')).save(ann/seq/f'{fi:05d}.png')
            for mi,m in enumerate(PRIMARY_SOURCE_METHODS):
                x=np.roll(y,shift=(mi%3)-1,axis=1).copy()
                if (mi+fi)%2==0: x[10+mi:12+mi,20:25]=1
                if (mi+seqi)%3==0: x[25:28,38:43]=0
                Image.fromarray((x*255).astype('uint8')).save(res/m/seq/f'{fi:05d}.png')
    tv=split_train_sequences(train,.25); splitseq={'train':tv['train'],'tune':tv['tune'],'test':test}
    methoddirs={m:res/m for m in PRIMARY_SOURCE_METHODS}
    cases={s:enumerate_cases(ann,methoddirs,splitseq[s],min_coverage=1.0) for s in splitseq}
    seqnames=sorted(allseq); seqidx={s:i for i,s in enumerate(seqnames)}; methidx={m:i for i,m in enumerate(PRIMARY_SOURCE_METHODS)}
    out.mkdir()
    for s in ('train','tune','test'): build_cache(s,cases[s],out,48,seqidx,methidx)
    sel=train_and_evaluate(out,48,seqnames,PRIMARY_SOURCE_METHODS,n_estimators=4); create_result_tex(out)
    required=['davis2016_summary.csv','davis2016_source_method_summary.csv','davis2016_sequence_clustered_ci.csv','davis2016_paired_comparisons.json','generated_davis2016.tex']
    assert all((out/f).exists() for f in required)
    print(json.dumps({'status':'PASS','case_counts':{k:len(v) for k,v in cases.items()},'selection':{'action_leaf':sel['action_leaf'],'direct_leaf':sel['direct_leaf']},'required_outputs':required},indent=2))
