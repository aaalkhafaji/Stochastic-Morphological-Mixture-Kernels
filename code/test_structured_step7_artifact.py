#!/usr/bin/env python3
"""Fast real-data compatibility preflight for Step 7.

Uses the frozen Step-5 GitHub Actions artifact, but deliberately avoids fitting the
five forests twice. It verifies artifact identity/structure, frozen selection
metadata, split sizes, and all nine structured-corruption -> 31-action -> quotient
paths on genuine held-out Oxford masks.
"""
from __future__ import annotations
import argparse, tempfile
from pathlib import Path
import numpy as np

from run_structured_step7 import (
    STEP5_EXPECTED_SHA256, PRIMARY_CONDITIONS, load_step5_artifact,
    structured_corrupt, rng_for,
)
from morphology import bank, features, NAMES
from output_space_learning import unique_output_rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("artifact", type=Path)
    a=ap.parse_args()
    with tempfile.TemporaryDirectory(prefix="step7_preflight_") as td:
        caches, selection, protocol = load_step5_artifact(a.artifact, Path(td))
        expected={"train":1840,"val":1840,"test":3669}
        for split,n in expected.items():
            got=len(caches[split]["clean_masks"])
            assert got==n, (split,got,n)
        assert selection["action_leaf"]==20
        assert selection["direct_leaf"]==8
        assert selection["best_validation_action"]=="area_connected16"
        assert len(protocol)==sum(expected.values()), len(protocol)
        clean=caches["test"]["clean_masks"].astype(bool)
        # Exercise all corruption families on different genuine held-out masks.
        distinct=[]
        for ci,name in enumerate(PRIMARY_CONDITIONS):
            truth=clean[ci]
            x=structured_corrupt(truth,name,rng_for(ci,ci))
            assert x.shape==(128,128) and x.dtype==bool
            assert np.any(x!=truth), name
            outs=bank(x)
            assert outs.shape==(len(NAMES),128,128)
            af=features(x,outs)
            assert af.shape[0]>0 and np.isfinite(af).all()
            uq=unique_output_rows(x,outs)
            assert 1 <= len(uq.representatives) <= len(NAMES)
            assert np.isfinite(uq.pair_features).all()
            distinct.append(int(len(uq.representatives)))
    print({"status":"PASS","artifact_sha256":STEP5_EXPECTED_SHA256,
           "split_sizes":expected,"conditions":PRIMARY_CONDITIONS,
           "distinct_output_counts":distinct})

if __name__=="__main__":
    main()
