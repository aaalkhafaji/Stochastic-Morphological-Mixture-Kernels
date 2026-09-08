#!/usr/bin/env python3
"""Step 6: genuine segmentation-error validation on DAVIS 2016.

Primary scientific question
---------------------------
Can the stochastic morphological post-processor improve *actual algorithmic
segmentation errors* on real video masks, rather than only synthetically
corrupted ground-truth masks?

Primary protocol
----------------
* Official DAVIS 2016 single-object annotations and official pre-computed
  segmentations from the original benchmark.
* Six unsupervised source methods from Perazzi et al. (CVPR 2016):
  NLC, FST, SAL, TRC, MSG, and CVOS.
* Official DAVIS train sequences are split deterministically, by sequence, into
  80% fit and 20% tuning subsets. Official DAVIS validation sequences are never
  used for fitting/tuning and form the held-out test benchmark.
* Ground truth and algorithm masks are nearest-neighbor letterboxed to 128x128;
  a native-vs-resized audit quantifies whether topology/error structure changes.
* No synthetic corruption is used anywhere in Step 6.
* Five estimator seeds are retained; statistical inference clusters by video
  sequence, not frame, so temporally dependent frames are never treated as
  independent replicates.
* Primary comparisons use family-wise 95% bootstrap intervals across four
  predeclared loss contrasts.

The RGB video frames are not used by the estimator. They may be present in the
DAVIS archive, but this script selectively extracts only annotations/ImageSets
plus benchmark segmentation outputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import urllib.request
import zipfile

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from scipy.special import softmax
from sklearn.ensemble import ExtraTreesRegressor

from morphology import NAMES, METRICS, bank, betti, features, measure
from output_space_learning import unique_output_rows

ROOT = Path(__file__).resolve().parents[1]
DAVIS_DATA_URL = "https://graphics.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip"
DAVIS_RESULTS_URL = "https://graphics.ethz.ch/Downloads/Data/Davis/DAVIS-results.zip"
DAVIS_DATA_MD5 = "0cb3cf9c5617209fa3cc4794e52a2ffa"  # official fperazzi/davis script
PRIMARY_SOURCE_METHODS = ["NLC", "FST", "SAL", "TRC", "MSG", "CVOS"]
PROTOCOL_SEED = 20260907
MODEL_SEEDS = [101, 102, 103, 104, 105]
PRIMARY_RESOLUTION = 128
STEP6_METRICS = METRICS + ["boundary_f08"]


def file_digest(path: Path, algo="sha256", chunk=1 << 20) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def stable_key(text: str, seed: int = PROTOCOL_SEED) -> str:
    return hashlib.sha256(f"{seed}:{text}".encode()).hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    print(f"Downloading {url}", flush=True)
    urllib.request.urlretrieve(url, dest)


def selective_extract_zip(archive: Path, dest: Path, keep_predicate) -> int:
    """Safely extract only members accepted by keep_predicate."""
    dest = dest.resolve()
    count = 0
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            name = info.filename.replace("\\", "/")
            if not keep_predicate(name):
                continue
            target = (dest / name).resolve()
            if dest not in target.parents and target != dest:
                raise RuntimeError(f"unsafe zip member: {name}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, target.open("wb") as out:
                    while True:
                        b = src.read(1 << 20)
                        if not b:
                            break
                        out.write(b)
            count += 1
    return count


def ensure_davis(data_root: Path, data_archive: Path | None, results_archive: Path | None,
                 do_download: bool) -> tuple[Path, Path, Path, Path]:
    """Return (DAVIS root, annotations480, imagesets480, results480)."""
    data_root.mkdir(parents=True, exist_ok=True)
    if data_archive is None:
        data_archive = data_root / "DAVIS-data.zip"
    if results_archive is None:
        results_archive = data_root / "DAVIS-results.zip"
    if do_download:
        download(DAVIS_DATA_URL, data_archive)
        download(DAVIS_RESULTS_URL, results_archive)
    if not data_archive.exists() or not results_archive.exists():
        raise FileNotFoundError("DAVIS archives missing; pass --data-archive/--results-archive or --download")
    md5 = file_digest(data_archive, "md5")
    if md5 != DAVIS_DATA_MD5:
        raise RuntimeError(f"DAVIS data MD5 mismatch: {md5} != {DAVIS_DATA_MD5}")

    # Selective extraction avoids unpacking RGB video frames, which Step 6 never uses.
    def keep_data(name: str) -> bool:
        s = "/" + name.strip("/")
        return ("/Annotations/480p/" in s or "/ImageSets/480p/" in s or
                s.endswith("/Annotations/db_info.yml") or s.endswith("/README.md"))
    def keep_results(name: str) -> bool:
        s = "/" + name.strip("/")
        return "/Results/Segmentations/480p/" in s

    marker = data_root / ".step6_extracted.json"
    if not marker.exists():
        n1 = selective_extract_zip(data_archive, data_root, keep_data)
        n2 = selective_extract_zip(results_archive, data_root, keep_results)
        marker.write_text(json.dumps({"dataset_members": n1, "result_members": n2}, indent=2) + "\n")

    candidates = [p for p in data_root.rglob("Annotations") if (p / "480p").is_dir()]
    if not candidates:
        raise RuntimeError("could not locate DAVIS/Annotations/480p after extraction")
    ann_parent = min(candidates, key=lambda p: len(p.parts))
    davis_root = ann_parent.parent
    ann480 = davis_root / "Annotations" / "480p"
    imagesets480 = davis_root / "ImageSets" / "480p"
    result480 = davis_root / "Results" / "Segmentations" / "480p"
    if not imagesets480.is_dir() or not result480.is_dir():
        # Results archive can in rare mirrors extract under another DAVIS root.
        r = [p for p in data_root.rglob("Segmentations") if (p / "480p").is_dir()]
        if r:
            result480 = min(r, key=lambda p: len(p.parts)) / "480p"
    if not imagesets480.is_dir() or not result480.is_dir():
        raise RuntimeError(f"DAVIS structure incomplete: ImageSets={imagesets480}, Results={result480}")
    return davis_root, ann480, imagesets480, result480


def read_sequence_list(path: Path) -> list[str]:
    """Read DAVIS split files and return unique video-sequence names.

    DAVIS 2016's original ``ImageSets/480p/{train,val}.txt`` files are
    frame lists, for example::

        /JPEGImages/480p/bear/00000.jpg /Annotations/480p/bear/00000.png

    while newer DAVIS layouts may store one sequence name per line.  This
    parser accepts both formats and deduplicates frame-list rows by sequence.
    """
    rows = []
    seen = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        token = line.split()[0].strip()
        clean = token.strip("/")
        if "/" not in clean:
            seq = clean
        else:
            parts = list(Path(clean).parts)
            seq = None
            for i, part in enumerate(parts[:-1]):
                if part.lower() == "480p" and i + 1 < len(parts) - 1:
                    seq = parts[i + 1]
                    break
            if seq is None:
                seq = Path(clean).parent.name
        if seq and seq not in seen:
            rows.append(seq)
            seen.add(seq)
    return rows


def locate_split_file(imagesets480: Path, split: str) -> Path:
    cands = [imagesets480 / f"{split}.txt", imagesets480 / f"{split.title()}.txt"]
    for p in cands:
        if p.exists():
            return p
    hits = [p for p in imagesets480.rglob("*.txt") if p.stem.lower() == split.lower()]
    if not hits:
        raise RuntimeError(f"could not locate DAVIS {split}.txt under {imagesets480}")
    return hits[0]


def split_train_sequences(train_sequences: list[str], tune_fraction=.20) -> dict[str, list[str]]:
    ordered = sorted(train_sequences, key=lambda s: stable_key("train:" + s))
    ntune = max(1, min(len(ordered)-1, int(round(tune_fraction * len(ordered)))))
    tune = sorted(ordered[:ntune])
    fit = sorted(ordered[ntune:])
    return {"train": fit, "tune": tune}


def norm_method(name: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", name.upper())


def discover_source_methods(result480: Path, requested: list[str]) -> dict[str, Path]:
    dirs = [p for p in result480.iterdir() if p.is_dir()]
    by_norm = {norm_method(p.name): p for p in dirs}
    out = {}
    for method in requested:
        key = norm_method(method)
        if key in by_norm:
            out[method] = by_norm[key]
            continue
        fuzzy = [p for p in dirs if key in norm_method(p.name) or norm_method(p.name) in key]
        if len(fuzzy) == 1:
            out[method] = fuzzy[0]
        else:
            raise RuntimeError(f"missing/ambiguous official method {method}; discovered={[p.name for p in dirs]}")
    return out


def png_map(folder: Path) -> dict[str, Path]:
    return {p.stem: p for p in folder.glob("*.png")}


def load_binary(path: Path) -> np.ndarray:
    a = np.asarray(Image.open(path))
    if a.ndim == 3:
        a = a[..., 0]
    return a != 0


def letterbox_bool(mask: np.ndarray, resolution: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    scale = min(resolution / h, resolution / w)
    nh = max(1, min(resolution, int(round(h * scale))))
    nw = max(1, min(resolution, int(round(w * scale))))
    im = Image.fromarray(mask.astype(np.uint8) * 255)
    resized = np.asarray(im.resize((nw, nh), resample=Image.Resampling.NEAREST)) > 0
    out = np.zeros((resolution, resolution), dtype=bool)
    y0, x0 = (resolution - nh)//2, (resolution - nw)//2
    out[y0:y0+nh, x0:x0+nw] = resized
    return out


def boundary_pixels(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, bool)
    er = ndi.binary_erosion(mask, structure=np.ones((3,3), bool), border_value=0)
    return mask & ~er


def disk(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    return (x*x + y*y) <= radius*radius


def boundary_f08(pred: np.ndarray, truth: np.ndarray) -> float:
    """Boundary F with a tolerance of 0.8% of image diagonal (DAVIS-style)."""
    pb, tb = boundary_pixels(pred), boundary_pixels(truth)
    npb, ntb = int(pb.sum()), int(tb.sum())
    if npb == 0 and ntb == 0:
        return 1.0
    if npb == 0 or ntb == 0:
        return 0.0
    r = max(1, int(math.ceil(0.008 * math.hypot(*pred.shape))))
    st = disk(r)
    p_match = pb & ndi.binary_dilation(tb, structure=st)
    t_match = tb & ndi.binary_dilation(pb, structure=st)
    precision = p_match.sum() / npb
    recall = t_match.sum() / ntb
    return float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def measure_step6(y: np.ndarray, truth: np.ndarray, true_betti=None) -> np.ndarray:
    base = measure(y, truth, true_betti)
    return np.r_[base, boundary_f08(y, truth)]


def extract_case(x: np.ndarray, truth: np.ndarray) -> dict:
    outputs = bank(x)
    tb = betti(truth)
    scores = np.stack([measure_step6(y, truth, tb) for y in outputs]).astype(np.float32)
    afeat = features(x, outputs).astype(np.float32)
    rows = unique_output_rows(x, outputs, scores[:, 0])
    m = len(NAMES)
    dfeat = np.zeros((m, rows.pair_features.shape[1]), dtype=np.float32)
    reps = np.full(m, -1, dtype=np.int16)
    dfeat[:len(rows.representatives)] = rows.pair_features
    reps[:len(rows.representatives)] = rows.representatives.astype(np.int16)
    return {"scores": scores, "action_features": afeat, "direct_features": dfeat,
            "direct_reps": reps, "n_distinct": np.int16(len(rows.representatives))}


def enumerate_cases(ann480: Path, method_dirs: dict[str, Path], sequences: list[str],
                    min_coverage=.95) -> list[dict]:
    cases = []
    for method, mroot in method_dirs.items():
        possible = matched = 0
        for seq in sequences:
            gt = png_map(ann480 / seq)
            pred = png_map(mroot / seq) if (mroot / seq).is_dir() else {}
            possible += len(gt)
            common = sorted(set(gt) & set(pred), key=lambda x: int(x) if x.isdigit() else x)
            matched += len(common)
            for stem in common:
                cases.append({"sequence": seq, "frame": stem, "base_method": method,
                              "gt": gt[stem], "pred": pred[stem]})
        coverage = matched / possible if possible else 0.0
        print(f"{method}: coverage {matched}/{possible} = {coverage:.4f}", flush=True)
        if coverage < min_coverage:
            raise RuntimeError(f"official segmentation coverage for {method} is only {coverage:.3f}")
    return cases


def save_protocol(outdir: Path, split_sequences: dict[str, list[str]], cases_by_split: dict[str, list[dict]],
                  data_archive: Path, results_archive: Path, method_dirs: dict[str,Path], resolution: int) -> None:
    rows = []
    for split, cases in cases_by_split.items():
        for c in cases:
            rows.append({"analysis_split": split, "sequence": c["sequence"], "frame": c["frame"],
                         "base_method": c["base_method"]})
    pd.DataFrame(rows).to_csv(outdir / "davis2016_protocol_cases.csv.gz", index=False)
    with (outdir / "davis2016_protocol_sequences.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["analysis_split", "sequence", "stable_key"])
        for split, seqs in split_sequences.items():
            for seq in seqs:
                w.writerow([split, seq, stable_key(split + ":" + seq)])
    meta = {
        "dataset": "DAVIS 2016 single-object video object segmentation",
        "dataset_url": "https://davischallenge.org/davis2016/code.html",
        "data_archive_url": DAVIS_DATA_URL,
        "results_archive_url": DAVIS_RESULTS_URL,
        "data_archive_md5": file_digest(data_archive, "md5"),
        "data_archive_sha256": file_digest(data_archive),
        "results_archive_sha256": file_digest(results_archive),
        "source_methods": {k: v.name for k,v in method_dirs.items()},
        "source_method_category": "unsupervised methods in the original DAVIS 2016 evaluation",
        "resolution": resolution,
        "resize": "aspect-ratio preserving nearest-neighbor letterbox",
        "protocol_seed": PROTOCOL_SEED,
        "model_seeds": MODEL_SEEDS,
        "sequence_counts": {k: len(v) for k,v in split_sequences.items()},
        "case_counts": {k: len(v) for k,v in cases_by_split.items()},
        "no_synthetic_corruption": True,
        "rgb_images_used_by_estimator": False,
        "statistical_cluster": "DAVIS video sequence",
        "primary_metric_family": ["loss", "iou (DAVIS region J)", "boundary_f08", "topology_exact"],
    }
    (outdir / "davis2016_provenance.json").write_text(json.dumps(meta, indent=2) + "\n")


def build_resize_audit(cases: list[dict], outdir: Path, resolution: int) -> dict:
    rows, seen_gt = [], set()
    for i,c in enumerate(cases):
        gt_native, pr_native = load_binary(c["gt"]), load_binary(c["pred"])
        if gt_native.shape != pr_native.shape:
            raise RuntimeError(f"shape mismatch {c['base_method']} {c['sequence']} {c['frame']}: {pr_native.shape} vs {gt_native.shape}")
        gt_small, pr_small = letterbox_bool(gt_native,resolution), letterbox_bool(pr_native,resolution)
        bgn, bpn = betti(gt_native), betti(pr_native)
        bgs, bps = betti(gt_small), betti(pr_small)
        native_err = np.abs(bpn-bgn); small_err = np.abs(bps-bgs)
        native_iou = measure(pr_native, gt_native, bgn)[3]
        small_iou = measure(pr_small, gt_small, bgs)[3]
        key=(c['sequence'],c['frame'])
        gt_pres = None
        if key not in seen_gt:
            gt_pres=int(np.array_equal(bgn,bgs)); seen_gt.add(key)
        rows.append({"sequence":c["sequence"],"frame":c["frame"],"base_method":c["base_method"],
                     "native_iou":native_iou,"resized_iou":small_iou,
                     "iou_abs_change":abs(native_iou-small_iou),
                     "beta_error_preserved":int(np.array_equal(native_err,small_err)),
                     "gt_topology_preserved_unique_flag":gt_pres if gt_pres is not None else -1})
        if i and i%1000==0: print(f"resize audit {i}/{len(cases)}",flush=True)
    df=pd.DataFrame(rows); df.to_csv(outdir/"davis2016_resize_audit.csv.gz",index=False,float_format="%.9g")
    uniq=df[df.gt_topology_preserved_unique_flag>=0]
    summary={
        "n_method_frame_cases":int(len(df)),"n_unique_groundtruth_frames":int(len(uniq)),
        "groundtruth_topology_preserved_fraction":float(uniq.gt_topology_preserved_unique_flag.mean()),
        "input_beta_error_preserved_fraction":float(df.beta_error_preserved.mean()),
        "mean_absolute_iou_change":float(df.iou_abs_change.mean()),
        "max_absolute_iou_change":float(df.iou_abs_change.max()),
        "native_resized_iou_correlation":float(df[["native_iou","resized_iou"]].corr().iloc[0,1]),
    }
    (outdir/"davis2016_resize_audit_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


def build_cache(split: str, cases: list[dict], outdir: Path, resolution: int,
                sequence_index: dict[str,int], method_index: dict[str,int]) -> None:
    path=outdir/f"davis2016_{split}_{resolution}.npz"
    if path.exists(): print(f"Using cache {path.name}",flush=True); return
    n=len(cases); m=len(NAMES)
    scores=np.empty((n,m,len(STEP6_METRICS)),np.float32)
    action_features=np.empty((n,5+5*m),np.float32)
    direct_features=np.empty((n,m,10),np.float32)
    direct_reps=np.empty((n,m),np.int16); n_distinct=np.empty(n,np.int16)
    seq_idx=np.empty(n,np.int16); meth_idx=np.empty(n,np.int8); frame_idx=np.empty(n,np.int32)
    t0=time.perf_counter()
    for i,c in enumerate(cases):
        truth=letterbox_bool(load_binary(c["gt"]),resolution)
        x=letterbox_bool(load_binary(c["pred"]),resolution)
        e=extract_case(x,truth)
        scores[i]=e["scores"]; action_features[i]=e["action_features"]
        direct_features[i]=e["direct_features"]; direct_reps[i]=e["direct_reps"]
        n_distinct[i]=e["n_distinct"]; seq_idx[i]=sequence_index[c["sequence"]]
        meth_idx[i]=method_index[c["base_method"]]
        frame_idx[i]=int(c["frame"]) if c["frame"].isdigit() else i
        if i and i%500==0: print(f"{split}: {i}/{n}",flush=True)
    np.savez_compressed(path,scores=scores,action_features=action_features,direct_features=direct_features,
                        direct_reps=direct_reps,n_distinct=n_distinct,sequence_index=seq_idx,
                        base_method_index=meth_idx,frame_index=frame_idx,resolution=np.array([resolution],np.int16))
    print(f"Extracted {split}: {n} genuine-error cases in {time.perf_counter()-t0:.1f}s",flush=True)


def fit_forest(X,y,seed,leaf,n_estimators=96):
    model=ExtraTreesRegressor(n_estimators=n_estimators,max_features=1.0,min_samples_leaf=leaf,
                              random_state=seed,n_jobs=1)
    model.fit(X,y); return model


def direct_table(cache):
    X=[];y=[]
    for i in range(len(cache["scores"])):
        k=int(cache["n_distinct"][i]); reps=cache["direct_reps"][i,:k]
        X.append(cache["direct_features"][i,:k]); y.append(cache["scores"][i,reps,0])
    return np.vstack(X),np.concatenate(y)


def direct_predict(model,cache):
    return [model.predict(cache["direct_features"][i,:int(cache["n_distinct"][i])]) for i in range(len(cache["scores"]))]


def direct_selected_metrics(cache,pred):
    out=np.empty((len(pred),len(STEP6_METRICS)),float)
    for i,pc in enumerate(pred):
        k=len(pc); reps=cache["direct_reps"][i,:k]; out[i]=cache["scores"][i,reps[int(np.argmin(pc))]]
    return out


def direct_expected_metrics(cache,pred,tau):
    out=np.empty((len(pred),len(STEP6_METRICS)),float)
    for i,pc in enumerate(pred):
        k=len(pc); reps=cache["direct_reps"][i,:k]
        p=softmax(-(pc-pc.min())/tau); out[i]=p@cache["scores"][i,reps]
    return out


def action_selected_metrics(cache,pred):
    idx=pred.argmin(axis=1); return cache["scores"][np.arange(len(idx)),idx]


def aggregate_rows(cache, method, seed, metrics, seq_names, method_names):
    f=pd.DataFrame(np.asarray(metrics),columns=STEP6_METRICS)
    f.insert(0,"frame",cache["frame_index"].astype(int))
    f.insert(0,"base_method",[method_names[int(i)] for i in cache["base_method_index"]])
    f.insert(0,"sequence",[seq_names[int(i)] for i in cache["sequence_index"]])
    f.insert(0,"seed",int(seed)); f.insert(0,"method",method); f.insert(0,"dataset","DAVIS2016")
    return f


def sequence_cluster_table(df,metric,base_method=None):
    d=df if base_method is None else df[df.base_method==base_method]
    return d.groupby(["method","sequence"],as_index=False)[metric].mean()


def bootstrap_mean(df,method,metric,n_boot=5000,seed=PROTOCOL_SEED):
    g=sequence_cluster_table(df[df.method==method],metric)
    vals=g[metric].to_numpy(float); rng=np.random.default_rng(seed+(int(hashlib.sha256((method+metric).encode()).hexdigest()[:8],16)&0xffff))
    boots=np.array([vals[rng.integers(0,len(vals),len(vals))].mean() for _ in range(n_boot)])
    lo,hi=np.quantile(boots,[.025,.975])
    return {"method":method,"metric":metric,"n_sequences":len(vals),"mean":vals.mean(),"ci95_low":lo,"ci95_high":hi,
            "bootstrap_resamples":n_boot,"cluster":"video sequence"}


def paired_bootstrap(df,a,b,metric="loss",n_boot=5000,seed=PROTOCOL_SEED,family_size=4):
    g=df.groupby(["method","sequence"],as_index=False)[metric].mean()
    aa=g[g.method==a].set_index("sequence")[metric]; bb=g[g.method==b].set_index("sequence")[metric]
    ids=sorted(set(aa.index)&set(bb.index)); diff=(aa.loc[ids]-bb.loc[ids]).to_numpy(float)
    rng=np.random.default_rng(seed+(int(hashlib.sha256((a+b+metric).encode()).hexdigest()[:8],16)&0xffff))
    boots=np.array([diff[rng.integers(0,len(diff),len(diff))].mean() for _ in range(n_boot)])
    alpha=.05/family_size
    return {"method_a":a,"method_b":b,"metric":metric,"n_sequences":len(diff),
            "mean_difference_a_minus_b":float(diff.mean()),"ci95":np.quantile(boots,[.025,.975]).tolist(),
            "familywise95_ci_four_comparisons":np.quantile(boots,[alpha/2,1-alpha/2]).tolist(),
            "bootstrap_resamples":n_boot,"cluster":"video sequence"}


def train_and_evaluate(outdir:Path,resolution:int,seq_names:list[str],method_names:list[str],n_estimators=96):
    c={s:dict(np.load(outdir/f"davis2016_{s}_{resolution}.npz")) for s in ("train","tune","test")}
    tr,va,te=c["train"],c["tune"],c["test"]; rows=[]; ntest=len(te["scores"])
    best_action=int(va["scores"][:,:,0].mean(axis=0).argmin())
    rows.append(aggregate_rows(te,"Input segmentation",-1,te["scores"][:,0],seq_names,method_names))
    rows.append(aggregate_rows(te,"Validation best",-1,te["scores"][:,best_action],seq_names,method_names))
    oi=te["scores"][:,:,0].argmin(axis=1)
    rows.append(aggregate_rows(te,"Oracle (diagnostic)",-1,te["scores"][np.arange(ntest),oi],seq_names,method_names))

    action_candidates=[]
    for leaf in (2,8,20):
        m=fit_forest(tr["action_features"],tr["scores"][:,:,0],99,leaf,n_estimators)
        pr=m.predict(va["action_features"]); action_candidates.append((leaf,float(action_selected_metrics(va,pr)[:,0].mean())))
    action_leaf=min(action_candidates,key=lambda x:x[1])[0]
    Xd,yd=direct_table(tr); direct_candidates=[]
    for leaf in (2,8,20):
        m=fit_forest(Xd,yd,99,leaf,n_estimators); pr=direct_predict(m,va)
        direct_candidates.append((leaf,float(direct_selected_metrics(va,pr)[:,0].mean())))
    direct_leaf=min(direct_candidates,key=lambda x:x[1])[0]

    temperatures=[.005,.02,.1]; runinfo=[]
    for seed in MODEL_SEEDS:
        t0=time.perf_counter(); am=fit_forest(tr["action_features"],tr["scores"][:,:,0],seed,action_leaf,n_estimators)
        ap=am.predict(te["action_features"]); rows.append(aggregate_rows(te,"Action-coordinate selector",seed,action_selected_metrics(te,ap),seq_names,method_names))
        dm=fit_forest(Xd,yd,seed,direct_leaf,n_estimators); vp=direct_predict(dm,va); tp=direct_predict(dm,te)
        tau=min(temperatures,key=lambda t:direct_expected_metrics(va,vp,t)[:,0].mean())
        rows.append(aggregate_rows(te,"Direct-output selector",seed,direct_selected_metrics(te,tp),seq_names,method_names))
        rows.append(aggregate_rows(te,"Direct-output sampled",seed,direct_expected_metrics(te,tp,tau),seq_names,method_names))
        runinfo.append({"seed":seed,"temperature":tau,"seconds":time.perf_counter()-t0}); print(f"fit seed {seed}",flush=True)
    df=pd.concat(rows,ignore_index=True); df.to_csv(outdir/"davis2016_test_metrics.csv.gz",index=False,float_format="%.9g")

    headline=["Input segmentation","Validation best","Action-coordinate selector","Direct-output selector","Direct-output sampled","Oracle (diagnostic)"]
    summary=[]
    for metric in ("loss","dice","iou","boundary_f08","beta0_error","beta1_error","topology_exact"):
        g=df.groupby(["method","sequence"],as_index=False)[metric].mean().groupby("method")[metric].agg(["mean","std","count"]).reset_index()
        g.insert(0,"metric",metric); summary.append(g)
    pd.concat(summary,ignore_index=True).to_csv(outdir/"davis2016_summary.csv",index=False,float_format="%.9g")

    # Source-method stratification, sequence-weighted within each original algorithm.
    byseq=df.groupby(["method","base_method","sequence"],as_index=False)[STEP6_METRICS].mean()
    bysource=byseq.groupby(["method","base_method"],as_index=False)[STEP6_METRICS].mean()
    bysource.to_csv(outdir/"davis2016_source_method_summary.csv",index=False,float_format="%.9g")

    ci=[]
    for method in headline:
        for metric in ("loss","iou","boundary_f08","topology_exact"):
            ci.append(bootstrap_mean(df,method,metric))
    pd.DataFrame(ci).to_csv(outdir/"davis2016_sequence_clustered_ci.csv",index=False,float_format="%.9g")

    comparisons=[
        paired_bootstrap(df,"Action-coordinate selector","Input segmentation"),
        paired_bootstrap(df,"Direct-output selector","Input segmentation"),
        paired_bootstrap(df,"Direct-output selector","Action-coordinate selector"),
        paired_bootstrap(df,"Direct-output selector","Validation best"),
    ]
    (outdir/"davis2016_paired_comparisons.json").write_text(json.dumps(comparisons,indent=2)+"\n")

    # Per-sequence loss deltas make adverse sequences visible rather than hiding them in means.
    g=df.groupby(["method","sequence"],as_index=False)["loss"].mean().pivot(index="sequence",columns="method",values="loss")
    for method in ("Validation best","Action-coordinate selector","Direct-output selector","Direct-output sampled"):
        if method in g:
            g[f"delta_{method}_minus_input"]=g[method]-g["Input segmentation"]
    g.reset_index().to_csv(outdir/"davis2016_sequence_effects.csv",index=False,float_format="%.9g")

    selection={"resolution":resolution,"best_validation_action_index":best_action,"best_validation_action":NAMES[best_action],
               "action_leaf_candidates":action_candidates,"action_leaf":action_leaf,"direct_leaf_candidates":direct_candidates,
               "direct_leaf":direct_leaf,"temperature_candidates":temperatures,"model_seeds":MODEL_SEEDS,"runs":runinfo,
               "n_estimators":n_estimators}
    (outdir/"davis2016_selection.json").write_text(json.dumps(selection,indent=2)+"\n")
    return selection


def create_result_tex(outdir:Path):
    summary=pd.read_csv(outdir/"davis2016_summary.csv")
    loss=summary[summary.metric=="loss"].set_index("method"); iou=summary[summary.metric=="iou"].set_index("method")
    bf=summary[summary.metric=="boundary_f08"].set_index("method"); topo=summary[summary.metric=="topology_exact"].set_index("method")
    methods=["Input segmentation","Validation best","Action-coordinate selector","Direct-output selector","Direct-output sampled","Oracle (diagnostic)"]
    lines=[r"\begin{table}[t]",r"\centering",r"\caption{DAVIS 2016 genuine segmentation-error validation on the official held-out validation sequences. The observed masks are pre-computed outputs of six unsupervised methods from the original DAVIS benchmark, not synthetically corrupted ground truth. Values first average frames, source methods, and fitted seeds within video sequence and then average equally over sequences.}",r"\label{tab:davisactual}",r"\small",r"\begin{tabular}{lcccc}",r"\toprule",r"Method & Loss $\downarrow$ & Region $J$ $\uparrow$ & Boundary $F$ $\uparrow$ & Topology exact $\uparrow$\\",r"\midrule"]
    for m in methods:
        if m in loss.index:
            lines.append(f"{m} & {loss.loc[m,'mean']:.4f} & {iou.loc[m,'mean']:.4f} & {bf.loc[m,'mean']:.4f} & {topo.loc[m,'mean']:.4f}\\\\")
    lines += [r"\bottomrule",r"\end{tabular}",r"\end{table}",""]
    comps=json.loads((outdir/"davis2016_paired_comparisons.json").read_text())
    lines.append("% Four predeclared sequence-clustered loss contrasts (A minus B):")
    for c in comps:
        lo,hi=c["familywise95_ci_four_comparisons"]
        lines.append(f"% {c['method_a']} vs {c['method_b']}: mean={c['mean_difference_a_minus_b']:.9g}; FWER95=[{lo:.9g},{hi:.9g}]; nseq={c['n_sequences']}")
    (outdir/"generated_davis2016.tex").write_text("\n".join(lines)+"\n")


def freeze_checksums(outdir:Path):
    files=sorted(p for p in outdir.iterdir() if p.is_file() and p.name!="SHA256SUMS")
    (outdir/"SHA256SUMS").write_text("".join(f"{file_digest(p)}  {p.name}\n" for p in files))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-root",type=Path,default=ROOT/"data"/"davis2016_official")
    ap.add_argument("--data-archive",type=Path,default=None)
    ap.add_argument("--results-archive",type=Path,default=None)
    ap.add_argument("--download",action="store_true")
    ap.add_argument("--output-dir",type=Path,default=ROOT/"results"/"step6_davis2016")
    ap.add_argument("--resolution",type=int,default=PRIMARY_RESOLUTION)
    ap.add_argument("--tune-fraction",type=float,default=.20)
    ap.add_argument("--methods",nargs="+",default=PRIMARY_SOURCE_METHODS)
    ap.add_argument("--n-estimators",type=int,default=96)
    ap.add_argument("--stage",choices=["all","extract","fit"],default="all")
    args=ap.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)

    data_archive=args.data_archive or args.data_root/"DAVIS-data.zip"
    results_archive=args.results_archive or args.data_root/"DAVIS-results.zip"
    _,ann480,imagesets480,result480=ensure_davis(args.data_root,data_archive,results_archive,args.download)
    train_off=read_sequence_list(locate_split_file(imagesets480,"train")); test_off=read_sequence_list(locate_split_file(imagesets480,"val"))
    tv=split_train_sequences(train_off,args.tune_fraction)
    split_sequences={"train":tv["train"],"tune":tv["tune"],"test":sorted(test_off)}
    method_dirs=discover_source_methods(result480,args.methods)
    seq_names=sorted(set(train_off)|set(test_off)); seq_index={s:i for i,s in enumerate(seq_names)}
    method_names=list(args.methods); method_index={m:i for i,m in enumerate(method_names)}
    cases={s:enumerate_cases(ann480,method_dirs,split_sequences[s]) for s in ("train","tune","test")}
    save_protocol(args.output_dir,split_sequences,cases,data_archive,results_archive,method_dirs,args.resolution)
    if args.stage in ("all","extract"):
        audit=build_resize_audit(cases["test"],args.output_dir,args.resolution)
        print("Resize audit:",json.dumps(audit,indent=2),flush=True)
        for s in ("train","tune","test"):
            build_cache(s,cases[s],args.output_dir,args.resolution,seq_index,method_index)
    if args.stage in ("all","fit"):
        train_and_evaluate(args.output_dir,args.resolution,seq_names,method_names,args.n_estimators)
        create_result_tex(args.output_dir)
    freeze_checksums(args.output_dir)
    print(json.dumps({"status":"PASS","output_dir":str(args.output_dir),"source_methods":method_names,
                      "sequence_counts":{k:len(v) for k,v in split_sequences.items()},
                      "case_counts":{k:len(v) for k,v in cases.items()}},indent=2))

if __name__=="__main__": main()
