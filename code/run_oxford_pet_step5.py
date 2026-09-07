#!/usr/bin/env python3
"""Step 5: higher-resolution Oxford-IIIT Pet mask validation.

The campaign needs only the official Oxford-IIIT Pet *annotation* archive, not
its RGB images.  The stochastic morphological method receives binary masks, so
human trimap annotations are the scientifically relevant source object.

Primary protocol
----------------
* official trainval.txt supplies a deterministic breed-stratified train/val pool;
* official test.txt remains the held-out benchmark source;
* all official samples: deterministic 50:50 train/validation split within each breed of trainval.txt and the full official test.txt partition;
* masks are aspect-ratio preserving, nearest-neighbor letterboxed to 128x128;
* trimap labels 1 and 3 are foreground in the primary inclusive-silhouette rule;
* six corruption conditions match the primary manuscript campaign;
* every held-out test mask is evaluated under all six conditions;
* method selection and hyperparameter tuning use train/validation only;
* repeated corruptions and estimator seeds are clustered by clean test mask.

No RGB image is used by the estimator and no test target enters fitting/tuning.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tarfile
import time
import urllib.request

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")

import numpy as np
import pandas as pd
from PIL import Image
from scipy.special import softmax
from sklearn.ensemble import ExtraTreesRegressor

from morphology import NAMES, METRICS, bank, betti, features, measure
from output_space_learning import unique_output_rows

ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_URLS = [
    "https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz",
    "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
]
ANNOTATION_URL = ANNOTATION_URLS[0]
ANNOTATION_MD5 = "95a8c909bbe2e81eed6a22bccdf3f68f"
PROTOCOL_SEED = 20260907
MODEL_SEEDS = [101, 102, 103, 104, 105]
CONDITIONS = [
    ("balanced04", .04, .04),
    ("balanced10", .10, .10),
    ("salt04", .01, .04),
    ("salt10", .01, .10),
    ("pepper04", .04, .01),
    ("pepper10", .10, .01),
]
PRIMARY_RESOLUTION = 128


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
    return hashlib.sha256(f"{seed}:{text}".encode("utf-8")).hexdigest()


def safe_extract_tar(archive: Path, dest: Path) -> None:
    dest = dest.resolve()
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf.getmembers():
            target = (dest / member.name).resolve()
            if dest not in target.parents and target != dest:
                raise RuntimeError(f"unsafe tar member: {member.name}")
        tf.extractall(dest)


def ensure_annotations(data_root: Path, archive: Path | None, download: bool) -> Path:
    """Return directory containing annotations/{trainval,test,trimaps}."""
    data_root.mkdir(parents=True, exist_ok=True)
    ann_dir = data_root / "annotations"
    if (ann_dir / "trainval.txt").exists() and (ann_dir / "test.txt").exists() and (ann_dir / "trimaps").is_dir():
        return ann_dir

    if archive is None:
        archive = data_root / "annotations.tar.gz"
    if not archive.exists():
        if not download:
            raise FileNotFoundError(
                "Oxford-IIIT Pet annotations not found. Supply --annotations-archive "
                "or use --download-annotations."
            )
        last_error = None
        for url in ANNOTATION_URLS:
            try:
                print(f"Downloading official annotations from {url}", flush=True)
                urllib.request.urlretrieve(url, archive)
                if file_digest(archive, "md5") == ANNOTATION_MD5:
                    break
                archive.unlink(missing_ok=True)
                last_error = RuntimeError("download completed but MD5 did not match")
            except Exception as exc:
                last_error = exc
                archive.unlink(missing_ok=True)
        if not archive.exists():
            raise RuntimeError(f"could not download the official annotation archive: {last_error}")

    md5 = file_digest(archive, "md5")
    if md5 != ANNOTATION_MD5:
        raise RuntimeError(f"annotation archive MD5 mismatch: {md5} != {ANNOTATION_MD5}")
    safe_extract_tar(archive, data_root)
    if not (ann_dir / "trainval.txt").exists():
        raise RuntimeError("archive extracted but annotations/trainval.txt is missing")
    return ann_dir


def parse_split(ann_dir: Path, split: str) -> list[dict]:
    rows = []
    for line in (ann_dir / f"{split}.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        image_id, class_id, species, breed_id = line.split()
        rows.append({
            "image_id": image_id,
            "class_id": int(class_id),
            "species": int(species),
            "breed_id": int(breed_id),
            "source_split": split,
        })
    return rows


def stratified_protocol(trainval: list[dict], test: list[dict], n_train=0, n_val=0, n_test=0, val_fraction=.5) -> dict[str, list[dict]]:
    """Deterministic breed-stratified protocol.

    Primary mode (all n_* equal zero) uses every official sample: each breed's
    trainval pool is split deterministically according to ``val_fraction`` and
    the entire official test partition is retained. Positive n_* values enable
    a small explicitly declared subsample, used only for offline smoke tests or
    resource-constrained sensitivity runs.
    """
    out = {"train": [], "val": [], "test": []}
    subsample = any(v > 0 for v in (n_train, n_val, n_test))
    if subsample and not all(v > 0 for v in (n_train, n_val, n_test)):
        raise ValueError("if subsampling, all three per-breed counts must be positive")
    for class_id in range(1, 38):
        pool = sorted((r for r in trainval if r["class_id"] == class_id), key=lambda r: stable_key(r["image_id"]))
        held = sorted((r for r in test if r["class_id"] == class_id), key=lambda r: stable_key("test:" + r["image_id"]))
        if subsample:
            if len(pool) < n_train + n_val:
                raise RuntimeError(f"class {class_id} has only {len(pool)} trainval examples")
            if len(held) < n_test:
                raise RuntimeError(f"class {class_id} has only {len(held)} test examples")
            out["train"].extend(pool[:n_train])
            out["val"].extend(pool[n_train:n_train+n_val])
            out["test"].extend(held[:n_test])
        else:
            nva = max(1, min(len(pool)-1, int(round(val_fraction * len(pool)))))
            out["val"].extend(pool[:nva])
            out["train"].extend(pool[nva:])
            out["test"].extend(held)
    return out


def load_native_binary_trimap(path: Path, include_boundary: bool = True) -> np.ndarray:
    a = np.asarray(Image.open(path))
    if a.ndim == 3:
        a = a[..., 0]
    vals = set(np.unique(a).tolist())
    if not vals.issubset({1, 2, 3}):
        raise RuntimeError(f"unexpected trimap values {sorted(vals)} in {path.name}")
    return np.isin(a, [1, 3]) if include_boundary else (a == 1)


def load_binary_trimap(path: Path, resolution: int, include_boundary: bool = True) -> np.ndarray:
    return letterbox_bool(load_native_binary_trimap(path, include_boundary), resolution)


def letterbox_bool(mask: np.ndarray, resolution: int) -> np.ndarray:
    """Nearest-neighbor resize with aspect-ratio preserving zero padding."""
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    scale = min(resolution / h, resolution / w)
    nh = max(1, min(resolution, int(round(h * scale))))
    nw = max(1, min(resolution, int(round(w * scale))))
    im = Image.fromarray(mask.astype(np.uint8) * 255)
    resized = np.asarray(im.resize((nw, nh), resample=Image.Resampling.NEAREST)) > 0
    out = np.zeros((resolution, resolution), dtype=bool)
    y0 = (resolution - nh) // 2
    x0 = (resolution - nw) // 2
    out[y0:y0+nh, x0:x0+nw] = resized
    return out


def corrupt(y: np.ndarray, d: float, a: float, rng: np.random.Generator) -> np.ndarray:
    u = rng.random(y.shape)
    return np.where(y, u >= d, u < a)


def case_rng(split_code: int, image_id: str, condition_index: int) -> np.random.Generator:
    image_hash = int(hashlib.sha256(image_id.encode()).hexdigest()[:16], 16)
    ss = np.random.SeedSequence([PROTOCOL_SEED, split_code, image_hash & 0xffffffff,
                                (image_hash >> 32) & 0xffffffff, condition_index])
    return np.random.default_rng(ss)


def extract_case(x: np.ndarray, truth: np.ndarray) -> dict:
    outputs = bank(x)
    tb = betti(truth)
    scores = np.stack([measure(y, truth, tb) for y in outputs]).astype(np.float32)
    afeat = features(x, outputs).astype(np.float32)
    rows = unique_output_rows(x, outputs, scores[:, 0])
    m = len(NAMES)
    dfeat = np.zeros((m, rows.pair_features.shape[1]), dtype=np.float32)
    reps = np.full(m, -1, dtype=np.int16)
    dfeat[:len(rows.representatives)] = rows.pair_features
    reps[:len(rows.representatives)] = rows.representatives.astype(np.int16)
    return {
        "scores": scores,
        "action_features": afeat,
        "direct_features": dfeat,
        "direct_reps": reps,
        "n_distinct": np.int16(len(rows.representatives)),
    }


def build_cache(ann_dir: Path, protocol: dict[str, list[dict]], outdir: Path, resolution: int,
                include_boundary=True) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    timings = {}
    split_code = {"train": 0, "val": 1, "test": 2}
    for split in ("train", "val", "test"):
        cache_path = outdir / f"oxford_pet_{split}_{resolution}.npz"
        if cache_path.exists():
            print(f"Using existing cache {cache_path.name}", flush=True)
            continue
        records = protocol[split]
        case_defs = []
        if split == "test":
            for i, rec in enumerate(records):
                for c in range(len(CONDITIONS)):
                    case_defs.append((i, rec, c))
        else:
            for i, rec in enumerate(records):
                # One preassigned condition per clean image, balanced by index.
                case_defs.append((i, rec, i % len(CONDITIONS)))
        n = len(case_defs)
        scores = np.empty((n, len(NAMES), len(METRICS)), dtype=np.float32)
        action_features = np.empty((n, 5 + 5 * len(NAMES)), dtype=np.float32)
        direct_features = np.empty((n, len(NAMES), 10), dtype=np.float32)
        direct_reps = np.empty((n, len(NAMES)), dtype=np.int16)
        n_distinct = np.empty(n, dtype=np.int16)
        image_index = np.empty(n, dtype=np.int16)
        condition = np.empty(n, dtype=np.int8)
        class_id = np.empty(n, dtype=np.int8)
        species = np.empty(n, dtype=np.int8)
        clean_masks = np.empty((len(records), resolution, resolution), dtype=np.uint8)
        loaded = np.zeros(len(records), dtype=bool)

        t0 = time.perf_counter()
        for qi, (i, rec, c) in enumerate(case_defs):
            if not loaded[i]:
                trimap_path = ann_dir / "trimaps" / f"{rec['image_id']}.png"
                truth = load_binary_trimap(trimap_path, resolution, include_boundary)
                clean_masks[i] = truth.astype(np.uint8)
                loaded[i] = True
            else:
                truth = clean_masks[i].astype(bool)
            name, d, a = CONDITIONS[c]
            x = corrupt(truth, d, a, case_rng(split_code[split], rec["image_id"], c))
            e = extract_case(x, truth)
            scores[qi] = e["scores"]
            action_features[qi] = e["action_features"]
            direct_features[qi] = e["direct_features"]
            direct_reps[qi] = e["direct_reps"]
            n_distinct[qi] = e["n_distinct"]
            image_index[qi] = i
            condition[qi] = c
            class_id[qi] = int(rec["class_id"])
            species[qi] = int(rec["species"])
            if qi and qi % 250 == 0:
                print(f"{split}: {qi}/{n}", flush=True)
        timings[split] = time.perf_counter() - t0
        np.savez_compressed(cache_path,
            scores=scores, action_features=action_features,
            direct_features=direct_features, direct_reps=direct_reps,
            n_distinct=n_distinct, image_index=image_index, condition=condition,
            class_id=class_id, species=species, clean_masks=clean_masks,
            resolution=np.array([resolution], dtype=np.int16))
        print(f"Extracted {split}: {n} cases in {timings[split]:.1f}s", flush=True)
    (outdir / "oxford_pet_extraction_timing.json").write_text(json.dumps(timings, indent=2) + "\n")
    return timings


def fit_forest(X, y, seed, leaf, n_estimators=96):
    m = ExtraTreesRegressor(n_estimators=n_estimators, max_features=1.0,
                            min_samples_leaf=leaf, random_state=seed, n_jobs=1)
    m.fit(X, y)
    return m


def direct_table(cache: dict) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(cache["scores"])):
        k = int(cache["n_distinct"][i])
        reps = cache["direct_reps"][i, :k]
        X.append(cache["direct_features"][i, :k])
        y.append(cache["scores"][i, reps, 0])
    return np.vstack(X), np.concatenate(y)


def direct_predict(model, cache: dict) -> list[np.ndarray]:
    pred = []
    for i in range(len(cache["scores"])):
        k = int(cache["n_distinct"][i])
        pred.append(model.predict(cache["direct_features"][i, :k]))
    return pred


def direct_expected_metrics(cache: dict, predicted_costs: list[np.ndarray], tau: float) -> np.ndarray:
    out = np.empty((len(predicted_costs), len(METRICS)), dtype=float)
    for i, pc in enumerate(predicted_costs):
        k = len(pc)
        reps = cache["direct_reps"][i, :k]
        p = softmax(-(pc - pc.min()) / tau)
        out[i] = p @ cache["scores"][i, reps]
    return out


def direct_selected_metrics(cache: dict, predicted_costs: list[np.ndarray]) -> np.ndarray:
    out = np.empty((len(predicted_costs), len(METRICS)), dtype=float)
    for i, pc in enumerate(predicted_costs):
        k = len(pc)
        reps = cache["direct_reps"][i, :k]
        out[i] = cache["scores"][i, reps[int(np.argmin(pc))]]
    return out


def action_selected_metrics(cache: dict, predicted_costs: np.ndarray) -> np.ndarray:
    idx = predicted_costs.argmin(axis=1)
    return cache["scores"][np.arange(len(idx)), idx]


def aggregate_rows(dataset, cache, method, seed, metrics) -> pd.DataFrame:
    f = pd.DataFrame(np.asarray(metrics), columns=METRICS)
    f.insert(0, "condition", [CONDITIONS[int(c)][0] for c in cache["condition"]])
    f.insert(0, "species", cache["species"].astype(int))
    f.insert(0, "class_id", cache["class_id"].astype(int))
    f.insert(0, "image", cache["image_index"].astype(int))
    f.insert(0, "seed", int(seed))
    f.insert(0, "method", method)
    f.insert(0, "dataset", dataset)
    return f


def clustered_image_summary(df: pd.DataFrame, metric: str, seeds=MODEL_SEEDS) -> pd.DataFrame:
    """Average conditions and model seeds within clean image before population mean."""
    d = df.copy()
    # Static controls have seed=-1. Trained methods average their five fitted seeds.
    by_image = d.groupby(["method", "image"], as_index=False)[metric].mean()
    return by_image.groupby("method")[metric].agg(["mean", "std", "count"]).reset_index()


def clustered_mean_bootstrap(df: pd.DataFrame, method: str, metric: str, n_boot=2000, seed=20260907) -> dict:
    g = df[df.method == method].groupby("image", as_index=False)[metric].mean()
    vals = g[metric].to_numpy(float)
    rng = np.random.default_rng(seed + (int(hashlib.sha256((method+metric).encode()).hexdigest()[:8],16) & 0xffff))
    boots = np.empty(n_boot)
    for j in range(n_boot):
        boots[j] = vals[rng.integers(0, len(vals), len(vals))].mean()
    lo, hi = np.quantile(boots, [.025, .975])
    return {"method": method, "metric": metric, "n_images": int(len(vals)),
            "mean": float(vals.mean()), "ci95_low": float(lo), "ci95_high": float(hi),
            "bootstrap_resamples": int(n_boot), "cluster": "clean test mask"}


def paired_cluster_bootstrap(df: pd.DataFrame, method_a: str, method_b: str, metric="loss",
                             n_boot=2000, seed=20260907) -> dict:
    g = df.groupby(["method", "image"], as_index=False)[metric].mean()
    a = g[g.method == method_a].set_index("image")[metric]
    b = g[g.method == method_b].set_index("image")[metric]
    ids = np.array(sorted(set(a.index) & set(b.index)), dtype=int)
    diff = (a.loc[ids] - b.loc[ids]).to_numpy(float)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for j in range(n_boot):
        s = rng.integers(0, len(diff), len(diff))
        boots[j] = diff[s].mean()
    lo, hi = np.quantile(boots, [.025, .975])
    slo, shi = np.quantile(boots, [.0125, .9875])  # Bonferroni family of two primary comparisons
    return {
        "method_a": method_a, "method_b": method_b, "metric": metric,
        "n_images": int(len(diff)), "mean_difference_a_minus_b": float(diff.mean()),
        "ci95": [float(lo), float(hi)],
        "simultaneous_familywise95_ci_two_comparisons": [float(slo), float(shi)],
        "bootstrap_resamples": int(n_boot), "cluster": "clean test mask"
    }


def train_and_evaluate(outdir: Path, resolution: int) -> dict:
    cache = {s: dict(np.load(outdir / f"oxford_pet_{s}_{resolution}.npz")) for s in ("train", "val", "test")}
    tr, va, te = cache["train"], cache["val"], cache["test"]
    rows = []
    nval = len(va["scores"])
    ntest = len(te["scores"])

    # Static controls.
    best_action = int(va["scores"][:, :, 0].mean(axis=0).argmin())
    rows.append(aggregate_rows("oxford_pet", te, "Input", -1, te["scores"][:, 0]))
    rows.append(aggregate_rows("oxford_pet", te, "Validation best", -1, te["scores"][:, best_action]))
    oracle_idx = te["scores"][:, :, 0].argmin(axis=1)
    rows.append(aggregate_rows("oxford_pet", te, "Oracle (diagnostic)", -1,
                               te["scores"][np.arange(ntest), oracle_idx]))

    # Tune leaf size separately for action-coordinate and direct-output learners.
    action_candidates = []
    for leaf in (2, 8, 20):
        m = fit_forest(tr["action_features"], tr["scores"][:, :, 0], 99, leaf)
        pr = m.predict(va["action_features"])
        val_loss = action_selected_metrics(va, pr)[:, 0].mean()
        action_candidates.append((leaf, float(val_loss)))
    action_leaf = min(action_candidates, key=lambda z: z[1])[0]

    Xd, yd = direct_table(tr)
    direct_candidates = []
    for leaf in (2, 8, 20):
        m = fit_forest(Xd, yd, 99, leaf)
        pr = direct_predict(m, va)
        val_loss = direct_selected_metrics(va, pr)[:, 0].mean()
        direct_candidates.append((leaf, float(val_loss)))
    direct_leaf = min(direct_candidates, key=lambda z: z[1])[0]

    runs = []
    temperatures = [.005, .02, .1]
    for model_seed in MODEL_SEEDS:
        t0 = time.perf_counter()
        action_model = fit_forest(tr["action_features"], tr["scores"][:, :, 0], model_seed, action_leaf)
        ap = action_model.predict(te["action_features"])
        rows.append(aggregate_rows("oxford_pet", te, "Action-coordinate selector", model_seed,
                                   action_selected_metrics(te, ap)))

        direct_model = fit_forest(Xd, yd, model_seed, direct_leaf)
        vp = direct_predict(direct_model, va)
        tp = direct_predict(direct_model, te)
        val_tau = min(temperatures,
                      key=lambda tau: direct_expected_metrics(va, vp, tau)[:, 0].mean())
        rows.append(aggregate_rows("oxford_pet", te, "Direct-output selector", model_seed,
                                   direct_selected_metrics(te, tp)))
        rows.append(aggregate_rows("oxford_pet", te, "Direct-output sampled", model_seed,
                                   direct_expected_metrics(te, tp, val_tau)))
        runs.append({"seed": model_seed, "temperature": val_tau,
                     "fit_predict_seconds": time.perf_counter() - t0})
        print(f"Completed seed {model_seed}", flush=True)

    df = pd.concat(rows, ignore_index=True)
    df.to_csv(outdir / "oxford_pet_test_metrics.csv.gz", index=False, float_format="%.9g")

    summaries = []
    for metric in ("loss", "dice", "iou", "beta0_error", "beta1_error", "topology_exact"):
        s = clustered_image_summary(df, metric)
        s.insert(0, "metric", metric)
        summaries.append(s)
    summary_df = pd.concat(summaries, ignore_index=True)
    summary_df.to_csv(outdir / "oxford_pet_summary.csv", index=False, float_format="%.9g")

    # Every predefined corruption is retained separately. Learned seeds are
    # averaged within clean image/condition before condition-level aggregation.
    by_case = df.groupby(["method", "image", "class_id", "species", "condition"], as_index=False)[METRICS].mean()
    condition_summary = by_case.groupby(["method", "condition"], as_index=False)[METRICS].mean()
    condition_summary.to_csv(outdir / "oxford_pet_condition_summary.csv", index=False, float_format="%.9g")
    # Macro-breed summary gives every one of the 37 breeds equal weight.
    by_image = df.groupby(["method", "image", "class_id"], as_index=False)[METRICS].mean()
    by_breed = by_image.groupby(["method", "class_id"], as_index=False)[METRICS].mean()
    macro_breed = by_breed.groupby("method", as_index=False)[METRICS].mean()
    macro_breed.to_csv(outdir / "oxford_pet_macro_breed_summary.csv", index=False, float_format="%.9g")

    ci_rows = []
    headline_methods = ["Input", "Validation best", "Action-coordinate selector",
                        "Direct-output selector", "Direct-output sampled", "Oracle (diagnostic)"]
    for method in headline_methods:
        for metric in ("loss", "dice", "iou", "topology_exact"):
            ci_rows.append(clustered_mean_bootstrap(df, method, metric))
    pd.DataFrame(ci_rows).to_csv(outdir / "oxford_pet_clustered_ci.csv", index=False, float_format="%.9g")

    comparisons = []
    for b in ("Validation best", "Action-coordinate selector"):
        comparisons.append(paired_cluster_bootstrap(df, "Direct-output selector", b, "loss"))
    (outdir / "oxford_pet_paired_comparisons.json").write_text(json.dumps(comparisons, indent=2) + "\n")

    selection = {
        "resolution": resolution,
        "best_validation_action_index": best_action,
        "best_validation_action": NAMES[best_action],
        "action_leaf_candidates": action_candidates,
        "action_leaf": action_leaf,
        "direct_leaf_candidates": direct_candidates,
        "direct_leaf": direct_leaf,
        "model_seeds": MODEL_SEEDS,
        "temperature_candidates": temperatures,
        "runs": runs,
    }
    (outdir / "oxford_pet_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    return selection


def save_protocol(protocol: dict[str, list[dict]], outdir: Path, ann_dir: Path,
                  archive: Path | None, resolution: int, include_boundary: bool) -> None:
    with (outdir / "oxford_pet_protocol_ids.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["analysis_split", "source_split", "image_id", "class_id", "species", "breed_id", "stable_key"])
        for split in ("train", "val", "test"):
            for r in protocol[split]:
                token = ("test:" if r["source_split"] == "test" else "") + r["image_id"]
                w.writerow([split, r["source_split"], r["image_id"], r["class_id"], r["species"], r["breed_id"], stable_key(token)])

    # Resampling is audited before fitting: native and letterboxed topology are
    # compared for every selected clean annotation. This is preprocessing
    # evidence only and is independent of estimator outcomes.
    audit_rows = []
    for split in ("train", "val", "test"):
        for r in protocol[split]:
            p = ann_dir / "trimaps" / f"{r['image_id']}.png"
            native = load_native_binary_trimap(p, include_boundary)
            resized = letterbox_bool(native, resolution)
            bn, br = betti(native), betti(resized)
            audit_rows.append({
                "analysis_split": split, "image_id": r["image_id"],
                "class_id": r["class_id"], "native_h": native.shape[0],
                "native_w": native.shape[1], "native_beta0": int(bn[0]),
                "native_beta1": int(bn[1]), "resized_beta0": int(br[0]),
                "resized_beta1": int(br[1]),
                "native_foreground_fraction": float(native.mean()),
                "resized_foreground_fraction": float(resized.mean()),
                "topology_preserved": int(np.array_equal(bn, br)),
            })
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(outdir / "oxford_pet_resize_audit.csv", index=False, float_format="%.9g")
    resize_summary = {
        "all_topology_preserved_fraction": float(audit.topology_preserved.mean()),
        "test_topology_preserved_fraction": float(audit.loc[audit.analysis_split == "test", "topology_preserved"].mean()),
        "n_all": int(len(audit)),
        "n_test": int((audit.analysis_split == "test").sum()),
    }

    meta = {
        "dataset": "Oxford-IIIT Pet",
        "official_dataset_page": "https://www.robots.ox.ac.uk/~vgg/data/pets/",
        "official_annotations_urls": ANNOTATION_URLS,
        "official_annotations_md5": ANNOTATION_MD5,
        "archive_sha256": file_digest(archive) if archive and archive.exists() else None,
        "license": "Creative Commons Attribution-ShareAlike 4.0 International; image copyrights remain with original owners",
        "resolution": resolution,
        "resize": "aspect-ratio preserving nearest-neighbor letterbox to square canvas",
        "resize_topology_audit": resize_summary,
        "trimap_primary_rule": "labels 1 and 3 foreground; label 2 background" if include_boundary else "label 1 foreground only",
        "protocol_seed": PROTOCOL_SEED,
        "counts": {k: len(v) for k, v in protocol.items()},
        "conditions": CONDITIONS,
        "operator_names": NAMES,
        "model_seeds": MODEL_SEEDS,
        "test_repeats_per_clean_mask": len(CONDITIONS),
        "no_rgb_images_used": True,
    }
    (outdir / "oxford_pet_provenance.json").write_text(json.dumps(meta, indent=2) + "\n")


def create_result_tex(outdir: Path) -> None:
    summary = pd.read_csv(outdir / "oxford_pet_summary.csv")
    comps = json.loads((outdir / "oxford_pet_paired_comparisons.json").read_text())
    loss = summary[summary.metric == "loss"].set_index("method")
    dice = summary[summary.metric == "dice"].set_index("method")
    iou = summary[summary.metric == "iou"].set_index("method")
    methods = ["Input", "Validation best", "Action-coordinate selector", "Direct-output selector", "Direct-output sampled", "Oracle (diagnostic)"]
    provenance = json.loads((outdir / "oxford_pet_provenance.json").read_text())
    resolution = int(provenance["resolution"])
    caption = (f"Oxford-IIIT Pet higher-resolution mask validation at ${{{resolution}}}\\times{{{resolution}}}$. "
               "Values average the six predeclared corruptions and, for learned estimators, five fitted seeds within each clean held-out mask before averaging over masks.")
    lines = [r"\begin{table}[t]", r"\centering", f"\\caption{{{caption}}}", r"\label{tab:oxfordpet}", r"\small", r"\begin{tabular}{lccc}", r"\toprule", r"Method & Loss $\downarrow$ & Dice $\uparrow$ & IoU $\uparrow$\\", r"\midrule"]
    for m in methods:
        if m in loss.index:
            lines.append(f"{m} & {loss.loc[m,'mean']:.4f} & {dice.loc[m,'mean']:.4f} & {iou.loc[m,'mean']:.4f}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    lines.append("% Paired clean-mask clustered bootstrap comparisons (Direct-output selector minus baseline):")
    for c in comps:
        lines.append(f"% vs {c['method_b']}: mean={c['mean_difference_a_minus_b']:.8g}, 95% CI=[{c['ci95'][0]:.8g},{c['ci95'][1]:.8g}], n={c['n_images']}")
    (outdir / "generated_oxford_pet.tex").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=ROOT / "data" / "oxford_pet_official")
    ap.add_argument("--annotations-archive", type=Path, default=None)
    ap.add_argument("--download-annotations", action="store_true")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "results" / "step5_oxford_pet")
    ap.add_argument("--resolution", type=int, default=PRIMARY_RESOLUTION)
    ap.add_argument("--strict-foreground", action="store_true", help="use trimap label 1 only; primary protocol includes label 3 boundary")
    ap.add_argument("--n-train-per-breed", type=int, default=0, help="0 = use all official trainval data")
    ap.add_argument("--n-val-per-breed", type=int, default=0, help="0 = use all official trainval data")
    ap.add_argument("--n-test-per-breed", type=int, default=0, help="0 = use the entire official test partition")
    ap.add_argument("--val-fraction", type=float, default=.5, help="primary split of each breed within official trainval")
    ap.add_argument("--stage", choices=["all", "extract", "fit"], default="all")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ann_dir = ensure_annotations(args.data_root, args.annotations_archive, args.download_annotations)
    trainval = parse_split(ann_dir, "trainval")
    test = parse_split(ann_dir, "test")
    protocol = stratified_protocol(trainval, test, args.n_train_per_breed, args.n_val_per_breed, args.n_test_per_breed, args.val_fraction)
    archive = args.annotations_archive or (args.data_root / "annotations.tar.gz")
    save_protocol(protocol, args.output_dir, ann_dir, archive if archive.exists() else None,
                  args.resolution, not args.strict_foreground)

    if args.stage in ("all", "extract"):
        build_cache(ann_dir, protocol, args.output_dir, args.resolution, not args.strict_foreground)
    if args.stage in ("all", "fit"):
        train_and_evaluate(args.output_dir, args.resolution)
        create_result_tex(args.output_dir)

    print(json.dumps({"status": "PASS", "output_dir": str(args.output_dir),
                      "counts": {k: len(v) for k, v in protocol.items()},
                      "resolution": args.resolution}, indent=2))


if __name__ == "__main__":
    main()
