"""Refit the published estimators from UCI files and verify paired losses.

Run from the project root via python -m replication.replay. A single
repetition or size can be selected without changing folds or preprocessing.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from p19.data_loading import load_superconductivity_x_only
from p19.l2_pipeline import evaluate_holdout, fit_training_prefix
from p19.l2_sampling import nested_folds_for_sample_size, source_permutation
from p19.superconductivity_split import group_exact_feature_rows, STORED_SPLIT_SEED

HERE = Path(__file__).resolve().parent


def materials(raw, meta):
    with raw.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != meta["sha256"]["train.csv"]:
            raise ValueError("unexpected materials CSV")
    x = load_superconductivity_x_only(raw)["X"]
    with raw.open(encoding="utf-8", newline="") as f:
        y = np.fromiter((float(r["critical_temp"]) for r in csv.DictReader(f)), float)
    groups = group_exact_feature_rows(x)
    perm = np.random.default_rng(STORED_SPLIT_SEED).permutation(len(groups))
    cut = len(groups)*4//5
    train, holdout = (np.concatenate([groups[i] for i in ix]) for ix in (perm[:cut],perm[cut:]))
    for ix,key in ((train,"training"),(holdout,"holdout")):
        if hashlib.sha256(ix.astype("<i8").tobytes()).hexdigest() != meta["split"][key+"_row_index_sha256"]:
            raise ValueError("ordered materials split mismatch")
    return x,y,train,x[holdout],y[holdout]


def music(raw):
    # Parse the distributed decimal file into the same float64 representation.
    data = np.loadtxt(raw, delimiter=",")
    if data.shape != (515345,91) or not np.isfinite(data).all():
        raise ValueError("unexpected YearPredictionMSD table")
    x,y = np.ascontiguousarray(data[:,1:]),data[:,0]
    return x,y,None,x[463715:],y[463715:]


def replay(args):
    with (args.release/"losses.csv").open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["dataset"]==args.dataset
                and (args.repetition is None or int(r["repetition"])==args.repetition)
                and (args.n is None or int(r["n"])==args.n)]
    if not rows:
        raise ValueError("selection contains no archived fits")
    if args.dataset == "YearPredictionMSD":
        x,y,train,h,target = music(args.raw)
    else:
        meta = json.loads((args.release/"provenance.json").read_text(encoding="utf-8"))
        x,y,train,h,target = materials(args.raw,meta)
        membership = np.empty(810,dtype=int)
        permutation = np.random.default_rng(820000001).permutation(810)
        for fold in range(5):
            membership[permutation[fold::5]] = fold
    maximum = 0.
    last_seed = None
    for row in rows:
        seed,r,n = int(row["seed"]),int(row["repetition"]),int(row["n"])
        if seed != last_seed:
            order = source_permutation(seed) if train is None else np.random.default_rng(seed).permutation(train)
            last_seed = seed
        ix = order[:n]
        folds = nested_folds_for_sample_size(n) if train is None else [np.flatnonzero(membership[:n]==i) for i in range(5)]
        fits = fit_training_prefix(x[ix],y[ix],repetition=r,sampling_seed=seed,sample_size=n,prefix_indices=ix,folds=folds)
        for name,model in fits.models.items():
            score = evaluate_holdout(fits.preprocessor,model,h,target)["mse"]
            error = abs(score/float(row[name])-1.)
            if not np.isfinite(error) or error > args.rtol:
                raise ValueError(f"{name}, repetition {r}, n={n}: archived={row[name]}, replay={score}, relative error={error}")
            maximum = max(maximum,error)
    print(f"{len(rows)} prefixes, {3*len(rows)} verified losses, maximum relative error {maximum:.4g}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset",choices=["YearPredictionMSD","Superconductivity"])
    p.add_argument("raw",type=Path)
    p.add_argument("--n",type=int)
    p.add_argument("--repetition",type=int)
    p.add_argument("--release",type=Path,default=HERE)
    p.add_argument("--rtol",type=float,default=1e-5)
    args = p.parse_args()
    if not np.isfinite(args.rtol) or args.rtol < 0:
        p.error("--rtol must be finite and nonnegative")
    replay(args)
