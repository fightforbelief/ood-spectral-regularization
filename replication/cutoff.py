"""Post-hoc tolerance sensitivity on the fixed Superconductivity holdout."""
import argparse
import csv
from decimal import Decimal
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from p19.data_loading import load_superconductivity_x_only
from p19.estimators import economy_svd, numerical_cutoff
from p19.preprocessing import fit_preprocessor
from p19.protocol import DRAFT_ZERO_VARIANCE, SUPER_SAMPLE_SIZE_LIST_ORIGINAL
from p19.superconductivity_split import group_exact_feature_rows, STORED_SPLIT_SEED

PRIME = 1_000_000_007
CUTOFFS = (0., 1e-12, 1e-10, 1e-8, 1e-6)
HERE = Path(__file__).resolve().parent


@lru_cache(maxsize=131072)
def field_value(token):
    a, b = Decimal(token).as_integer_ratio()
    return a * pow(b, -1, PRIME) % PRIME


def rank_certificate(tokens):
    # Distinct decimal vectors give an upper bound after centering. Differences
    # from a reference row span exactly the centered row space over the reals.
    distinct = {tuple(Decimal(s) for s in row) for row in tokens}
    field = np.array([[field_value(s) for s in row] for row in tokens], dtype=np.int64)
    matrix = (field[1:] - field[0]) % PRIME
    rank = 0
    for j in range(matrix.shape[1]):
        pivots = np.flatnonzero(matrix[rank:, j])
        if not len(pivots):
            continue
        k = rank + pivots[0]
        matrix[[rank, k]] = matrix[[k, rank]]
        row = (matrix[rank, j:] * pow(int(matrix[rank, j]), -1, PRIME)) % PRIME
        # Products of two residues fit in signed int64.
        tail = matrix[rank+1:, j:]
        tail[:] = (tail - tail[:, :1] * row) % PRIME
        rank += 1
    return rank, min(len(distinct) - 1, matrix.shape[1])


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def run(raw, dest, release):
    info = json.loads((release / "provenance.json").read_text(encoding="utf-8"))
    if sha(raw) != info["sha256"]["train.csv"]:
        raise ValueError("unexpected raw CSV checksum")
    X = load_superconductivity_x_only(raw)["X"]
    with raw.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        if header[-1] != "critical_temp" or len(header) != 82:
            raise ValueError("unexpected data schema")
        raw_rows = list(reader)
    y = np.array([float(row[-1]) for row in raw_rows])
    groups = group_exact_feature_rows(X)
    order = np.random.default_rng(STORED_SPLIT_SEED).permutation(len(groups))
    cut = 4 * len(groups) // 5
    train, test = (np.concatenate([groups[j] for j in idx]) for idx in (order[:cut], order[cut:]))
    for arr, key in [(train, "training"), (test, "holdout")]:
        if hashlib.sha256(arr.astype("<i8").tobytes()).hexdigest() != info["split"][key+"_row_index_sha256"]:
            raise ValueError("ordered split mismatch")
    x_test, y_test = X[test], y[test]
    with (release / "losses.csv").open(encoding="utf-8") as f:
        records = [r for r in csv.DictReader(f) if r["dataset"] == "Superconductivity"]
    by_cell = {(int(r["repetition"]), int(r["n"])): r for r in records}
    grid = SUPER_SAMPLE_SIZE_LIST_ORIGINAL
    if len(records) != 1500 or set(by_cell) != {(r, n) for r in range(100) for n in grid}:
        raise ValueError("incomplete repetition grid")
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(".partial")
    with partial.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["repetition", "n", "relative_cutoff", "rank", "mse", "field_rank_lower", "rank_upper"])
        for r in range(100):
            seeds = {int(by_cell[r, n]["seed"]) for n in grid}
            if len(seeds) != 1:
                raise ValueError("nested prefixes require one seed per repetition")
            order = np.random.default_rng(seeds.pop()).permutation(train)
            for n in grid:
                prefix = order[:n]
                x0, y0 = X[prefix], y[prefix]
                pre = fit_preprocessor(x0, y0, DRAFT_ZERO_VARIANCE)
                u, s, vt = economy_svd(pre.transform_X(x0))
                default = numerical_cutoff(s, n, X.shape[1])
                z, h = pre.transform_y(y0), pre.transform_X(x_test)
                numerical_rank = int(np.count_nonzero(s > default))
                lower = upper = ""
                if s[numerical_rank-1] < 1e-10 * s[0]:
                    lower, upper = rank_certificate([raw_rows[i][:-1] for i in prefix])
                previous_rank, loss = None, None
                for c in CUTOFFS:
                    keep = s > max(default, c * s[0])
                    rank = int(np.count_nonzero(keep))
                    if rank != previous_rank:
                        weights = np.zeros_like(s)
                        weights[keep] = (u[:, keep].T @ z) / s[keep]
                        pred = pre.restore_y(h @ (vt.T @ weights))
                        loss = float(np.mean((y_test - pred)**2))
                        if not math.isfinite(loss):
                            raise ValueError("nonfinite loss")
                    if c == 0 and not math.isclose(loss, float(by_cell[r,n]["min_norm"]), rel_tol=1e-5):
                        raise ValueError(f"archived fit mismatch at {r}, {n}: {loss} vs {by_cell[r,n]['min_norm']}")
                    writer.writerow([r, n, c, rank, loss, lower, upper])
                    previous_rank = rank
    partial.replace(dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--release", type=Path, default=HERE)
    args = parser.parse_args()
    run(args.raw, args.output, args.release)
