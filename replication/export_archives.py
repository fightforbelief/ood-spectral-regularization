"""Extract paired losses from the two completed runs in the portfolio archive."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import yaml

YEAR = "20261030T085223Z_l2_28abcf33d8e4"
SUPER = "20261102T141929Z_super_b3c9fff27517"


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def export(root, staging, output):
    output.mkdir(parents=True, exist_ok=True)
    year_file = root / "artifacts/runs" / YEAR / "l2_run_manifest.json"
    super_dir = staging / "artifacts/superconductivity_runs" / SUPER
    super_file = super_dir / "holdout_metrics.jsonl"
    manifest_file = super_dir / "superconductivity_run_manifest.json"
    candidate = staging / "reports/STAGE5N6_SUPERCONDUCTIVITY_PROTOCOL_FROZEN.yaml"
    baseline = root / "results/posthoc/null_baseline_repetitions.csv"
    year = json.loads(year_file.read_text(encoding="utf-8"))
    seeds = yaml.safe_load(candidate.read_text(encoding="utf-8"))["random_seed_list"]
    year_seeds = {r["repetition"]: r["sampling_seed"] for r in year["source_cells"]}
    groups = {"YearPredictionMSD": year["holdout_metrics"],
              "Superconductivity": [json.loads(s) for s in super_file.read_text(encoding="utf-8").splitlines()]}
    losses = {}
    for dataset, rows in groups.items():
        for row in rows:
            r, n = int(row["repetition"]), int(row["sample_size"])
            key = dataset, r, n
            values = losses.setdefault(key, {"seed": (seeds if dataset == "Superconductivity" else year_seeds)[r]})
            if row["estimator"] in values:
                raise ValueError(f"duplicate metric at {key}")
            values[row["estimator"]] = row["mse"]
    with baseline.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            values = losses[row["dataset"], int(row["repetition"]), int(row["sample_size"])]
            if float(row["ridge_mse"]) != values["ridge"] or "null" in values:
                raise ValueError("baseline pairing failed")
            values["null"] = float(row["null_mse"])
    if len(losses) != 3000:
        raise ValueError("incomplete archive")
    fields = ["dataset", "repetition", "n", "seed", "min_norm", "ridge", "pcr", "null"]
    with (output / "losses.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fields, lineterminator="\n")
        writer.writeheader()
        for (dataset, r, n), values in sorted(losses.items()):
            if values.keys() != set(fields[3:]):
                raise ValueError("missing estimator")
            writer.writerow(dict(dataset=dataset, repetition=r, n=n, **values))
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    meta = {"year_run": YEAR, "super_run": SUPER, "sha256": {
        p.name: digest(p) for p in (year_file, super_file, manifest_file, candidate, baseline,
                                     root / "data/raw/train.csv", root / "reports/protocol_freeze.yaml")},
        "split": {k: manifest["split_identity"][k] for k in ("training_row_index_sha256", "holdout_row_index_sha256")}}
    (output / "provenance.json").write_text(json.dumps(meta, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "staging", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    export(args.root, args.staging, args.output)
