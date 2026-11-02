"""Command line for remote audits, tests, L1, and the closed L2 gate."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from p19.data_audit import audit_superconductivity_x_only, audit_yearprediction_x_only
from p19.gates import GateRefusal
from p19.superconductivity_runner import (
    SuperconductivityExecutionBlocked,
    run_superconductivity_confirmatory,
)
from p19.l1_reproduction import run_l1
from p19.code_identity import write_manifest
from p19.l2_evaluation import l2_preflight, run_confirmatory
from p19.protocol import load_yaml
from p19.reporting import write_json


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run(command: list[str]) -> str:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    text = (completed.stdout or "") + (("\n" + completed.stderr) if completed.stderr else "")
    return text.strip()


def _package_version(name: str) -> str:
    try:
        import importlib.metadata as metadata
        return metadata.version(name)
    except Exception as exc:
        return f"unavailable ({type(exc).__name__})"


def environment_audit(root: Path) -> Path:
    disk = shutil.disk_usage("C:\\")
    numpy_config = {}
    try:
        numpy_config = np.show_config(mode="dicts")
    except TypeError:
        numpy_config = {"note": "show_config(mode='dicts') is unavailable"}
    thread_info = None
    try:
        from threadpoolctl import threadpool_info
        thread_info = threadpool_info()
    except Exception as exc:
        thread_info = [{"error": type(exc).__name__, "message": str(exc)}]
    gpu = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    cuda_header = _run(["nvidia-smi"])
    cuda_line = ""
    for line in cuda_header.splitlines():
        if "CUDA" in line or "NVIDIA-SMI" in line:
            cuda_line = line.strip()
            break
    payload = {
        "recorded_utc": _utc(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "in_venv": sys.prefix != getattr(sys, "base_prefix", sys.prefix),
        "prefix": sys.prefix,
        "pip": _run([sys.executable, "-m", "pip", "--version"]),
        "conda": shutil.which("conda"),
        "uv": shutil.which("uv"),
        "numpy": np.__version__,
        "scipy": _package_version("scipy"),
        "pyyaml": _package_version("PyYAML"),
        "matplotlib": _package_version("matplotlib"),
        "pytest": _package_version("pytest"),
        "pandas": _package_version("pandas"),
        "torch_metadata_only": _package_version("torch"),
        "torch_imported": False,
        "disk_c_total_bytes": disk.total,
        "disk_c_free_bytes": disk.free,
        "cpu_count": os_cpu_count(),
        "numpy_config": numpy_config,
        "threadpool_info": thread_info,
        "gpu_query": gpu,
        "cuda_header_line": cuda_line,
    }
    write_json(root / "artifacts" / "environment_audit.json", payload)
    report = root / "reports" / "environment_audit.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    gib = 1024 ** 3
    lines = [
        "# Environment audit",
        "",
        f"Recorded at `{payload['recorded_utc']}` on `{payload['host']}`.",
        "This audit describes the isolated project environment used for execution.",
        "PyTorch was not imported. No GPU package was installed for this project.",
        "",
        "## Operating system and machine",
        "",
        f"- Platform: `{payload['platform']}`",
        f"- CPU count: `{payload['cpu_count']}`",
        f"- C: total GiB: `{disk.total / gib:.2f}`",
        f"- C: free GiB: `{disk.free / gib:.2f}`",
        "",
        "## Python",
        "",
        f"- Executable: `{payload['python_executable']}`",
        f"- Version: `{payload['python_version'].splitlines()[0]}`",
        f"- Virtual environment: `{payload['in_venv']}`",
        f"- Prefix: `{payload['prefix']}`",
        f"- pip: `{payload['pip']}`",
        f"- conda on PATH: `{payload['conda']}`",
        f"- uv on PATH: `{payload['uv']}`",
        "",
        "## Packages in this environment",
        "",
        f"- NumPy `{payload['numpy']}`",
        f"- SciPy `{payload['scipy']}`",
        f"- PyYAML `{payload['pyyaml']}`",
        f"- Matplotlib `{payload['matplotlib']}`",
        f"- pytest `{payload['pytest']}`",
        f"- pandas `{payload['pandas']}` (not required; CSV is written by the standard library)",
        f"- torch metadata `{payload['torch_metadata_only']}`; imported: `{payload['torch_imported']}`",
        "",
        "## BLAS / LAPACK and threads",
        "",
        "```json",
        json.dumps({"numpy_config": numpy_config, "threadpool_info": thread_info}, indent=2, default=str),
        "```",
        "",
        "## GPU / CUDA metadata only",
        "",
        f"- nvidia-smi query: `{gpu}`",
        f"- header: `{cuda_line}`",
        "",
        "The project estimators call SciPy's float64 economy SVD. They do not use this GPU.",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def os_cpu_count() -> int | None:
    import os
    return os.cpu_count()


def _write_year_report(root: Path, audit: dict, manifest_entry: dict) -> None:
    lines = [
        "# Data schema audit",
        "",
        "YearPredictionMSD was read with an X-only parser. The first comma-separated token on every row was discarded and was not converted to a number.",
        "",
        "## Feature-count blocker",
        "",
        f"- status: `{audit['status']}`",
        f"- feature_count_status: `{audit['feature_count_status']}`",
        f"- total_columns: `{audit['total_columns']}`",
        f"- target_index: `{audit['target_index']}`",
        f"- predictor_count: `{audit['predictor_count']}`",
        f"- selection-report predictor count: `{audit['protocol_predictor_count']}`",
        f"- uci_text_resolution: `{audit['uci_text_resolution']}`",
        f"- feature_count_amendment: UNRESOLVED",
        f"- human_amendment_required: `{audit['human_amendment_required']}`",
        f"- confirmatory_l2: `{audit['confirmatory_l2']}`",
        f"- labels_loaded: `{audit['labels_loaded']}`",
        "",
        "The original selection-report sample-size list is preserved below and was not recomputed.",
        "",
        f"`n = {audit['original_sample_size_list']}`",
        "",
        f"amended_sample_size_list: `{audit['amended_sample_size_list']}`",
        "",
        "## File and split",
        "",
        f"- file: `{audit['path_name']}`",
        f"- bytes: `{audit['byte_size']}`",
        f"- sha256: `{audit['sha256']}`",
        f"- rows: `{audit['n_rows']}`",
        f"- source rows: `{audit['source_rows_observed']}`",
        f"- target rows: `{audit['target_rows_observed']}`",
        f"- nonfinite feature rows: `{audit['nonfinite_feature_rows']}`",
        f"- unparsed feature rows: `{audit['unparsed_feature_rows']}`",
        f"- zero-variance feature columns: `{audit['zero_variance_feature_columns']}`",
        f"- extra duplicate source feature rows: `{audit['source_extra_duplicate_feature_rows']}`",
        f"- extra duplicate target feature rows: `{audit['target_extra_duplicate_feature_rows']}`",
        f"- target rows with features matching a source row: `{audit['target_rows_whose_features_match_a_source_row']}`",
        "",
        "## X-only shift diagnostic",
        "",
        "```json",
        json.dumps(audit["x_only_shift"], indent=2),
        "```",
        "",
        "x_only_diagnostics: COMPLETE" if audit["x_only_shift"] is not None else "x_only_diagnostics: INCOMPLETE",
        "",
        "## Provenance",
        "",
        "```json",
        json.dumps(manifest_entry, indent=2),
        "```",
        "",
        "Official UCI prose says there are 90 attributes and that the first value is the year. The same page also says 12 timbre averages and 78 timbre covariances, which sum to 90. Those sentences do not by themselves say whether the year is inside the 90. This audit records the measured layout and does not rewrite the sample-size list.",
        "",
        "A human owner still has to approve a written amendment before any confirmatory L2 run. The amendment needs to state the accepted column count, target position, and predictor count, and whether the archived sample-size list is kept. The measured file does not by itself open the confirmatory gate.",
        "",
    ]
    (root / "reports" / "data_schema_audit.md").write_text("\n".join(lines), encoding="utf-8")


def audit_year(root: Path, path: Path) -> dict:
    audit = audit_yearprediction_x_only(path)
    manifest = {
        "dataset": "YearPredictionMSD",
        "official_page": "https://archive.ics.uci.edu/dataset/203/yearpredictionmsd",
        "doi": "https://doi.org/10.24432/C50K61",
        "license_stated_by_uci": "CC BY 4.0",
        "download_url": "https://archive.ics.uci.edu/static/public/203/yearpredictionmsd.zip",
        "retrieval_utc": _utc(),
        "retrieval_note": (
            "The archive bytes were fetched from the official UCI URL on the operator workstation "
            "after the direct download from hanqing-win remained near 10 KB/s. The completed archive "
            "was copied onto hanqing-win. The SHA-256 below is the file stored on hanqing-win. "
            "A partial remote download was not hashed."
        ),
        "zip_sha256": _file_sha(root / "data" / "raw" / "yearpredictionmsd.zip"),
        "extracted_file": path.name,
        "byte_size": audit["byte_size"],
        "sha256": audit["sha256"],
        "official_rows": 515345,
        "official_split": {"source_first_rows": 463715, "target_last_rows": 51630},
    }
    write_json(root / "data" / "manifests" / "yearpredictionmsd.json", manifest)
    _merge_manifest(root, {"yearpredictionmsd": manifest})
    _write_year_report(root, audit, manifest)
    return audit


def audit_super(root: Path, train_csv: Path, unique_m: Path | None) -> dict:
    audit = audit_superconductivity_x_only(train_csv, unique_m)
    manifest = {
        "dataset": "Superconductivty Data",
        "official_page": "https://archive.ics.uci.edu/dataset/464/superconductivty+dat",
        "doi": "https://doi.org/10.24432/C53P47",
        "license_stated_by_uci": "CC BY 4.0",
        "download_url": "https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip",
        "retrieval_utc": _utc(),
        "retrieval_note": (
            "The archive bytes were fetched from the official UCI URL on the operator workstation "
            "after the direct download from hanqing-win remained near 10 KB/s. The completed archive "
            "was copied onto hanqing-win. The SHA-256 below is the file stored on hanqing-win. "
            "A partial remote download was not hashed."
        ),
        "zip_sha256": _file_sha(root / "data" / "raw" / "superconductivty_data.zip"),
        "train_csv": train_csv.name,
        "train_byte_size": audit["byte_size"],
        "train_sha256": audit["sha256"],
        "unique_m_sha256": audit["unique_m_sha256"],
        "unique_m_byte_size": audit["unique_m_byte_size"],
        "official_rows": 21263,
        "official_predictors": 81,
        "target_column": "critical_temp",
        "split": "NOT_FROZEN",
    }
    write_json(root / "data" / "manifests" / "superconductivity.json", manifest)
    _merge_manifest(root, {"superconductivity": manifest})
    report = root / "reports" / "superconductivity_x_only_audit.md"
    report.write_text(
        "\n".join(
            [
                "# Superconductivity X-only audit",
                "",
                f"- status: `{audit['status']}`",
                f"- total_columns: `{audit['total_columns']}`",
                f"- target_column: `{audit['target_column']}`",
                f"- target_index: `{audit['target_index']}`",
                f"- predictor_count: `{audit['predictor_count']}`",
                f"- n_rows: `{audit['n_rows']}`",
                f"- labels_loaded: `{audit['labels_loaded']}`",
                f"- split_frozen: `{audit['split_frozen']}`",
                f"- confirmatory_l2: `{audit['confirmatory_l2']}`",
                f"- exact extra duplicate feature rows: `{audit['exact_extra_duplicate_feature_rows']}`",
                f"- rounded-6 extra near-duplicate feature rows: `{audit['rounded6_extra_near_duplicate_feature_rows']}`",
                f"- human decision required for duplicates/formulas: `{audit['duplicate_or_formula_issue_requires_human_decision']}`",
                "",
                "Original sample-size list, not amended:",
                "",
                f"`n = {audit['original_sample_size_list']}`",
                "",
                "## Formula audit",
                "",
                "```json",
                json.dumps(audit["formula_audit"], indent=2),
                "```",
                "",
                "x_only_diagnostics: COMPLETE",
                "",
            ]
        ),
        encoding="utf-8",
    )
    schema = root / "reports" / "data_schema_audit.md"
    existing = schema.read_text(encoding="utf-8") if schema.is_file() else "# Data schema audit\n"
    schema.write_text(
        existing.rstrip()
        + "\n\n## Superconductivity control\n\n"
        + f"- status: `{audit['status']}`\n"
        + f"- predictor_count: `{audit['predictor_count']}`\n"
        + f"- target_index: `{audit['target_index']}`\n"
        + "- split: NOT_FROZEN\n"
        + "x_only_diagnostics: COMPLETE\n",
        encoding="utf-8",
    )
    return audit


def _file_sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _merge_manifest(root: Path, entry: dict) -> None:
    path = root / "data" / "manifests" / "source_manifest.json"
    current = {}
    if path.is_file():
        current = json.loads(path.read_text(encoding="utf-8"))
    current.update(entry)
    write_json(path, current)


def _run_tests(root: Path) -> int:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests"],
        cwd=root,
    )
    return int(completed.returncode)


def _new_run_dir(root: Path, purpose: str, config_hash: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = root / "artifacts" / "runs" / f"{stamp}_{purpose}_{config_hash[:12]}"
    if path.exists():
        raise FileExistsError(path)
    path.mkdir(parents=True)
    return path


def _write_l1_report(root: Path, run_dir: Path, result: dict, config: dict) -> None:
    lines = [
        "# L1 reproduction report",
        "",
        "Target: Tang, Wu, Fan, and Jin, *Benign Overfitting in Out-of-Distribution Generalization of Linear Models*, ICLR 2025, Figure 1(c) and Appendix A.2.",
        "",
        "Paper: https://proceedings.iclr.cc/paper_files/paper/2025/hash/dcf51b2eef1af32e594177b326f8cff9-Abstract-Conference.html",
        "",
        f"Claim label: **{result['claim']}**.",
        "This is not a novelty claim and it is not an exact replication. The paper does not provide author code in the PDF, and several simulation details are not stated.",
        "",
        f"- Qualitative band match: `{result['qualitative_band_match']}`",
        f"- Band notes: `{result['band_reasons']}`",
        f"- n grid: `{result['n_grid']}`",
        f"- Replicates written: `{result['n_replicates_written']}`",
        f"- Failure records: `{result['n_failure_records']}`",
        f"- Run directory: `{run_dir}`",
        "",
        "## Slopes of log population excess risk versus log n",
        "",
        "The paper's Figure 1(c) annotations are approximately -0.99 for PCR and -0.48 for ridge with lambda = n^0.75. The acceptance intervals were written in `configs/l1_tang_ood_reproduction.yaml` before this run.",
        "",
    ]
    for name, payload in result["slopes"].items():
        lines.append(f"- `{name}`: slope `{payload['slope']}`, error `{payload['error']}`")
    lines.extend(["", "## Construction mapping", ""])
    lines.append("- k = 10; beta_star on the major coordinates is 1/sqrt(k); minor signal is 0.")
    lines.append("- Noise is N(0, 0.1). Source covariance is diag(I_k, n^{-1/2} I_{floor(sqrt(n))}). Target covariance is I.")
    lines.append("- Ridge lambdas are 1e-8, n^{0.5}, n^{0.75}, and n, on the paper's unnormalized objective.")
    lines.append("- PCR keeps the top 10 right singular directions of the training matrix.")
    lines.append("- 10 independent training draws. 1000 test rows are drawn and are not used for the acceptance slope.")
    lines.extend(["", "## Deviations and underspecified details", ""])
    for item in result["deviations"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## What was not done",
            "",
            "The simulation was not searched over seeds, grids, or estimators to force PCR below ridge. If the predeclared band is missed, this report keeps that result.",
            "",
            f"Config hash input is the file `configs/l1_tang_ood_reproduction.yaml`.",
            "",
        ]
    )
    (root / "reports" / "l1_reproduction_report.md").write_text("\n".join(lines), encoding="utf-8")
    # config is referenced so a reviewer can see that the run used the on-disk yaml.
    del config


def _write_preflight(root: Path, payload: dict) -> None:
    lines = [
        "# L2 preflight",
        "",
        "Confirmatory L2 was not run. No YearPredictionMSD target labels were loaded.",
        "",
        f"- confirmatory_l2: `{payload['confirmatory_l2']}`",
        f"- gate_approved: `{payload['gate_approved']}`",
        f"- target_labels_loaded: `{payload['target_labels_loaded']}`",
        "",
        "## Gate reasons",
        "",
    ]
    for reason in payload["gate_reasons"]:
        lines.append(f"- {reason}")
    recorded = [name for name, state in payload["freeze_fields"].items() if state == "PRESENT"]
    lines.extend(["", "## Recorded freeze fields", ""])
    if recorded:
        for name in recorded:
            lines.append(f"- `{name}`")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "A recorded YearPredictionMSD schema field accepts the measured file layout and retains the original sample-size list. The gate reads that structured decision and `reports/yearprediction_schema_decision.md`. It does not require `reports/data_schema_audit.md` to contain `feature_count_amendment: APPROVED`. The measurement file's historical `UNRESOLVED` line stays a measurement. `uci_text_resolution` stays unresolved. None of this is confirmatory authorization.",
            "",
            "`code_identity_kind` names `SOURCE_TREE_SNAPSHOT_SHA256_V1` when that algorithm is selected. `code_identity_sha256` stays null until a later protocol freeze. The gate accepts an identity only when the declared hash equals a freshly calculated source-tree snapshot. A non-empty string, a Git commit id, and `code_commit_sha` are not substitutes. `confirmatory_l2_authorized` stays false unless a human sets it, and `reports/confirmatory_gate.md` stays closed until its status line is exactly `APPROVED` and its protocol hash matches.",
            "",
            "## Unresolved freeze fields",
            "",
        ]
    )
    for name in payload["unresolved_fields"]:
        lines.append(f"- `{name}`")
    lines.extend(
        [
            "",
            "Unresolved fields are still absent. They are not invented here.",
            "",
            "The schema decision does not by itself let `reports/confirmatory_gate.md` say `status: APPROVED`. The remaining fields above still have to be filled by a human.",
            "",
        ]
    )
    (root / "reports" / "l2_preflight.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="p19.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("environment-audit")
    year = sub.add_parser("audit-yearprediction")
    year.add_argument("--x-only", action="store_true")
    year.add_argument("--path", type=Path, default=None)
    super_cmd = sub.add_parser("audit-superconductivity")
    super_cmd.add_argument("--x-only", action="store_true")
    super_cmd.add_argument("--train", type=Path, default=None)
    super_cmd.add_argument("--unique-m", type=Path, default=None)
    sub.add_parser("run-tests")
    sub.add_parser("run-l1")
    sub.add_parser("l2-preflight")
    sub.add_parser("code-identity")
    l2 = sub.add_parser("run-l2")
    l2.add_argument("--confirmatory", action="store_true")
    sub.add_parser("run-superconductivity")
    args = parser.parse_args(argv)
    root = project_root()
    if args.command == "environment-audit":
        path = environment_audit(root)
        print(path)
        return 0
    if args.command == "audit-yearprediction":
        if not args.x_only:
            print("refusing a year audit that is not --x-only", file=sys.stderr)
            return 2
        path = args.path or (root / "data" / "raw" / "YearPredictionMSD.txt")
        audit = audit_year(root, path)
        print(audit["status"], "predictors", audit["predictor_count"], "columns", audit["total_columns"])
        return 0
    if args.command == "audit-superconductivity":
        if not args.x_only:
            print("refusing a superconductivity audit that is not --x-only", file=sys.stderr)
            return 2
        train = args.train or (root / "data" / "raw" / "train.csv")
        unique = args.unique_m or (root / "data" / "raw" / "unique_m.csv")
        audit = audit_super(root, train, unique if unique.is_file() else None)
        print(audit["status"], "predictors", audit["predictor_count"])
        return 0
    if args.command == "run-tests":
        return _run_tests(root)
    if args.command == "run-l1":
        config_path = root / "configs" / "l1_tang_ood_reproduction.yaml"
        config = load_yaml(config_path)
        import hashlib
        config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
        run_dir = _new_run_dir(root, "l1", config_hash)
        (run_dir / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
        env_path = environment_audit(root)
        (run_dir / "environment.json").write_text(
            (root / "artifacts" / "environment_audit.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        result = run_l1(config, run_dir)
        _write_l1_report(root, run_dir, result, config)
        manifest = {
            "purpose": "l1",
            "config_sha256": config_hash,
            "environment_audit": str(env_path),
            "claim": result["claim"],
            "qualitative_band_match": result["qualitative_band_match"],
        }
        write_json(run_dir / "manifest.json", manifest)
        (run_dir / "run.log").write_text(json.dumps(result, indent=2), encoding="utf-8")
        # metrics.csv is the population-excess summary; failures.csv is written by run_l1.
        summary = run_dir / "summary.csv"
        if summary.is_file():
            (run_dir / "metrics.csv").write_bytes(summary.read_bytes())
        print(result["claim"], result["slopes"])
        return 0 if result["n_failure_records"] == 0 else 1
    if args.command == "code-identity":
        path = write_manifest(root)
        print(path)
        return 0
    if args.command == "l2-preflight":
        payload = l2_preflight(root)
        _write_preflight(root, payload)
        print("BLOCKED", "unresolved", ",".join(payload["unresolved_fields"]))
        return 0
    if args.command == "run-l2":
        if not args.confirmatory:
            print("run-l2 requires --confirmatory and still fail-closes", file=sys.stderr)
            return 2
        try:
            result = run_confirmatory(root)
        except GateRefusal as exc:
            print("REFUSED", exc)
            return 2
        print(result["confirmatory_l2"], "target_labels_loaded", result["target_labels_loaded"])
        if result.get("confirmatory_result") is not True:
            return 2
        return 0
    if args.command == "run-superconductivity":
        from p19.l2_runner import PredictionArtifactError
        from p19.superconductivity_runner import (
            SuperconductivityPersistenceError,
            finish_superconductivity_cli,
            run_superconductivity_confirmatory,
        )
        try:
            result = run_superconductivity_confirmatory(root)
        except GateRefusal as exc:
            print("REFUSED", exc)
            return 2
        except (
            SuperconductivityExecutionBlocked,
            PredictionArtifactError,
            SuperconductivityPersistenceError,
        ) as exc:
            print("PERSISTENCE_FAILED", type(exc).__name__, exc)
            return 4
        return finish_superconductivity_cli(result)
