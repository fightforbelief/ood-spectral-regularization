"""Build both review manuscripts, their supplements, and the editor's letter."""
from hashlib import sha256
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# Explicit public inputs keep local archives and identifying notes out of review files.
MODELS = ("__init__", "code_identity", "data_loading", "estimators", "gates",
          "l2_pipeline", "l2_sampling", "metrics", "preprocessing", "protocol",
          "superconductivity_split", "tuning")
ANALYSES = ("cutoff.py", "replay.py", "summarize.py", "diagnostics.py", "test_rank.py",
            "losses.csv", "cutoff.csv", "provenance.json", "spectra_YearPredictionMSD.csv",
            "spectra_Superconductivity.csv")
SUMMARY = ("risk.csv", "paired.csv", "sensitivity.csv", "curves.pdf", "cutoffs.pdf")


def compile_pdf(directory, name, source):
    target = directory / (name + ".tex")
    target.write_text(source, encoding="utf-8")
    subprocess.run(["tectonic", "--keep-logs", target.name], cwd=directory, check=True,
                   env={**os.environ, "SOURCE_DATE_EPOCH": "1795219200"})
    log = target.with_suffix(".log").read_text(encoding="utf-8")
    if any(s in log for s in ("Overfull \\hbox", "undefined references", "Citation(s) may have changed")):
        raise ValueError(f"Check typesetting in {name}: " + "\n".join(
            line for line in log.splitlines() if "Overfull" in line or "undefined" in line))
    return target.with_suffix(".pdf")


def write_zip(path, files):
    manifest = "".join(f"{sha256(data).hexdigest()}  {name}\n" for name,data in sorted(files.items()))
    with ZipFile(path, "w") as z:
        for name, data in sorted({**files, "SHA256SUMS.txt": manifest.encode()}.items()):
            info = ZipInfo(name, date_time=(2026, 11, 21, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            z.writestr(info, data, compress_type=ZIP_DEFLATED)


def letter_source():
    text = (HERE / "cover-letter.txt").read_text(encoding="utf-8")
    text = text.replace("%", r"\%").strip()
    paragraphs = [para.replace("\n", r"\\") for para in text.split("\n\n")]
    return (r"\documentclass[12pt,letterpaper]{article}\usepackage[margin=1.1in]{geometry}"
            r"\usepackage{microtype}\pagestyle{empty}\setlength{\parindent}{0pt}\setlength{\parskip}{1em}"
            r"\begin{document}" + "\n" + "\n\n".join(paragraphs) + "\n" + r"\end{document}")


def build():
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        staging = workspace / "manuscript"
        shutil.copytree(ROOT / "manuscript", staging)
        shutil.copytree(ROOT / "replication/summary", workspace / "replication/summary")
        for name in ("review.tex", "title.tex", "tas_references.tex", "appendix.tex"):
            shutil.copyfile(HERE / name, staging / name)
        names = ([f"p19/{name}.py" for name in MODELS] + [f"replication/{name}" for name in ANALYSES]
                 + [f"replication/summary/{name}" for name in SUMMARY])
        names.append("manuscript/tables.py")
        files = {name: (ROOT / name).read_bytes() for name in names}
        pins = (ROOT / "requirements.txt").read_text().splitlines()
        files["requirements.txt"] = ("\n".join(line for line in pins if line and not line.startswith("#"))+"\n").encode()
        guide = (HERE / "supplement-guide.md").read_bytes()
        outputs = []
        for version in ("anonymous", "identified"):
            flag = r"\def\Anonymous{1}" if version == "anonymous" else ""
            outputs.append(compile_pdf(staging, f"manuscript-{version}", flag+r"\input{review}"))
            appendix = compile_pdf(staging, f"appendix-{version}", flag+r"\input{appendix}")
            credit = b"" if version == "anonymous" else b"\nPrepared by Yunjie Guo and Hanqing Yang.\n"
            package = staging / f"replication-{version}.zip"
            write_zip(package, {**files, "online-appendix.pdf": appendix.read_bytes(), "README.md": guide+credit})
            outputs.append(package)
        outputs.append(compile_pdf(staging, "cover-letter", letter_source()))
        for path in outputs:
            shutil.copyfile(path, HERE / path.name)


if __name__ == "__main__":
    build()
