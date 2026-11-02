"""Run the post-hoc tasks. Tasks 1–3 wait on a passing regeneration check."""

from __future__ import annotations

from analysis.posthoc.extended_precision import main as extended_main
from analysis.posthoc.hyperparams import main as hyper_main
from analysis.posthoc.null_baseline import main as null_main
from analysis.posthoc.number_audit import main as audit_main
from analysis.posthoc.pool_spectrum import main as pool_main
from analysis.posthoc.regeneration import main as regeneration_main
from analysis.posthoc.spectra import main as spectra_main
from analysis.posthoc.summary import main as summary_main


def main() -> int:
    status = regeneration_main()
    audit_main()
    null_main()
    hyper_main()
    extended_main()
    if status == 0:
        spectra_main()
        pool_main()
    else:
        print("Skipping spectral and pool-spectrum tasks because regeneration did not pass.")
    summary_main()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
