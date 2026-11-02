"""Task 2: extended-precision check when a wider floating type is available."""

from __future__ import annotations

import importlib.util

import numpy as np

from analysis.posthoc.common import RESULTS, ensure_results, print_map, write_markdown


def main() -> None:
    print_map()
    ensure_results()
    longdouble_is_float64 = np.dtype(np.longdouble) == np.dtype(np.float64)
    mpmath_available = importlib.util.find_spec("mpmath") is not None
    lines = [
        "# Extended-precision check",
        "",
        "Target cell: Superconductivity repetition 5, n=73, minimum-norm, same retained directions as the float64 fit.",
        "",
        f"numpy.longdouble dtype: `{np.dtype(np.longdouble)}`.",
        f"longdouble is float64 on this build: {longdouble_is_float64}.",
        f"mpmath importable: {mpmath_available}.",
        "",
    ]
    if not mpmath_available and longdouble_is_float64:
        lines.append(
            "No extended-precision arithmetic is available. mpmath is not installed, and this Windows NumPy build implements longdouble as float64. "
            "The relative change ⟨Z⟩ was therefore not computed. No package was installed."
        )
    else:
        lines.append("An extended-precision backend is present, but this script currently records availability only.")
    write_markdown(RESULTS / "extended_precision.md", "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
