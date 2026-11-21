# Superconductivity execution snapshot

This directory is the Python package used for the Superconductivity confirmatory arm. It is not on the import path of the YearPredictionMSD package in the repository root. The root `p19/` tree stays at the frozen YearPredictionMSD source identity `04ad1f655c48bc6a4caa11b46a869f8327b7b4311eb12d47388dcb848bb395d9`.

From the staging root with that snapshot on the import path, the historical confirmatory invocation is `python -u -c "from p19.cli import main; raise SystemExit(main())" run-superconductivity`. The module only defines the entry function. It refuses to run unless the local Superconductivity authorization files are present. Official tables and prediction artifacts are not included.
