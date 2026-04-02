# MFA-Temoa Quickstart

This repository provides a stable Conda environment and onboarding steps for running Temoa from the submodule at `2_Temoa_submodule/`.

## 1. Clone and Pull Submodules

If you cloned this repo without `--recurse-submodules`, run:

```bash
cd MFA-Temoa
git submodule update --init --recursive
```

## 2. Create and Activate Environment

Run from the `MFA-Temoa` root:

```bash
conda env create -f environment.yml
conda activate temoa-py3
```

## 3. Initialize the Sample Database

Before the first run, build `utopia.sqlite` from `utopia.sql`:

```bash
sqlite3 ./2_Temoa_submodule/data_files/example_dbs/utopia.sqlite < ./2_Temoa_submodule/data_files/example_dbs/utopia.sql
```

## 4. Run the Sample Configuration

```bash
python ./2_Temoa_submodule/main.py --config ./2_Temoa_submodule/data_files/my_configs/config_sample.toml
```

If the run succeeds, Temoa will create a timestamped output folder under `2_Temoa_submodule/output_files/`.

## What This Environment Pins and Why

This Conda setup is intentionally pinned to avoid the exact incompatibilities encountered during first-time setup on macOS arm64 with Python 3.12:
`numpy<2`, `setuptools<81`, `pandas<3`, `pyam-iamc>=3.3,<4`, and `salib<1.5`.

These pins keep the current Temoa + Pyomo + Gravis + pyam workflow stable (including Excel export) and prevent known import/runtime breakages caused by newer upstream package changes.

## Troubleshooting

For known errors, causes, and exact fix commands, see:

- [TROUBLESHOOTING.md](./docs/TROUBLESHOOTING.md)
