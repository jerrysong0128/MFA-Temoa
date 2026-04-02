# MFA-Temoa Quickstart

This folder provides a stable Conda environment definition for running the Temoa sample model in this workspace.

## 1. Create and Activate Environment

Run from this folder:

```bash
cd /Users/jerrysong/Study_PKU/MFA_Workspace/ODYM_Application/summer-school/MFA-Temoa
conda env create -f environment.yml
conda activate temoa-py3
```

## 2. Switch to the Temoa Project Root

`main.py` is in `summer-school/temoa`, not in `MFA-Temoa`.

```bash
cd /Users/jerrysong/Study_PKU/MFA_Workspace/ODYM_Application/summer-school/temoa
```

## 3. Initialize the Sample Database

Before the first run, build `utopia.sqlite` from `utopia.sql`:

```bash
sqlite3 ./data_files/example_dbs/utopia.sqlite < ./data_files/example_dbs/utopia.sql
```

## 4. Run the Sample Configuration

```bash
python main.py --config data_files/my_configs/config_sample.toml
```

If the run succeeds, Temoa will create a timestamped output folder under `temoa/output_files/`.

## What This Environment Pins and Why

This Conda setup is intentionally pinned to avoid the exact incompatibilities encountered during first-time setup on macOS arm64 with Python 3.12:
`numpy<2`, `setuptools<81`, `pandas<3`, `pyam-iamc>=3.3,<4`, and `salib<1.5`.

These pins keep the current Temoa + Pyomo + Gravis + pyam workflow stable (including Excel export) and prevent known import/runtime breakages caused by newer upstream package changes.

## Troubleshooting

For known errors, causes, and exact fix commands, see:

- [TROUBLESHOOTING.md](./docs/TROUBLESHOOTING.md)
