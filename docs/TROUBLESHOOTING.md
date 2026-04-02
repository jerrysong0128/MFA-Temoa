# Troubleshooting

This guide maps common setup/run errors to their root causes and fixes.

## 1) `AttributeError: np.float_ was removed in the NumPy 2.0 release`

### Cause
Current Pyomo usage in this stack still references NumPy APIs removed in NumPy 2.x.

### Fix
Pin NumPy below 2.0 (already handled in `environment.yml`):

```bash
conda install -n temoa-py3 "numpy<2"
```

If already inside the env:

```bash
python -m pip install "numpy<2"
```

## 2) `ModuleNotFoundError: No module named 'pkg_resources'`

### Cause
`gravis` imports `pkg_resources`, but `setuptools` 81+ no longer ships it.

### Fix
Pin setuptools below 81 (already handled in `environment.yml`):

```bash
conda install -n temoa-py3 "setuptools<81"
python -c "import sys,pkg_resources; print(sys.executable); print(pkg_resources.__file__)"
```

## 3) `TypeError: NDFrame.to_excel() takes 2 positional arguments but 3 ... were given`

### Cause
A pandas 3.x API change conflicts with the pyam export path used during Temoa Excel output.

### Fix
Use pandas 2.x and pyam-iamc 3.x compatibility range (already pinned in `environment.yml`):

```bash
conda install -n temoa-py3 "pandas<3" "pyam-iamc>=3.3,<4"
```

## 4) `FileNotFoundError: could not locate the input database: data_files/example_dbs/utopia.sqlite`

### Cause
The sample config points to `utopia.sqlite`, but that file does not exist until you build it from `utopia.sql`.

### Fix
From `MFA-Temoa/2_Temoa_submodule`:

```bash
sqlite3 ./data_files/example_dbs/utopia.sqlite < ./data_files/example_dbs/utopia.sql
python main.py --config data_files/my_configs/config_sample.toml
```

## 5) `zsh: command not found: temoa`

### Cause
`temoa` is not a shell command; it was typed as if it were one.

### Fix
Run Python directly:

```bash
python main.py --config data_files/my_configs/config_sample.toml
```

## 6) `python: can't open file '.../MFA-Temoa/main.py': [Errno 2] No such file or directory`

### Cause
The command was run from the wrong directory. `main.py` is in `MFA-Temoa/2_Temoa_submodule`.

### Fix

```bash
cd 2_Temoa_submodule
python main.py --config data_files/my_configs/config_sample.toml
```

## Non-Fatal Warnings

These warnings can appear in successful runs:

- `InsecureKeyLengthWarning` from `jwt` about short HMAC key length.
- `pkg_resources is deprecated as an API` from `gravis` import path.

They are warnings, not run-stopping errors. The current environment keeps `setuptools<81` for compatibility until `gravis` no longer requires `pkg_resources`.
