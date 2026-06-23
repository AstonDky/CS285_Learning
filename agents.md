# CS285 Project Agent Rules

## Environment

- Always use the conda environment named `CS285` for work in this repository.
- Activate it before running any Python command:
  - `conda activate CS285`
- Create or refresh the full environment from the repository root with:
  - `conda env update -n CS285 -f environment-cs285.yml --prune`
- After the conda-managed base is present, refresh the pip-managed layer with:
  - `python -m pip install -r requirements-cs285.txt`

## Homework Layout

- Run homework commands from the specific homework directory under `homework/homework_fall2023-main/`.
- Do not install multiple homework folders with `pip install -e .` at the same time, because each homework defines the same package name: `cs285`.
- Prefer executing scripts directly inside the target homework folder, for example:
  - `cd homework/homework_fall2023-main/hw3`
  - `python cs285/scripts/run_hw3_dqn.py ...`

## Dependency Baseline

- `environment-cs285.yml` is the source of truth for rebuilding the full `CS285` environment.
- `requirements-cs285.txt` is the pip-managed dependency layer shared by `hw1` through `hw5`.
- `swig` and `box2d-py` are intentionally managed by conda for better Windows compatibility.
- On this Windows laptop with an NVIDIA RTX 5070 Ti Laptop GPU, use the GPU-capable `torch==2.7.1` build in the `CS285` environment instead of the course's older `torch==1.13.1`, because the older binaries predate RTX 50-series Blackwell GPUs.
- When new packages are needed for this repository, add them there and keep using the `CS285` environment.
