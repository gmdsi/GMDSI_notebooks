# GitHub Copilot Code Review Instructions

## Excluded Paths

Do **not** review changes in the `dependencies/` folder. These contain external packages (`flopy` and `pyemu`) installed as editable dependencies and are maintained in their own upstream repositories.

## General Guidance

- This repository contains educational Jupyter notebooks for groundwater modelling with PEST++/pyEMU.
- Notebooks are committed with outputs cleared. Do not flag missing cell outputs as an issue.
- Focus reviews on correctness, clarity, and pedagogical quality of notebook content.
- PEST++ binaries in `bin_new/` are pre-compiled executables; do not flag binary files.
