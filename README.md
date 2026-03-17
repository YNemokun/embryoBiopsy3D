# EmbryoBiopsy3D

`embryoBiopsy3D` is a Python library that constructs an spherical embryo, placing cells with respect to their lineage history and simulating spatial dispersal movements. It then allows biopsy operations, including rebiopsy of the same embryo, and returns categorized results of the sample. This simulation aims to assist understanding and improvement of IVF's genetic testing process.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Usage

**TO BE IMPLEMENTED**

## Tests

With the project root on `PYTHONPATH` (via `pyproject.toml`), run:

```bash
pytest test/
```

If using conda, activate your env first: `conda activate embryo-simulator`

## Dependencies

- numpy
- scipy

## File Overview

- **lineage_simulator.py** — Builds a binary lineage tree, generates/annotates aneuploidy, and places leaf cells on a sphere
- **biopsy.py** — Sampling helpers for selecting cell clusters on the embryo surface
- **rebiopsy.py** — Rebiopsy simulation
