# embryoBiopsy3D 

[![GitHub Repo](https://img.shields.io/badge/GitHub-embryoBiopsy3D-181717?logo=github)](https://github.com/YNemokun/embryoBiopsy3D)

[![CI](https://github.com/YNemokun/embryoBiopsy3D/actions/workflows/python-package.yml/badge.svg)](https://github.com/YNemokun/embryoBiopsy3D/actions/workflows/python-package.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

`embryoBiopsy3D` is a Python library that constructs an spherical embryo, placing cells with respect to their lineage history and simulating spatial dispersal movements. It then allows biopsy operations, including rebiopsy of the same embryo, and returns categorized results of the sample. This simulation aims to assist understanding and improvement of IVF's genetic testing process.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[test]"
```

Same as: `make install` (from the repo root).

For the interactive visualization demo, install the optional visualization stack:

```bash
pip install -e ".[viz]"
```

## Primary API

Typical workflows build a lineage tree, place cells on the sphere, then run biopsy / rebiopsy logic. The functions below are the usual entry points (import from the submodules shown).

**Lineage and geometry** (`embryobiopsy3d.lineage_simulator`)

- **`generate_tree`** — Build a binary division tree to a fixed number of generations (structure only; positions not set until placement).
- **`build_embryo`** — Construct an **`Embryo`**: optional tree generation, meiotic/mitotic error assignment, and spherical placement of leaves (or supply fixed coordinates).
- **`apply_error_rates`** / **`reset_flags`** — Draw aneuploidy on an existing tree and clear flags between repeated trials (used by rebiopsy batch helpers).

**Biopsy sampling** (`embryobiopsy3d.biopsy`)

- **`Sampling`** — Pick spatial clusters of cells on the sphere and **categorize** a sample (euploid / mosaic / aneuploid).

**Rebiopsy simulation** (`embryobiopsy3d.rebiopsy`)

- **`rebiopsy_single_embryo`** — Two biopsies on one embryo at a target angular separation; returns concordance and category metadata (used in notebooks and one-off analyses).
- **`rebiopsy_at_error_rate`** — Run many trials with fixed meiotic/mitotic rates and shared or cached geometry; returns a list of per-trial result dicts.
- **`simulate_experiment`** — Higher-level sweep over dispersal, rebiopsy distance, and sampled error rates (large batch summaries).


## Tests

With the package installed in editable mode, run:

```bash
pytest tests/
```

Same as `make test`

## Visualization Demo

Launch the browser demo with:

```bash
streamlit run app/streamlit_app.py
```

The demo shows:

- a 3D embryo view with sampled rebiopsy leaves highlighted
- a 2D lineage tree aligned to the same embryo state
- controls for dispersal, rebiopsy distance, random meiotic/mitotic errors, and generation-targeted aneuploidy

Visualization helpers live in `embryobiopsy3d.visualization`:

- `scene.py` builds reusable, serializable embryo / biopsy scene data
- `plotly_views.py` turns those scenes into Plotly figures for the Streamlit app

## Dependencies

- numpy
- scipy
- pandas (optional, for the visualization demo)
- plotly (optional, for the visualization demo)
- streamlit (optional, for the visualization demo)

## File Overview

- **lineage_simulator.py** — Builds a binary lineage tree, generates/annotates aneuploidy, and places leaf cells on a sphere
- **biopsy.py** — Sampling helpers for selecting cell clusters on the embryo surface
- **rebiopsy.py** — Rebiopsy simulation
- **visualization/scene.py** — Reusable scene-building helpers for embryo, lineage, and rebiopsy views
- **visualization/plotly_views.py** — Plotly figure builders used by the interactive demo
- **app/streamlit_app.py** — Streamlit entrypoint for the browser-shareable visualization demo

## Terminology (simulation)

- **Mosaic**: any sample that contains both euploid and aneuploid cells is labeled mosaic (no percentage threshold in the current assignment logic). Number of aneuploidy cells for each biopsy are included in the trial data.
- **Dispersal**: controls how far daughter cells move from the parent-centered ideal after division (placement metric).
- **Distance**: minimum separation between first and second biopsy centers (sampling metric).

Trials can reuse one tree structure with flags reset between trials; independence depends on that reset being correct.
