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

For the interactive visualization demo, install the optional visualization stack:

```bash
pip install -e ".[viz]"
```

## Quickstart

### Python

Build an embryo, mark an aneuploid subtree, and run a rebiopsy:

```python
from embryobiopsy3d.lineage_simulator import build_embryo
from embryobiopsy3d.rebiopsy import rebiopsy_single_embryo

embryo = build_embryo(
    generations=8,
    meio_rate=0.0,
    mito_rate=0.0,
    placement_dispersal=0.25,
    seed=7,
)
embryo.set_aneuploid_by_generation_index(4, 0, is_aneuploid=True, include_subtree=True)

result = rebiopsy_single_embryo(embryo, distance=0.5, return_metadata=True, seed=11)
print(result["standard_category"], "->", result["second_category"],
      "match:", result["match"])
```

For random (rather than targeted) aneuploidy, pass non-zero `meio_rate` /
`mito_rate`. See the API section below for details.

### Command line

The package installs an `embryobiopsy3d` CLI with two subcommands:

```bash
embryobiopsy3d demo  --mito-rate 0.15 --dispersal 0  # sanity check
embryobiopsy3d sweep --n-trials 100 --out-dir out/   # small parameter sweep
```

`demo` builds one embryo, runs a single biopsy, and prints a compact summary.
`sweep` reproduces the full parameter sweep used for the paper figures,
writing `rebiopsy_trials.csv` and `rebiopsy_transition_summary.csv` into
`--out-dir`.

Run `embryobiopsy3d --help`, `embryobiopsy3d demo --help`, or
`embryobiopsy3d sweep --help` for the full list of options.

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
pytest -n 4 --verbose
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

## Deploying the demo (Streamlit Community Cloud)

The repo is ready to deploy on [Streamlit Community Cloud](https://share.streamlit.io)
for free, with auto-redeploy on every `git push`.

1. Sign in to Streamlit Community Cloud with GitHub and click **New app**.
2. Point it at this repo, pick the branch (e.g. `main`), and set the main file to
   `app/streamlit_app.py`.
3. In **Advanced settings**, select Python **3.12** (to match `requires-python`
   in `pyproject.toml`).
4. Click **Deploy**.

Streamlit Cloud installs from the top-level `requirements.txt`, which pulls in
`numpy`, `scipy`, `pandas`, `plotly`, `streamlit`, and installs the local
`embryobiopsy3d` package in editable mode (`-e .`) so the app's
`from embryobiopsy3d.visualization...` imports resolve. After the first deploy,
editing the live app is just: commit locally, push to the chosen branch, and
the app rebuilds automatically.

## File Overview

- **lineage_simulator.py** — Builds a binary lineage tree, generates/annotates aneuploidy, and places leaf cells on a sphere
- **biopsy.py** — Sampling helpers for selecting cell clusters on the embryo surface
- **rebiopsy.py** — Rebiopsy simulation
- **visualization/scene.py** — Reusable scene-building helpers for embryo, lineage, and rebiopsy views
- **visualization/plotly_views.py** — Plotly figure builders used by the interactive demo
- **app/streamlit_app.py** — Streamlit entrypoint for the browser-shareable visualization demo

## Terminology (simulation)

- **Mosaic**: any sample that contains *both* euploid and aneuploid cells is labeled as mosaic (there is no percentage threshold in the current assignment logic). Number of aneuploidy cells within each biopsy are included in the trial data.
- **Dispersal**: controls how far daughter cells move from the parent-centered ideal after division (placement metric).
- **Distance**: minimum separation between first and second biopsy centers (sampling metric).

Trials can reuse one tree structure with flags reset between trials; independence depends on that reset being correct.
