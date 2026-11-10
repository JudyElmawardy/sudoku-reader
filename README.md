Handwritten Sudoku Reader and Solver

A system that reads a Sudoku written with handwritten digits (MNIST-style), detects and corrects the most likely misreads using search, and solves the resulting puzzle with a CSP solver that supports Backtracking, Forward Checking and AC-3.

## Project Structure

```
sudoku_project/
├── data/               # Dataset download, preprocessing, puzzle & mosaic generators
├── nn/                 # Digit classifier (NN + logistic baseline), calibration
├── search/             # Correction search: UCS, Greedy, A*
├── csp/                # Sudoku CSP solver (BT / FC / AC-3)
├── integration/        # End-to-end pipeline
├── eval/               # Experiments and metrics
├── tests/              # Unit & integration tests
├── models/             # Saved models
├── figures/            # Plots
├── artifacts/          # Generated data, logs, results
├── make_data.py        # Reproducible data pipeline
├── run_demo.py         # Integrated demo
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Exact Run Commands

```bash
# 1. Download & preprocess data + generate puzzles/mosaics (fixed seed)
python make_data.py --seed 42

# 2. Train digit classifier + baseline
python -m nn.train --seed 42

# 3. Run unit tests
pytest tests/ -v

# 4. Integrated demo (shows input mosaic, NN reading, corrections, solution)
python run_demo.py --seed 42 --misreads 2

# 5. Run the three required experiments
python -m eval.experiments --seed 42
```

## Common Checklist Mapping

1. Source modules: `data/`, `nn/`, `search/`, `csp/`, `integration/`, `eval/`, `tests/`
2. `make_data.py` with fixed seed
3. Data dictionary: `data/DATA_DICTIONARY.md`
4. Trained NN + logistic regression baseline
5. Search with node counters, tested on small known example
6. CSP with switchable BT / FC / AC-3 + counters
7. `run_demo.py`
8. This README + pinned requirements
9. Tests (≥5 normal, 3 edge, 2 impossible)
10. Experiments (3 experiments, ≥5 seeds, mean ± std)
11. Report (to be written)
12. Presentation + live demo
13. Contribution statements + git log

## Minimum Version Covered

- NN digit classifier (1-9)
- Full CSP solver with BT + FC + AC-3
- UCS correction search (goal = puzzle becomes solvable)
- Up to 2 injected misreads
- Puzzle & mosaic generator with seed

## Optional Extensions (implemented where useful)

- A* with conflict heuristic
- Greedy baseline
- Temperature scaling for calibration
- Uniqueness check (optional flag)

## Licence / Data

- MNIST: public (Yann LeCun)
- Generated Sudoku grids and mosaics: project-specific (seeded)
