# Backup Control Barrier Functions with Memory for Online Set Expansion

This repository contains the implementation of **Backup Control Barrier Functions with Memory for Online Set Expansion**

M. Yusuf Uzun and Ersin Daş

Paper: [arXiv:2610.05381](https://arxiv.org/abs/2610.05381)

## Prerequisites

- Python 3.10 or later
- Required Python libraries: `numpy`, `numba`, `scipy`, `matplotlib`, `shapely`, `pytest`
- LaTeX on the PATH, for the figures

## Usage

```
python simulate.py          # closed-loop runs of Section VI
python fig2_quadrotor.py    # Fig. 2
python fig1_regions.py      # Fig. 1
python timing.py            # computation times of Section VI
python -m pytest test_certificates.py -q
```

## Files

- `model.py`: planar quadrotor, Section VI
- `family.py`: ball family (Definition 3) and links along backup arcs (Theorem 2)
- `interval.py`: interval bounds used by `family.py`
- `simulate.py`: safety filter (19), demand-driven growth (Section V-B), closed loops
- `results/`: stored runs, figures, `summary.txt` and `timing.txt`
