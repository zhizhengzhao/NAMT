# NAMT

NAMT reconstructs a scattering field from detector hits using a calibrated instrument response and a momentum-marginalized likelihood.

## Structure

| Path | Content |
|---|---|
| `simulation/` | Geant4 detector simulation and dataset conversion |
| `namt/` | PoCA, ASR, MLSD-EM, NAMT-3P, and NAMT-4P |
| `evaluation/` | Reconstruction loading, AUC, CNR, and visualization |
| `data/` | Detector-hit datasets |
| `assets/` | Momentum spectrum and blank calibrations |
| `configs/` | Reconstruction configuration |
| `reconstructions/` | Two-dimensional reconstruction arrays |

## Installation

```bash
pip install -r requirements.txt
```

## Simulation

The simulation requires Geant4, ROOT, yaml-cpp, and CRY.

```bash
cmake -S simulation -B simulation/build -DCRY_ROOT=/path/to/cry
cmake --build simulation/build -j
MUPOS_BUILD_DIR=simulation/build NPROC=24 bash simulation/run_all.sh 0 110 1000000
```

```bash
python simulation/dataset.py convert \
  --glob 'simulation/work/A/u_pb_soil/root_file/job_*.root' \
  --out data/seed_42/A_u_pb_soil.npz \
  --target-count 200000 \
  --selection-seed 42 \
  --expected-nbeam 1000000 \
  --expected-job-count 110
```

## Reconstruction

```bash
python calibrate.py --device cuda:0
python run.py --method namt_3p --scene u_pb_soil --condition 150k_1mm --seed 42 --device cuda:0
python run_all.py --device cuda:0
```

## Evaluation

```bash
python -m evaluation.evaluate \
  --input-root reconstructions \
  --runs-output metrics.csv \
  --summary-output metrics_summary.csv
```

```bash
python -m evaluation.plot \
  --input-root reconstructions \
  --output-root figures
```

The mathematical formulation is given in [THEORY.md](THEORY.md).
