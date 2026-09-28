# NAMT

Code and data for **Nuisance-Aware Muon Tomography with Momentum Marginalization**.

## Results

[Results](results/) includes all 750 benchmark runs and 75 ablation runs:

- [Benchmark metrics](results/benchmark_metrics.csv) and [summary](results/benchmark_summary.csv)
- [Ablation metrics](results/ablation_metrics.csv) and [summary](results/ablation_summary.csv)
- [Reference images](results/fig3_reference_gallery.pdf) and [robustness plots](results/fig4_robustness.pdf)
- Tables II-V in `results/table*.tex`; underlying arrays in [reference](reference/)

At 150,000 events and 1 mm hit error, mean AUC across six scenes is:

| PoCA | ASR | MLSD | NAMT-3P | NAMT-4P |
|---:|---:|---:|---:|---:|
| 0.732 | 0.839 | 0.875 | 0.969 | 0.971 |

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python reproduce.py --device cuda:0
```

This reconstructs all 825 runs from the included hit data and writes metrics,
tables, and figures to `outputs/reports/`. Existing outputs are reused;
`--force` reruns them. Use `--suite benchmark` or `--suite ablation` for one
experiment group, `--device cpu` for CPU execution, or repeat `--device` for
multiple GPUs.

To regenerate reports from the stored reference arrays:

```bash
python reproduce.py --evaluate-only --input-root reference
```

The benchmark covers five methods, six scenes, five realizations, and five
acquisition conditions. The ablation compares the full model, Gaisser prior,
fixed momentum, nominal instrument response, and no instrument response.
Summaries report means and sample standard deviations over five realizations.

Verified with Python 3.13.2, NumPy 2.2.6, PyTorch 2.10.0/CUDA 12.9,
Matplotlib 3.10.8, and NVIDIA A100 GPUs. Floating-point results can vary across
hardware and library versions; figure fonts can also differ.

## Files

| Directory | Contents |
|---|---|
| `data/` | Detector-hit datasets for seeds 42-46 |
| `assets/` | Momentum priors and blank calibrations |
| `namt/` | Reconstruction methods and calibration |
| `ablation/` | Likelihood ablations |
| `evaluation/` | Metrics, tables, and figures |
| `reference/` | 825 reference reconstruction arrays |
| `results/` | Computed metrics, tables, and figures |
| `simulation/` | Geant4/CRY simulation and ROOT conversion |

The five realizations are sampled from shared simulation event pools.
Reconstruction uses detector hits and geometry, not truth momentum or reference images.

To rebuild blank calibrations in `outputs/calibrations/`:

```bash
python calibrate.py --device cuda:0
```

Reconstruction uses the supplied calibrations in `assets/`; the ablation
fits its calibration during reconstruction. Figures 1-2 are explanatory
illustrations; Figures 3-4 are generated from reconstruction results.

## Verify

```bash
python -m unittest discover -s tests
shasum -a 256 -c SHA256SUMS
```
