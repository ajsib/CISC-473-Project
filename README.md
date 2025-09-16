# Face Restoration: GFPGAN vs CodeFormer

## Purpose

Comparative study of two face-restoration models—GFPGAN and CodeFormer—on synthetic degradations and real historical photos. Measure the trade-off between perceptual quality and identity preservation using PSNR/SSIM/LPIPS and ArcFace cosine similarity. Deliver a reproducible pipeline, analysis, and research-style report.

## Course Admin (CISC 473 — Fall 2025)

* Course: Deep Learning (CISC 473), Queen’s University
* Meeting: Tue 9:30–10:30, Thu 8:30–9:30, Fri 10:30–11:30 (KINES 100)
* Project weight: 20%
* Key dates (America/Toronto):

  * Proposal: September 15, 2025
  * Midterm project report: October 18, 2025
  * Final deliverables (paper ≥6 pages PDF, code, results, presentation): November 30, 2025

## Project Objectives

1. Build a consistent evaluation pipeline for GFPGAN and CodeFormer on aligned face crops.
2. Create reproducible synthetic degradations with seeded manifests (blur, JPEG, noise).
3. Quantify fidelity, perceptual quality, and identity retention across model settings.
4. Curate qualitative comparisons on historical public-domain images.
5. Produce a research-style report, figures, code, and a minimal demo.

## Tentative Architecture

* `src/detect_align.py`: face detection+alignment (MTCNN), 256×256 crops.
* `src/degrade.py`: seeded synthetic degradations and manifest generation.
* `src/run_gfpgan.py` / `src/run_codeformer.py`: batched inference; CodeFormer exposes fidelity knob `w`.
* `src/metrics_pixel.py`: PSNR/SSIM.
* `src/metrics_lpips.py`: LPIPS.
* `src/metrics_identity.py`: ArcFace embeddings and cosine similarity.
* `scripts/eval_all.sh`: matrix executor (methods × presets × fidelity).
* `notebooks/`: sanity checks, identity eval, user-study aggregation.
* `results/`: outputs, tables, figures, logs.

## Minimal Repo Layout

```
project/
  README.md
  run.sh
  config.json
  env.yml
  .gitignore
  src/
    detect_align.py
    degrade.py
    run_gfpgan.py
    run_codeformer.py
    metrics_pixel.py
    metrics_lpips.py
    metrics_identity.py
  scripts/
    eval_all.sh
    make_synth_testset.sh
    make_figures.sh
  data/
    raw/
    processed/
    historical/
    manifests/
  results/
    outputs/
    tables/
    figures/
    logs/
  notebooks/
    sanity_checks.ipynb
    identity_eval.ipynb
    user_study_agg.ipynb
  docs/
    TASK_BOARD.md
```

## Onboarding

* Primary stack: Python 3.10–3.11, PyTorch 2.3.x, Conda environment.
* Execution targets: Local GPU, Google Colab, Kaggle, CPU fallback.
* Determinism: fixed seeds, deterministic CuDNN settings.

## System Package Prereqs

Linux (APT):

```
sudo apt update && sudo apt install -y git wget ffmpeg build-essential libgl1
```

Linux (DNF):

```
sudo dnf install -y git wget ffmpeg @development-tools mesa-libGL
```

macOS (Homebrew):

```
brew update
brew install git wget ffmpeg
```

Windows:

* Use WSL2 (Ubuntu recommended) and follow APT steps above; or Git Bash for Git-only tasks.
* For native Windows Python, install Miniconda and ensure `conda` is on PATH.

## Conda Environment

* Conda required. If missing, install Miniconda:

  * Linux/macOS:

    ```
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    eval "$($HOME/miniconda/bin/conda shell.bash hook)"
    conda init
    ```

    (Use the macOS installer URL on macOS.)
* Environment creation (auto-run by `run.sh`):

  ```
  conda env create -f env.yml
  conda activate face-restore
  ```

## Milestones and Roadmap

* Week of Sep 16: Repo online, baseline inference both models on 50–100 images, metric smoke tests.
* Week of Sep 23: Full synthetic evaluation (≥1k images), standardized panels, initial identity tables.
* Week of Sep 30: Historical gallery, draft midterm figures.
* Oct 18: Midterm report submitted.
* Oct 19–Nov 8: Fidelity sweeps, optional light fine-tuning, updated tables/plots.
* Nov 1–15: User study run and aggregation.
* Nov 15–29: Final report, cleaned repo, slides.
* Nov 30: Final submission.

## Contributors

* Monica Stef
* Enqi Liang
* Aidan Sibley
