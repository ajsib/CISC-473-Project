# Face Restoration: GFPGAN vs CodeFormer — Implementation Overview

## 1. Project Goal

Comparative study of two pretrained face restoration models (GFPGAN and CodeFormer) on CelebA, using a deterministic pipeline with seven stages (S1–S7):

* S1: data ingestion
* S2: alignment verification
* S3: synthetic degradation
* S4A: GFPGAN inference on raw aligned CelebA
* S4B: CodeFormer inference on synthetically degraded aligned faces
* S5: metric evaluation on a unified aligned output space
* S6: visualization
* S7: logging and provenance

GFPGAN consumes the aligned CelebA faces directly and produces restored faces.
CodeFormer consumes degraded aligned faces from S3.
All evaluation occurs in the 256×256 canonical alignment space.

The codebase is organized around these stages and driven by a single CLI entry point.

---

## 2. Repository Layout (Top Level)

Root directories and files:

* `config.json`
  Global configuration and experiment matrix.

* `env.yml`
  Conda environment specification.

* `data/`
  Immutable ground-truth aligned CelebA images and metadata.

  * `img_align_celeba/` — aligned images (202,599 JPEGs)
  * `list_attr_celeba.csv`
  * `list_bbox_celeba.csv`
  * `list_eval_partition.csv`
  * `list_landmarks_align_celeba.csv`
  * Dataset notes and planning files

* `results/`
  All derived artifacts.

  * `outputs/` — LQ images and restored outputs
  * `tables/` — CSV metrics
  * `figures/` — PNG plots
  * `logs/` — per-stage logs and manifests

* `docs/`
  Written material (proposal, reports, diagrams).

* `notebooks/`
  Sanity checks and analysis notebooks.

* `src/`
  Pipeline implementation and CLI.

* `scripts/`
  Legacy Bash utilities superseded by the CLI.

* `README.md`
  Setup and usage instructions.

---

## 3. `src/` Layout and Stage Modules

The `src/` directory contains one subdirectory per pipeline stage, plus shared utilities and the CLI.

* `src/`

  * `cli/`
    CLI entry point invoking stages with `config.json`.

  * `s1_data/`
    Data presence and integrity checks. No modification to the dataset.

  * `s2_align/`
    Verifies that CelebA images are already in the canonical 256×256 alignment using landmarks.

  * `s3_degrade/`
    Synthetic degradation of aligned CelebA images according to presets in `config.json`.
    Produces:

    * `results/outputs/s3-degrade/<preset>/imgs/`
    * `results/logs/s3_degrade_manifest.csv`

  * `s4_gfpgan/`
    Pretrained GFPGAN inference on the aligned CelebA images.
    Outputs restored aligned faces:

    * `results/outputs/s4-gfpgan/<preset>/imgs/`
    * `results/logs/s4_gfpgan_manifest.json`

  * `s4_codeformer/`
    Pretrained CodeFormer inference using fidelity values from `config.json`, consuming degraded aligned images:

    * `results/outputs/s4-codeformer/<preset>/w_<value>/`
    * `results/logs/s4_codeformer_manifest.json`

  * `s5_metrics/`
    Computes PSNR, SSIM, LPIPS, and ArcFace identity on the aligned outputs of both models.

  * `s6_figures/`
    Produces figure-ready PNGs: panels, bar charts, trade-off plots.

  * `s7_logging/`
    Aggregates manifests, config hash, environment summary into a unified run manifest.

  * `utils/`
    Shared helpers for config handling, logging, paths, and metric preparation.

---

## 4. CLI Design and Usage

Single entry point: `src/cli/main.py`.

Usage:

* Full pipeline:
  `python -m src.cli.main`

* Single stage:
  `python -m src.cli.main --stage s3`

* Sequential up to a stage:
  `python -m src.cli.main --up-to s5`

Characteristics:

* Defaults to running S1→S7
* Stages are idempotent within a run and overwrite previous results
* Strict use of `config.json`
* Pipeline aborts on structural errors
* Reports stage status, counts, output paths, and timing

---

## 5. Configuration and Environment

### 5.1 `config.json`

Central configuration including:

* metadata
* versions and environment details
* deterministic seeds
* GFPGAN and CodeFormer upstreams
* CelebA path settings
* synthetic degradation presets
* fidelity grid for CodeFormer
* metric and figure settings
* experiment matrix across methods and presets

### 5.2 `env.yml`

Canonical conda environment:

* Python 3.10
* PyTorch 2.3.1 + CUDA 11.8
* facenet-pytorch, lpips, opencv, scikit-image, pandas, PyYAML, tqdm, matplotlib, pillow

---

## 6. Data and Results Contracts

### 6.1 Data Contract

* `data/` is immutable.
* No stage mutates CelebA or CSVs.
* S3 uses metadata but preserves it unmodified.

### 6.2 Results Contract

All output lives under `results/`.

* **S1**
  `results/outputs/s1-validated-pruned-dataset/`

* **S2**
  `results/outputs/s2-processed-size-bb/`

* **S3**
  `results/outputs/s3-degrade/<preset>/imgs/`
  `results/logs/s3_degrade_manifest.csv`

* **S4A (GFPGAN)**
  `results/outputs/s4-gfpgan/<preset>/imgs/`
  `results/logs/s4_gfpgan_manifest.json`

* **S4B (CodeFormer)**
  `results/outputs/s4-codeformer/<preset>/w_<value>/`
  `results/logs/s4_codeformer_manifest.json`

* **S5**
  `results/tables/*.csv`

* **S6**
  `results/figures/*.png`

* **S7**
  `results/logs/run_manifest.json`

---

## 7. Stage-by-Stage Summary

### S1 Data (Ingestion and Verification)

Checks dataset presence, counts, and metadata consistency.
Produces a summary log.

### S2 Align (Verification Only)

Verifies CelebA alignment and canonical cropping through landmark checks.

### S3 Degrade (Synthetic LQ Generation)

Applies Gaussian blur, JPEG compression, and additive noise to aligned CelebA.
Creates degraded aligned LQ sets and a manifest linking GT→LQ.

### S4A GFPGAN (Pretrained Inference)

Runs GFPGAN on the aligned CelebA images.
Outputs restored aligned faces.
Logs per-sample results and runtime metadata.

### S4B CodeFormer (Pretrained Inference with Fidelity Sweep)

Runs CodeFormer on each degraded aligned image for each fidelity value.
Produces aligned restored outputs for direct comparison.

### S5 Metrics (Pixel, Perceptual, Identity)

Computes:

* PSNR, SSIM
* LPIPS
* ArcFace cosine similarity

Uses aligned GT and aligned restored outputs from both models.
Produces both sample-level CSVs and aggregated summaries.

### S6 Figures (Panels and Plots)

Generates:

* GT / LQ / GFPGAN / CodeFormer composite panels
* bar charts for PSNR/SSIM
* perceptual vs identity trade-off plots

### S7 Logging (Provenance Capture)

Consolidates:

* stage manifests
* configuration hash
* runtime environment info
* seed state

Writes a unified `run_manifest.json`.

---

## 8. Execution Model and Assumptions

* Target: local workstation with GPU
* CPU fallback possible
* Stages assume upstream success
* One experiment configuration at a time
* `config.json` is the single definition of the active experiment

This structure maintains a clear, deterministic pipeline for aligned face restoration, supports controlled synthetic degradations, and ensures that both GFPGAN and CodeFormer are evaluated in a unified, directly comparable canonical space.
