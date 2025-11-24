# Face Restoration: GFPGAN vs CodeFormer — Implementation Overview

## 1. Project Goal

Comparative study of two pretrained face restoration models (GFPGAN and CodeFormer) on CelebA-aligned faces, using a deterministic pipeline with seven stages (S1–S7):

* S1: data ingestion
* S2: alignment verification
* S3: synthetic degradation
* S4A: GFPGAN inference
* S4B: CodeFormer inference
* S5: metric evaluation
* S6: visualization
* S7: logging and provenance

The codebase is organized around these stages and driven by a single CLI entry point that can execute the full pipeline or individual stages.

---

## 2. Repository Layout (Top Level)

Root directories and files:

* `config.json`
  Global configuration and experiment matrix. Single source of truth. No CLI overrides.

* `env.yml`
  Conda environment specification. Canonical environment used for development and evaluation.

* `data/`
  Immutable ground-truth and metadata.

  * `img_align_celeba/` — aligned CelebA images (202,599 JPEGs).
  * `list_attr_celeba.csv` — per-image attributes.
  * `list_bbox_celeba.csv` — bounding boxes.
  * `list_eval_partition.csv` — train/val/test split (0/1/2).
  * `list_landmarks_align_celeba.csv` — five-point facial landmarks.
  * `README.md`, `TaskList.md` — dataset notes and planning.

* `results/`
  All derived artifacts. Overwritten on each run.

  * `outputs/` — model outputs (LQ inputs, GFPGAN, CodeFormer).
  * `tables/` — metrics tables (CSV).
  * `figures/` — plots and panels (PNG).
  * `logs/` — plain-text logs and per-stage manifests.

* `docs/`
  Written material for the course.

  * `CISC 473 Proposal.pdf`
  * `midterm-report/` (previous report and diagrams)
  * `final-report/` (this document and final writeup)
  * `TASK_BOARD.md` — task planning.

* `notebooks/`
  One-off analysis, sanity checks, and user study aggregation.

  * `sanity_checks.ipynb`
  * `identity_eval.ipynb`
  * `user_study_agg.ipynb`

* `src/`
  All executable pipeline code and CLI.

* `scripts/`
  Legacy bash stubs retained for historical reference, superseded by the Python CLI:

  * `make_synth_testset.sh`
  * `eval_all.sh`
  * `make_figures.sh`

* `README.md`
  High-level instructions for cloning, environment setup, and basic usage.

---

## 3. `src/` Layout and Stage Modules

The `src/` directory is organized into one subdirectory per stage, plus shared utilities and the CLI.

Proposed layout:

* `src/`

  * `cli/`

    * `__init__.py`
    * `main.py`
      CLI entry point. Parses arguments, loads `config.json`, dispatches to stages, and coordinates logging.
  * `s1_data/`

    * `__init__.py`
    * `stage.py`
      Validates the presence and integrity of CelebA data and CSVs. Produces a short summary and logs, but no new files.
  * `s2_align/`

    * `__init__.py`
    * `stage.py`
      Optional alignment verification using five-point landmarks. Confirms that CelebA crops are already 256×256 and aligned. No modification of `data/`.
  * `s3_degrade/`

    * `__init__.py`
    * `stage.py`
      Implements synthetic degradation (Gaussian blur, JPEG compression, Gaussian noise) according to `config.json`.
      Outputs:

      * `results/outputs/lq/<preset>/imgs/` — degraded images.
      * `results/logs/s3_degrade.log` — operational log.
      * `results/logs/s3_degrade_manifest.csv` — pairs mapping (id, path_gt, path_deg, degradation, split).
  * `s4_gfpgan/`

    * `__init__.py`
    * `stage.py`
      Batched inference for GFPGAN using the pretrained GFPGANv1.4 checkpoint.
      Inputs:

      * LQ images and manifest from S3.
        Outputs:
      * `results/outputs/gfpgan/<preset>/` — restored images.
      * `results/logs/s4_gfpgan.log` — run log.
      * `results/logs/s4_gfpgan_manifest.json` — mapping between inputs, outputs, and config hash.
  * `s4_codeformer/`

    * `__init__.py`
    * `stage.py`
      Batched inference for CodeFormer with fidelity parameter `w` swept over values from `config.json` (e.g., 0.3, 0.5, 0.7, 0.9).
      Inputs:

      * Same LQ images and manifest as S4A.
        Outputs:
      * `results/outputs/codeformer/<preset>/w_<value>/` — restored images per fidelity setting.
      * `results/logs/s4_codeformer.log`
      * `results/logs/s4_codeformer_manifest.json`
  * `s5_metrics/`

    * `__init__.py`
    * `stage.py`
      Central orchestration for PSNR/SSIM, LPIPS, and ArcFace identity metrics.
      Internally wraps:

      * `metrics_pixel.py` — PSNR, SSIM.
      * `metrics_lpips.py` — LPIPS.
      * `metrics_identity.py` — ArcFace cosine similarity.
        Inputs:
      * GT images from `data/img_align_celeba/`
      * LQ manifest from S3
      * model outputs from S4A/S4B
        Outputs:
      * `results/tables/metrics_pixel.csv`
      * `results/tables/metrics_lpips.csv`
      * `results/tables/metrics_identity.csv`
      * `results/tables/metrics_summary.csv` — aggregated mean/std per method, preset, and fidelity.
      * `results/logs/s5_metrics.log`
  * `s6_figures/`

    * `__init__.py`
    * `stage.py`
      Consumes metric tables and selected images to generate figure-ready PNGs.
      Outputs:

      * `results/figures/panel_<preset>.png` — GT vs LQ vs GFPGAN vs CodeFormer side-by-side.
      * `results/figures/bar_psnr_ssim.png` — bar chart of PSNR/SSIM across methods.
      * `results/figures/tradeoff_lpips_arcface.png` — identity vs perceptual trade-off plot.
      * `results/logs/s6_figures.log`
  * `s7_logging/`

    * `__init__.py`
    * `stage.py`
      Final consolidation of run metadata:

      * collects per-stage manifests, config hash, and environment info
      * writes a single top-level manifest:

        * `results/logs/run_manifest.json`
  * `utils/`

    * `__init__.py`
    * `config.py` — loads and validates `config.json`; exposes typed accessors for paths and experiment matrix.
    * `logging.py` — sets up a shared plain-text logger per stage and a root `results/logs/pipeline.log`.
    * `io.py` — filesystem helpers for safe read/write and path construction.
    * `metrics_common.py` — utilities shared across metric modules (normalization, tensor handling, device selection).

Legacy `src/*.py` placeholders (`degrade.py`, `detect_align.py`, `run_gfpgan.py`, `run_codeformer.py`, `metrics_*.py`) are conceptually replaced by the `stage.py` modules and internal utilities described above. Their logic is migrated, not duplicated.

---

## 4. CLI Design and Usage

The CLI is the single entry point to the pipeline, exposed through `src/cli/main.py`.

Typical usage patterns:

* Full pipeline (S1–S7), default behavior:
  `python -m src.cli.main`

* Run a specific stage only, using a `--stage` flag:
  `python -m src.cli.main --stage s3`
  `python -m src.cli.main --stage s4_gfpgan`

* Run up to a given stage sequentially:
  `python -m src.cli.main --up-to s5`

Characteristics:

* If no flags are supplied, all stages run in order.
* Stages are assumed to be idempotent within a run but overwrite previous run outputs in `results/`.
* Config is loaded strictly from `./config.json`. No runtime overrides, no environment-variable indirection.
* On any serious error (missing files, mismatch between manifest and outputs, metric failure), the CLI aborts and surfaces a clear error message. No silent skipping.

The CLI reports:

* which stage is running
* basic counts (number of images, presets, fidelities)
* where outputs are written
* high-level timing information

---

## 5. Configuration and Environment

### 5.1 `config.json`

Single global configuration file. Responsibilities:

* Project metadata (name, license, contributors).
* Stack information (Python version, PyTorch version, CUDA versions, Python packages).
* Determinism settings:

  * global seed
  * torch/cuDNN flags
  * numpy seed
  * PYTHONHASHSEED
* Upstream model references (GFPGAN and CodeFormer repos, checkpoints, optional commit pins, fidelity knob range).
* Data settings:

  * data roots (logical, not all currently used)
  * alignment policy and parameters
  * degradation presets (blur, JPEG, noise) and random seed
  * manifest schema for degraded pairs.
* Evaluation settings:

  * list of metrics
  * locations of tables and figures
  * figure ordering for panels.
* Experiment matrix:

  * methods = {gfpgan, codeformer}
  * degradations = selected subset of presets
  * codeformer fidelity values `w`.

The pipeline treats `config.json` as read-only during execution. Any change in experimental setup is performed by editing this file directly.

### 5.2 `env.yml`

Conda environment specification:

* Python 3.10
* PyTorch 2.3.1, torchvision 0.18.1, CUDA 11.8
* Core packages: facenet-pytorch, lpips, opencv, scikit-image, pandas, PyYAML, tqdm, matplotlib, pillow.

This is the canonical environment. Optional future `requirements.txt` may mirror the pip dependencies for non-conda systems, but the primary supported flow remains conda-based.

---

## 6. Data and Results Contracts

### 6.1 Data Contract

* `data/img_align_celeba/` and its CSVs are treated as immutable ground truth.
* No stage writes into `data/`.
* All derived artifacts are generated under `results/`.

The degradation stage S3 uses the original metadata (partition and landmarks) but never mutates these files.

### 6.2 Results Contract

Results are logically partitioned by stage:

* S3 — synthetic degradation

  * `results/outputs/lq/<preset>/imgs/`
  * `results/logs/s3_degrade.log`
  * `results/logs/s3_degrade_manifest.csv`

* S4A — GFPGAN

  * `results/outputs/gfpgan/<preset>/`
  * `results/logs/s4_gfpgan.log`
  * `results/logs/s4_gfpgan_manifest.json`

* S4B — CodeFormer

  * `results/outputs/codeformer/<preset>/w_<value>/`
  * `results/logs/s4_codeformer.log`
  * `results/logs/s4_codeformer_manifest.json`

* S5 — metrics

  * `results/tables/metrics_pixel.csv`
  * `results/tables/metrics_lpips.csv`
  * `results/tables/metrics_identity.csv`
  * `results/tables/metrics_summary.csv`
  * `results/logs/s5_metrics.log`

* S6 — figures

  * `results/figures/*.png`
  * `results/logs/s6_figures.log`

* S7 — run manifest

  * `results/logs/run_manifest.json`
  * `results/logs/s7_logging.log`

All files are overwritten on each full run, ensuring that `results/` always reflects the latest executed experiment consistent with `config.json`.

---

## 7. Stage-by-Stage Summary

### S1 Data (Ingestion and Verification)

* Confirms presence of CelebA-aligned images and all four CSVs.
* Verifies basic consistency (counts, unique IDs, split coverage).
* Logs a short dataset summary and marks the data as ready.
* No outputs beyond logs.

### S2 Align (Verification Only)

* Reads landmark CSV and samples images to verify 256×256 alignment and reasonable crop framing.
* Optionally runs a lightweight alignment consistency check.
* Does not write new images; used strictly as a sanity gate before degradation.

### S3 Degrade (Synthetic LQ Generation)

* Applies degradations defined in `config.json`:

  * Gaussian blur (σ values)
  * JPEG compression (quality levels)
  * Gaussian noise (σ values)
* For each preset, generates LQ versions of selected GT images and writes them to `results/outputs/lq/<preset>/imgs/`.
* Produces `s3_degrade_manifest.csv` linking: id, path_gt, path_deg, degradation preset, and split.
* Forms the core paired dataset used by S4A/S4B and all metrics.

### S4A GFPGAN (Pretrained Inference)

* Uses pretrained GFPGANv1.4.
* Reads S3 manifest and LQ images.
* Restores each degraded face and writes outputs to `results/outputs/gfpgan/<preset>/`.
* Records any failures and configurations (batch size, device, checkpoint path) into `s4_gfpgan_manifest.json`.

### S4B CodeFormer (Pretrained Inference with Fidelity Sweep)

* Uses pretrained CodeFormer (sczhou) with fidelity knob `w`.
* For each preset and each `w` value from `config.json`, restores all LQ images.
* Writes outputs under `results/outputs/codeformer/<preset>/w_<value>/`.
* Logs runtime, device, and experiment grid into `s4_codeformer_manifest.json`.

### S5 Metrics (Pixel, Perceptual, Identity)

* Using GT images, S3 manifest, and S4 outputs, computes:

  * PSNR and SSIM (pixel metrics)
  * LPIPS (perceptual metric)
  * ArcFace cosine similarity (identity metric)
* Operates across all methods, presets, and CodeFormer `w` values, generating both per-sample and aggregated statistics.
* Writes separate metric CSVs plus a compact `metrics_summary.csv` used by S6 for plotting.

### S6 Figures (Panels and Plots)

* Uses `metrics_summary.csv`, selected exemplar images, and experiment matrix to construct:

  * Multi-column panels (GT, LQ, GFPGAN, CodeFormer at chosen `w`).
  * Bar charts for PSNR/SSIM.
  * LPIPS vs ArcFace trade-off plots across fidelity values.
* Produces figure files directly usable in the final report and slides.

### S7 Logging (Provenance Capture)

* Aggregates:

  * project name and version
  * config hash
  * environment summary (Python, PyTorch, CUDA)
  * seeds and determinism flags
  * per-stage manifests and basic statistics
* Writes a single `run_manifest.json` as the canonical record of the experiment.

---

## 8. Execution Model and Assumptions

* Primary target: local workstation with NVIDIA GPU (e.g., RTX 3060) and CUDA configured.
* CPU fallback is possible but not optimized; GFPGAN and CodeFormer are expected to run on GPU.
* Each stage expects the previous stage to have completed successfully; errors cause early termination with an explanatory log message.
* No attempt to maintain multiple simultaneous experiment versions. The pipeline represents one current experiment configuration determined by `config.json`.

This architecture keeps the codebase aligned with the conceptual S1–S7 pipeline, provides a clear CLI-driven execution model, and maintains a strictly organized separation between immutable data, derived results, and documentation.
