# Task Board / Roadmap (Skeleton)

## Key Dates (America/Toronto)
- Proposal: Sep 15, 2025
- Midterm Report: Oct 18, 2025
- Final Deliverables: Nov 30, 2025

## Phase 1 — Baselines & Setup
- [ ] Repo, env, determinism flags
- [ ] Alignment pipeline
- [ ] Synthetic degradation + manifest

## Phase 2 — Metrics & Evaluation
- [ ] PSNR/SSIM/LPIPS scripts
- [ ] ArcFace identity pipeline
- [ ] First metric tables

## Phase 3 — Historical & Qualitative
- [ ] Curate public-domain set (+urls.txt)
- [ ] Panels and failure cases

## Phase 4 — Midterm Report
- [ ] Pipeline diagram
- [ ] Tables/plots
- [ ] Risks/next steps

## Phase 5 — Sweeps & Fine-tuning
- [ ] CodeFormer fidelity sweeps
- [ ] Optional light fine-tuning

## Phase 6 — User Study
- [ ] A/B survey
- [ ] Aggregation plots

## Phase 7 — Final Integration
- [ ] Final paper (≥6 pages)
- [ ] Slides
- [ ] Code and figures locked



---
```
.
├── README.md                     # What this repo does, 5-minute run path (Stage 0: orientation)
├── env.yml                       # Reproducible env (PyTorch, LPIPS, ArcFace); freeze minimal deps
├── config.json                   # Single source of truth: paths, seeds, presets, model knobs
├── TaskList.md                   # Two-day checklist; who does what; DONE/TODO only

├── data/                         # DATA JOBS: acquire → standardize GT → synthesize LQ (immutable → derived)
│   ├── raw/                      # Stage 1: ingest as-is (CelebA drops + historical originals); never mutate
│   │   ├── celeba/               # In-The-Wild, Aligned, and all annotations placed verbatim
│   │   └── historical/           # Public-domain originals for qualitative only (no GT use)
│   ├── processed/                # Stage 2: standardized GT space (single truth for metrics)
│   │   ├── celeba256/            # 256×256 aligned crops from raw/celeba (train/val/test respected)
│   │   └── historical_aligned/   # 256×256 aligned crops for qualitative panels (no metrics)
│   ├── manifests/                # Stage 3a: JSONL recipes for degradations (params + RNG seeds)
│   │   └── celeba256/            # One file per preset+seed; file-wise ops recorded
│   └── lq/                       # Stage 3b: materialized degraded inputs (consumer-ready)
│       └── celeba256/
│           └── <preset>_<seed>/  # e.g., blur_jpeg40_s123/
│               ├── imgs/         # LQ images to feed models
│               └── pairs.csv     # relative LQ ↔ GT mapping (for metrics)

├── src/                          # EXECUTION JOBS: one verb per stage; pure functions w/ CLI
│   ├── detect_align.py           # Stage 2: raw → processed/celeba256 + align_logs; deterministic
│   ├── degrade.py                # Stage 3: manifests → lq/<preset>_<seed>/imgs + pairs.csv
│   ├── run_gfpgan.py             # Stage 4A: infer GFPGAN on LQ → results/outputs/gfpgan/...
│   ├── run_codeformer.py         # Stage 4B: infer CodeFormer (sweep w) → results/outputs/codeformer/...
│   ├── metrics_pixel.py          # Stage 5A: PSNR/SSIM using pairs.csv and GT
│   ├── metrics_lpips.py          # Stage 5B: LPIPS (net selectable in config.json)
│   └── metrics_identity.py       # Stage 5C: ArcFace cosine; batched; same pairs.csv contract

├── scripts/                      # ORCHESTRATION JOBS: glue only; no logic
│   ├── make_synth_testset.sh     # Calls degrade.py for all presets/seeds in config.json (Stage 3)
│   ├── eval_all.sh               # Runs Stage 4/5 matrix: methods × presets × fidelity (writes tables)
│   └── make_figures.sh           # Stage 6: aggregates → plots (reads results/tables → results/figures)

├── results/                      # OUTPUT JOBS: everything reproducible and disposable
│   ├── outputs/                  # Restored images by method/preset/fidelity (for panels)
│   │   ├── gfpgan/
│   │   └── codeformer/
│   ├── tables/                   # CSVs: per-image and aggregates (by method/preset/fidelity)
│   ├── figures/                  # PNG/SVG panels and charts for report
│   └── logs/                     # run logs, env snapshot, seeds, timing

├── docs/                         # COMMUNICATION JOBS: course artifacts
│   ├── CISC 473 Proposal.pdf     # Provided proposal
│   └── TASK_BOARD.md             # Milestones checklist (mirrors TaskList.md if needed by course)

└── notebooks/                    # DIAGNOSTIC JOBS (optional to pipeline)
    ├── sanity_checks.ipynb       # Quick visual checks: align/degrade/infer samples
    ├── identity_eval.ipynb       # Inspect ArcFace distributions, sanity on thresholds
    └── user_study_agg.ipynb      # If user study happens; can be left empty for 2-day scope
```


