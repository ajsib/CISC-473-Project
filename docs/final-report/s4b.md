# S4B — CodeFormer Inference with Fidelity Sweep

## Executive Summary

S4B runs the pretrained CodeFormer model on the same degraded faces used by GFPGAN, but exposes the fidelity–quality knob `w`.

It does four things:

1. Reads the S3 manifest to obtain GT paths, degraded input paths, and degradation labels.
2. Loads the pretrained CodeFormer model and initializes it on GPU.
3. For each fidelity value `w` from `config.json`, restores all degraded images and writes outputs into a structured results tree.
4. Emits a manifest and logs binding each restored image to its input, ground truth, degradation preset, split, and `w`.

S4B is deterministic given the model weights, `w` grid, and S3 manifest. It defines the CodeFormer branch and the fidelity sweep used for trade-off analysis.

---

## Given Structures

### Inputs

* S3 manifest:

  * `results/logs/s3_degrade_manifest.csv`
  * Columns:

    * `id` — image identifier
    * `path_gt` — aligned GT path
    * `path_deg` — degraded input path
    * `degradation` — preset name
    * `split` — partition label

* Degraded images:

  * `results/outputs/s3-degraded/<preset>/imgs/<id>.jpg`
  * 256×256 low-quality faces.

* CodeFormer upstream configuration (`config.json.upstreams.codeformer`):

  * `repo` — upstream GitHub repository (reference).
  * `default_checkpoint` — checkpoint name (e.g., `"codeformer-v0.1.0"`).
  * `commit_pin` — optional commit hash (documentation only).
  * `fidelity_knob`:

    * `name` — `"w"`
    * `range` — allowed values
    * `default` — default `w` for single-run scenarios.

* Experiment fidelity grid (`config.json.experiments.matrix.codeformer_fidelity_w`):

  * Explicit list of `w` values to sweep over (e.g., `[0.3, 0.5, 0.7, 0.9]`).

S4B does not depend on GFPGAN configuration. It shares S3’s manifest and data only.

### Output Locations (Contract)

S4B writes under `results/`:

* Restored images:

  * `results/outputs/s4-codeformer/<degradation>/w_<w>/imgs/<id>.jpg`

* Logs and manifests:

  * `results/logs/s4_codeformer.log`
  * `results/logs/s4_codeformer_manifest.json` (or CSV), including `w`.

---

## Transformations and Checks

S4B wraps CodeFormer in a matrix over degradation presets × fidelity values.

1. **Model initialization**

   * Resolve checkpoint path for CodeFormer.
   * Initialize the model on GPU in eval mode.
   * Log:

     * checkpoint path
     * model version
     * device (`cuda:0`)
     * list of `w` values that will be used.

2. **Input enumeration**

   * Load `s3_degrade_manifest.csv`.
   * Group rows by `degradation` preset for consistent output layout.
   * For each group, S4B will iterate over all configured `w` values.

3. **Fidelity sweep loop**

   For each degradation preset `d` and each fidelity value `w` in `codeformer_fidelity_w`:

   * Create an output directory:

     * `results/outputs/codeformer/<d>/w_<w>/imgs/`.

   * Process images in batches:

     * Read degraded images from `path_deg`.
     * Convert to CodeFormer input format (tensor, normalization).
     * Run CodeFormer forward pass with fidelity knob `w`.
     * Convert outputs back to 8-bit images, enforcing 256×256 shape.

   * For each `(id, d, w)` triplet:

     * Write restored image to:

       * `results/outputs/codeformer/<d>/w_<w>/imgs/<id>.jpg`.
     * Collect per-sample metadata.

4. **Per-sample metadata collection**

   For each restored sample, record:

   * `id`
   * `path_gt` — from S3 manifest
   * `path_deg` — from S3 manifest
   * `path_restored` — CodeFormer output path
   * `degradation`
   * `split`
   * `method` = `"codeformer"`
   * `w` — fidelity knob value used for this output
   * optionally:

     * runtime per image
     * GPU name
     * any internal CodeFormer settings (e.g., upscale flags, encoder choice).

   All entries are written into `results/logs/s4_codeformer_manifest.json` (or `.csv`).

5. **Sanity checks**

   After inference:

   * For each `(degradation, w)` pair, verify:

     * number of restored images equals number of `path_deg` entries processed for that `(d, w)` cell.
     * all `path_restored` files exist and can be opened.
     * images have shape 256×256 and valid pixel values.

   * On mismatch or persistent failures, log and terminate to prevent invalid metric runs.

---

## Outputs and End State

If S4B completes successfully:

* For every combination of:

  * `id` from S3 manifest

  * degradation preset `degradation`

  * fidelity value `w` from `config.json.experiments.matrix.codeformer_fidelity_w`
    there is a corresponding CodeFormer output image at:

  * `results/outputs/codeformer/<degradation>/w_<w>/imgs/<id>.jpg`.

* `results/logs/s4_codeformer_manifest.*` provides the authoritative mapping back to GT and degraded images, including the `w` value used for each restoration.

* `results/logs/s4_codeformer.log` records:

  * model path and version
  * device information
  * the full grid of `(degradation, w)` combinations
  * counts of processed images per combination
  * any errors or skipped cases.

Downstream impact:

* S5 (metrics) uses `s4_codeformer_manifest` to compute metric tables indexed by `(method="codeformer", degradation, w)`.
* S6 (figures) uses these outputs to build:

  * side-by-side panels comparing GFPGAN vs CodeFormer at specific `w`.
  * trade-off plots of LPIPS vs ArcFace similarity across different `w`.

S4B’s functional role is to generate the complete CodeFormer restoration matrix over the chosen fidelity values, making the fidelity–quality trade-off observable and measurable.
