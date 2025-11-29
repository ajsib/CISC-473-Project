# S4A — GFPGAN Inference (Pretrained Restoration)

## Executive Summary

S4A runs the pretrained GFPGAN model on the degraded faces produced by S3.

It does four things:

1. Reads the S3 manifest to obtain GT paths, degraded input paths, and degradation labels.
2. Loads the GFPGAN v1.4 pretrained checkpoint and initializes the restoration pipeline on GPU.
3. Restores each degraded image and writes the restored outputs into a structured results tree.
4. Emits a manifest and logs that bind each restored image to its input, ground truth, and configuration.

S4A is deterministic given the model weights, config, and S3 manifest. It defines the GFPGAN branch of the comparative study.

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
  * 256×256 LQ images compatible with GFPGAN input assumptions.

* GFPGAN upstream configuration (from `config.json.upstreams.gfpgan`):

  * `repo` — upstream GitHub repository URL (for reference).
  * `default_checkpoint` — checkpoint name (e.g., `"GFPGANv1.4"`).
  * `commit_pin` — optional specific commit hash (documentation only).

* Hardware and environment:

  * CUDA-enabled GPU configured via `env.yml`.
  * PyTorch and required libs already installed.

S4A does not depend on CodeFormer configuration. It is purely the GFPGAN branch.

### Output Locations (Contract)

S4A writes under `results/`:

* Restored images:

  * `results/outputs/s4-gfpgan/<degradation>/imgs/<id>.jpg`

* Logs and manifests:

  * `results/logs/s4_gfpgan.log`
  * `results/logs/s4_gfpgan_manifest.json` (or CSV), per-sample metadata.

---

## Transformations and Checks

S4A wraps GFPGAN inference in a uniform batch pipeline.

1. **Model initialization**

   * Resolve checkpoint location for GFPGAN v1.4:

     * either download from upstream or load from local cache.
   * Initialize GFPGAN restoration model on GPU:

     * set evaluation mode
     * disable gradients
   * Log:

     * checkpoint path
     * model version
     * device (e.g., `cuda:0`).

2. **Input enumeration**

   * Load `s3_degrade_manifest.csv` into memory or stream it.
   * For each unique `degradation` preset, group rows so outputs can be written under:

     * `results/outputs/s4-gfpgan/<degradation>/imgs/`.
   * Optionally filter by split (e.g., evaluation-only), if configured.

3. **Batched inference**

   For each degradation group:

   * Create the output directory for that preset.

   * Process images in mini-batches:

     * Read degraded images from `path_deg`.
     * Convert to the format expected by GFPGAN (tensor, normalization).
     * Run GFPGAN forward pass to obtain restored faces.
     * Convert outputs back to 8-bit images, enforcing 256×256 spatial size.

   * For each image in the batch:

     * Write restored image to:

       * `results/outputs/s4-gfpgan/<degradation>/imgs/<id>.jpg`.
     * Record per-sample metadata for the manifest.

4. **Per-sample metadata collection**

   For each `(id, degradation)` pair, S4A records:

   * `id`
   * `path_gt` — from S3 manifest
   * `path_deg` — from S3 manifest
   * `path_restored` — GFPGAN output path
   * `degradation`
   * `split`
   * `method` = `"gfpgan"`
   * optionally:

     * runtime per image
     * GPU name
     * GFPGAN internal config (e.g., upscale factor, version tag).

   This is written to `results/logs/s4_gfpgan_manifest.json` (or `.csv`) as a machine-readable structure.

5. **Sanity checks**

   After inference:

   * Confirm that:

     * number of restored images matches the number of degraded images processed.
     * all `path_restored` paths in manifest exist on disk.
     * each restored image is 256×256 and has valid pixel ranges.

   * Log any failures:

     * missing file
     * I/O errors
     * model failures.

   * On critical mismatch, S4A stops with a clear error so metrics are never run on incomplete outputs.

---

## Outputs and End State

If S4A completes successfully:

* For every `(id, degradation)` in the S3 manifest (respecting any split filters), there is a corresponding GFPGAN output:

  * `results/outputs/s4-gfpgan/<degradation>/imgs/<id>.jpg`.

* `results/logs/s4_gfpgan_manifest.*` provides an authoritative mapping linking:

  * GT image
  * degraded input
  * GFPGAN-restored output
  * degradation preset
  * split
  * method name.

* `results/logs/s4_gfpgan.log` records:

  * model path and version
  * device information
  * total images processed
  * per-degradation counts
  * any errors or skipped samples.

Downstream impact:

* S5 (metrics) uses `s4_gfpgan_manifest` combined with `s3_degrade_manifest` to construct GFPGAN-specific metric tables (PSNR, SSIM, LPIPS, ArcFace).
* S6 (figures) loads GFPGAN outputs alongside CodeFormer outputs and GT images to build visual comparisons and trade-off plots.

S4A’s functional role is to produce the GFPGAN branch of restored faces, fully paired and traceable back to both the degraded inputs and the ground truth.
