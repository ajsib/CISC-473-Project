# CelebA Data Reference (Project: GFPGAN vs CodeFormer)

All files in this directory come from the **CelebA dataset** (Large-scale CelebFaces Attributes).  
Used here as the **ground-truth source** for synthetic degradation and model evaluation.

---

## Directory Contents

### `img_align_celeba/`
Aligned and cropped facial images — 202,599 JPEG files.  
Each image is centered on a single face, used as the **base ground-truth (GT)** for restoration experiments.

---

### `list_attr_celeba.csv`
Facial attribute annotations (binary labels for each image).  
Columns:
- `image_id`: filename of the image (e.g., `000001.jpg`)  
- Remaining 40 columns: attributes ∈ {−1, 1}, where 1 = attribute present.  
  Examples: `Smiling`, `Male`, `Eyeglasses`, `Young`, etc.  
**Usage in project:** not used for training; only for potential filtering or stratified sampling.

---

### `list_bbox_celeba.csv`
Face bounding box metadata.  
Columns:
- `image_id`: image filename  
- `x_1`, `y_1`: top-left corner  
- `width`, `height`: bounding box dimensions  
**Usage in project:** optional for validation of alignment or region cropping; not essential for pretrained model inference.

---

### `list_eval_partition.csv`
Official dataset split indicator.  
Columns:
- `image_id`: image filename  
- `partition`: integer {0, 1, 2} corresponding to  
  - 0 = train  
  - 1 = validation  
  - 2 = test  
**Usage in project:** define evaluation subsets; ensures deterministic split when generating degraded sets.

---

### `list_landmarks_align_celeba.csv`
Facial landmark coordinates for aligned images.  
Columns:
- `image_id`: image filename  
- `lefteye_x`, `lefteye_y`  
- `righteye_x`, `righteye_y`  
- `nose_x`, `nose_y`  
- `leftmouth_x`, `leftmouth_y`  
- `rightmouth_x`, `rightmouth_y`  
**Usage in project:** basis for verifying face alignment, guiding region-aware losses or degradation centering.

---

## Project Context

- **Input domain:** `img_align_celeba/` serves as pristine face data.
- **Synthetic degradations:** applied to these aligned images to produce low-quality (LQ) inputs.
- **Ground truth:** same images in original form for metric comparisons (PSNR, SSIM, LPIPS, ArcFace similarity).
- **No fine-tuning:** pretrained GFPGAN and CodeFormer are evaluated directly on these pairs.

---

## Minimal Usage Flow

1. Use `list_eval_partition.csv` to select subset indices.  
2. Read corresponding images from `img_align_celeba/`.  
3. Optionally reference landmarks for alignment validation.  
4. Apply degradations → produce `(LQ, GT)` pairs.  
5. Evaluate both models using identical splits and metrics.
