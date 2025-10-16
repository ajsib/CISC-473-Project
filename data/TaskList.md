# 1) Where is the data coming from

- Scraping 
- Synthetic images

# 2) How is data ingested

```
# DATA DIRECTORY (flat layout for minimalism)
# All CelebA data stored directly under ./data — no raw/processed nesting.
# Each file below serves a distinct role in the CelebA dataset.
#
# ────────────────────────────────────────────────────────────────
# img_align_celeba/                 → aligned and cropped face images (≈202,599 JPEGs, 178×218 each)
# list_attr_celeba.csv              → 40 binary facial attributes per image (e.g., Smiling, Eyeglasses)
# list_bbox_celeba.csv              → bounding box coordinates (x, y, width, height) per face
# list_landmarks_align_celeba.csv   → 5 facial landmarks per image (left_eye, right_eye, nose, left_mouth, right_mouth)
# list_eval_partition.csv           → official split of dataset into train / validation / test sets (0/1/2)
# ────────────────────────────────────────────────────────────────
# No raw/processed separation; “data/” itself is the working root for all later stages.
# Any derived data (e.g., 256×256 aligned crops, synthetic degradations) will be created in sibling directories
# such as ./lq/ and ./results/, never altering the original CelebA data.

data/
├── img_align_celeba/               # aligned CelebA images (primary input)
├── list_attr_celeba.csv            # 40 facial attributes
├── list_bbox_celeba.csv            # face bounding boxes
├── list_eval_partition.csv         # train/val/test split indicators
└── list_landmarks_align_celeba.csv # 5 keypoint_
```