# src/s3_degrade/__init__.py

"""S3: Synthetic degradation of aligned faces into low-quality inputs.

This package takes 256×256 aligned CelebA faces from `data/processed/aligned/`,
applies configured degradation presets (blur, JPEG, noise), and writes
low-quality counterparts plus a manifest under `results/`.
"""
