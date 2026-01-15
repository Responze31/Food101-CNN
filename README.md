# Food101-CNN (PyTorch)

This repo trains a custom CNN on the **Food-101** dataset using **PyTorch**, while loading the dataset via **TensorFlow Datasets (TFDS)** and wrapping it into a PyTorch `Dataset`.

## Why this project is intentionally “small”
Training on the full Food-101 dataset can be heavy on **RAM/VRAM** and **runtime**, especially on free/limited environments (e.g., Colab sessions, smaller GPUs, or local machines). To keep the notebook runnable end-to-end, I **intentionally limited** the workload:

- **Capped training samples:** `MAX_TRAIN_SAMPLES = 10000` (out of 75,750)
- **Capped validation samples:** `MAX_VAL_SAMPLES = 2000` (out of 25,250)
- **Reduced image size:** `IMG_SIZE = 112` to cut compute + memory usage
- **Small batch size:** `BATCH_SIZE = 16` to fit memory limits
- **Short training:** `EPOCHS = 10` (with early stopping + LR scheduler)

These constraints make results **not representative of full-dataset performance**, but they make the notebook practical to run without crashing or taking hours.

## What’s inside
- TFDS download (`food101`) and conversion to a cached PyTorch dataset
- Data augmentation + normalization
- A custom 5-block CNN (Conv/BN/ReLU/Pool/Dropout)
- Training loop with:
  - AMP mixed precision (CUDA)
  - gradient clipping
  - ReduceLROnPlateau scheduler
  - checkpoint saving (`checkpoints/best_food101_model.pth`)
  - history export (`checkpoints/history.json`)

## How to run
Open and run the notebook:

- `food101.ipynb`

It will download Food-101 via TFDS and start training immediately.

## Notes
If you want better accuracy, increase `IMG_SIZE`, `EPOCHS`, and remove/raise the sample caps—but expect **significantly higher memory use and much longer training time**.
