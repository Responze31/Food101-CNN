# Food101-CNN (PyTorch)

This repo trains a custom CNN on the **Food-101** dataset using **PyTorch**, while loading the dataset via **TensorFlow Datasets (TFDS)** and wrapping it into a PyTorch `Dataset`.

## Why this project is intentionally “small”
Training on the full Food-101 dataset can be heavy on **RAM/VRAM** and **runtime**, especially on free/limited environments (e.g., Colab sessions, smaller GPUs, or local machines). To keep the notebook practical to run without crashing or taking hours, the training is intentionally constrained:

- **Capped training samples:** `MAX_TRAIN_SAMPLES = 10000` (out of 75,750)
- **Capped validation samples:** `MAX_VAL_SAMPLES = 2000` (out of 25,250)
- **Reduced image size:** `IMG_SIZE = 112` to cut compute + memory usage
- **Small batch size:** `BATCH_SIZE = 16` to fit memory limits
- **Short training:** `EPOCHS = 10` (with early stopping + LR scheduler)

These constraints make results **not representative of full-dataset performance**, but they make the notebook easy to run in constrained environments.

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

## How to run (detailed)

### 1) Get the notebook
Clone the repo (or download as ZIP) and open:

- `food101.ipynb`

If you’re using **Google Colab**, upload the notebook (and ideally the whole repo folder) or open it directly from GitHub.

### 2) Create an environment (recommended)
You can run this in **Colab** (GPU recommended) or locally.

**Local (conda example):**
```bash
conda create -n food101cnn python=3.10 -y
conda activate food101cnn
```

**Local (venv example):**
```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate
```

Then install the dependencies used by the notebook:
- PyTorch (CUDA version depends on your GPU)
- TensorFlow + TFDS
- Common utilities (numpy, matplotlib, tqdm, etc.)

> If you run into install issues, the simplest path is usually **Colab** (it already has most packages) and then `pip install tensorflow-datasets` if needed.

### 3) Enable GPU (strongly recommended)
- **Colab:** `Runtime → Change runtime type → Hardware accelerator → GPU`
- **Local:** confirm your PyTorch sees CUDA:
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

### 4) First run will download Food-101 (one-time cost)
When you run the notebook the first time:
- TFDS will download the **Food-101** dataset.
- This can take a while and uses significant disk space.
- The notebook then converts/wraps it into a PyTorch-friendly dataset and caches artifacts (so subsequent runs are faster).

**Common first-run gotchas**
- **Disk space:** make sure you have enough space (Food-101 is large).
- **Slow download:** TFDS download speed depends on your environment/network.
- **Colab reset:** if your Colab runtime resets, you may need to re-download unless you store data in Drive.

### 5) Configure the “small-mode” settings (optional)
The notebook is designed to run with caps such as:
- `MAX_TRAIN_SAMPLES = 10000`
- `MAX_VAL_SAMPLES = 2000`
- `IMG_SIZE = 112`
- `BATCH_SIZE = 16`
- `EPOCHS = 10`

You can edit these constants in the notebook to trade speed vs performance (see “Notes” below).

### 6) Run training
Run all cells top-to-bottom. During training you should see:
- Per-epoch metrics (train/val accuracy + loss)
- Learning rate adjustments (ReduceLROnPlateau)
- Early stopping behavior (if validation stops improving)
- Periodic checkpointing

Artifacts produced:
- Best checkpoint: `checkpoints/best_food101_model.pth`
- Training history: `checkpoints/history.json`
- Any plots saved/printed by the notebook

### 7) Resume / reuse the model (optional)
- To evaluate later without retraining, load the saved checkpoint from `checkpoints/best_food101_model.pth`.
- If you change model architecture or class mapping, old checkpoints may not load cleanly.

### 8) Troubleshooting quick tips
- **CUDA OOM:** lower `BATCH_SIZE`, keep `IMG_SIZE` small, or disable heavy augmentations.
- **Very slow training:** confirm you’re on GPU; keep caps enabled.
- **Validation accuracy looks “weird”:** with small sample caps, class coverage is limited and variance is higher.

## Interpreting the training curves (what you’re seeing)
The plot shows both **accuracy** and **loss** improving over ~10 epochs (train acc up, train loss down; val acc up, val loss down). A few notes:

### 1) Why validation can be higher than training
In your plot, **validation accuracy is consistently higher than training accuracy**. That can happen and is not automatically a bug. Common reasons in this setup:

- **Training has augmentation; validation does not.**  
  Augmentations (random crops/flip/color jitter, etc.) intentionally make training images harder, so training accuracy can be lower while still generalizing well.
- **Dropout / BatchNorm behavior differs.**  
  Dropout is active during training (making predictions noisier), but turned off at evaluation time, which can boost validation accuracy.
- **Small sample caps increase noise.**  
  With only 10k train / 2k val, the particular subset you get may be “easier” in validation than in training, purely by chance.

### 2) Loss decreasing smoothly is a good sign
Both train and val loss decrease steadily, suggesting:
- The optimizer is making stable progress (no divergence).
- There’s no strong sign of overfitting within 10 epochs (you’d often see train loss dropping while val loss rises).

### 3) Accuracy is still very low overall (single digits)
Food-101 has **101 classes**. Random guessing is ~**1%** top-1 accuracy. Your curve ending around ~8–11% means the model is learning something, but it’s still in an early/underpowered regime due to:
- training on a capped subset,
- reduced resolution,
- a relatively small custom CNN compared to standard transfer learning baselines,
- only 10 epochs.

## What would change with “no limitations” and 50 epochs?
You asked how much impact removing limits and training for ~50 epochs would have. The honest answer: **it depends heavily on compute, resolution, and model capacity**, but directionally:

### Removing the sample caps (use full 75,750 train / 25,250 val)
**Expected impact:**
- Usually a **large** improvement in generalization because the model sees more intra-class variation (different lighting, angles, backgrounds, plating styles).
- More stable curves (less variance from subset selection).
- But training time scales roughly with dataset size:  
  75,750 / 10,000 ≈ **7.6× more steps per epoch** (and validation ~12.6×).

### Increasing epochs to 50
**Expected impact:**
- With the *same* small subset, 50 epochs may start to **overfit** (train acc up, val stagnates/drops) unless regularization + augmentation are strong.
- With the *full* dataset, 50 epochs is much more likely to continue improving (though gains may slow after some point, and LR scheduling becomes important).

### Removing the resolution limit (use larger IMG_SIZE)
**Expected impact:**
- Often a **significant accuracy jump** on fine-grained visual categories (food types can be subtle).
- But compute/VRAM increases quickly (roughly proportional to pixel count and feature map sizes).  
  Going 112 → 224 is **4× pixels**, so it can feel like ~2–4×+ training cost depending on architecture.

### Realistic “how much better” (ballpark)
With a small custom CNN from scratch, full data + more epochs + higher resolution can move you from “single digits” to something meaningfully higher—but the biggest leap on Food-101 typically comes from **transfer learning** (e.g., ResNet/EfficientNet pretrained), where top-1 accuracy can become dramatically higher.

So:
- **Full dataset + 50 epochs**: likely noticeably better than your current curves (and less noisy), assuming your model capacity and LR schedule can keep up.
- **Full dataset + 50 epochs + larger images**: stronger still, but much heavier.
- **Biggest practical improvement** (if allowed): keep the full dataset and switch to a pretrained backbone.

## Notes
If you want better accuracy, increase `IMG_SIZE`, `EPOCHS`, and remove/raise the sample caps—but expect **significantly higher memory use and much longer training time**.
