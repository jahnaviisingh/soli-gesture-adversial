# Soli Gesture Recognition: Unified ACGAN + DANN Adversarial Training

**ELL784/ELL7286 Assignment 3 — IIT Delhi**

> Training is adversarial but the results are not!

## Overview

This repository contains the implementation of a unified adversarial training framework for hand gesture recognition using Google's Soli radar sensor. The method combines:

- **ACGAN** (Auxiliary Classifier GAN) in feature space — improves fine-grained gesture accuracy
- **DANN** (Domain Adversarial Neural Network) with Gradient Reversal Layer — improves cross-subject generalization
- **Three-phase training curriculum** — ensures GAN stability on sparse radar data

### Results (Two-fold Cross-subject Validation)

| Method | Fold 1 | Fold 2 | Mean Overall | Mean Fine-grained |
|--------|--------|--------|--------------|-------------------|
| Baseline CNN-LSTM | 70.76% | 70.33% | 70.55% | 68.20% |
| + DANN only | 83.78% | 71.71% | 77.75% | — |
| **Ours (ACGAN + DANN)** | **87.71%** | **80.15%** | **83.93%** | **83.60%** |

---

## Dataset

This code uses the [Soli dataset](https://github.com/simonwsw/deep-soli) which contains range-Doppler sequences for 11 gestures collected from 10 subjects.

Download the dataset and place the `.h5` files in your input directory. On Kaggle, the dataset is available at `/kaggle/input/`.

**Dataset structure:**
```
dataset/
  ├── 0_2_0.h5       # gesture_subject_instance.h5
  ├── 0_2_1.h5
  ├── ...
```

Each file contains:
- `ch0`, `ch1`, `ch2`, `ch3` — range-Doppler channels (shape: T × 1024, reshaped to T × 32 × 32)
- `label` — gesture label (0–11)

---

## Requirements

```bash
pip install torch torchvision h5py numpy
```

Tested with:
- Python 3.8+
- PyTorch 1.12+
- CUDA 11.3+ (GPU recommended)

---

## Training

### Train the final model (ACGAN + DANN):

```bash
python soli_acgan_dann_v1.py
```

This runs two-fold cross-validation automatically. Training takes approximately **30–40 minutes per fold** on a GPU (tested on Kaggle T4).

**Key hyperparameters** (edit at top of file):
```python
SEQUENCE_LENGTH = 60      # frames per sequence
BATCH_SIZE      = 32
EPOCHS_PHASE1   = 15      # classifier only
EPOCHS_PHASE2   = 10      # + DANN
EPOCHS_PHASE3   = 30      # + ACGAN
GAN_LAMBDA      = 0.3     # GAN loss weight
DOMAIN_LAMBDA   = 0.5     # domain adversarial weight
```

**Output:** Saves `best_model_fold1.pth` and `best_model_fold2.pth` with best test accuracy per fold.

### Train the baseline (CNN+LSTM only):

```bash
python baseline_cnn_lstm.py
```

---

## Testing / Inference

To evaluate a saved model on the test set:

```python
import torch
from soli_acgan_dann_v1 import SoliEncoder, GestureClassifier, SoliDataset
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
encoder    = SoliEncoder().to(device)
classifier = GestureClassifier().to(device)

ckpt = torch.load('best_model_fold1.pth', map_location=device)
encoder.load_state_dict(ckpt['encoder'])
classifier.load_state_dict(ckpt['classifier'])
encoder.eval()
classifier.eval()

# Run inference
test_dataset = SoliDataset(test_files)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

correct = total = 0
with torch.no_grad():
    for data, labels, _ in test_loader:
        data, labels = data.to(device), labels.to(device)
        preds = classifier(encoder(data)).argmax(1)
        correct += preds.eq(labels).sum().item()
        total   += labels.size(0)

print(f"Test accuracy: {100. * correct / total:.2f}%")
```

---

## Final Model Weights

Pre-trained model weights (best fold results: Fold 1 = 87.71%, Fold 2 = 80.15%):

| File | Fold | Test Accuracy | Link |
|------|------|---------------|------|
| `best_model_fold1.pth` | Fold 1 (train: 2,3,5,6,8 / test: 9,10,11,12,13) | **87.71%** | [Download](https://drive.google.com/file/d/1BcDEpPFBeizxsvGOj0pD5ZtIn2v96Ggr/view?usp=sharing) |
| `best_model_fold2.pth` | Fold 2 (train: 9,10,11,12,13 / test: 2,3,5,6,8) | **80.15%** | [Download](https://drive.google.com/file/d/1ePhKtJkygKwGbsm3rsGPT-5aidhNQloC/view?usp=sharing) |

> Replace `YOUR_GOOGLE_DRIVE_LINK_FOLD1` and `YOUR_GOOGLE_DRIVE_LINK_FOLD2` with your actual Google Drive shareable links.

---

## Repository Structure

```
soli-gesture-adversarial/
├── soli_acgan_dann_v1.py     # Final model: ACGAN + DANN (best results)
├── baseline_cnn_lstm.py      # Baseline: CNN+LSTM only (no adversarial)
├── README.md                 # This file
```

---

## Method Summary

### Architecture

```
Input: Range-Doppler sequence (T × 1 × 32 × 32)
              │
    ┌─────────▼──────────┐
    │   CNN + LSTM        │  ← Shared Encoder (256-d features)
    └─────────┬──────────┘
              │
    ┌─────────┼──────────────┐
    │         │              │
    ▼         ▼              ▼
Gesture    Domain         ACGAN
Classifier Classifier   Generator +
(CE loss)  (GRL+DANN)   Discriminator
```

### Three-Phase Training
1. **Phase 1** (15 epochs): Classifier only — builds stable feature space
2. **Phase 2** (10 epochs): Add DANN — makes features subject-invariant
3. **Phase 3** (30 epochs): Add ACGAN — augments fine-grained classes

---

## Citation

If you use this code, please cite:

```
Wang, S., et al. "Interacting with Soli: Exploring Fine-Grained Dynamic 
Gesture Recognition in the Radio-Frequency Spectrum." UIST 2016.

Ganin, Y., et al. "Domain-Adversarial Training of Neural Networks." JMLR 2016.

Odena, A., et al. "Conditional Image Synthesis With Auxiliary Classifier GANs." 
ICML 2017.
```
