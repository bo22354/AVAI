
# AVAI — Super-Resolution Experiments (GAN / INR / SWIN)

## Project Overview

This repository collects implementations and training pipelines for three super-resolution approaches evaluated on DIV2K with various synthetic noise levels:
- `GAN` — a GAN-based super-resolution model and trainer
- `INR` — a continuous implicit representation method (LIIF)
- `SWIN` — a SwinIR transformer-based super-resolution model

Each method includes training scripts, evaluation utilities, and example model checkpoints trained at different noise levels and scales. The project is organised to make side-by-side comparison and evaluation straightforward.

---

## Project Structure

```
code/
├── GAN/
│   ├── dataLoader.py
│   ├── Generator.py
│   ├── Discriminator.py
│   ├── DiscriminatorX16.py
│   ├── Trainer.py
│   ├── train_DIV2K.py        # training entry for GAN experiments
│   ├── evaluate.py          # evaluation utilities for GAN
│   └── Models/              # GAN checkpoints by noise/scale
├── INR/
│   ├── LIIF.py
│   ├── dataLoader.py
│   ├── Trainer.py
│   ├── train_DIV2K.py       # training entry for LIIF
│   ├── evaluateINR.py       # evaluation utilities for LIIF
│   └── Models/              # LIIF checkpoints by noise/scale
├── SWIN/
│   ├── SwinIR.py
│   ├── dataLoader.py
│   ├── Trainer.py
│   ├── train_DIV2K.py       # training entry for SwinIR
│   ├── evaluateSWIN.py      # evaluation utilities for SwinIR
│   └── Models/              # SwinIR checkpoints by noise/scale
└── Utils/
		└── calculatePSNR.py     # PSNR / simple metrics helpers

data/
├── Train/
│   ├── DIV2K_train_HR/
│   └── DIV2K_train_LR_bicubic/  # contains X8 etc.
├── Valid/
│   └── DIV2K_valid_HR/
└── Test/

README.md
```

---

## Installation & Setup

### Requirements
- Python 3.8+
- PyTorch (with CUDA support recommended)
- NumPy, Pillow, Matplotlib

### Install dependencies
```bash
pip install torch torchvision
pip install numpy pillow matplotlib
```

If you use a virtual environment, create and activate it before installing.

---

## Running Experiments

Each method has a `train_DIV2K.py` script under its folder. The scripts accept dataset paths and common training arguments (batch size, epochs, learning rate, etc.). Example commands below assume you run from the repository root.

**1) Train GAN model**

```bash
cd code/GAN
python train_DIV2K.py \
	--dataset-root ../../data \
	--batch-size 32 \
	--epochs 500 \
	--learning-rate 1e-4 \
	--scale 8 \
	--noise 0.0
```

**2) Train LIIF (INR) model**

```bash
cd code/INR
python train_DIV2K.py \
	--dataset-root ../../data \
	--batch-size 32 \
	--epochs 500 \
	--learning-rate 1e-4 \
	--scale 8 \
	--noise 10.0
```

**3) Train SwinIR model**

```bash
cd code/SWIN
python train_DIV2K.py \
	--dataset-root ../../data \
	--batch-size 32 \
	--epochs 500 \
	--learning-rate 2e-4 \
	--scale 8 \
	--noise 30.0
```

Note: exact flag names and defaults may differ per script. Run `python train_DIV2K.py --help` inside each folder to inspect available options.

---

## Evaluation

Each method provides evaluation scripts:
- `code/GAN/evaluate.py`
- `code/INR/evaluateINR.py`
- `code/SWIN/evaluateSWIN.py`

Example (generic):

```bash
cd code/GAN
python evaluate.py --model-path Models/noise0/scale8/last_GAN.pth --dataset-root ../../data --batch-size 32
```

Or call the evaluation functions from Python to compute PSNR, SSIM and save output images.

---

## Model Checkpoints

Trained model files live in each method's `Models/` directory, organised by noise level and scale. Example paths:
- `code/GAN/Models/noise0/scale8/last_GAN.pth`
- `code/INR/Models/noise10.0/scale8/last_LIIF.pth`
- `code/SWIN/Models/noise0/scale8/25.17_SWIN.pth`

Load a checkpoint for inference (PyTorch):

```python
import torch
# example for GAN
from code.GAN.Generator import Generator

model = Generator()  # adapt constructor to your model signature
ckpt = torch.load('code/GAN/Models/noise0/scale8/last_GAN.pth', map_location='cpu')
model.load_state_dict(ckpt)
model.eval()
```

---

## Dataset Format

The repository expects `data/` to contain DIV2K HR and LR folders. For example:

```
data/
├── DIV2K_train_HR/
└── DIV2K_train_LR_bicubic/X8/
```

If you use custom datasets, ensure `train_DIV2K.py` and the dataloader scripts point to the correct paths or adapt the loaders accordingly.

---

## Metrics & Utilities

- **PSNR / SSIM**: use `code/Utils/calculatePSNR.py` for single-image metric computation.

---

## Notes & Tips

- The codebase contains scripts for different noise levels (check `Models/` subfolders). Use `--noise` or equivalent flags to match training and evaluation noise settings.
- Many scripts accept different hyperparameters; run `--help` to list options.
- For faster experiments, reduce `--batch-size` or `--epochs`, or use a single GPU via `CUDA_VISIBLE_DEVICES`.

---

## Authors

Jack Wayt
Course: Year 4 Advanced Visual AI
Institution: University of Bristol

## License

This project is part of academic coursework at the University of Bristol