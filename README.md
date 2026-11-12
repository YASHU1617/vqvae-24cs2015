# VQ-VAE: Neural Discrete Representation Learning (reduced-scale re-implementation)

**Author:** G. Sri Yasaswitha, Roll No. 24CS2015 (IDD)
**Course:** Deep Learning 2026, Paper Implementation Project (Paper ID DL-081)
**Paper:** van den Oord, Vinyals, Kavukcuoglu. *Neural Discrete Representation Learning.* NeurIPS 2017. [arXiv:1711.00937](https://arxiv.org/abs/1711.00937)

## What this repo contains

| File | Description |
|---|---|
| `VQVAE_24CS2015.ipynb` | Runnable notebook with the full implementation, ablations and prior |
| `VQVAE_24CS2015_Report.pdf` | Project report (IEEE two-column) |
| `VQVAE_24CS2015_Presentation.pptx` | Presentation slides |

## What I implemented (PyTorch)

- **Vector quantiser** with nearest-codeword lookup, straight-through estimator, codebook loss, commitment loss (beta = 0.25) and EMA codebook updates (decay 0.99)
- **Encoder and decoder** with residual blocks (about 0.96 M parameters in total)
- **Codebook usage tools:** perplexity and number of codes used
- **PixelCNN prior** (masked convolutions, types A and B) over the 8x8 code grid, used to sample new images

Standard building blocks (Conv2d, Adam, torchvision CIFAR-10) are used as provided.

## Ablations

1. EMA codebook versus gradient-trained codebook
2. Codebook size K = 64, 512, 2048

## Results (CIFAR-10 test set, 10 epochs, seed 0)

| Run | Test MSE | PSNR (dB) | Perplexity | Codes used |
|---|---|---|---|---|
| K=512, EMA (main model) | 0.01535 | 24.16 | 407.0 | 512 / 512 |
| K=512, gradient codebook | 0.05160 | 18.89 | 8.4 | 18 / 512 |
| K=64, EMA | 0.02143 | 22.71 | 57.0 | 64 / 64 |
| K=2048, EMA | 0.01521 | 24.20 | 449.8 | 602 / 2048 |

PSNR is derived from the MSE as 10 log10(4 / MSE) on the [-1, 1] pixel range.

**PixelCNN prior:** best test NLL 4.632 nats per code at epoch 8 (uniform guess: 6.24). The test NLL rises to 4.671 by epoch 15, so the prior overfits.

**Main findings**
- The gradient-based codebook collapses to 18 of 512 codes and gives 3.4x higher MSE than EMA.
- K = 2048 improves MSE by only 0.9% over K = 512, and about 70% of its codes stay unused.
- Samples from the prior show plausible colours and layouts but few recognisable objects.

## What was changed from the paper

This is a reduced-scale reproduction. The network is small, training runs for 10 epochs, and only CIFAR-10 is used. The paper trains larger models for much longer on more datasets, so the numbers here are not directly comparable with the paper's.

## How to run

**Option 1: Google Colab (recommended)**
1. Open the notebook in Colab (upload `VQVAE_24CS2015.ipynb` at colab.research.google.com).
2. Go to Runtime, then Change runtime type, and choose a **T4 GPU**.
3. Click Runtime, then Run all.

**Option 2: Local**
```bash
pip install torch torchvision numpy matplotlib
jupyter notebook VQVAE_24CS2015.ipynb
```

CIFAR-10 is downloaded automatically by torchvision.

**Settings**
- Random seed: `SEED = 0` (Python, NumPy and PyTorch)
- `EPOCHS = 10` at the top of the notebook (use 2 for a quick test)
- Batch size 128, Adam, learning rate 2e-4 (VQ-VAE) and 1e-3 (prior, 15 epochs)

## Limitations

Single seed, short training, one dataset, no hyperparameter tuning, no FID. Small differences (for example K = 512 versus 2048) should not be over-interpreted.

## References

1. A. van den Oord, O. Vinyals, K. Kavukcuoglu. Neural Discrete Representation Learning. NeurIPS 2017.
2. A. van den Oord et al. Conditional Image Generation with PixelCNN Decoders. NeurIPS 2016.
3. Add here any GitHub repository or tutorial you consulted, as the assignment requires.
