# Representational Difference Distillation (RDD)

Code for **Representational Difference Distillation**
([paper](https://drive.google.com/file/d/1Is9-a2ZMwBdPYS7piE1mn8NuHTZYFg_p/view),
[blog post](https://divinrkz.com/blog/research/rdd-distillation)), a knowledge distillation method
that uses [Representational Difference Explanations (RDX)](https://arxiv.org/abs/2505.23917) to find
where the teacher's and student's representations disagree, and focuses the distillation loss there.

![RDD overview](assets/rdd-overview.png)

## How it works

**A. RDX diagnostic (every R epochs, no gradient).** Both networks embed the training set. For every
pair of samples, RDX compares the neighbor rank of `j` around `i` in the teacher and in the student,
and turns the difference into two directional affinities:

- `A01`: **teacher-unique** pairs (close for the teacher, far for the student)
- `A10`: **student-unique** pairs (close for the student, far for the teacher)

From these, each sample `i` gets a positive `p_i = argmax_j A01[i, j]`, a pool of `P` candidate
negatives (top `A10[i, :]`), and a confusion weight `c_i = mean_j (A01[i, j] + A10[i, j])`.

**B. Training step (every mini-batch).** As in [CRD](https://arxiv.org/abs/1910.10699), student
features are projected and contrasted against a momentum memory bank of projected teacher features.
RDD changes *which* entries are contrasted (pull toward `p_i`, push away from negatives sampled from
the pool) and *how much* each anchor counts (loss weighted by `c_i / mean(c)`).

The full N x N affinity matrix is never materialized: RDX is computed one block of rows at a time,
optionally against a random subset of anchor samples (`--rdx_anchor_n`).

## Results

CIFAR-100, resnet32x4 → resnet8x4, final-epoch top-1 test accuracy (%). **Single run per method**:
the RDD vs. CRD gap is within typical seed-to-seed variation, so treat the two as on par.

| Method | Test acc. |
| --- | --- |
| KD | 72.93 |
| RKD | 71.55 |
| CRD | 75.45 |
| RDD sampling (15% subset, refreshed) | 59.49 |
| RDD curriculum | 72.96 |
| **RDD supervision** | 75.58 |

Teacher (resnet32x4): 79.42% ([Tian et al., 2020](https://arxiv.org/abs/1910.10699)).

## Installation

Tested with Python 3.8+, PyTorch ≥ 1.10, and CUDA 11.x on Ubuntu.

```bash
pip install -r requirements.txt
sh scripts/fetch_pretrained_teachers.sh   # downloads pretrained teachers to save/models/
```

## Reproducing the table

All runs use the same recipe: SGD (momentum 0.9, weight decay 5e-4), lr 0.05 decayed by 0.1 at
epochs 150/180/210, 240 epochs, batch size 64. `-r`, `-a`, `-b` are the weights of the CE, KD and
method-specific losses.

```bash
T=./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth

# RDD supervision (same loss weights as CRD: no KD term; RDX beta 0.8, refreshed every 20 epochs)
python train_student.py --path_t $T --model_s resnet8x4 --distill rdx_contrast -r 1 -a 0 -b 0.8 \
  --rdx_beta 0.8 --rdx_refresh_epochs 20 --trial 1

# RDD curriculum (KD loss + RDX easiest-first curriculum)
python train_student.py --path_t $T --model_s resnet8x4 --distill kd -r 0.1 -a 0.9 -b 0 \
  --sampling curriculum --curriculum_start_frac 0.3 --curriculum_pace_epochs 100 --trial 1

# Baselines
python train_student.py --path_t $T --model_s resnet8x4 --distill kd  -r 0.1 -a 0.9 -b 0   --trial 1
python train_student.py --path_t $T --model_s resnet8x4 --distill crd -r 1   -a 0   -b 0.8 --trial 1
python train_student.py --path_t $T --model_s resnet8x4 --distill rkd -r 1   -a 0   -b 1   --trial 1
```

To combine RDD with KD, set `-a` to a non-zero value (e.g. `-a 1`).

### RDX options

| Flag | Description |
| --- | --- |
| `--rdx_gamma` | Saturation γ of the locally-biased difference (default: `0.1`) |
| `--rdx_beta` | Affinity sharpness β; only affects the weights `c_i` (default: `5.0`) |
| `--rdx_start_epoch` | First epoch at which RDX is computed (default: `1`) |
| `--rdx_refresh_epochs` | Recompute RDX every N epochs (`0` = compute once; default: `0`) |
| `--rdx_anchor_n` | Number of anchor samples for RDX (`0` = all; default: `0`) |
| `--rdx_kna_k` | Top-k affinities summed for curriculum difficulty scores (default: `8`) |
| `--sampling` | `standard` (default) or `curriculum` |
| `--curriculum_start_frac` | Initial fraction of easiest samples (default: `0.3`) |
| `--curriculum_pace_epochs` | Epochs to ramp to the full training set (default: `100`) |

Example: same as the paper's RDD run, but scoring against 2048 anchors (much cheaper than all 50k samples):

```bash
python train_student.py --path_t $T --model_s resnet8x4 --distill rdx_contrast -r 1 -a 0 -b 0.8 \
  --rdx_beta 0.8 --rdx_refresh_epochs 20 --rdx_anchor_n 2048 --trial 1
```

To train teachers from scratch instead of downloading them: `sh scripts/run_cifar_vanilla.sh`.

## Project structure

```
├── train_student.py              # Main training script
├── distiller_zoo/
│   ├── RDXContrast.py            # RDD loss: RDX-weighted contrastive loss over the memory bank
│   └── ...                       # Other distillation methods (KD, CRD, RKD, ...)
├── rdx/sampling.py               # RDX math: ranks, locally-biased difference, A01/A10, lookup tables
├── helper/rdx_utils.py           # Embedding extraction + RDX refresh orchestration
├── crd/                          # CRD memory bank and projection heads
├── models/                       # Teacher and student architectures
├── dataset/                      # CIFAR-100 loaders (incl. RDX-indexed variants)
├── scripts/                      # Training, export and figure scripts
└── save/                         # Checkpoints and TensorBoard logs
```

The overview figure is generated by `scripts/plot_overview_figure_v4.py` (uses the bundled Fira Sans fonts in `assets/fonts/`, SIL Open Font License).

## Maintainers

- **Abdulkarim Mugisha**
- **Divin Irakiza**: [divinrkz](https://divinrkz.com)

## Acknowledgements

The starter code was adapted from [RepDistiller](https://github.com/HobbitLong/RepDistiller) by
Yonglong Tian (BSD 2-Clause License, see `LICENSE`), which accompanies *Contrastive Representation
Distillation* (ICLR 2020). We thank the authors for releasing their code and pretrained teachers.

The code in `rdx/` reimplements the rank normalization, locally-biased difference and affinities of
*Representational Difference Explanations* (Kondapaneni, Mac Aodha and Perona, NeurIPS 2025).

## Citation

```bibtex
@misc{mugisha2026rdd,
  title  = {Representational Difference Distillation},
  author = {Mugisha, Abdulkarim and Irakiza, Divin},
  year   = {2026}
}
```
