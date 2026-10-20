# Generative synthesis enables universal brain segmentation across the lifespan and imaging modalities

## Release notes

## Introduction

This repository is for the proposed UBrainSeg used for automatic brain tissue segmentation (labels: 0 = background, 1 = CSF, 2 = GM, 3 = WM) in MR images. It contains the inference code, the model weights, and the Image Synthesis Module (ISM) used to generate the synthetic pre-training data. Please refer to the below paper for the detailed description.

Chen, Y., Sun, T., Wang, X., Li, X., et al. Generative synthesis enables universal brain segmentation across the lifespan and imaging modalities. Under review.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python >= 3.9, PyTorch >= 2.0 (CUDA recommended), and
[dynamic-network-architectures](https://pypi.org/project/dynamic-network-architectures/) (pip installable).

## Model weights

The weights (~120 MB) exceed GitHub's file size limit and are hosted on the
[Releases](../../releases) page: download `ubrainseg_release_weights.zip` and unzip it into the
repository root (`weights/`). The frozen CLIP text encoder is not included; it is loaded from
the public checkpoint `openai/clip-vit-base-patch32` (auto-downloaded on first use) or from a
local directory via the `clip_ckpt` argument.

## Inference

```bash
python inference.py \
    --image input.nii.gz \
    --output segmentation.nii.gz \
    --weights_dir weights \
    --age "54 years" --gender male --modality DWI \
    --gpu 0
```

Or in Python:

```python
import SimpleITK as sitk
from ubrainseg import UBrainSegModel

model = UBrainSegModel()
model.load(model_pth='weights/ubrainseg_state_dict.pth',
           plans_pth='weights/deploy_plans.json',
           gpu_id=0, clip_ckpt='openai/clip-vit-base-patch32')

text = {'age': '54 years', 'gender': 'male', 'modality': 'DWI'}
seg = model(sitk.ReadImage('input.nii.gz'), text)  # text=None to bypass the text branch
sitk.WriteImage(seg, 'segmentation.nii.gz')
```

`age`, `gender`, and `modality` are optional — if any field is missing, the case is segmented
from visual features alone. Run `python inference.py --help` for all options.

**CT scans:** pass `--ct` (and omit `--modality`) — the input is preprocessed
specifically for the CT modality:

```bash
python inference.py --image ct_scan.nii.gz --output segmentation.nii.gz --ct
```

## Image Synthesis Module (ISM)

`ubrainseg.ism` generates synthetic (image, label) pairs from tissue segmentation masks,
following the deform → sample → augment process described in the paper:

```bash
python synthesize.py \
    --mask tissue_mask.nii.gz \
    --output_dir out/ \
    --count 5 --seed 42
```

## Demo data

The `data/` directory contains 50 demo cases — 7 per MR modality (T1, T2, SWI, DWI, Phase, FLAIR) and 8 CT — with ground-truth segmentations (`labels/`) and per-case metadata (`texts/`).
