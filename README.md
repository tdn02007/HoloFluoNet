# HoloFluoNet

HoloFluoNet is a PyTorch-based framework for multi-task learning from quantitative phase images. It predicts semantic segmentation masks (live, dead, background, nuclei) and distance maps, and post-processes results using marker-based watershed and image composition.

---

## 🧪 Training (`train.py`)

```bash
python ./model/train.py --model_name HoloFluoNet
```

### Key Arguments

| Argument        | Description                           | Default |
|-----------------|---------------------------------------|---------|
| `--aspp`        | Use ASPP module                       | True    |
| `--cbam`        | Use CBAM module                       | True    |
| `--inclusion`   | Use inclusion loss                    | True    |
| `--exclusion`   | Use exclusion loss                    | True    |
| `--batch_size`  | Training batch size                   | 24      |
| `--num_epoch`   | Number of training epochs             | 100     |

---

## 🧪 Inference (`test.py`)

```bash
python ./model/test.py --model_name HoloFluoNet
```

- Outputs predictions to: `../result_mask_HoloFluoNet/model_result/`
- Generates:
  - `fake/`: predicted [input | background | live | dead | nuclei | distance]
  - `label/`: ground truth comparison

---

## 🧩 Post-processing Pipeline

### 1. Split prediction into components

```bash
python ./post-processing/split_data.py
```
→ Saves individual channels into `split_data/`

### 2. Apply watershed (per class)

```bash
python ./post-processing/watershed.py
```
→ Uses distance + live/dead masks to refine boundaries

### 3. Overlay on background

```bash
python ./post-processing/concat_img.py
```
→ Produces RGB overlays in `final_results/`

---

## 📦 Dataset Format

```
data/
├── train_data/
│   ├── phase/
│   ├── background/
│   ├── live/
│   ├── dead/
│   ├── nuclei/
│   └── distance/
├── test_data/
    └── ...
```

- Images should be `.png` or `.tif`, and spatially aligned by filename.

---

## 🧠 Loss Functions

- `DiceCELoss`: Main loss for semantic masks
- `InclusionLoss`: Penalizes missing nuclei inside live cells
- `ExclusionLoss`: Penalizes false positive dead predictions
- `GANLoss`: PatchGAN-based adversarial supervision

---

## ✍️ Author

This framework was developed for automated segmentation of label-free cell imaging using multi-task learning and adversarial refinement.

---