# Design: `datasets/landmark_dataset.py`

## Context

MSc thesis — adapting EoMT from semantic segmentation to heatmap-based landmark detection
for fetal biometry (BPD & OFD). This document covers the design of the data loading module.

---

## 1. Data Format

### CSV Schema

```
image_name, scale, center_w, center_h,
ofd_1_x, ofd_1_y, ofd_2_x, ofd_2_y,
bpd_1_x, bpd_1_y, bpd_2_x, bpd_2_y,
SubjectID, px_to_mm_rate, Algo, Split
```

- Landmark coordinates are in the **original image pixel space**
- `scale`, `center_w/h` are HRNet-style crop parameters — not used for initial training
- `px_to_mm_rate` is kept in records for NME evaluation but not used during training

### Files

| File | Rows (excl. header) | Purpose |
|------|---------------------|---------|
| `annotations/UCL/Head_Train.csv` | 110 | → split into train + val |
| `annotations/UCL/Head_Test.csv`  |  49 | test set (fixed) |

### Images

- Location: `images/UCL/Head/`
- Format: JPEG (`.jpg` or `.jpeg`)
- Dimensions: **variable** — most are 960×720, but e.g. `001_HC.jpg` is 1136×783
- Total: 159 images

---

## 2. Module Structure

```
datasets/landmark_dataset.py
├── generate_heatmap(x, y, heatmap_size, sigma)   — pure function, single Gaussian blob
├── LandmarkDataset(torch.utils.data.Dataset)      — core dataset, one split
└── HeadLandmarkDataModule(lightning.LightningDataModule)  — wraps train/val/test
```

**Rationale for not reusing existing EOMT classes:**

- `datasets/dataset.py` — tightly coupled to segmentation masks, zip files, `tv_tensors.Mask`
- `datasets/transforms.py` — operates on `tv_tensors.Mask`; incompatible with heatmap targets
- `datasets/lightning_data_module.py` — `train_collate` keeps targets as a list; we want to stack heatmaps too

Both `LandmarkDataset` and `HeadLandmarkDataModule` are written fresh and self-contained.

---

## 3. `generate_heatmap`

Replicates `generate_target()` from HRNet (`lib/utils/transforms.py`), σ=1.0 on 64×64:

```python
def generate_heatmap(
    x: float,          # landmark x in heatmap coordinate space
    y: float,          # landmark y in heatmap coordinate space
    heatmap_size: tuple[int, int] = (64, 64),
    sigma: float = 1.0,
) -> np.ndarray:       # shape (H, W), float32, values in [0, 1]
```

**Algorithm** (integer-rounded Gaussian patch, clamped at boundaries):

```
mu_x = round(x),  mu_y = round(y)
patch_radius = 3 * sigma             # → 3 pixels for sigma=1
patch_size   = 2 * patch_radius + 1  # → 7×7

# Build 7×7 Gaussian kernel centered at (patch_radius, patch_radius)
# Copy into heatmap, clipping at image boundaries
```

Peak value = 1.0. Out-of-bounds landmarks produce an all-zero heatmap (no crash).

---

## 4. `LandmarkDataset`

### Constructor

```python
LandmarkDataset(
    records:       list[dict],     # see record schema below
    img_size:      tuple[int,int], # model input, e.g. (512, 512)
    heatmap_size:  tuple[int,int], # e.g. (64, 64)
    sigma:         float,          # Gaussian sigma in heatmap space
    augment:       bool,           # True for train split only
)
```

### Record schema

```python
{
    "img_path":      Path,                     # absolute path to JPEG
    "landmarks":     np.ndarray,               # shape (num_landmarks, 2), float, original pixels
    "px_to_mm_rate": float | None,             # for NME; None if missing
}
```

### `__getitem__` pipeline

```
1. Load image
   PIL.open(img_path).convert("RGB") → (orig_w, orig_h)

2. Resize to img_size
   PIL resize (BILINEAR) → (512, 512)

3. Scale landmarks to img_size space
   x_new = x_orig × (512 / orig_w)
   y_new = y_orig × (512 / orig_h)

4. [Train only] Random horizontal flip  (p=0.5)
   img = hflip(img)
   x_new = 512 − 1 − x_new            # flip x; y unchanged

5. DOD sort: sort landmarks by x_new ascending
   channel 0 = left endpoint (lower x)
   channel 1 = right endpoint (higher x)
   → ensures consistent query-to-landmark assignment after any flip

6. [Train only] Color jitter
   Applied only to image tensor; heatmaps unaffected.
   Conservative params for grayscale ultrasound:
     brightness ±0.2, contrast ±0.2, saturation ±0.1

7. Map landmarks to heatmap coordinate space
   x_hm = x_new × (64 / 512)
   y_hm = y_new × (64 / 512)

8. Generate heatmaps
   For each landmark → generate_heatmap(x_hm, y_hm, heatmap_size, sigma)
   Stack → heatmaps: float32 tensor (num_landmarks, 64, 64)

9. Normalize image
   ToTensor (→ float32 [0,1]) + ImageNet mean/std
   → img: float32 tensor (3, 512, 512)

10. Return
    (img, heatmaps)
```

### Why DOD sort (step 5)?

BPD and OFD are diameters — the two endpoints have no intrinsic "identity" in the CSV. After a horizontal flip, the left and right endpoints swap. Without sorting, the network would need to learn two conflicting mappings (query 0 → left and query 0 → right, depending on augmentation). Sorting by x enforces: **query 0 always = left endpoint**, making the training signal consistent. This is a simple implicit normalization; full DOD reassignment (across the whole dataset, before training) can be added as a later ablation.

---

## 5. `HeadLandmarkDataModule`

### Constructor

```python
HeadLandmarkDataModule(
    images_dir:      str | Path,   # path to UCL Head image folder
    ann_train_csv:   str | Path,   # Head_Train.csv
    ann_test_csv:    str | Path,   # Head_Test.csv
    task:            str,          # "bpd" or "ofd"
    img_size:        tuple[int,int] = (512, 512),
    heatmap_size:    tuple[int,int] = (64, 64),
    sigma:           float          = 1.0,
    val_fraction:    float          = 0.1,
    val_split_seed:  int            = 42,
    batch_size:      int            = 16,
    num_workers:     int            = 4,
)
```

### Task → CSV columns mapping

| task  | landmark 0 cols       | landmark 1 cols       |
|-------|----------------------|-----------------------|
| `bpd` | `bpd_1_x`, `bpd_1_y` | `bpd_2_x`, `bpd_2_y` |
| `ofd` | `ofd_1_x`, `ofd_1_y` | `ofd_2_x`, `ofd_2_y` |

### Val split strategy

Split is done **by subject** (group of frames from the same patient), not by frame, to prevent data leakage.

```
Subject ID = numeric prefix of image filename
  e.g. "002_HC.jpeg" → subject "002"
       "022_3HC.jpeg" → subject "022"

Algorithm:
  1. Collect unique subject IDs from Head_Train.csv
  2. Shuffle subjects with val_split_seed
  3. Last ceil(N_subjects × val_fraction) subjects → val
  4. Remaining subjects → train
```

For 110 frames from ~15–20 subjects, `val_fraction=0.1` gives roughly 10–15 val frames.

### `setup()`

```python
def setup(self, stage=None):
    train_records, val_records = self._parse_and_split(self.ann_train_csv)
    test_records = self._parse_csv(self.ann_test_csv)

    self.train_dataset = LandmarkDataset(train_records, ..., augment=True)
    self.val_dataset   = LandmarkDataset(val_records,   ..., augment=False)
    self.test_dataset  = LandmarkDataset(test_records,  ..., augment=False)
```

### DataLoaders

```python
train_dataloader() → shuffle=True,  drop_last=True,  collate_fn=landmark_collate
val_dataloader()   → shuffle=False, drop_last=False, collate_fn=landmark_collate
test_dataloader()  → shuffle=False, drop_last=False, collate_fn=landmark_collate
```

### `landmark_collate`

Since all images are resized to the same `img_size`, both images and heatmaps can be stacked:

```python
def landmark_collate(batch):
    imgs, heatmaps = zip(*batch)
    return torch.stack(imgs), torch.stack(heatmaps)
    # → (B, 3, 512, 512),  (B, num_landmarks, 64, 64)
```

---

## 6. Open Questions (to confirm before coding)

| # | Question | Recommendation |
|---|----------|----------------|
| 1 | `img_size`? | `(512, 512)` — matches EOMT segmentation baseline |
| 2 | Val split: by subject or random? | **By subject** — prevents same-patient leakage |
| 3 | DOD sort (x ascending) in initial version? | **Yes** — required for flip augmentation consistency |
| 4 | Start with `task="bpd"` or `"ofd"`, or support both via parameter? | **Parameter** — one `HeadLandmarkDataModule` covers both |
| 5 | `num_workers` default? | `4` (AutoDL), set to `0` for local CPU debug |

---

## 7. What's NOT included (future ablations)

| Item | Reason deferred |
|------|-----------------|
| Random scale jitter + crop | Adds complexity; pure resize is simpler to debug first |
| Random rotation | Rotated ultrasound is less common; add if NME is high |
| Full DOD reassignment | Requires cross-dataset analysis; x-sort covers the flip case |
| Multi-task (BPD + OFD simultaneously) | Out of scope for initial architecture |
| `px_to_mm_rate` in loss | Not needed for training; only for physical-unit NME reporting |

---

## 8. File Dependencies

```
datasets/
├── landmark_dataset.py   ← new file (self-contained)
├── ade20k_semantic.py    ← unchanged
├── dataset.py            ← unchanged
├── transforms.py         ← unchanged
└── lightning_data_module.py  ← unchanged
```

`landmark_dataset.py` imports only: `pathlib`, `csv`, `numpy`, `PIL`, `torch`, `torchvision`, `lightning`.

---

## 9. Verification Plan (local CPU)

```python
# Quick smoke test — run on Windows/WSL before AutoDL
dm = HeadLandmarkDataModule(
    images_dir   = "d:/download/Project coding/msc/Muti/MultiCentre-Fetal-Biometry-2025/images/UCL/Head",
    ann_train_csv= "d:/download/.../annotations/UCL/Head_Train.csv",
    ann_test_csv = "d:/download/.../annotations/UCL/Head_Test.csv",
    task         = "bpd",
    batch_size   = 4,
    num_workers  = 0,
)
dm.setup()
imgs, heatmaps = next(iter(dm.train_dataloader()))
assert imgs.shape     == (4, 3, 512, 512)
assert heatmaps.shape == (4, 2, 64, 64)
assert heatmaps.max() <= 1.0
assert heatmaps.max() > 0.0   # at least one landmark visible
```

Also: visualize one sample (overlay heatmap argmax on image) to sanity-check coordinate scaling.
