# ---------------------------------------------------------------
# MODIFIED: new file — heatmap-based landmark detection dataset
# for UCL fetal biometry (BPD / OFD), replacing segmentation data pipeline.
# ---------------------------------------------------------------

import csv
import math
import re
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader
import lightning


# CSV column pairs for each task (x_col, y_col) per landmark
TASK_COLS = {
    "bpd": [("bpd_1_x", "bpd_1_y"), ("bpd_2_x", "bpd_2_y")],
    "ofd": [("ofd_1_x", "ofd_1_y"), ("ofd_2_x", "ofd_2_y")],
}


def generate_heatmap(
    x: float,
    y: float,
    heatmap_size: tuple[int, int] = (64, 64),
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Single Gaussian heatmap centred at (x, y) in heatmap coordinate space.
    Replicates HRNet lib/utils/transforms.py:generate_target() exactly.

    Args:
        x, y       : landmark position in heatmap pixel space (float)
        heatmap_size: (H, W)
        sigma      : Gaussian std in heatmap pixels (HRNet default = 1.0)

    Returns:
        float32 ndarray of shape (H, W), peak value = 1.0.
        All-zero if landmark is outside heatmap bounds.
    """
    H, W = heatmap_size
    heatmap = np.zeros((H, W), dtype=np.float32)

    mu_x = int(x + 0.5)  # column index
    mu_y = int(y + 0.5)  # row index

    tmp_size = int(sigma * 3)
    size = 2 * tmp_size + 1

    xs = np.arange(0, size, dtype=np.float32)
    ys = xs[:, np.newaxis]
    x0 = y0 = tmp_size
    g = np.exp(-((xs - x0) ** 2 + (ys - y0) ** 2) / (2 * sigma ** 2))

    ul = [mu_x - tmp_size, mu_y - tmp_size]
    br = [mu_x + tmp_size + 1, mu_y + tmp_size + 1]

    # source patch slice (g)
    g_x0 = max(0, -ul[0]);  g_x1 = min(br[0], W) - ul[0]
    g_y0 = max(0, -ul[1]);  g_y1 = min(br[1], H) - ul[1]
    # destination slice (heatmap)
    h_x0 = max(0, ul[0]);   h_x1 = min(br[0], W)
    h_y0 = max(0, ul[1]);   h_y1 = min(br[1], H)

    if g_x1 > g_x0 and g_y1 > g_y0:
        heatmap[h_y0:h_y1, h_x0:h_x1] = g[g_y0:g_y1, g_x0:g_x1]

    return heatmap


class LandmarkDataset(torch.utils.data.Dataset):
    """
    Per-split dataset for fetal landmark detection.

    Each item: (img_tensor [3, H, W],  heatmaps [num_landmarks, hm_H, hm_W])

    records   : list of dicts produced by HeadLandmarkDataModule._parse_csv()
                Required keys: img_path (Path), landmarks (ndarray [N,2] float32
                in original image pixel space).
    img_size  : (H, W) — model input resolution; images are resized to this.
    heatmap_size: (H, W) — heatmap output resolution, typically (64, 64).
    augment   : if True, applies random hflip + conservative colour jitter (train only).
    """

    def __init__(
        self,
        records: list[dict],
        img_size: tuple[int, int] = (512, 512),
        heatmap_size: tuple[int, int] = (64, 64),
        sigma: float = 1.0,
        augment: bool = False,
    ):
        self.records = records
        self.img_size = img_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        rec = self.records[index]

        # 1. load image
        img = Image.open(rec["img_path"]).convert("RGB")
        orig_w, orig_h = img.size  # PIL convention: (W, H)

        # 2. resize to model input size
        ih, iw = self.img_size
        img = img.resize((iw, ih), Image.BILINEAR)

        # 3. scale landmarks to img_size pixel space
        lms = rec["landmarks"].copy()  # (N, 2) float32: col 0 = x, col 1 = y
        lms[:, 0] *= iw / orig_w
        lms[:, 1] *= ih / orig_h

        # 4. random horizontal flip (train only)
        if self.augment and torch.rand(()) < 0.5:
            img = TF.hflip(img)
            lms[:, 0] = iw - 1.0 - lms[:, 0]

        # 5. DOD sort: channel 0 = left endpoint (lower x), channel 1 = right
        #    ensures consistent query-to-landmark assignment after any flip
        order = np.argsort(lms[:, 0])
        lms = lms[order]

        # 6. map landmarks to heatmap coordinate space
        hm_h, hm_w = self.heatmap_size
        lms_hm = lms.copy()
        lms_hm[:, 0] *= hm_w / iw
        lms_hm[:, 1] *= hm_h / ih

        # 7. generate heatmaps
        heatmaps = np.stack([
            generate_heatmap(lms_hm[i, 0], lms_hm[i, 1], (hm_h, hm_w), self.sigma)
            for i in range(len(lms_hm))
        ])  # (N, hm_h, hm_w)

        # 8. image → float tensor [0, 1]
        img_t = TF.to_tensor(img)  # (3, H, W) float32

        # 9. colour jitter (train only, conservative for ultrasound)
        if self.augment:
            if torch.rand(()) < 0.5:
                img_t = TF.adjust_brightness(
                    img_t, 1.0 + float(torch.empty(1).uniform_(-0.2, 0.2)))
            if torch.rand(()) < 0.5:
                img_t = TF.adjust_contrast(
                    img_t, 1.0 + float(torch.empty(1).uniform_(-0.2, 0.2)))
            if torch.rand(()) < 0.5:
                img_t = TF.adjust_saturation(
                    img_t, 1.0 + float(torch.empty(1).uniform_(-0.1, 0.1)))
            img_t = img_t.clamp(0.0, 1.0)

        # 10. Return [0, 1] float tensor — EoMT encoder applies ImageNet normalisation
        #     internally via pixel_mean / pixel_std buffers (models/vit.py).
        #     Applying it here as well would double-normalise.
        # lms_hm: float GT coords in heatmap space — used for NME (avoids quantisation
        # error from decoding heatmaps with argmax).
        return img_t, torch.from_numpy(heatmaps), torch.from_numpy(lms_hm).float()


class HeadLandmarkDataModule(lightning.LightningDataModule):
    """
    LightningDataModule for UCL Head fetal biometry landmark detection.

    Reads Head_Train.csv / Head_Test.csv, splits train by subject (to prevent
    same-patient leakage), and exposes train / val / test DataLoaders.

    Args:
        images_dir    : directory containing the JPEG images
        ann_train_csv : path to Head_Train.csv
        ann_test_csv  : path to Head_Test.csv
        task          : "bpd" or "ofd"
        img_size      : (H, W) model input, default (512, 512)
        heatmap_size  : (H, W) heatmap output, default (64, 64)
        sigma         : Gaussian sigma in heatmap pixels, default 1.0
        val_fraction  : fraction of *subjects* (not frames) held out for val
        val_split_seed: RNG seed for reproducible subject shuffle
        batch_size    : dataloader batch size
        num_workers   : dataloader workers (0 for CPU-only / debug)
        pin_memory    : passed to DataLoader
    """

    def __init__(
        self,
        images_dir: Union[str, Path],
        ann_train_csv: Union[str, Path],
        ann_test_csv: Union[str, Path],
        task: str = "bpd",
        img_size: tuple[int, int] = (512, 512),
        heatmap_size: tuple[int, int] = (64, 64),
        sigma: float = 1.0,
        val_fraction: float = 0.1,
        val_split_seed: int = 42,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()
        assert task in TASK_COLS, f"task must be one of {list(TASK_COLS)}, got {task!r}"

        self.images_dir    = Path(images_dir)
        self.ann_train_csv = Path(ann_train_csv)
        self.ann_test_csv  = Path(ann_test_csv)
        self.task          = task
        self.img_size      = img_size
        self.heatmap_size  = heatmap_size
        self.sigma         = sigma
        self.val_fraction  = val_fraction
        self.val_split_seed = val_split_seed
        self.dataloader_kwargs = {
            "batch_size":        batch_size,
            "num_workers":       num_workers,
            "pin_memory":        pin_memory,
            "persistent_workers": num_workers > 0,
        }
        self.save_hyperparameters(ignore=["_class_path"])

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _subject_id(img_name: str) -> str:
        """
        Extract numeric subject prefix from filename.
        e.g. '022_3HC.jpeg' → '022',  '002_HC.jpeg' → '002'
        """
        m = re.match(r"^(\d+)", img_name)
        return m.group(1) if m else img_name

    def _parse_csv(self, csv_path: Path) -> list[dict]:
        """Read a CSV file and return a list of record dicts."""
        cols = TASK_COLS[self.task]
        records = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                img_path = self.images_dir / row["image_name"]
                if not img_path.exists():
                    continue  # tolerate missing images without crashing

                lms = np.array(
                    [[float(row[xc]), float(row[yc])] for xc, yc in cols],
                    dtype=np.float32,
                )  # (num_landmarks, 2)

                px_to_mm: float | None = None
                raw = row.get("px_to_mm_rate", "")
                if raw:
                    try:
                        px_to_mm = float(raw)
                    except ValueError:
                        pass

                records.append({
                    "img_path":      img_path,
                    "img_name":      row["image_name"],
                    "landmarks":     lms,
                    "px_to_mm_rate": px_to_mm,
                })
        return records

    def _split_by_subject(
        self, records: list[dict]
    ) -> tuple[list[dict], list[dict]]:
        """
        Group frames by subject, shuffle subjects with fixed seed, assign the
        last ceil(N_subjects × val_fraction) subjects to val.
        """
        subjects = sorted({self._subject_id(r["img_name"]) for r in records})
        rng = np.random.default_rng(self.val_split_seed)
        rng.shuffle(subjects)
        n_val = max(1, math.ceil(len(subjects) * self.val_fraction))
        val_set = set(subjects[-n_val:])

        train_recs = [r for r in records if self._subject_id(r["img_name"]) not in val_set]
        val_recs   = [r for r in records if self._subject_id(r["img_name"]) in val_set]
        return train_recs, val_recs

    # ------------------------------------------------------------------
    # Lightning interface
    # ------------------------------------------------------------------

    def setup(self, stage: Union[str, None] = None):
        all_train = self._parse_csv(self.ann_train_csv)
        test_recs = self._parse_csv(self.ann_test_csv)
        train_recs, val_recs = self._split_by_subject(all_train)

        ds_kw = dict(
            img_size=self.img_size,
            heatmap_size=self.heatmap_size,
            sigma=self.sigma,
        )
        self.train_dataset = LandmarkDataset(train_recs, augment=True,  **ds_kw)
        self.val_dataset   = LandmarkDataset(val_recs,   augment=False, **ds_kw)
        self.test_dataset  = LandmarkDataset(test_recs,  augment=False, **ds_kw)
        return self

    @staticmethod
    def collate(batch):
        """Stack images, heatmaps, and GT coords into batched tensors."""
        imgs, heatmaps, coords = zip(*batch)
        return torch.stack(imgs), torch.stack(heatmaps), torch.stack(coords)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            drop_last=False,
            collate_fn=self.collate,
            **self.dataloader_kwargs,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            drop_last=False,
            collate_fn=self.collate,
            **self.dataloader_kwargs,
        )
