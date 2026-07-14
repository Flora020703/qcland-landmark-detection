# ---------------------------------------------------------------
# MODIFIED: new file — 300W face landmark detection dataset (68 points),
# Split 2 protocol (train=AFW+LFPW-train+HELEN-train, test=common+challenging).
# Mirrors datasets/landmark_dataset.py's (img, heatmaps, coords) interface so
# LandmarkDetection / main_landmark.py work with this DataModule unmodified.
# ---------------------------------------------------------------

from pathlib import Path
from typing import Optional, Union

import numpy as np
import scipy.io as sio
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader
import lightning

from datasets.landmark_dataset import generate_heatmap, seed_worker

# 68-point horizontal-flip symmetric pairs (0-indexed).
# Verified against HRNet-Facial-Landmark-Detection's MATCHED_PARTS['300W']
# (there given 1-indexed; converted to 0-indexed here). Earlier drafts of
# this table were missing (65, 67) - the inner-lip-bottom symmetric pair -
# which would silently mislabel those two points on every horizontal flip.
FLIP_PAIRS_68 = [
    (0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9),  # jaw
    (17, 26), (18, 25), (19, 24), (20, 23), (21, 22),  # eyebrows
    (31, 35), (32, 34),  # nose (nostrils)
    (36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46),  # eyes
    (48, 54), (49, 53), (50, 52),  # outer lip (top/corners)
    (55, 59), (56, 58),  # outer lip (bottom)
    (60, 64), (61, 63), (65, 67),  # inner lip
]

# Standard 300W "inter-ocular distance" = outer eye corners, NOT eye-centre
# means. Confirmed against HRNet-Facial-Landmark-Detection lib/core/
# evaluation.py: `interocular = np.linalg.norm(pts_gt[36] - pts_gt[45])`.
# Pass this as LandmarkDetection's nme_norm_pair for 300W configs.
INTER_OCULAR_PAIR = (36, 45)

# source dir (relative to data_root) -> which final test bucket it feeds
_TRAIN_DIRS = ["afw", "lfpw/trainset", "helen/trainset"]              # 337+811+2000 = 3148
_TEST_DIRS_COMMON = ["lfpw/testset", "helen/testset"]                  # 224+330 = 554
_TEST_DIRS_CHALLENGING = ["ibug"]                                      # 135

_BBOX_FILES = {
    "afw": "bounding_boxes_afw.mat",
    "lfpw/trainset": "bounding_boxes_lfpw_trainset.mat",
    "lfpw/testset": "bounding_boxes_lfpw_testset.mat",
    "helen/trainset": "bounding_boxes_helen_trainset.mat",
    "helen/testset": "bounding_boxes_helen_testset.mat",
    "ibug": "bounding_boxes_ibug.mat",
}

BBOX_PAD_FACTOR = 1.25  # matches HRNet-Facial-Landmark-Detection's crop padding


def _load_pts(path: Path) -> np.ndarray:
    """
    Parse an iBUG/300W .pts file into a (68, 2) float32 array in 0-indexed
    pixel coordinates.

    MODIFIED: 300W .pts coordinates use the MATLAB 1-indexed convention
    (top-left pixel = (1,1)) - subtract 1 to align with Python/PIL's
    0-indexed convention (confirmed against ibug.doc.ic.ac.uk documentation).
    """
    lines = path.read_text().splitlines()
    coord_lines = [
        l for l in lines
        if l.strip() and l.strip() not in ("{", "}")
        and not l.startswith("version") and not l.startswith("n_points")
    ]
    pts = np.array([[float(v) for v in l.split()] for l in coord_lines], dtype=np.float32)
    assert pts.shape == (68, 2), f"{path}: expected 68 points, got {pts.shape}"
    return pts - 1.0


_BBOX_DIR_CANDIDATES = ["Bounding Boxes", "Bounding_Boxes", "bounding_boxes"]


def _resolve_bbox_dir(data_root: Path) -> Path:
    """
    The 300W bounding-box folder is distributed as "Bounding Boxes" (with a
    space), but scp/zip/unzip round-trips across Windows<->Linux commonly
    turn that space into an underscore (or something else). Try the known
    variants rather than hardcoding one exact name.
    """
    for name in _BBOX_DIR_CANDIDATES:
        candidate = data_root / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find a bounding-box folder under {data_root} "
        f"(tried: {_BBOX_DIR_CANDIDATES})"
    )


def _load_bbox_map(mat_path: Path) -> dict:
    """
    Parse a 300W `bounding_boxes_*.mat` file into {image_name: [x1,y1,x2,y2]}
    using the ground-truth box (bb_ground_truth), not the detector box.

    MODIFIED: some of these .mat files contain more entries than their
    corresponding image folder - bounding_boxes_ibug.mat has 337 entries
    (135 actually named like the ibug/ folder's images + 202 AFW-named
    entries, likely a distribution artifact), not just 135. Build a
    name->bbox lookup and let the caller only query the names it actually
    has images for; never assume len(mat) == number of images on disk.
    """
    mat = sio.loadmat(str(mat_path))
    bb = mat["bounding_boxes"]
    out = {}
    for i in range(bb.shape[1]):
        entry = bb[0, i][0, 0]
        name = str(entry["imgName"][0])
        box = np.asarray(entry["bb_ground_truth"], dtype=np.float32).reshape(4)
        out[name] = box
    return out


def bbox_to_center_side(box: np.ndarray) -> tuple[float, float, float]:
    """
    Convert [x1,y1,x2,y2] ground-truth bbox to (center_x, center_y, side),
    where `side` is the padded square crop side length in original-image
    pixels. Matches HRNet-Facial-Landmark-Detection's convention (scale in
    units of 200px, x1.25 padding), so crops are comparable to the cited
    baseline: scale = max(w,h)/200, side = scale * 1.25 * 200 = max(w,h)*1.25.
    """
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * BBOX_PAD_FACTOR
    return cx, cy, side


class Face300WDataset(torch.utils.data.Dataset):
    """
    Per-split dataset for 300W face landmark detection (68 points).

    Each item: (img_tensor [3, H, W], heatmaps [68, hm_H, hm_W], coords [68, 2])
    - same interface as datasets.landmark_dataset.LandmarkDataset.

    records: list of dicts with img_path (Path), landmarks (68,2 float32,
             0-indexed original-image pixel space), bbox (4,) [x1,y1,x2,y2].

    Unlike LandmarkDataset's BPD/OFD 2-point case (which sorts landmarks by
    x after a flip, since the two points are interchangeable), 300W's 68
    points are each a fixed, semantically distinct query target - a flip
    must swap each FLIP_PAIRS_68 pair explicitly rather than re-sorting.
    """

    def __init__(
        self,
        records: list[dict],
        img_size: tuple[int, int] = (256, 256),
        heatmap_size: tuple[int, int] = (64, 64),
        sigma: float = 1.5,
        augment: bool = False,
        pixel_center_align: bool = True,
        flip_prob: float = 0.5,
    ):
        self.records = records
        self.img_size = img_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.augment = augment
        self.pixel_center_align = pixel_center_align
        self.flip_prob = flip_prob

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        rec = self.records[index]
        img = Image.open(rec["img_path"]).convert("RGB")
        orig_w, orig_h = img.size

        # 1. bbox -> square crop region (integer-pixel, black-padded if it
        #    extends past the original image edges so the crop is never
        #    aspect-distorted and never index-errors near the border).
        cx, cy, side = bbox_to_center_side(rec["bbox"])
        side_i = max(1, int(round(side)))
        x0 = int(round(cx - side / 2.0))
        y0 = int(round(cy - side / 2.0))

        crop = Image.new("RGB", (side_i, side_i))
        src_x0, src_y0 = max(0, x0), max(0, y0)
        src_x1, src_y1 = min(orig_w, x0 + side_i), min(orig_h, y0 + side_i)
        dst_x0, dst_y0 = src_x0 - x0, src_y0 - y0
        if src_x1 > src_x0 and src_y1 > src_y0:
            crop.paste(img.crop((src_x0, src_y0, src_x1, src_y1)), (dst_x0, dst_y0))

        # 2. resize crop to model input size
        ih, iw = self.img_size
        crop_resized = crop.resize((iw, ih), Image.BILINEAR)

        # 3. transform landmarks: original -> crop-local (subtract the same
        #    integer x0/y0 used for the image crop, not the float bbox
        #    centre, so image and label stay pixel-consistent) -> img_size
        lms = rec["landmarks"].copy()  # (68, 2)
        lms[:, 0] = lms[:, 0] - x0
        lms[:, 1] = lms[:, 1] - y0
        if self.pixel_center_align:
            lms[:, 0] = (lms[:, 0] + 0.5) * (iw / side_i) - 0.5
            lms[:, 1] = (lms[:, 1] + 0.5) * (ih / side_i) - 0.5
        else:
            lms[:, 0] *= iw / side_i
            lms[:, 1] *= ih / side_i

        img_pil = crop_resized

        # 4. random horizontal flip (train only) - explicit FLIP_PAIRS_68
        #    swap so each of the 68 query tokens keeps a fixed anatomical
        #    identity (see class docstring).
        if self.augment and torch.rand(()) < self.flip_prob:
            img_pil = TF.hflip(img_pil)
            lms[:, 0] = iw - 1.0 - lms[:, 0]
            flipped = lms.copy()
            for a, b in FLIP_PAIRS_68:
                flipped[a], flipped[b] = lms[b].copy(), lms[a].copy()
            lms = flipped

        # 5. map to heatmap coordinate space
        hm_h, hm_w = self.heatmap_size
        lms_hm = lms.copy()
        if self.pixel_center_align:
            lms_hm[:, 0] = (lms_hm[:, 0] + 0.5) * (hm_w / iw) - 0.5
            lms_hm[:, 1] = (lms_hm[:, 1] + 0.5) * (hm_h / ih) - 0.5
        else:
            lms_hm[:, 0] *= hm_w / iw
            lms_hm[:, 1] *= hm_h / ih

        # 6. one Gaussian heatmap per landmark
        heatmaps = np.stack([
            generate_heatmap(lms_hm[i, 0], lms_hm[i, 1], (hm_h, hm_w), self.sigma)
            for i in range(len(lms_hm))
        ])  # (68, hm_h, hm_w)

        img_t = TF.to_tensor(img_pil)  # (3, H, W) float32 [0, 1]

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

        return img_t, torch.from_numpy(heatmaps), torch.from_numpy(lms_hm).float()


class Face300WDataModule(lightning.LightningDataModule):
    """
    LightningDataModule for 300W face landmark detection (68 points).

    Split 2 protocol (the one HRNet and most modern 300W papers report):
      train = AFW (337) + LFPW-trainset (811) + HELEN-trainset (2000) = 3148
      test  = common (LFPW-testset 224 + HELEN-testset 330 = 554)
              + challenging (iBUG, 135) = 689 total

    val is carved out of the train pool by a plain image-level random split
    (val_split_seed) - unlike UCL fetal data, 300W images are independent
    photos of different people with no repeated-subject leakage risk, so
    there's no need for BPD/OFD's subject-grouped split.

    Args:
        data_root   : root dir containing afw/, helen/, ibug/, lfpw/, and a
                      bounding-box folder (see _BBOX_DIR_CANDIDATES for the
                      accepted name variants - "Bounding Boxes" vs
                      "Bounding_Boxes" etc., since scp/zip round-trips often
                      mangle the space) (e.g. D:/.../300w or /root/autodl-tmp/300w)
        test_subset : "common" | "challenging" | "full" - selects which test
                      images populate test_dataset, so common/challenging/full
                      NME can each be obtained via a separate `main_landmark.py
                      test` invocation against the same trained checkpoint,
                      mirroring how the BPD/OFD ablation scripts test the same
                      checkpoint against multiple criteria.
        pixel_center_align: see datasets.landmark_dataset.LandmarkDataset -
                      same UDP-style fix, default True here (300W is a new
                      pipeline with no historical runs to stay bit-compatible
                      with, so there's no reason to default it off).
        augment     : if True, enables hflip + colour jitter (see
                      Face300WDataset). Default False - ship an
                      augmentation-free baseline first per the project's own
                      "one variable at a time" rule; flip-pair correctness is
                      still needed even with augment=False verified upfront,
                      since it's the kind of subtle bug that's easy to miss
                      once augmentation is switched on later.
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        img_size: tuple[int, int] = (256, 256),
        heatmap_size: tuple[int, int] = (64, 64),
        sigma: float = 1.5,
        val_fraction: float = 0.1,
        val_split_seed: int = 42,
        test_subset: str = "full",
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = True,
        pixel_center_align: bool = True,
        augment: bool = False,
        loader_seed: Optional[int] = None,
    ):
        super().__init__()
        assert test_subset in ("common", "challenging", "full"), test_subset
        self.data_root = Path(data_root)
        self.img_size = img_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.val_fraction = val_fraction
        self.val_split_seed = val_split_seed
        self.test_subset = test_subset
        self.pixel_center_align = pixel_center_align
        self.augment = augment
        self.loader_seed = loader_seed
        self.dataloader_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "persistent_workers": num_workers > 0,
        }
        self.save_hyperparameters(ignore=["_class_path"])

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _load_dir(self, rel_dir: str) -> list[dict]:
        d = self.data_root / rel_dir
        bbox_map = _load_bbox_map(_resolve_bbox_dir(self.data_root) / _BBOX_FILES[rel_dir])
        records = []
        for pts_path in sorted(d.glob("*.pts")):
            img_path = pts_path.with_suffix(".jpg")
            if not img_path.exists():
                img_path = pts_path.with_suffix(".png")
            if not img_path.exists():
                continue
            name = img_path.name
            if name not in bbox_map:
                continue  # tolerate missing bbox entries without crashing
            records.append({
                "img_path": img_path,
                "img_name": name,
                "landmarks": _load_pts(pts_path),
                "bbox": bbox_map[name],
            })
        return records

    # ------------------------------------------------------------------
    # Lightning interface
    # ------------------------------------------------------------------

    def setup(self, stage: Optional[str] = None):
        train_pool = []
        for rel_dir in _TRAIN_DIRS:
            train_pool.extend(self._load_dir(rel_dir))

        rng = np.random.default_rng(self.val_split_seed)
        idx = np.arange(len(train_pool))
        rng.shuffle(idx)
        n_val = max(1, int(round(len(train_pool) * self.val_fraction)))
        val_idx = set(idx[:n_val].tolist())

        train_recs = [r for i, r in enumerate(train_pool) if i not in val_idx]
        val_recs   = [r for i, r in enumerate(train_pool) if i in val_idx]

        common_recs = []
        for rel_dir in _TEST_DIRS_COMMON:
            common_recs.extend(self._load_dir(rel_dir))
        challenging_recs = []
        for rel_dir in _TEST_DIRS_CHALLENGING:
            challenging_recs.extend(self._load_dir(rel_dir))

        if self.test_subset == "common":
            test_recs = common_recs
        elif self.test_subset == "challenging":
            test_recs = challenging_recs
        else:
            test_recs = common_recs + challenging_recs

        ds_kw = dict(
            img_size=self.img_size,
            heatmap_size=self.heatmap_size,
            sigma=self.sigma,
            pixel_center_align=self.pixel_center_align,
        )
        self.train_dataset = Face300WDataset(train_recs, augment=self.augment, **ds_kw)
        self.val_dataset   = Face300WDataset(val_recs,   augment=False, **ds_kw)
        self.test_dataset  = Face300WDataset(test_recs,  augment=False, **ds_kw)
        return self

    @staticmethod
    def collate(batch):
        imgs, heatmaps, coords = zip(*batch)
        return torch.stack(imgs), torch.stack(heatmaps), torch.stack(coords)

    def _loader_extra_kwargs(self, offset: int) -> dict:
        if self.loader_seed is None:
            return {}
        generator = torch.Generator()
        generator.manual_seed(self.loader_seed + offset)
        return {"generator": generator, "worker_init_fn": seed_worker}

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.collate,
            **self.dataloader_kwargs,
            **self._loader_extra_kwargs(offset=0),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            drop_last=False,
            collate_fn=self.collate,
            **self.dataloader_kwargs,
            **self._loader_extra_kwargs(offset=1),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            drop_last=False,
            collate_fn=self.collate,
            **self.dataloader_kwargs,
            **self._loader_extra_kwargs(offset=2),
        )
