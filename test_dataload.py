"""
Smoke-tests for both data pipelines.

Run from the eomt/ directory:
    python test_dataload.py           # both tests
    python test_dataload.py ade20k    # segmentation only
    python test_dataload.py landmark  # landmark only
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from torch.utils.data import DataLoader

from datasets.dataset import Dataset
from datasets.ade20k_semantic import ADE20KSemantic

DATA_ROOT = Path(__file__).parent.parent / "data" / "ADEChallengeData2016" / "ADE20K_mini"
BATCH_SIZE = 2


def main():
    print(f"Data root : {DATA_ROOT}")
    assert DATA_ROOT.exists(), f"Not found: {DATA_ROOT}"

    dataset_kwargs = dict(
        data_path=DATA_ROOT,
        target_data_path=DATA_ROOT,
        img_suffix=".jpg",
        target_suffix=".png",
        target_parser=ADE20KSemantic.target_parser,
        check_empty_targets=False,  # skip pixel-range check to speed up init
    )

    train_ds = Dataset(
        img_folder_path=Path("train/images"),
        target_folder_path=Path("train/masks"),
        **dataset_kwargs,
    )
    val_ds = Dataset(
        img_folder_path=Path("val/images"),
        target_folder_path=Path("val/masks"),
        **dataset_kwargs,
    )

    print(f"Train samples : {len(train_ds)}")
    print(f"Val   samples : {len(val_ds)}")

    # --- single item ---
    img, target = train_ds[0]
    print("\n[single item]")
    print(f"  img    shape={tuple(img.shape)}  dtype={img.dtype}")
    print(f"  masks  shape={tuple(target['masks'].shape)}")
    print(f"  labels={target['labels'].tolist()}")

    # --- batch (plain list collate avoids stacking variable-size images) ---
    def list_collate(batch):
        return batch

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, num_workers=0, collate_fn=list_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, num_workers=0, collate_fn=list_collate
    )

    train_batch = next(iter(train_loader))
    val_batch   = next(iter(val_loader))

    print(f"\n[train batch — {len(train_batch)} samples]")
    for i, (img, tgt) in enumerate(train_batch):
        print(f"  [{i}] img={tuple(img.shape)}  masks={tuple(tgt['masks'].shape)}  n_labels={len(tgt['labels'])}")

    print(f"\n[val batch — {len(val_batch)} samples]")
    for i, (img, tgt) in enumerate(val_batch):
        print(f"  [{i}] img={tuple(img.shape)}  masks={tuple(tgt['masks'].shape)}  n_labels={len(tgt['labels'])}")

    print("\nData loading OK")


def test_landmark():
    """Smoke-test HeadLandmarkDataModule + LandmarkDataset."""
    from datasets.landmark_dataset import HeadLandmarkDataModule, generate_heatmap

    _win = Path("d:/download/Project coding/msc/Muti/MultiCentre-Fetal-Biometry-2025")
    _wsl = Path("/mnt/d/download/Project coding/msc/Muti/MultiCentre-Fetal-Biometry-2025")
    DATA_BASE = _win if _win.exists() else _wsl
    IMAGES_DIR    = DATA_BASE / "images" / "UCL" / "Head"
    TRAIN_CSV     = DATA_BASE / "annotations" / "UCL" / "Head_Train.csv"
    TEST_CSV      = DATA_BASE / "annotations" / "UCL" / "Head_Test.csv"

    print("\n" + "=" * 60)
    print("LANDMARK SMOKE-TEST")
    print("=" * 60)

    # --- generate_heatmap unit check ---
    hm = generate_heatmap(32.0, 32.0, heatmap_size=(64, 64), sigma=1.0)
    assert hm.shape == (64, 64), f"unexpected shape {hm.shape}"
    assert abs(hm[32, 32] - 1.0) < 1e-5, f"peak should be 1.0, got {hm[32,32]}"
    assert hm[32, 35] < 0.02, "value 3px away should be ~exp(-4.5)≈0.011 for sigma=1"
    # out-of-bounds should not crash and return all zeros
    hm_oob = generate_heatmap(-5.0, -5.0, heatmap_size=(64, 64), sigma=1.0)
    assert hm_oob.max() == 0.0, "out-of-bounds landmark should give all-zero heatmap"
    print("[generate_heatmap] OK")

    for task in ("bpd", "ofd"):
        print(f"\n[task={task}]")
        dm = HeadLandmarkDataModule(
            images_dir=IMAGES_DIR,
            ann_train_csv=TRAIN_CSV,
            ann_test_csv=TEST_CSV,
            task=task,
            img_size=(512, 512),
            heatmap_size=(64, 64),
            sigma=1.0,
            val_fraction=0.1,
            batch_size=4,
            num_workers=0,   # CPU / local debug
            pin_memory=False,
        )
        dm.setup()

        n_train = len(dm.train_dataset)
        n_val   = len(dm.val_dataset)
        n_test  = len(dm.test_dataset)
        print(f"  split sizes  train={n_train}  val={n_val}  test={n_test}")
        assert n_train + n_val == len(dm._parse_csv(TRAIN_CSV)), \
            "train+val should equal all Head_Train rows that have images"
        assert n_test == len(dm._parse_csv(TEST_CSV)), \
            "test size mismatch"

        # single item — now returns (img, heatmaps, coords)
        img, heatmaps, coords_item = dm.train_dataset[0]
        assert img.shape      == (3, 512, 512),  f"img shape wrong: {img.shape}"
        assert heatmaps.shape == (2, 64, 64),    f"heatmap shape wrong: {heatmaps.shape}"
        assert coords_item.shape == (2, 2),      f"coords shape wrong: {coords_item.shape}"
        assert heatmaps.max() <= 1.0 + 1e-5,     "heatmap max should be ≤ 1"
        assert heatmaps.max() > 0.0,             "heatmap should have at least one peak"
        print(f"  single item  img={tuple(img.shape)}  heatmaps={tuple(heatmaps.shape)}"
              f"  hm_max={heatmaps.max():.4f}  hm_min={heatmaps.min():.4f}"
              f"  coords={coords_item.tolist()}")

        # batched train loader — batch is (imgs, heatmaps, coords)
        imgs_b, hms_b, coords_b = next(iter(dm.train_dataloader()))
        assert imgs_b.shape   == (4, 3, 512, 512), f"batch img shape wrong: {imgs_b.shape}"
        assert hms_b.shape    == (4, 2, 64, 64),   f"batch hm shape wrong: {hms_b.shape}"
        assert coords_b.shape == (4, 2, 2),         f"batch coords shape wrong: {coords_b.shape}"
        print(f"  train batch  imgs={tuple(imgs_b.shape)}  heatmaps={tuple(hms_b.shape)}  coords={tuple(coords_b.shape)}")

        # val loader (drop_last=False, may be < batch_size)
        imgs_v, hms_v, coords_v = next(iter(dm.val_dataloader()))
        print(f"  val   batch  imgs={tuple(imgs_v.shape)}  heatmaps={tuple(hms_v.shape)}  coords={tuple(coords_v.shape)}")

        # test loader
        imgs_t, hms_t, coords_t = next(iter(dm.test_dataloader()))
        print(f"  test  batch  imgs={tuple(imgs_t.shape)}  heatmaps={tuple(hms_t.shape)}  coords={tuple(coords_t.shape)}")

        # channel 0 should always be the left endpoint (x0 ≤ x1)
        # verify via argmax of each heatmap channel
        for b in range(hms_b.shape[0]):
            coords = []
            for c in range(2):
                flat_idx = hms_b[b, c].argmax().item()
                x = flat_idx % 64
                coords.append(x)
            assert coords[0] <= coords[1], \
                f"DOD sort failed: ch0.x={coords[0]} > ch1.x={coords[1]}"
        print(f"  DOD sort     OK (channel-0 x ≤ channel-1 x in all batch items)")

    print("\nLandmark data loading OK")


def test_training_utils():
    """
    Smoke-test heatmap_to_coords and compute_nme without needing a GPU or model.
    Verifies shapes, NME=0 for perfect prediction, and sub-pixel shift direction.
    """
    from training.landmark_detection import heatmap_to_coords, compute_nme

    print("\n" + "=" * 60)
    print("TRAINING UTILS SMOKE-TEST")
    print("=" * 60)

    B, N, H, W = 4, 2, 64, 64
    heatmap_size = (H, W)
    img_size = (512, 512)

    # build fake GT heatmaps: sharp spike at known integer positions
    gt_hm = torch.zeros(B, N, H, W)
    gt_x = [10, 50]   # landmark x positions (col)
    gt_y = [20, 40]   # landmark y positions (row)
    for n in range(N):
        gt_hm[:, n, gt_y[n], gt_x[n]] = 1.0

    # --- heatmap_to_coords: perfect prediction ---
    coords = heatmap_to_coords(gt_hm)
    assert coords.shape == (B, N, 2), f"shape wrong: {coords.shape}"
    for n in range(N):
        assert abs(coords[0, n, 0].item() - gt_x[n]) < 0.5, \
            f"x wrong: {coords[0, n, 0].item()} vs {gt_x[n]}"
        assert abs(coords[0, n, 1].item() - gt_y[n]) < 0.5, \
            f"y wrong: {coords[0, n, 1].item()} vs {gt_y[n]}"
    print("[heatmap_to_coords] integer spike → coords OK")

    # --- compute_nme: perfect prediction should give NME ≈ 0 ---
    nme = compute_nme(coords, coords, heatmap_size, img_size)
    assert nme.shape == (B,), f"NME shape wrong: {nme.shape}"
    assert nme.max().item() < 1e-5, f"NME for perfect pred should be 0, got {nme}"
    print("[compute_nme]       perfect prediction → NME=0 OK")

    # --- NME with known offset: shift pred by +4 hm pixels in x for lm-0 ---
    pred_hm = torch.zeros(B, N, H, W)
    pred_x = [gt_x[0] + 4, gt_x[1]]   # landmark 0 shifted right by 4px
    pred_y = [gt_y[0],     gt_y[1]]
    for n in range(N):
        pred_hm[:, n, pred_y[n], pred_x[n]] = 1.0

    pred_coords = heatmap_to_coords(pred_hm)
    nme_shifted = compute_nme(pred_coords, coords, heatmap_size, img_size)

    # expected: error for lm-0 = 4 * (512/64) = 32 px, lm-1 = 0
    # mean error = 16 px; inter-lm dist = dist((10,20),(50,40)) in img space
    lm0_img = torch.tensor([gt_x[0] * (512/64), gt_y[0] * (512/64)])
    lm1_img = torch.tensor([gt_x[1] * (512/64), gt_y[1] * (512/64)])
    expected_norm = (lm0_img - lm1_img).norm().item()
    expected_nme = 16.0 / expected_norm
    assert abs(nme_shifted[0].item() - expected_nme) < 1e-4, \
        f"NME value wrong: {nme_shifted[0].item():.5f} vs {expected_nme:.5f}"
    print(f"[compute_nme]       4-px shift → NME={nme_shifted[0].item():.4f} (expected {expected_nme:.4f}) OK")

    print("\nTraining utils OK")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("all", "ade20k"):
        main()
    if mode in ("all", "landmark"):
        test_landmark()
    if mode in ("all", "training"):
        test_training_utils()
