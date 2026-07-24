"""
Smoke-tests for both data pipelines.

Run from the eomt/ directory:
    python scripts/test_dataload.py           # both tests
    python scripts/test_dataload.py ade20k    # segmentation only
    python scripts/test_dataload.py landmark  # landmark only
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from datasets.dataset import Dataset
from datasets.ade20k_semantic import ADE20KSemantic

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "ADEChallengeData2016" / "ADE20K_mini"
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

    # --- endpoint_order_invariant=True: published (Di Vece et al.) swap-min
    #     two-endpoint NME. Uses the same B,N=2 fake heatmaps as above. ---

    # (1) same channel order as GT: swap-min result must equal the
    #     fixed-channel result (no crossover to resolve).
    nme_fixed = compute_nme(pred_coords, coords, heatmap_size, img_size)
    nme_swapmin_same_order = compute_nme(
        pred_coords, coords, heatmap_size, img_size,
        endpoint_order_invariant=True,
    )
    assert torch.allclose(nme_fixed, nme_swapmin_same_order, atol=1e-6), (
        f"swap-min should equal fixed-channel NME when channel order "
        f"already matches GT: {nme_swapmin_same_order} vs {nme_fixed}"
    )
    print("[compute_nme]       swap-min == fixed-channel when order matches OK")

    # (2) predicted endpoints swapped (channel 0 <-> channel 1): swap-min
    #     result must be IDENTICAL to the unswapped case (it resolves the
    #     crossover), while the fixed-channel metric must increase a lot
    #     (it penalises the crossover as if it were a coordinate error).
    pred_coords_swapped = pred_coords.flip(dims=[1])   # (B, N, 2), N=2 swap
    nme_swapmin_swapped = compute_nme(
        pred_coords_swapped, coords, heatmap_size, img_size,
        endpoint_order_invariant=True,
    )
    assert torch.allclose(nme_swapmin_swapped, nme_swapmin_same_order, atol=1e-6), (
        f"swap-min must be invariant to which predicted channel holds which "
        f"endpoint: {nme_swapmin_swapped} vs {nme_swapmin_same_order}"
    )
    nme_fixed_swapped = compute_nme(pred_coords_swapped, coords, heatmap_size, img_size)
    assert nme_fixed_swapped.min().item() > nme_fixed.max().item(), (
        "fixed-channel NME should increase sharply once predicted "
        f"endpoints are swapped: {nme_fixed_swapped} vs baseline {nme_fixed}"
    )
    print("[compute_nme]       swap-min invariant to channel swap; "
          "fixed-channel is NOT (penalises it) OK")

    # Pixel error must make the same correspondence choice as swap-min NME.
    from training.landmark_detection import compute_pixel_error
    pe_same = compute_pixel_error(
        pred_coords, coords, heatmap_size, img_size,
        endpoint_order_invariant=True,
    )
    pe_swapped = compute_pixel_error(
        pred_coords_swapped, coords, heatmap_size, img_size,
        endpoint_order_invariant=True,
    )
    assert torch.allclose(pe_same, pe_swapped, atol=1e-6), (
        "swap-min pixel error must be invariant to predicted-channel swap: "
        f"{pe_same} vs {pe_swapped}"
    )
    print("[compute_pixel_error] swap-min correspondence matches NME and is "
          "channel-swap invariant OK")

    # (3) per-sample manual computation matches the batched result (sample 0
    #     of the "4-px shift" case: lm-0 shifted +4hm-px in x, lm-1 exact).
    d_std_manual = (
        (pred_coords[0, 0] - coords[0, 0]).norm()
        + (pred_coords[0, 1] - coords[0, 1]).norm()
    ) * (512 / 64)   # heatmap px -> image px
    d_swap_manual = (
        (pred_coords[0, 0] - coords[0, 1]).norm()
        + (pred_coords[0, 1] - coords[0, 0]).norm()
    ) * (512 / 64)
    diameter_manual = expected_norm  # already computed in image-pixel space above
    expected_swapmin = min(d_std_manual.item(), d_swap_manual.item()) / (2.0 * diameter_manual)
    assert abs(nme_swapmin_same_order[0].item() - expected_swapmin) < 1e-4, (
        f"batched swap-min NME doesn't match manual per-sample computation: "
        f"{nme_swapmin_same_order[0].item():.5f} vs {expected_swapmin:.5f}"
    )
    print("[compute_nme]       batched swap-min matches manual per-sample calc OK")

    # (4) N != 2 (the 300W path) must reject endpoint_order_invariant=True
    #     outright, rather than silently computing something wrong -- 300W
    #     must never pass this flag, and this guard is what enforces that.
    coords68 = torch.randn(B, 68, 2)
    try:
        compute_nme(coords68, coords68, heatmap_size, img_size, endpoint_order_invariant=True)
        raise AssertionError("expected ValueError for N=68 with endpoint_order_invariant=True")
    except ValueError:
        print("[compute_nme]       endpoint_order_invariant=True correctly "
              "rejects N!=2 (300W path unaffected) OK")

    print("\nTraining utils OK")


def test_face300w():
    """
    Smoke-test Face300WDataModule + Face300WDataset: split sizes, shapes,
    NME norm pair, FLIP_PAIRS_68 completeness, and a visual landmark-overlay
    check saved to disk (can't eyeball correctness from asserts alone).
    """
    import numpy as np
    from datasets.face300w_dataset import (
        Face300WDataModule, FLIP_PAIRS_68, INTER_OCULAR_PAIR, _load_pts,
    )

    _win    = Path("D:/download/Project coding/msc/300w")
    _wsl    = Path("/mnt/d/download/Project coding/msc/300w")
    _server = Path("/root/autodl-tmp/300W/300w")
    DATA_ROOT = next((p for p in (_win, _wsl, _server) if p.exists()), _server)

    print("\n" + "=" * 60)
    print("300W SMOKE-TEST")
    print("=" * 60)
    print(f"Data root: {DATA_ROOT}")
    assert DATA_ROOT.exists(), f"Not found: {DATA_ROOT}"

    # --- FLIP_PAIRS_68 completeness: every one of the 68 indices must be
    #     either paired exactly once, or be one of the 10 known midline
    #     points (chin=8, nose bridge=27-30, nose tip=33, lip mids=51/57,
    #     inner-lip mids=62/66). Anything else uncovered = a real bug. ---
    paired = set()
    for a, b in FLIP_PAIRS_68:
        assert a not in paired and b not in paired, f"index reused: {a},{b}"
        paired.add(a); paired.add(b)
    expected_midline = {8, 27, 28, 29, 30, 33, 51, 57, 62, 66}
    unpaired = set(range(68)) - paired
    assert unpaired == expected_midline, (
        f"FLIP_PAIRS_68 coverage wrong — unpaired={sorted(unpaired)} "
        f"expected_midline={sorted(expected_midline)}"
    )
    assert len(paired) == 58 and len(FLIP_PAIRS_68) == 29
    print(f"[FLIP_PAIRS_68] OK — {len(FLIP_PAIRS_68)} pairs, "
          f"{len(unpaired)} midline points, all 68 indices covered exactly once")

    # --- .pts 1-indexed -> 0-indexed sanity: no coordinate should be
    #     negative after the -1 shift for a real annotation file ---
    sample_pts = next((DATA_ROOT / "afw").glob("*.pts"))
    pts = _load_pts(sample_pts)
    assert pts.shape == (68, 2)
    assert pts.min() >= -1.0, f"suspiciously negative coords after -1 shift: {pts.min()}"
    print(f"[_load_pts] OK — sample {sample_pts.name}: range "
          f"x[{pts[:,0].min():.1f},{pts[:,0].max():.1f}] "
          f"y[{pts[:,1].min():.1f},{pts[:,1].max():.1f}]")

    for test_subset, expected_test_n in [("common", 554), ("challenging", 135), ("full", 689)]:
        dm = Face300WDataModule(
            data_root=DATA_ROOT,
            img_size=(256, 256),
            heatmap_size=(64, 64),
            sigma=1.5,
            val_fraction=0.1,
            val_split_seed=42,
            test_subset=test_subset,
            batch_size=4,
            num_workers=0,
            pin_memory=False,
            augment=False,
        )
        dm.setup()
        n_train, n_val, n_test = len(dm.train_dataset), len(dm.val_dataset), len(dm.test_dataset)
        print(f"\n[test_subset={test_subset}] train={n_train} val={n_val} test={n_test}")
        assert n_test == expected_test_n, f"expected {expected_test_n} test images, got {n_test}"
        if test_subset == "full":
            assert n_train + n_val == 3148, f"train pool should be 3148, got {n_train + n_val}"
            # same val_split_seed must give the same split regardless of test_subset
            base_train, base_val = n_train, n_val

    # single item shape + heatmap sanity
    img, heatmaps, coords = dm.train_dataset[0]
    assert img.shape == (3, 256, 256), f"img shape wrong: {img.shape}"
    assert heatmaps.shape == (68, 64, 64), f"heatmap shape wrong: {heatmaps.shape}"
    assert coords.shape == (68, 2), f"coords shape wrong: {coords.shape}"
    assert heatmaps.max() <= 1.0 + 1e-5 and heatmaps.max() > 0.0
    print(f"\n[single item] img={tuple(img.shape)} heatmaps={tuple(heatmaps.shape)} "
          f"coords={tuple(coords.shape)} hm_max={heatmaps.max():.4f}")

    # batched loaders
    imgs_b, hms_b, coords_b = next(iter(dm.train_dataloader()))
    assert imgs_b.shape == (4, 3, 256, 256) and hms_b.shape == (4, 68, 64, 64) and coords_b.shape == (4, 68, 2)
    print(f"[train batch] imgs={tuple(imgs_b.shape)} heatmaps={tuple(hms_b.shape)} coords={tuple(coords_b.shape)}")
    imgs_v, hms_v, coords_v = next(iter(dm.val_dataloader()))
    print(f"[val batch]   imgs={tuple(imgs_v.shape)} heatmaps={tuple(hms_v.shape)} coords={tuple(coords_v.shape)}")
    imgs_t, hms_t, coords_t = next(iter(dm.test_dataloader()))
    print(f"[test batch]  imgs={tuple(imgs_t.shape)} heatmaps={tuple(hms_t.shape)} coords={tuple(coords_t.shape)}")

    # --- NME norm pair sanity: inter-ocular distance on a real sample should
    #     be a plausible fraction of the 256px crop (not ~0, not > crop size) ---
    from training.landmark_detection import compute_nme
    i0, i1 = INTER_OCULAR_PAIR
    img_coords = coords.clone()
    img_coords[:, 0] *= 256 / 64
    img_coords[:, 1] *= 256 / 64
    d = (img_coords[i0] - img_coords[i1]).norm().item()
    print(f"\n[inter-ocular] landmarks[{i0}]-landmarks[{i1}] distance in 256px crop = {d:.1f}px")
    assert 20 < d < 200, f"inter-ocular distance implausible for a 256px face crop: {d:.1f}px"

    nme = compute_nme(coords.unsqueeze(0), coords.unsqueeze(0), (64, 64), (256, 256), norm_pair=INTER_OCULAR_PAIR)
    assert nme.item() < 1e-5, f"NME for identical pred/gt should be ~0, got {nme.item()}"
    print(f"[compute_nme]  identical pred/gt -> NME={nme.item():.6f} OK")

    # --- flip augmentation: force a flip, check known-pair identities swap ---
    dm_flip = Face300WDataModule(
        data_root=DATA_ROOT, img_size=(256, 256), heatmap_size=(64, 64),
        sigma=1.5, batch_size=1, num_workers=0, augment=True,
    )
    dm_flip.setup()
    ds = dm_flip.train_dataset
    ds.flip_prob = 1.0  # force flip for this check
    torch.manual_seed(0)
    _, _, coords_flip = ds[0]
    torch.manual_seed(0)
    ds.augment = False
    _, _, coords_noflip = ds[0]
    ds.augment = True
    # after flip: landmark 36 (one outer eye corner) should land near where
    # 45 (the other outer eye corner) was pre-flip, mirrored in x
    x36_pre_mirrored = 64 - 1.0 - coords_noflip[36, 0]
    assert abs(coords_flip[45, 0].item() - x36_pre_mirrored.item()) < 2.0, (
        "flip pair (36,45) identity not preserved correctly"
    )
    print("[flip augment] OK — (36,45) eye-corner identity preserved across hflip")

    # --- visual check: overlay landmarks on the actual crop, save to disk ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_dir = Path(__file__).resolve().parent.parent / "docs" / "static"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "face300w_dataload_check.png"

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for ax, i in zip(axes, [0, 1, 2, 3]):
            img_i, _, coords_i = dm.train_dataset[i]
            im = img_i.permute(1, 2, 0).numpy()
            ax.imshow(im)
            xs = coords_i[:, 0].numpy() * (256 / 64)
            ys = coords_i[:, 1].numpy() * (256 / 64)
            ax.scatter(xs, ys, s=8, c="lime", edgecolors="black", linewidths=0.3)
            ax.scatter(xs[[36, 45]], ys[[36, 45]], s=20, c="red")  # inter-ocular pair highlighted
            ax.set_title(f"train[{i}]")
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        print(f"\n[visual check] saved landmark overlay to {out_path}")
    except ImportError:
        print("\n[visual check] matplotlib not available — skipped (shapes/asserts above already passed)")

    print("\n300W data loading OK")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("all", "ade20k"):
        main()
    if mode in ("all", "landmark"):
        test_landmark()
    if mode in ("all", "training"):
        test_training_utils()
    if mode in ("all", "face300w"):
        test_face300w()
