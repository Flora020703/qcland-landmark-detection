"""Verifies endpoint_order.canonical_order() against HRNet's OWN real,
already-canonicalised per-image output -- not just synthetic cases -- so a
future edit to this function cannot silently drift from HRNet's actual
behaviour without a test failing.

Reference values below were read directly from the audited upstream's
`fixed_channel_per_image.csv` for UCL brain/BPD seed 42 (extracted from
checkpoint_backups/hrnet-512-fixed-50runs.tar) and from the corresponding
raw rows of the released UCL Head_Test.csv. Do not "simplify" these two
real-data cases into synthetic round numbers; their value is that they are
verified against an actual trained run, including one case (004_HC.jpeg)
where the raw CSV order and the canonical order genuinely differ.

Run directly: python rtmpose_reproduction/test_endpoint_order.py
"""

from __future__ import annotations

from dod_vectors import get_d_vect
from endpoint_order import canonical_order

TOLERANCE = 1e-6


def _assert_points_close(actual, expected, label):
    for a, e in zip(actual, expected):
        for av, ev in zip(a, e):
            assert abs(av - ev) < TOLERANCE, f"[{label}] {actual} != {expected}"


def test_real_hrnet_ucl_bpd_no_swap_case():
    # 001_HC.jpg raw UCL Head_Test.csv row: bpd_1=(565,112), bpd_2=(587.0,464.0).
    # HRNet's own fixed_channel_per_image.csv gt0/gt1 for this file:
    #   gt0=(565.0,112.0), gt1=(587.0,464.0) -- i.e. NO swap.
    d_vect = get_d_vect("UCL", "BPD")
    p0, p1 = (565.0, 112.0), (587.0, 464.0)
    ordered = canonical_order(p0, p1, d_vect)
    _assert_points_close(ordered, ((565.0, 112.0), (587.0, 464.0)), "001_HC.jpg")
    print("[PASS] test_real_hrnet_ucl_bpd_no_swap_case")


def test_real_hrnet_ucl_bpd_swap_case():
    # 004_HC.jpeg raw UCL Head_Test.csv row: bpd_1=(403,532), bpd_2=(399.0,133.0).
    # HRNet's own fixed_channel_per_image.csv gt0/gt1 for this file:
    #   gt0=(399.0,133.0), gt1=(403.0,532.0) -- i.e. the raw order IS swapped.
    d_vect = get_d_vect("UCL", "BPD")
    p0, p1 = (403.0, 532.0), (399.0, 133.0)  # raw CSV order (bpd_1, bpd_2)
    ordered = canonical_order(p0, p1, d_vect)
    _assert_points_close(ordered, ((399.0, 133.0), (403.0, 532.0)), "004_HC.jpeg")
    print("[PASS] test_real_hrnet_ucl_bpd_swap_case")


def test_tie_break_keeps_original_order():
    # Exact tie (proj0 == proj1): must keep original order, never swap.
    d_vect = ((0.0, 0.0), (1.0, 0.0))  # horizontal direction vector
    p0, p1 = (5.0, 100.0), (5.0, -100.0)  # equal x-projection regardless of y
    ordered = canonical_order(p0, p1, d_vect)
    _assert_points_close(ordered, (p0, p1), "exact tie")
    print("[PASS] test_tie_break_keeps_original_order")


def test_symmetry_swapping_inputs_gives_same_canonical_pair():
    d_vect = get_d_vect("MULTICENTRE", "TAD")
    p0, p1 = (100.0, 400.0), (300.0, 50.0)
    ordered_a = canonical_order(p0, p1, d_vect)
    ordered_b = canonical_order(p1, p0, d_vect)
    _assert_points_close(ordered_a, ordered_b, "order-independence")
    print("[PASS] test_symmetry_swapping_inputs_gives_same_canonical_pair")


def main():
    test_real_hrnet_ucl_bpd_no_swap_case()
    test_real_hrnet_ucl_bpd_swap_case()
    test_tie_break_keeps_original_order()
    test_symmetry_swapping_inputs_gives_same_canonical_pair()
    print("[ALL ENDPOINT-ORDER TESTS PASSED]")


if __name__ == "__main__":
    main()
