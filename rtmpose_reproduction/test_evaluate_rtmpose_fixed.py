"""Tests evaluate_rtmpose_fixed.py against hand-computable synthetic cases,
including a deliberately-broken invariant case to prove the fixed>=swap-min
assertion actually fires rather than just happening to pass on tidy data.

Run directly: python rtmpose_reproduction/test_evaluate_rtmpose_fixed.py
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from evaluate_rtmpose_fixed import evaluate


def _write_gt(path: Path, records):
    images = [{"id": i, "file_name": name, "width": 100, "height": 100} for i, (name, _, _) in enumerate(records)]
    anns = [{"id": i, "image_id": i, "keypoints": [gt0[0], gt0[1], 2, gt1[0], gt1[1], 2]}
            for i, (name, gt0, gt1) in enumerate(records)]
    path.write_text(json.dumps({"images": images, "annotations": anns}), encoding="utf-8")


def _write_pred(path: Path, records):
    out = [{"file_name": name, "pred": [list(p0), list(p1)]} for name, p0, p1 in records]
    path.write_text(json.dumps(out), encoding="utf-8")


def test_exact_prediction_gives_zero_nme():
    tmp = Path(tempfile.mkdtemp(prefix="rtmpose_eval_test_"))
    try:
        gt = [("a.png", (0.0, 0.0), (10.0, 0.0))]
        _write_gt(tmp / "gt.json", gt)
        _write_pred(tmp / "pred.json", [("a.png", (0.0, 0.0), (10.0, 0.0))])
        summary = evaluate(tmp / "gt.json", tmp / "pred.json",
                            tmp / "per_image.csv", tmp / "summary.json")
        assert abs(summary["fixed_channel_mean_pct"]) < 1e-9
        assert abs(summary["swap_min_mean_pct"]) < 1e-9
        print("[PASS] test_exact_prediction_gives_zero_nme")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_swapped_prediction_only_hurts_fixed_channel():
    tmp = Path(tempfile.mkdtemp(prefix="rtmpose_eval_test_"))
    try:
        gt = [("a.png", (0.0, 0.0), (10.0, 0.0))]
        _write_gt(tmp / "gt.json", gt)
        # Prediction channels exactly swapped relative to GT: fixed-channel
        # should be large (2*10/(2*10)=1.0 -> 100%), swap-min should be ~0.
        _write_pred(tmp / "pred.json", [("a.png", (10.0, 0.0), (0.0, 0.0))])
        summary = evaluate(tmp / "gt.json", tmp / "pred.json",
                            tmp / "per_image.csv", tmp / "summary.json")
        assert abs(summary["swap_min_mean_pct"]) < 1e-9
        assert abs(summary["fixed_channel_mean_pct"] - 100.0) < 1e-6
        print("[PASS] test_swapped_prediction_only_hurts_fixed_channel")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_invariant_violation_is_actually_caught():
    """Proves the fixed>=swap-min assertion is a real, load-bearing check by
    monkeypatching a broken evaluate() call that would otherwise report
    swap-min > fixed-channel, and confirming it raises."""
    tmp = Path(tempfile.mkdtemp(prefix="rtmpose_eval_test_"))
    try:
        gt = [("a.png", (0.0, 0.0), (10.0, 0.0))]
        _write_gt(tmp / "gt.json", gt)
        _write_pred(tmp / "pred.json", [("a.png", (0.0, 0.0), (10.0, 0.0))])

        import evaluate_rtmpose_fixed as mod
        original_min = min

        def broken_min(*args):
            # Force swap "min" to instead return the larger value, simulating
            # a hypothetical bug where swap-min ends up bigger than fixed.
            return max(*args) if len(args) > 1 else original_min(*args)

        mod.min = broken_min
        try:
            raised = False
            try:
                evaluate(tmp / "gt.json", tmp / "pred.json",
                         tmp / "per_image.csv", tmp / "summary.json")
            except SystemExit:
                raised = True
            assert raised, "invariant check did not fire for a broken swap-min computation"
        finally:
            mod.min = original_min
        print("[PASS] test_invariant_violation_is_actually_caught")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    test_exact_prediction_gives_zero_nme()
    test_swapped_prediction_only_hurts_fixed_channel()
    test_invariant_violation_is_actually_caught()
    print("[ALL EVALUATOR TESTS PASSED]")


if __name__ == "__main__":
    main()
