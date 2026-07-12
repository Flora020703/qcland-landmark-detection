#!/usr/bin/env python3
# ---------------------------------------------------------------
# MODIFIED: new file — materialize a testable checkpoint from an
# EMACallback shadow (training/ema.py).
#
# EMACallback never swaps the shadow into the live model (see that file's
# docstring for why) — it stashes the shadow inside the checkpoint payload
# under "ema_state_dict" instead. This script takes a normal .ckpt, replaces
# the matching state_dict entries with the EMA shadow values, and writes a
# new .ckpt that can be tested exactly like any other checkpoint:
#
# Usage (from repo root):
#   python3 ablation/apply_ema.py checkpoints/.../seed42_final.ckpt checkpoints/.../seed42_ema.ckpt
#   python3 main_landmark.py test --config ... --ckpt_path checkpoints/.../seed42_ema.ckpt
# ---------------------------------------------------------------

import sys

import torch


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python3 ablation/apply_ema.py <in_ckpt> <out_ckpt>")
        sys.exit(1)

    in_path, out_path = sys.argv[1], sys.argv[2]
    ckpt = torch.load(in_path, map_location="cpu", weights_only=False)

    ema_shadow = ckpt.get("ema_state_dict")
    if not ema_shadow:
        print(f"[ERROR] no 'ema_state_dict' in {in_path} — was EMACallback enabled during training?")
        sys.exit(1)

    state_dict = ckpt["state_dict"]
    applied = 0
    for name, ema_val in ema_shadow.items():
        if name in state_dict:
            state_dict[name] = ema_val
            applied += 1
        else:
            print(f"[WARN] EMA key not found in checkpoint state_dict: {name}")

    print(f"[OK] applied {applied}/{len(ema_shadow)} EMA parameters (step={ckpt.get('ema_step', '?')})")
    torch.save(ckpt, out_path)
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
