# EoMT final-comparison reuse audit

Protocol locked on 2026-07-31: measurement-specific models, released splits, 512×512 inputs, final checkpoint, fixed-channel NME, seeds 42/0/123/2024/3407.

## Reusable result groups

| Dataset | Backbone | Tasks | Training status | Numeric status | Raw-evidence status |
|---|---|---|---|---|---|
| UCL | DINOv2 | BPD/OFD/APAD/TAD/FL | complete | five-seed final summaries recorded in the chronological summary | authoritative seed-level TSV and per-image files must be copied back from the server/archive before final table generation |
| UCL | DINOv3 | BPD/OFD/APAD/TAD/FL | complete | five-seed final summaries recorded in the chronological summary | authoritative seed-level TSV and per-image files must be copied back from the server/archive before final table generation |
| Multicentre | DINOv2 | BPD/OFD/APAD/TAD/FL | complete | exact seed-level best/final swap-min and fixed-channel values recorded | checkpoint tar archives were verified locally; reconcile their CSV/log contents with the server results TSV |
| Multicentre | DINOv3 | BPD/OFD/APAD/TAD/FL | complete | exact seed-level best/final swap-min and fixed-channel values recorded | checkpoint tar archives were verified locally; reconcile their CSV/log contents with the server results TSV |

These 20 method×dataset×task groups are reusable and require no retraining. “Reusable” does not mean that aggregate values alone are sufficient for paired tests or figure generation. The final data freeze must include the exact seed-level TSV, per-image predictions/NME, generated config and test log for every reported cell.

## Exclusions

- The completed Multicentre HRNet reproduction at 256×256 with native swap-min evaluation is not a matched final-comparison cell.
- No EoMT seed may be selected retrospectively by its test score.
- Best-checkpoint diagnostic values must not replace the inherited final-checkpoint reporting convention.
