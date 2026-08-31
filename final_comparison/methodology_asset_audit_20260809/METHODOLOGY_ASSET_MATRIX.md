# BPD/OFD methodology asset matrix (2026-08-09)

This is an asset audit, not a results table. No missing result is inferred from an old aggregate number alone. The final fetal metric is permutation-invariant NME, evaluated from final checkpoints as five-seed single-model mean plus seed-level sample SD unless explicitly labelled as a historical single-seed screen.

## Decision states

- `aggregate`: suitable per-image permutation-invariant outputs already exist.
- `inference-only`: checkpoint and configuration exist, but formal per-image outputs must be generated.
- `unresolved`: assets or provenance remain insufficient; do not train until the remaining checks finish.
- `retrain-candidate`: use only after the unresolved checks confirm that no recoverable checkpoint exists.

## Audited matrix

| Scope | Condition | Evidence found | Current state | Required action |
|---|---|---|---|---|
| BPD early screen | sigma 1 / sigma 4 | Server has best, last, final and config for each sigma; these are single-run screens | inference-only, single-run exploratory evidence; no five-seed extension | Do not retrain. Rescore each existing final checkpoint once with PI NME because Chapter 5 will retain the numeric comparison. Report no seed SD and explicitly label it as a limited single-run collapse/target-width diagnostic, not a robust performance comparison and not part of the Chapter 3 method definition |
| BPD core loss chain | plain MSE -> weighted MSE -> hybrid WMSE + coordinate L1, all with original einsum head | Controlled five-seed final-checkpoint runs, fixed/PI companion dumps, and original-image-space aggregation are complete | complete / aggregate | Official PI-NME: plain MSE `36.65+/-17.28%`, weighted MSE `13.75+/-2.90%`, hybrid `11.23+/-2.06%` (n=49 each). Preserve the archived checkpoints and official per-image outputs |
| BPD historical loss exploration | refinement variants, 400 epochs, Adaptive Wing alone, hybrid AWing, triple loss, temperature/lambda sweeps | Historical single-seed aggregate notes only; not selected in the final recipe | prose only | Do not rerun. Remove exact values from the formal table and summarise briefly as investigated but not retained |
| BPD core | original einsum head with selected hybrid loss | Controlled five-seed final-checkpoint run and original-image-space aggregation complete | complete / aggregate | Official PI-NME `11.23+/-2.06%` (n=49) |
| BPD core | DeconvHeadV2 with selected hybrid loss | Controlled five-seed final-checkpoint run and original-image-space aggregation complete | complete / aggregate | Official PI-NME `8.12+/-1.42%` (n=49) |
| BPD core | DeconvHeadV2 + FPN with selected hybrid loss | Controlled five-seed final-checkpoint run and original-image-space aggregation complete | complete / aggregate | Official PI-NME `9.00+/-2.20%` (n=49); FPN alone did not improve the DeconvHeadV2 mean |
| BPD core | DeconvHeadV2 + FPN + UDP, loader-seed-controlled | Five archived final checkpoints were restored, audited, rescored, and removed from the server after local verification | complete / aggregate | Official PI-NME `8.96+/-2.30%` (n=49); UDP was numerically near-neutral relative to FPN alone |
| BPD augmentation | DINOv2 rotation + scale | Five-seed final results and per-image PI outputs already archived | aggregate | Reuse the frozen official output |
| BPD augmentation | DINOv3 no augmentation | Historical log/result files exist but no checkpoint group was found in the complete inventory | unresolved | Do not quote under the new metric unless recoverable predictions/checkpoints are found |
| BPD augmentation | DINOv3 rotation only | Historical log/result files exist but no checkpoint group was found | unresolved | Same as above |
| BPD augmentation | DINOv3 rotation + scale | Five-seed final results and per-image PI outputs already archived | aggregate | Reuse the frozen official output |
| BPD resolution | 256 and 512 | The original audit found seeds 42, 0, 123 and 3407 in both conditions; the two genuinely missing seed2024 runs were subsequently trained with the matched configurations and exported with full per-image coordinates | complete / aggregate | Official original-image-space PI-NME: 256 input `7.30+/-1.50%`; 512 input `6.61+/-0.91%` (five seeds, n=49 each). Scoring used `model_input_size=256/512` as appropriate, `heatmap_size=64`, and `pixel_center_align=true`. The 0.70-point lower mean at 512 is descriptive until the matched-seed comparison is calculated |
| BPD EMA | raw-best and materialized EMA | Server has five raw-best and five materialized-EMA checkpoints | excluded from quantitative core | Do not add an EMA row to the core methodology table and do not spend further inference time on it. At most state that EMA was investigated but not retained because it did not give a sufficiently clear and consistent contribution |
| BPD TTA | flip TTA | No separate training checkpoint is required; any selected retained checkpoint can support this evaluation | inference-only experiment | Apply identical PI evaluator with/without TTA to a pre-declared checkpoint set; retain as a negative auxiliary comparison, not a new training rung |
| BPD not retained | LoRA / frozen backbone | Configs exist but no checkpoint group was found in the complete checkpoint inventory | unresolved | Search run metadata; otherwise omit quantitative claim or retrain only if scientifically necessary |
| BPD not retained | 128x128 heatmap | Configs exist but no checkpoint group was found | unresolved | Search run metadata; if retrained/evaluated, record `heatmap_size=[128,128]` and the corresponding pixel-centre recovery explicitly |
| OFD methodology chain | DeconvHeadV2 / FPN+UDP / augmentation variants | Recoverable five-seed assets exist for several OFD conditions | intentionally not pursued | Do not repeat the BPD model-development chain on OFD and do not add an OFD ablation table merely for symmetry. OFD remains in the final multi-task comparison under the common PI-NME evaluator; existing assets are retained as provenance/recovery material only |

## Coordinate-recovery requirement

The evaluator must read the saved YAML for every run. Pre-UDP einsum/Deconv/FPN experiments historically use `pixel_center_align=false`, hence no UDP offset. UDP runs generally use `pixel_center_align=true`; the recovery depends on both model-input and heatmap size. With a 64 heatmap the offset is 1.5 pixels for a 256 input and 3.5 pixels for a 512 input; with a 512 input and 128 heatmap it is 1.5 pixels. `model_input_size`, `heatmap_size` and `pixel_center_align` must therefore be derived from metadata rather than assigned from the experiment name.

## Assets still requiring identification

Two unarchived local checkpoints were inspected:

- `/mnt/d/download/Project coding/EOMT/epoch=4-step=125.ckpt` is a 150-class semantic-segmentation checkpoint (`num_classes=150`, segmentation mask/class heads), not a fetal-landmark methodology asset. Its neighbouring `metrics.csv` is likewise a segmentation log and must be excluded from this audit.
- `/mnt/d/download/Project coding/EOMT/run2_best.ckpt` is confirmed by its datamodule metadata as UCL BPD (`Head_Train.csv`/`Head_Test.csv`), 512-to-64, sigma 4, validation split seed 42, and by its state dict as the original class/mask-head (einsum-style) landmark model with no Deconv/FPN keys. It is a validation-best checkpoint at epoch 109 (`val_nme=0.1768`), not a final checkpoint. The embedded metadata still does not establish the training seed, full loss configuration, or a final-checkpoint counterpart. It is useful historical provenance but cannot serve as the five-seed final-checkpoint core rung.

The second checkpoint no longer blocks the asset identification audit, but it does not remove the need for a controlled five-seed final-checkpoint einsum rung if that rung is retained in the formal core table.

## Current execution rule (updated 2026-08-10)

The 25 controlled BPD core runs are complete: 15 loss-axis runs and 10 additional architecture-axis runs, with hybrid-einsum shared between the axes. Sigma rescoring and the matched five-seed 256/512 resolution study are also complete. No further fetal-methodology training is currently required. The OFD development chain has been removed from the completion plan by design: repeating the same optimisation and architecture search on a second fetal-head diameter adds limited independent methodological evidence. OFD remains a required task in the final UCL/Multicentre method comparison, not a second ablation dataset.

## Locked thesis evidence allocation (2026-08-10)

- **BPD is the sole fetal-domain methodology-development dataset.** It carries the sigma screen, controlled loss chain, controlled architecture chain, augmentation, resolution, and concise EMA/TTA sensitivity evidence where retained.
- **300W provides the cross-domain test.** Its landmarks retain their semantic identities and therefore use the appropriate standard 300W evaluation rather than the unordered two-endpoint PI matching introduced for fetal diameters.
- **OFD is not a missing methodology experiment.** It remains, together with BPD/APAD/TAD/FL, in the final multi-task comparison of the proposed DINOv2/DINOv3 models, HRNet and RTMPose under the supervisor-approved fetal PI-NME protocol.
- Existing OFD methodology checkpoints are not deleted merely because the ablation is no longer planned; they remain recovery/provenance assets unless a separately verified local backup permits later cleanup.

## 300W hand-off status

The completed recursive audit verified the DINOv2 archive: complete five-seed best/final checkpoint sets exist for four ablation rungs, but no per-image predictions were saved and seed42 lacks neighbouring configs/logs. The user confirmed that DINOv3 300W was never run; its prepared config/driver do not constitute a missing result, and no DINOv3 training is planned. The thesis will use the existing recorded DINOv2 aggregate results for cross-domain validation; uploading the 21.7 GB archive and performing a new inference-only consolidation are not completion requirements. Consequently, no new per-image or paired-image statistical claims will be made for 300W. Rotation+scale is not added because 300W validates the architecture across domains rather than repeating every fetal training choice, and the current 300W augmentation implementation is not equivalent to the fetal rotation/scale pipeline. EMA is prose-only (`investigated but not retained`). Fetal permutation-invariant endpoint matching remains out of scope for semantically identified 300W landmarks.
