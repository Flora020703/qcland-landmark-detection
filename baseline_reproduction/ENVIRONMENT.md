# Environment note: why `environment.yml` can't be used verbatim

The source repo's `environment.yml` pins:

```
python=3.6.15
torch==1.0.0
torchvision==0.2.1
numpy==1.19.5
opencv-python==4.4.0.42
scipy==1.2.0
```

`torch==1.0.0` predates CUDA 11 entirely and has no kernels for Ada Lovelace
(RTX 4090, compute capability 8.9) — it cannot allocate a CUDA context on
that GPU at all, let alone train. This is a hardware compatibility wall, not
a methodology choice: there is no way to honour this exact pin on the AutoDL
RTX 4090 instance this project trains on.

**What to do instead:** install a modern `torch`/`torchvision` build matching
whatever CUDA toolkit the AutoDL image already has (the same one used for
this project's own EoMT training, i.e. current AutoDL image = PyTorch
2.8/CUDA 12.8 per CLAUDE.md), plus current `numpy`/`opencv-python`/`scipy`,
and `tensorboardX`, `yacs`, `hdf5storage`, `pandas`, `scikit-learn`, and
`matplotlib`. The resolved versions must be recorded: forward-compatible
syntax does not establish numerical equivalence across dependency versions.

**Checked before recommending this**: grepped the entire `lib/`/`tools/` tree
for the classic PyTorch-1.0-era / Python-2-era patterns that break on a
modern stack — `torch._six`, `np.math` (beyond the one already-patched
call), `np.float`/`np.int`/`np.bool` bare aliases (removed in numpy>=1.24),
`xrange`, Python-2 `print` statements, `F.upsample`/`nn.Upsample` with the
old positional-arg signature, `Variable(...)`, `.data[0]`,
`size_average=`/`reduce=` loss kwargs, `volatile=`. **None found** beyond the
one already-fixed `np.math.floor` call. `cudnn.benchmark`/`cudnn.deterministic`
are read from config, not hardcoded. This lowers the expected compatibility
risk, but does not replace a real GPU canary under the resolved environment.

**Recommended env for the AutoDL server** (adjust exact pins to whatever
resolves cleanly against the already-installed CUDA 12.8 driver):

```
torch / torchvision   # match the CUDA 12.8 build already used for EoMT training
numpy
opencv-python
scipy
pandas
scikit-learn
matplotlib
tensorboardx
yacs
hdf5storage
```

If training breaks on the modern stack, fix only the specific compatibility
error and document it; do not change architecture, data handling or training
hyperparameters. The driver records the exact versions, CUDA/cuDNN, GPU,
source commit and pretrained-weight hash on every invocation. The latest
snapshot is stored in `environment_audit.txt`, while
`environment_audit_history.txt` retains the canary/formal history so an
environment change between them cannot be silently hidden.

**Document this substitution in Appendix B** alongside the existing
HRNet-Facial-Landmark-Detection commit-pinning TODO: state plainly that the
reproduction ran under a modernized PyTorch/numpy/opencv stack rather than
the repo's original 2019-era pins, because the original pins cannot execute
on the training GPU, and that this was verified not to touch any
hyperparameter, architecture, loss, or data-handling code.
