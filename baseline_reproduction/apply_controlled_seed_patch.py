#!/usr/bin/env python3
"""Apply the audited, idempotent seed-control patch to upstream tools/train.py."""

import argparse
from pathlib import Path


REPLACEMENTS = (
    (
        'warnings.filterwarnings("ignore")\n',
        'warnings.filterwarnings("ignore")\nimport random\nimport numpy as np\n',
    ),
    (
        'import torch.nn.functional as F\n\ndef parse_args():',
        'import torch.nn.functional as F\n\n'
        'def _seed_worker(worker_id):\n'
        '    worker_seed = torch.initial_seed() % (2**32)\n'
        '    np.random.seed(worker_seed)\n'
        '    random.seed(worker_seed)\n\n'
        'def parse_args():',
    ),
    (
        '    args = parse_args()\n'
        "    logger, final_output_dir, tb_log_dir = utils.create_logger(config, args.cfg, 'train')\n",
        '    args = parse_args()\n'
        '    run_seed = int(os.environ["HRNET_RUN_SEED"])\n'
        '    random.seed(run_seed)\n'
        '    np.random.seed(run_seed)\n'
        '    torch.manual_seed(run_seed)\n'
        '    torch.cuda.manual_seed_all(run_seed)\n'
        '    loader_generator = torch.Generator()\n'
        '    loader_generator.manual_seed(run_seed)\n'
        "    logger, final_output_dir, tb_log_dir = utils.create_logger(config, args.cfg, 'train')\n",
    ),
    (
        '    cudnn.benchmark = config.CUDNN.BENCHMARK\n'
        '    cudnn.deterministic = config.CUDNN.DETERMINISTIC\n',
        '    cudnn.benchmark = False\n'
        '    cudnn.deterministic = True\n',
    ),
    (
        '        pin_memory=config.PIN_MEMORY)\n',
        '        pin_memory=config.PIN_MEMORY,\n'
        '        worker_init_fn=_seed_worker,\n'
        '        generator=loader_generator)\n',
    ),
    (
        '        pin_memory=config.PIN_MEMORY\n'
        '    )\n',
        '        pin_memory=config.PIN_MEMORY,\n'
        '        worker_init_fn=_seed_worker,\n'
        '        generator=loader_generator\n'
        '    )\n',
    ),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo")
    args = parser.parse_args()
    path = Path(args.repo).resolve() / "tools" / "train.py"
    text = path.read_text(encoding="utf-8")
    changed = False
    for old, new in REPLACEMENTS:
        if new in text:
            continue
        count = text.count(old)
        if count != 1:
            raise SystemExit(f"refusing to patch {path}: expected one exact source block, found {count}")
        text = text.replace(old, new, 1)
        changed = True
    if changed:
        path.write_text(text, encoding="utf-8")
        print(f"[OK] applied controlled-seed patch to {path}")
    else:
        print(f"[OK] controlled-seed patch already present in {path}")


if __name__ == "__main__":
    main()
