#!/usr/bin/env bash
set -euo pipefail

BACKUP_DIR="${1:-/mnt/d/download/Project coding/EOMT/checkpoint_backups}"

if [[ ! -d "$BACKUP_DIR" ]]; then
  echo "ERROR: backup directory not found: $BACKUP_DIR" >&2
  exit 1
fi

found_archives=0
found_checkpoints=0

while IFS= read -r -d '' archive; do
  found_archives=$((found_archives + 1))
  matches="$(
    tar -tf "$archive" 2>/dev/null \
      | grep -Ei '\.ckpt$' \
      | grep -vi 'multicentre' \
      | grep -Ei '(bpd|ofd|apad|tad|fl).*(dinov2|dinov3)|(dinov2|dinov3).*(bpd|ofd|apad|tad|fl)' \
      || true
  )"
  if [[ -n "$matches" ]]; then
    echo "=== $archive ==="
    printf '%s\n' "$matches"
    count="$(printf '%s\n' "$matches" | grep -cE '(_final|/last)\.ckpt$' || true)"
    found_checkpoints=$((found_checkpoints + count))
  fi
done < <(find "$BACKUP_DIR" -maxdepth 1 -type f -name '*.tar' -print0 | sort -z)

echo "=== audit summary ==="
echo "tar archives scanned: $found_archives"
echo "candidate final/last checkpoints listed: $found_checkpoints"
