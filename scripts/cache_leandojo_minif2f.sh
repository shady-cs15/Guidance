#!/usr/bin/env bash
set -euo pipefail

# Usage:
# bash scripts/cache_leandojo_minif2f.sh export s3://my-bucket/lean-dojo-cache
# bash scripts/cache_leandojo_minif2f.sh import s3://my-bucket/lean-dojo-cache


CACHE_DIR="${LEAN_DOJO_CACHE:-$HOME/.cache/lean_dojo}"
ARCHIVE_NAME="leandojo_minif2f_cache.tar.zst"
# Set this to your S3/GCS bucket or shared storage path
REMOTE="${2:-${LEANDOJO_CACHE_REMOTE:-}}"

if [ -z "$REMOTE" ]; then
  echo "Error: No remote path provided."
  echo "Usage: $0 {export|import} <s3://bucket/path>"
  echo "  Or set LEANDOJO_CACHE_REMOTE env var."
  exit 1
fi

case "${1:-}" in
  export)
    echo "Packing LeanDojo cache from $CACHE_DIR ..."
    tar -C "$(dirname "$CACHE_DIR")" -cf - "$(basename "$CACHE_DIR")" \
      | zstd -T0 -3 > "/tmp/$ARCHIVE_NAME"
    echo "Uploading to $REMOTE/$ARCHIVE_NAME ..."
    aws s3 cp "/tmp/$ARCHIVE_NAME" "$REMOTE/$ARCHIVE_NAME"
    rm "/tmp/$ARCHIVE_NAME"
    echo "Cache exported."
    ;;
  import)
    if [ -d "$CACHE_DIR" ] && [ "$(ls -A "$CACHE_DIR" 2>/dev/null)" ]; then
      echo "Cache already exists at $CACHE_DIR — skipping. Use 'import --force' to overwrite."
      exit 0
    fi
    echo "Downloading from $REMOTE/$ARCHIVE_NAME ..."
    aws s3 cp "$REMOTE/$ARCHIVE_NAME" "/tmp/$ARCHIVE_NAME"
    echo "Extracting to $CACHE_DIR ..."
    mkdir -p "$(dirname "$CACHE_DIR")"
    zstd -d "/tmp/$ARCHIVE_NAME" --stdout | tar -C "$(dirname "$CACHE_DIR")" -xf -
    rm "/tmp/$ARCHIVE_NAME"
    echo "Cache imported to $CACHE_DIR"
    ;;
  *)
    echo "Usage: $0 {export|import}"
    exit 1
    ;;
esac