#!/usr/bin/env bash
# Download the UMD Part Affordance Dataset (tools subset, ~5 GB).
# Idempotent: skips if data/umd/tools/<categories> already populated.
set -euo pipefail

DEST="${DEST:-data/umd}"
mkdir -p "$DEST/raw" "$DEST/tools"

# Primary mirror (UMIACS).
URL_TOOLS="https://obj.umiacs.umd.edu/part-affordance/part-affordance-dataset-tools.tar.gz"

if [ "$(find "$DEST/tools" -maxdepth 2 -name "*_rgb.jpg" -print -quit)" ]; then
    echo "[skip] $DEST/tools already populated"
    exit 0
fi

if [ ! -f "$DEST/raw/tools.tar.gz" ]; then
    echo "[download] $URL_TOOLS"
    wget --tries=3 --continue -O "$DEST/raw/tools.tar.gz" "$URL_TOOLS"
fi

echo "[extract] -> $DEST/tools"
tar -xzf "$DEST/raw/tools.tar.gz" -C "$DEST"
# Tarball layout is part-affordance-dataset/tools/...; flatten one level.
if [ -d "$DEST/part-affordance-dataset/tools" ]; then
    mv "$DEST/part-affordance-dataset/tools/"* "$DEST/tools/"
    rm -rf "$DEST/part-affordance-dataset"
fi

echo "[done] $(find "$DEST/tools" -name "*_rgb.jpg" | wc -l) RGB images"
