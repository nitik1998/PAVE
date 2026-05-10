#!/usr/bin/env bash
# Print a concise inventory of all assets in outputs/ for quick navigation.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Probing tables (val) ==="
for f in outputs/tables/*_overall.csv; do
  [ -f "$f" ] || continue
  m=$(basename "$f" _overall.csv)
  miou=$(python3 -c "import csv; r=next(csv.DictReader(open('$f'))); print(f\"{float(r['mIoU']):.3f}\")")
  printf "  %-22s  mIoU=%s\n" "$m" "$miou"
done

echo
echo "=== Probing tables (test) ==="
for f in outputs/tables_test/*_overall.csv; do
  [ -f "$f" ] || continue
  m=$(basename "$f" _overall.csv)
  miou=$(python3 -c "import csv; r=next(csv.DictReader(open('$f'))); print(f\"{float(r['mIoU']):.3f}\")")
  printf "  %-22s  mIoU=%s\n" "$m" "$miou"
done

echo
echo "=== Figures ==="
ls -1 outputs/figures/*.png outputs/figures/*.mp4 2>/dev/null | sed 's|outputs/figures/|  |'

echo
echo "=== Docs ==="
ls -1 outputs/*.md 2>/dev/null | sed 's|outputs/|  |'

echo
echo "=== Predictions ==="
for d in outputs/predictions/*/; do
  [ -d "$d" ] || continue
  count=$(ls "$d" 2>/dev/null | wc -l)
  printf "  %-30s  %d files\n" "${d#outputs/}" "$count"
done
for d in outputs/predictions/test/*/; do
  [ -d "$d" ] || continue
  count=$(ls "$d" 2>/dev/null | wc -l)
  printf "  %-30s  %d files\n" "${d#outputs/}" "$count"
done
