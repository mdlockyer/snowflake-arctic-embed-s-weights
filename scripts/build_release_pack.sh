#!/usr/bin/env bash
# Builds the deterministic Int8 SmoothQuant model pack (`dist/model.ae8`) and packages the
# release tarball (`dist/arctic-embed-s-ae8.tar.gz`) with `vocab.txt`, `test-data/`, and `SHA256SUMS`.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WEIGHTS_DIR="${ARCTIC_EMBED_SOURCE_WEIGHTS:-$ROOT/weights}"
DIST_DIR="${1:-$ROOT/dist}"
EXPECTED_AE8_SHA="d68b00aca78b143fb2fe3f8116eb2330d175f2a8318774725cf40bc906d564f1"

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

if [[ ! -f "$WEIGHTS_DIR/model.safetensors" ]]; then
  "$ROOT/scripts/download_weights.sh" "$WEIGHTS_DIR"
fi

rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR/test-data"

swift run -c release --package-path "$ROOT" arctic-embed-convert \
  "$WEIGHTS_DIR/model.safetensors" \
  "$DIST_DIR/model.ae8"

GOT_SHA="$(sha256 "$DIST_DIR/model.ae8")"
if [[ "$GOT_SHA" != "$EXPECTED_AE8_SHA" ]]; then
  echo "error: model.ae8 SHA-256 mismatch (got $GOT_SHA, expected $EXPECTED_AE8_SHA)" >&2
  exit 1
fi
echo "verified  dist/model.ae8 ($GOT_SHA)"

cp "$ROOT/vocab.txt" "$DIST_DIR/vocab.txt"
cp "$ROOT/LICENSE" "$DIST_DIR/LICENSE"
cp "$ROOT/NOTICE" "$DIST_DIR/NOTICE"
cp "$ROOT/test-data/"* "$DIST_DIR/test-data/"

(
  cd "$DIST_DIR"
  for f in model.ae8 vocab.txt test-data/reference_test_vectors.json test-data/long_reference_vectors.json test-data/long_texts.txt; do
    echo "$(sha256 "$f")  $f"
  done > SHA256SUMS
  tar -czf arctic-embed-s-ae8.tar.gz model.ae8 vocab.txt LICENSE NOTICE SHA256SUMS test-data
)

echo "packaged  $DIST_DIR/arctic-embed-s-ae8.tar.gz"
