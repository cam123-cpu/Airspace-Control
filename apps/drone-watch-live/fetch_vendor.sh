#!/usr/bin/env bash
# Vendors ONNX Runtime Web next to the app so nothing loads from a third-party CDN at runtime.
# npm verifies the package tarball against the registry's sha512 integrity automatically.
set -euo pipefail
VER="1.20.1"
cd "$(dirname "$0")"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
( cd "$tmp" && npm pack "onnxruntime-web@${VER}" --silent >/dev/null )
tar -xzf "$tmp"/onnxruntime-web-${VER}.tgz -C "$tmp"
mkdir -p vendor
cp "$tmp"/package/dist/ort.min.js "$tmp"/package/dist/*.wasm vendor/
cp "$tmp"/package/dist/ort-wasm-simd-threaded*.mjs vendor/ 2>/dev/null || true
( cd vendor && sha256sum * > SHA256SUMS )
echo "Vendored onnxruntime-web ${VER}:"; cat vendor/SHA256SUMS
