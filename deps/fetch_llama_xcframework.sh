#!/usr/bin/env bash
# Fetch llama.cpp's prebuilt Apple XCFramework for the on-device model runtime
# (src/ios/LocalModels).
#
# Produces deps/frameworks/llama.xcframework, which the Minis target links and
# embeds. llama.cpp publishes this archive with every release
# (https://github.com/ggml-org/llama.cpp/releases), built with Metal enabled
# and with slices for iOS devices, the iOS Simulator and macOS.
#
# Pinned release; override with LLAMA_CPP_TAG=bNNNN. Set LLAMA_CPP_SHA256 to
# verify the archive. Pass --force to replace an existing framework.
#
# The C API this app uses (llama.h): model/context lifecycle, vocab,
# chat templates, tokenize/detokenize, llama_decode, the sampler chain and
# llama_memory_*. If a newer tag renames any of these, LlamaEngine.swift will
# fail to compile — pin back to the tag below or adjust the wrapper.
set -euo pipefail

TAG="${LLAMA_CPP_TAG:-b11669}"
FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/deps/frameworks"
DEST="$OUT/llama.xcframework"
CACHE="$ROOT/deps/.cache/llama.cpp"
ARCHIVE="$CACHE/llama-$TAG-xcframework.zip"
URL="https://github.com/ggml-org/llama.cpp/releases/download/$TAG/llama-$TAG-xcframework.zip"

if [[ -d "$DEST" && $FORCE -eq 0 ]]; then
  echo "llama.xcframework already present at $DEST (use --force to replace)"
  exit 0
fi

mkdir -p "$CACHE" "$OUT"

if [[ ! -f "$ARCHIVE" ]]; then
  echo "Downloading $URL"
  # GitHub's release CDN occasionally answers 503; retry generously.
  curl -fL --retry 6 --retry-all-errors --retry-delay 15 --connect-timeout 30 -o "$ARCHIVE.part" "$URL"
  mv "$ARCHIVE.part" "$ARCHIVE"
fi

if [[ -n "${LLAMA_CPP_SHA256:-}" ]]; then
  echo "${LLAMA_CPP_SHA256}  $ARCHIVE" | shasum -a 256 -c -
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
unzip -q "$ARCHIVE" -d "$WORK"

FOUND="$(find "$WORK" -type d -name 'llama.xcframework' -prune | head -n 1)"
if [[ -z "$FOUND" ]]; then
  echo "error: llama.xcframework not found inside $ARCHIVE" >&2
  exit 1
fi

rm -rf "$DEST"
cp -R "$FOUND" "$DEST"

echo "Installed llama.cpp $TAG -> $DEST"
ls "$DEST"
