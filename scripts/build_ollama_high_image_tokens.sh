#!/usr/bin/env bash
# Build a custom Ollama with Gemma 4 image-token budget bumped to 1120,
# AND optionally install + restart + verify in one shot.
#
# Why: Ollama's gemma4 image preprocessor hardcodes maxTokens := 280 in
# model/models/gemma4/process_image.go. The Gemma 4 architecture supports
# 70/140/280/560/1120 token budgets but Ollama doesn't expose any way to
# pick a higher one — no Modelfile PARAMETER, no /api/chat option, no
# env var. This script clones the exact source for the version we have
# installed, patches that constant, runs the FULL build (cmake for the
# native CUDA libs + go for the orchestrator), and (with --install)
# rolls it out and runs the verification benchmark.
#
# Safe to re-run: idempotent. Patch is skipped if already applied; the
# build incrementally rebuilds only what's stale; the install backs up
# the stock binary on first run and respects existing backups.
#
# Usage:
#   bash scripts/build_ollama_high_image_tokens.sh             # build only
#   bash scripts/build_ollama_high_image_tokens.sh --install   # build + install + restart + bench
#   bash scripts/build_ollama_high_image_tokens.sh --install --skip-restore   # don't reset to stock first
#
# With --install you'll be prompted for your sudo password ONCE at the
# start (sudo -v caches it). The build can take 10-15 min so the script
# refreshes the sudo cache right before the install step too. If your
# default sudo timeout is shorter than 15 min you may need to re-enter.
#
# Reverting after --install:
#   sudo systemctl stop ollama
#   sudo install -m 0755 /usr/local/bin/ollama.orig /usr/local/bin/ollama
#   [[ -d /usr/local/lib/ollama.orig ]] && sudo rm -rf /usr/local/lib/ollama && sudo cp -av /usr/local/lib/ollama.orig /usr/local/lib/ollama
#   sudo systemctl start ollama

set -euo pipefail

OLLAMA_VERSION="${OLLAMA_VERSION:-v0.20.2}"
OLLAMA_SRC_DIR="${OLLAMA_SRC_DIR:-$HOME/ollama-src}"
TARGET_FILE="model/models/gemma4/process_image.go"

# Match these to whatever you want — defaults are 4x the upstream values.
NEW_MIN_TOKENS="${NEW_MIN_TOKENS:-160}"
NEW_MAX_TOKENS="${NEW_MAX_TOKENS:-1120}"

DO_INSTALL=0
DO_RESTORE=1
for arg in "$@"; do
  case "$arg" in
    --install)       DO_INSTALL=1 ;;
    --skip-restore)  DO_RESTORE=0 ;;
    --help|-h)
      sed -n '1,32p' "$0" | grep -E '^#' | sed 's/^# //; s/^#//'
      exit 0
      ;;
    *) echo "unknown arg: $arg" >&2; exit 1 ;;
  esac
done

# Resolve gen-harness root from this script's location so we can find
# bench_ollama_quick.sh after we cd into the ollama src dir.
GEN_HARNESS_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH_SCRIPT="$GEN_HARNESS_ROOT/scripts/bench_ollama_quick.sh"

step()  { printf '\n\033[1;36m==>\033[0m %s\n' "$*"; }
info()  { printf '    %s\n' "$*"; }
warn()  { printf '\033[1;33m!!\033[0m  %s\n' "$*" >&2; }
fail()  { printf '\033[1;31mxx\033[0m  %s\n' "$*" >&2; exit 1; }

# ---- 0. Pre-flight: cache sudo creds + restore stock so we have a known
#         baseline to swap from. Only when --install was passed.
if [[ $DO_INSTALL -eq 1 ]]; then
  step "Pre-flight: caching sudo credentials"
  if ! sudo -v; then
    fail "sudo authentication failed"
  fi
  info "sudo cached; we'll refresh again right before install"

  if [[ $DO_RESTORE -eq 1 && -f /usr/local/bin/ollama.orig ]]; then
    step "Restoring stock Ollama (so we have a known baseline)"
    sudo systemctl stop ollama 2>/dev/null || true
    sudo install -m 0755 /usr/local/bin/ollama.orig /usr/local/bin/ollama
    if [[ -d /usr/local/lib/ollama.orig ]]; then
      sudo rm -rf /usr/local/lib/ollama
      sudo cp -av /usr/local/lib/ollama.orig /usr/local/lib/ollama >/dev/null
    fi
    sudo systemctl start ollama
    sleep 3
    info "stock daemon restored: $(curl -s --max-time 3 http://127.0.0.1:11434/api/version || echo unreachable)"
  fi
fi

# ---- 1. Go toolchain ------------------------------------------------------
step "Checking Go toolchain"
if command -v go >/dev/null 2>&1; then
  GO_BIN="$(command -v go)"
  GO_VER="$($GO_BIN version | awk '{print $3}')"
  info "found: $GO_BIN ($GO_VER)"
else
  warn "go not installed"
  info "this will run: sudo apt update && sudo apt install -y golang-go"
  read -rp "    proceed? [y/N] " yn
  [[ "$yn" =~ ^[Yy]$ ]] || fail "aborted by user"
  sudo apt update
  sudo apt install -y golang-go
  GO_BIN="$(command -v go)"
  GO_VER="$($GO_BIN version | awk '{print $3}')"
  info "installed: $GO_VER"
fi

# Ollama needs Go 1.22+ as of v0.20. Ubuntu 24.04 ships 1.22.x — fine.
case "$GO_VER" in
  go1.2[2-9]*|go1.[3-9][0-9]*|go[2-9]*)
    info "go version OK"
    ;;
  *)
    warn "go version may be too old for Ollama (need >= 1.22)"
    warn "continuing anyway — ask me to switch to a newer Go if build fails"
    ;;
esac

# ---- 2. Source clone -----------------------------------------------------
step "Cloning ollama@$OLLAMA_VERSION → $OLLAMA_SRC_DIR"
if [[ -d "$OLLAMA_SRC_DIR/.git" ]]; then
  info "directory already exists, fetching + checking out $OLLAMA_VERSION"
  pushd "$OLLAMA_SRC_DIR" >/dev/null
  git fetch --depth 1 origin "tag" "$OLLAMA_VERSION" || git fetch origin
  git checkout "$OLLAMA_VERSION"
  popd >/dev/null
else
  git clone --branch "$OLLAMA_VERSION" --depth 1 \
    https://github.com/ollama/ollama.git "$OLLAMA_SRC_DIR"
fi
info "checked out: $(cd "$OLLAMA_SRC_DIR" && git describe --tags 2>/dev/null || git rev-parse --short HEAD)"

# ---- 3. Patch ------------------------------------------------------------
step "Patching $TARGET_FILE: $NEW_MIN_TOKENS / $NEW_MAX_TOKENS"
cd "$OLLAMA_SRC_DIR"
[[ -f "$TARGET_FILE" ]] || fail "$TARGET_FILE not found in this Ollama version"

# Show current values
info "current source:"
grep -nE 'minTokens *:=|maxTokens *:=' "$TARGET_FILE" | sed 's/^/      /'

if grep -q "maxTokens := $NEW_MAX_TOKENS" "$TARGET_FILE"; then
  info "patch already applied — skipping sed"
else
  # Back up unpatched original on first run
  if [[ ! -f "$TARGET_FILE.orig" ]]; then
    cp "$TARGET_FILE" "$TARGET_FILE.orig"
    info "saved original at $TARGET_FILE.orig"
  fi

  sed -i -E \
    -e "s/(minTokens *:= *)[0-9]+/\1$NEW_MIN_TOKENS/" \
    -e "s/(maxTokens *:= *)[0-9]+/\1$NEW_MAX_TOKENS/" \
    "$TARGET_FILE"

  info "after patch:"
  grep -nE 'minTokens *:=|maxTokens *:=' "$TARGET_FILE" | sed 's/^/      /'

  # Sanity-check both lines now match the requested values
  grep -q "minTokens := $NEW_MIN_TOKENS" "$TARGET_FILE" || fail "patch did not apply minTokens"
  grep -q "maxTokens := $NEW_MAX_TOKENS" "$TARGET_FILE" || fail "patch did not apply maxTokens"
fi

# ---- 4. Build ------------------------------------------------------------
#
# Ollama is a TWO-STAGE build:
#   1. cmake → produces the native .so files (libggml-cuda.so, libcublas, etc)
#      compiled against your local CUDA toolkit + CPU SIMD level. These hold
#      the actual GPU inference kernels — they are where performance lives.
#      WITHOUT this step the orchestrator falls back to slow CPU paths and
#      runs ~16x slower (we measured this the hard way).
#   2. go build → produces the orchestrator binary that loads those .so files
#      at runtime via dlopen(). This is where our process_image.go patch
#      compiles in.
#
# We need BOTH stages, in this order. The .so files end up in build/lib/ollama/
# and we install them alongside the binary into /usr/local/lib/ollama/.

# Detect CUDA toolkit so we can pick the right CMake preset.
CUDA_PRESET=""
if command -v nvcc >/dev/null 2>&1; then
  cuda_ver=$(nvcc --version 2>/dev/null | grep -oE 'release [0-9]+' | grep -oE '[0-9]+' | head -1 || echo 0)
  case "$cuda_ver" in
    11) CUDA_PRESET="CUDA 11" ;;
    12) CUDA_PRESET="CUDA 12" ;;
    13) CUDA_PRESET="CUDA 13" ;;
    *)  warn "unrecognized nvcc version '$cuda_ver' — defaulting to CUDA 12 preset" ; CUDA_PRESET="CUDA 12" ;;
  esac
  info "detected CUDA toolkit: $cuda_ver → preset '$CUDA_PRESET'"
elif [[ -d /usr/local/cuda ]] || [[ -d /usr/local/cuda-12.8 ]] || [[ -d /usr/local/cuda-12 ]]; then
  CUDA_PRESET="CUDA 12"
  info "no nvcc on PATH but /usr/local/cuda* exists → assuming CUDA 12"
else
  warn "no CUDA toolkit detected — building CPU-only (will be slow)"
  CUDA_PRESET="CPU"
fi

step "Stage 1/2: cmake configure + build native libs (5-10 min)"
info "preset: $CUDA_PRESET"
info "running: cmake --preset='$CUDA_PRESET' && cmake --build build --parallel"
# Configure into ./build/. The preset sets CMAKE_BUILD_TYPE=Release and
# enables shared libs, plus the CUDA arch flags.
cmake --preset="$CUDA_PRESET"
time cmake --build build --parallel

# Verify the .so files we care about actually got produced. The build
# output puts everything flat under build/lib/ollama/ — the cuda_v12/
# subdir convention only kicks in at the install location.
# libggml-base.so is a symlink so check its target.
expected_libs=(
  "build/lib/ollama/libggml-base.so.0.0.0"
  "build/lib/ollama/libggml-cpu-x64.so"
)
if [[ "$CUDA_PRESET" != "CPU" ]]; then
  expected_libs+=("build/lib/ollama/libggml-cuda.so")
fi
for lib in "${expected_libs[@]}"; do
  if [[ ! -f "$lib" ]]; then
    fail "expected library missing after cmake build: $lib"
  fi
  info "  produced: $lib  ($(du -h "$lib" | cut -f1))"
done

step "Stage 2/2: go build orchestrator (1-2 min)"
info "GOCACHE=$(go env GOCACHE)"
info "running: go build ."
time CGO_ENABLED=1 go build -ldflags="-s -w" -o "$OLLAMA_SRC_DIR/ollama" .

[[ -x "$OLLAMA_SRC_DIR/ollama" ]] || fail "go build did not produce ./ollama"

NEW_VER="$($OLLAMA_SRC_DIR/ollama --version 2>&1 | head -1)"
info "built binary: $OLLAMA_SRC_DIR/ollama"
info "reports: $NEW_VER"
info "size: $(du -h "$OLLAMA_SRC_DIR/ollama" | cut -f1)"

# ---- 5. Install + verify (only with --install) ---------------------------
if [[ $DO_INSTALL -eq 0 ]]; then
  step "Build complete (build only)"
  info "to install: bash scripts/build_ollama_high_image_tokens.sh --install"
  info "or run these manually:"
  cat <<EOF

  sudo systemctl stop ollama
  [[ -f /usr/local/bin/ollama.orig ]] || sudo cp -v /usr/local/bin/ollama /usr/local/bin/ollama.orig
  [[ -d /usr/local/lib/ollama.orig ]] || sudo cp -av /usr/local/lib/ollama /usr/local/lib/ollama.orig
  sudo install -m 0755 "$OLLAMA_SRC_DIR/ollama" /usr/local/bin/ollama
  sudo cp -av "$OLLAMA_SRC_DIR/build/lib/ollama/." /usr/local/lib/ollama/
  sudo systemctl start ollama
  sleep 3
  bash scripts/bench_ollama_quick.sh
EOF
  exit 0
fi

step "Installing — refreshing sudo cache"
if ! sudo -v; then
  fail "sudo cache expired and re-auth failed"
fi

step "Stopping daemon"
sudo systemctl stop ollama

step "Backing up stock binary + libs (idempotent)"
if [[ ! -f /usr/local/bin/ollama.orig ]]; then
  sudo cp -v /usr/local/bin/ollama /usr/local/bin/ollama.orig
else
  info "/usr/local/bin/ollama.orig already exists — keeping"
fi
if [[ ! -d /usr/local/lib/ollama.orig ]]; then
  sudo cp -av /usr/local/lib/ollama /usr/local/lib/ollama.orig >/dev/null
  info "saved /usr/local/lib/ollama → /usr/local/lib/ollama.orig"
else
  info "/usr/local/lib/ollama.orig already exists — keeping"
fi

step "Installing new binary"
sudo install -m 0755 "$OLLAMA_SRC_DIR/ollama" /usr/local/bin/ollama
info "installed: $(ls -la /usr/local/bin/ollama)"

step "Installing new native libs into Ollama's expected layout"
# Official install layout:
#   /usr/local/lib/ollama/{libggml-base.so*, libggml-cpu-*.so}
#   /usr/local/lib/ollama/cuda_v12/libggml-cuda.so
#   /usr/local/lib/ollama/cuda_v13/libggml-cuda.so
#
# Our build produces everything flat under build/lib/ollama/ and only
# builds one CUDA variant (matching the CUDA toolkit on this box).
# We install into the matching cuda_v* subdir based on the preset.
BUILD_LIB_DIR="$OLLAMA_SRC_DIR/build/lib/ollama"

# Top-level CPU + base libs (preserve symlink chain via cp -av).
for f in libggml-base.so libggml-base.so.0 libggml-base.so.0.0.0 \
         libggml-cpu-alderlake.so libggml-cpu-haswell.so \
         libggml-cpu-icelake.so libggml-cpu-sandybridge.so \
         libggml-cpu-skylakex.so libggml-cpu-sse42.so libggml-cpu-x64.so; do
  if [[ -e "$BUILD_LIB_DIR/$f" ]]; then
    sudo cp -av "$BUILD_LIB_DIR/$f" "/usr/local/lib/ollama/$f" >/dev/null
  fi
done

# CUDA lib goes under the cuda_v* subdir matching the build preset.
case "$CUDA_PRESET" in
  "CUDA 11") cuda_subdir="cuda_v11" ;;
  "CUDA 12") cuda_subdir="cuda_v12" ;;
  "CUDA 13") cuda_subdir="cuda_v13" ;;
  *)         cuda_subdir="" ;;
esac
if [[ -n "$cuda_subdir" && -f "$BUILD_LIB_DIR/libggml-cuda.so" ]]; then
  sudo mkdir -p "/usr/local/lib/ollama/$cuda_subdir"
  sudo install -m 0755 "$BUILD_LIB_DIR/libggml-cuda.so" \
    "/usr/local/lib/ollama/$cuda_subdir/libggml-cuda.so"
  info "installed CUDA lib to /usr/local/lib/ollama/$cuda_subdir/libggml-cuda.so"
  info "  size: $(du -h /usr/local/lib/ollama/$cuda_subdir/libggml-cuda.so | cut -f1)"

  # Also overwrite the OTHER cuda_v* subdir if it exists, so the runtime
  # picks our build regardless of which CUDA major it auto-detects.
  for other in cuda_v11 cuda_v12 cuda_v13; do
    if [[ "$other" != "$cuda_subdir" && -d "/usr/local/lib/ollama/$other" ]]; then
      sudo install -m 0755 "$BUILD_LIB_DIR/libggml-cuda.so" \
        "/usr/local/lib/ollama/$other/libggml-cuda.so"
      info "  also overwrote /usr/local/lib/ollama/$other/libggml-cuda.so"
    fi
  done
fi

info "lib dir contents:"
ls -la /usr/local/lib/ollama/ /usr/local/lib/ollama/cuda_v* 2>/dev/null | sed 's/^/      /' | head -40

step "Starting daemon"
sudo systemctl start ollama
# Wait up to 30s for /api/version to respond
ready=0
for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  sleep 2
  if curl -s --max-time 3 http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
    ready=1
    break
  fi
done
if [[ $ready -ne 1 ]]; then
  warn "daemon didn't respond within 30s — printing recent journal logs"
  sudo journalctl -u ollama -n 30 --no-pager
  fail "daemon down after install — try manual revert (see top of script)"
fi
info "daemon up: $(curl -s http://127.0.0.1:11434/api/version)"

step "Running verification benchmark (~30s)"
if [[ -f "$BENCH_SCRIPT" ]]; then
  ( cd "$GEN_HARNESS_ROOT" && bash "$BENCH_SCRIPT" ) || warn "bench reported a non-zero exit"
else
  warn "$BENCH_SCRIPT missing — skipping verification"
fi

step "Done"
info "patched binary installed at /usr/local/bin/ollama"
info "stock backup at /usr/local/bin/ollama.orig"
info "stock libs at /usr/local/lib/ollama.orig"
echo
info "expected: prompt_eval_count ~1130 per image (was ~290)"
info "expected: prefill rate close to stock (~7000 tok/s, slightly less due to 4x tokens)"
echo
info "to revert:"
info "  sudo systemctl stop ollama"
info "  sudo install -m 0755 /usr/local/bin/ollama.orig /usr/local/bin/ollama"
info "  sudo rm -rf /usr/local/lib/ollama && sudo cp -av /usr/local/lib/ollama.orig /usr/local/lib/ollama"
info "  sudo systemctl start ollama"
