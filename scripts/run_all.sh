#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/opt/venv/bin/python}"
GPU=0
SIZE=$((256*1024*1024))
WARMUP=5
ITERS=10
FULL=0
BIN="tests/local/gpu_kernel_rw"

usage() {
  cat <<'EOF'
usage: scripts/run_all.sh [--gpu N] [--size BYTES] [--warmup N] [--iters N] [--full]

--full   additionally runs the lightweight connector / communication benchmarks.
PY       override the Python interpreter, e.g. PY=python3.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --size)
      SIZE="$2"
      shift 2
      ;;
    --warmup)
      WARMUP="$2"
      shift 2
      ;;
    --iters)
      ITERS="$2"
      shift 2
      ;;
    --full)
      FULL=1
      shift
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "$ROOT"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc is required to run the local memory benchmark." >&2
  exit 1
fi

if [[ ! -x "$PY" ]]; then
  echo "Python interpreter $PY is not available." >&2
  exit 1
fi

run_local() {
  if [[ ! -x "$BIN" ]]; then
    nvcc -O3 -std=c++17 -arch=native \
         tests/local/gpu_kernel_rw.cu \
         -o "$BIN"
  fi

  log "local RAW CUDA kernel"
  "$BIN" --gpu "$GPU" --size "$SIZE" --warmup "$WARMUP" --iters "$ITERS"

  log "local Triton kernel"
  "$PY" tests/local/gpu_kernel_rw_triton.py \
        --gpu "$GPU" --size "$SIZE" --warmup "$WARMUP" --iters "$ITERS"

  log "local H2D/D2H"
  "$PY" tests/local/gpu_mem_read.py \
        --gpu "$GPU" --size "$SIZE" --warmup "$WARMUP" --iters "$ITERS"

  log "local PyTorch copy"
  "$PY" tests/local/torch_mem_read.py \
        --gpu "$GPU" --size "$SIZE" --warmup "$WARMUP" --iters "$ITERS"

  log "local pinned pool H2D"
  "$PY" tests/local/cpu_pinned_pool.py \
        --gpu "$GPU" --pool-size "$SIZE" --chunks 64 \
        --warmup "$WARMUP" --iters "$ITERS" --direction h2d

  log "local pinned pool D2H"
  "$PY" tests/local/cpu_pinned_pool.py \
        --gpu "$GPU" --pool-size "$SIZE" --chunks 64 \
        --warmup "$WARMUP" --iters "$ITERS" --direction d2h
}

run_full() {
  log "CUDA IPC"
  "$PY" bench/gpu_ipc_bw.py

  log "Mooncake transfer"
  "$PY" -m pytest -q bench/mooncake_transfer_test.py

  log "NCCL"
  "$PY" -m torch.distributed.run --nproc_per_node=2 bench/nccl_bw.py

  log "CUDA P2P"
  ./bench/p2p_bw 0 1
}

log() {
  echo "=== $* ==="
}

if [[ "$FULL" -eq 1 ]]; then
  run_local
  run_full
else
  run_local
fi
