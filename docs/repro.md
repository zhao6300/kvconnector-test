# Repro and one-click runner

## Quick start

This is enough to reproduce the local memory benchmark:

```bash
./scripts/run_all.sh
```

To include the lightweight connector/communication benchmark:

```bash
./scripts/run_all.sh --full
```

## Tuned run

If you want to pin down the parameters:

```bash
./scripts/run_all.sh \
  --gpu 0 \
  --size $((256*1024*1024)) \
  --warmup 5 \
  --iters 10
```

To override the Python interpreter:

```bash
PY=/usr/bin/python3 ./scripts/run_all.sh
```

## Reading the output

- Each raw number is `GiB/s`, and `1 GiB = 2^30 bytes`.
- `raw CUDA kernel` goes through explicit device-side load/store.
- `Triton kernel` uses explicit `tl.load`/`tl.store`.
- `local H2D/D2H` uses standard CUDA memory copy semantics.
- `local PyTorch copy` adds the framework layer on top of the same basic copy.
- `local pinned pool` uses a contiguous pinned host pool and moves through the CUDA DMA path.

## What is reproduced

The runner is not a general benchmark suite. It keeps the set of commands small so that the same local-memory data points can be regenerated:

| Command | Purpose |
|---|---|
| `./scripts/run_all.sh` | local memory paths only |
| `./scripts/run_all.sh --full` | local memory paths + connector communication paths |
| `PY=/path/to/python ./scripts/run_all.sh` | same run with a different Python environment |

If you want a completely fixed environment, use the same `PY` path, same GPU, same warmup/iterations, and run from the same directory. The script intentionally does not hide those knobs.

## Data

The latest result summary lives here:

- [`results.md`](results.md)
- [`dvram_results.md`](dvram_results.md)
