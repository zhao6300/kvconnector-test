# comm

This repository holds small, focused GPU communication and KV-transfer connector tests divided across a few scripts in the `bench/` directory. The main summary is in [`bench/results.md`](bench/results.md).

## Layout

- `bench/` - benchmark scripts for CUDA IPC, Mooncake, NIXL, CUDA P2P, NCCL, and FlashInfer-based paths.
- `bench/results.md` - consolidated benchmark and connector-semantic notes.
- `run_logs/` - stdout/stderr snapshots from a few representative runs.

## Run benchmarks

The scripts expect the existing local environment at `/opt/venv`.

```bash
/opt/venv/bin/python bench/gpu_ipc_bw.py
/opt/venv/bin/python -m pytest -q bench/mooncake_transfer_test.py
torchrun --nproc_per_node=2 bench/nccl_bw.py
```

For the NIXL benchmark, start the target and initiator separately:

```bash
/opt/venv/bin/python bench/nixl_bw.py --mode target --gpu 1 --size 67108864 --warmup 2 --iters 10
/opt/venv/bin/python bench/nixl_bw.py --mode initiator --gpu 0 --ip 127.0.0.1 --size 67108864 --warmup 2 --iters 10
```

For Mooncake transfer testing:

```bash
/opt/venv/bin/python bench/mooncake_transfer_cross_gpu.py --source-gpu 0 --target-gpu 1 --size 67108864 --warmup 10 --iters 50
```

## Reading results

`bench/results.md` contains the measured bandwidth and a short interpretation of what each path does. It also has a separate connector comparison covering Nixl, Mooncake, LMCache, and FlexKV.

## Notes

- The focus is on explaining and measuring communication paths, not on packaging a library.
- Running a benchmark will create a few lightweight local artifacts such as Python/pytest caches and small log files. Those are intentionally not kept in version control.
