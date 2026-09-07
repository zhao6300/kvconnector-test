#!/usr/bin/env python3
"""FlashInfer PCIe IPC AllReduce Benchmark.
Uses the same backend vLLM uses for Blackwell (SM12x) without NVLink.
"""
import os
import time
import torch
import torch.distributed as dist

def main():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "2"))
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(rank)

    import flashinfer.comm as fi_comm
    ws_cls = fi_comm.pcie_ipc_ar.PcieIpcAllReduceWorkspace

    hidden = 2048
    # max_numel covers the biggest tensor we'll reduce
    max_numel = (1 << 24)  # 16M elements * bf16 = 32MB
    dtype = torch.bfloat16

    sizes = [4096, 65536, 1048576, 4194304, 16777216, 67108864]

    ws = ws_cls(group=dist.group.WORLD, max_numel=max_numel // dtype.itemsize, dtype=dtype)
    dist.barrier()
    torch.cuda.synchronize()
    if rank == 0:
        print(f"workspace created, world={world_size}, max_numel={max_numel//2}")

    num_its = 50
    warmup_its = 10
    results = []
    try:
        for nbytes in sizes:
            nelem = nbytes // dtype.itemsize
            if nelem > max_numel // dtype.itemsize:
                if rank == 0:
                    print(f"skip size {nbytes} (larger than workspace)")
                continue
            x = torch.full((nelem,), 1.0, dtype=dtype, device="cuda")

            if not ws.supports(x):
                if rank == 0:
                    print(f"skip size {nbytes} (not supported by pcie_ipc AR)")
                continue

            for _ in range(warmup_its):
                ws.all_reduce(x)
            torch.cuda.synchronize()
            dist.barrier()

            start = time.perf_counter()
            for _ in range(num_its):
                ws.all_reduce(x)
            torch.cuda.synchronize()
            dist.barrier()
            elapsed = time.perf_counter() - start
            # algebraic bandwidth
            bw = nbytes * num_its / 1e9 / elapsed
            if rank == 0:
                results.append((nbytes, elapsed / num_its * 1e6, bw))
            del x
            torch.cuda.empty_cache()
        dist.barrier()
    finally:
        ws.destroy()

    if rank == 0:
        print(f"# FlashInfer PCIe IPC AR (world_size={world_size})")
        print(f"{'size(B)':>12} {'time(us)':>10} {'BW(Gb/s)':>14}")
        for nbytes, us, bw in results:
            print(f"{nbytes:>12} {us:>10.2f} {bw:>14.2f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
