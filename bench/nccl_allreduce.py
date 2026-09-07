#!/usr/bin/env python3
"""NCCL AllReduce benchmark (Qtorch, multi-GPU ring)."""
import os
import time
import torch
import torch.distributed as dist

def main():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "2"))
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(rank)

    sizes = [1 << 12, 1 << 16, 1 << 20, 1 << 22, 1 << 24, 1 << 26, 1 << 28]
    results = []
    for nbytes in sizes:
        nelem = nbytes // 2  # bf16
        buf = torch.randn(nelem, dtype=torch.bfloat16, device="cuda")
        buf2 = buf
        for _ in range(5):
            dist.all_reduce(buf2)
        torch.cuda.synchronize()
        dist.barrier()
        start = time.perf_counter()
        for _ in range(20):
            buf2 = buf
            dist.all_reduce(buf2)
        torch.cuda.synchronize()
        dist.barrier()
        elapsed = time.perf_counter() - start
        # effective bandwidth (algbw)
        eff_bw = nbytes * 20 / 1e9 / elapsed
        if rank == 0:
            results.append((nbytes, elapsed/20*1e6, eff_bw))
        del buf
        torch.cuda.empty_cache()

    if rank == 0:
        print(f"world_size={world_size}, dtype=bf16")
        print(f"{'size(B)':>12} {'time(us)':>10} {'eff_BW(Gb/s)':>14}")
        for nbytes, us, bw in results:
            print(f"{nbytes:>12} {us:>10.2f} {bw:>14.2f}")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
