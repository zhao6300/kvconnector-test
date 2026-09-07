#!/usr/bin/env python3
"""NCCL benchmark: point-to-point send/recv + all_reduce (PyTorch + NCCL)."""
import os
import time
import torch
import torch.distributed as dist


def bench_p2p(rank, world_size, sizes, warmup=10, iters=50):
    """Two-rank send/recv bandwidth (rank0 -> rank1)."""
    is_src = rank == 0
    dst = 1 if is_src else 0
    stream = torch.cuda.current_stream()

    results = []
    for size in sizes:
        nbytes = size
        nelem = nbytes // 4  # float32
        if nelem == 0:
            continue
        buf = torch.empty(nelem, dtype=torch.float32, device="cuda")
        recv = torch.empty(nelem, dtype=torch.float32, device="cuda") if not is_src else None

        for _ in range(warmup):
            if is_src:
                dist.send(buf, dst=dst)
            else:
                dist.recv(recv, src=dst)

        torch.cuda.synchronize()
        dist.barrier()
        start = time.perf_counter()
        for _ in range(iters):
            if is_src:
                dist.send(buf, dst=dst)
            else:
                dist.recv(recv, src=dst)
        torch.cuda.synchronize()
        dist.barrier()
        elapsed = time.perf_counter() - start
        if is_src:
            bw = nbytes * iters / 1e9 / elapsed
            results.append((nbytes, elapsed / iters * 1e6, bw))
    return results


def bench_allreduce(rank, world_size, sizes, warmup=5, iters=20):
    """AllReduce bandwidth (2 GPUs, ring algorithm)."""
    results = []
    for size in sizes:
        nbytes = size
        nelem = nbytes // 4
        if nelem == 0:
            continue
        buf = torch.randn(nelem, dtype=torch.float32, device="cuda")
        for _ in range(warmup):
            dist.all_reduce(buf)
        torch.cuda.synchronize()
        dist.barrier()
        start = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(buf)
        torch.cuda.synchronize()
        dist.barrier()
        elapsed = time.perf_counter() - start
        # effective bandwidth: 2*(n-1)/n * size / time
        if world_size == 2:
            effective = 2 * (world_size - 1) / world_size * nbytes * iters / 1e9 / elapsed
        else:
            effective = nbytes * iters / 1e9 / elapsed  # lower bound
        if rank == 0:
            results.append((nbytes, elapsed / iters * 1e6, effective))
    return results


def main():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "2"))
    master_port = os.environ.get("MASTER_PORT", "29500")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    torch.cuda.synchronize()

    sizes = [4096, 65536, 1048576, 1048576 * 4, 1048576 * 16, 1048576 * 64, 1048576 * 256]

    if rank == 0:
        print(f"world_size={world_size}, NCCL backend")
    torch.cuda.synchronize()
    dist.barrier()

    p2p_results = bench_p2p(rank, world_size, sizes)
    if rank == 0:
        print("\n# NCCL P2P send/recv (rank0 -> rank1)")
        print("size(B)    time(us)   BW_GB/s")
        for nbytes, us, bw in p2p_results:
            print(f"{nbytes:<10} {us:<10.2f} {bw:<10.2f}")

    ar_results = bench_allreduce(rank, world_size, sizes)
    if rank == 0:
        print("\n# NCCL AllReduce (ring, 2 GPUs)")
        print("size(B)    time(us)   eff_BW_GB/s")
        for nbytes, us, bw in ar_results:
            print(f"{nbytes:<10} {us:<10.2f} {bw:<10.2f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
