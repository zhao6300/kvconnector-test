#!/usr/bin/env python3
"""Benchmark CUDA IPC shared-memory device-to-device copies on one GPU."""

import argparse
from multiprocessing import get_context
import time
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    return parser.parse_args()


def worker(rank: int, size: int, warmup: int, iters: int,
           send_q, result_q, event) -> None:
    device = torch.device("cuda:0")

    if rank == 0:
        src = torch.empty(size, dtype=torch.uint8, device=device)
        src.fill_(123)
        send_q.put(src)
        event.wait()
        del src
        return

    remote = send_q.get()
    dst = torch.empty_like(remote)

    for _ in range(warmup):
        dst.copy_(remote)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        dst.copy_(remote)
    end.record()
    torch.cuda.synchronize()

    elapsed_seconds = start.elapsed_time(end) * 1e-3
    bandwidth = size * iters / elapsed_seconds
    result_q.put({"elapsed_seconds": elapsed_seconds, "bandwidth": bandwidth})

    del remote
    del dst
    event.set()


def main() -> None:
    args = parse_args()
    ctx = get_context()
    send_q = ctx.Queue()
    result_q = ctx.Queue()
    event = ctx.Event()

    processes = [
        ctx.Process(
            target=worker,
            args=(rank, args.size, args.warmup, args.iters, send_q, result_q, event),
        ) for rank in range(2)
    ]
    for process in processes:
        process.start()

    result = result_q.get()
    print(f"elapsed={result['elapsed_seconds']:.6f}s")
    print(f"per_copy_us={result['elapsed_seconds'] / args.iters * 1e6:.2f}")
    print(f"bandwidth_GiB_s={result['bandwidth'] / (1 << 30):.3f}")
    for process in processes:
        process.join()


if __name__ == "__main__":
    main()
