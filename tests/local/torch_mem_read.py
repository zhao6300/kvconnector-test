#!/usr/bin/env python3
"""Benchmark host-memory to GPU loads through PyTorch copy paths."""

import argparse
import time

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--size", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    return parser.parse_args()


def make_source(nbytes: int, pinned: bool) -> torch.Tensor:
    source = torch.empty(nbytes, dtype=torch.uint8, device="cpu")
    if pinned:
        source = source.pin_memory()
    source.fill_(91)
    return source


def transfer_once(method: str, source: torch.Tensor,
                  target: torch.Tensor) -> None:
    if method.endswith("_to"):
        target.copy_(source.to(target.device))
    else:
        target.copy_(source)


def bench(method: str, size: int, gpu: int, warmup: int, iters: int) -> float:
    torch.cuda.set_device(gpu)
    pinned = method.startswith("pinned")
    source = make_source(size, pinned)
    target = torch.empty(size, dtype=torch.uint8, device=f"cuda:{gpu}")

    for _ in range(warmup):
        transfer_once(method, source, target)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        transfer_once(method, source, target)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return size * iters / elapsed


def main() -> None:
    args = parse_args()
    methods = ("pageable_copy", "pageable_to", "pinned_copy", "pinned_to")
    samples = {}
    print(f"device=cuda:{args.gpu}")
    print("method            bandwidth_GiB_s  per_copy_us")
    for method in methods:
        sample = bench(method, args.size, args.gpu, args.warmup, args.iters)
        samples[method] = sample
        print(f"{method:<18} {sample / (1 << 30):>10.3f}     "
              f"{args.size / sample * 1e6:>6.2f}")

    best = max(samples.items(), key=lambda item: item[1])
    print(f"best={best[0]} bandwidth_GiB_s={best[1] / (1 << 30):.3f}")


if __name__ == "__main__":
    main()
