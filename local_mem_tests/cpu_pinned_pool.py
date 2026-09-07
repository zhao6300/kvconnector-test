#!/usr/bin/env python3
"""Benchmark a CPU pinned memory pool against CUDA H2D/D2H DMA."""

import argparse
import random
import time

import numpy as np
import cupy as cp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--pool-size", type=int, default=268_435_456)
    parser.add_argument("--chunks", type=int, default=64)
    parser.add_argument("--scatter", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--direction", choices=("h2d", "d2h"), default="h2d")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def makePool(nbytes: int) -> int:
    """Allocate a contiguous pinned host pool."""
    return cp.cuda.runtime.hostAlloc(nbytes, 0)


def makeDevice(nbytes: int) -> int:
    """Allocate a contiguous device pool."""
    return cp.cuda.runtime.malloc(nbytes)


def makePoolPlan(pool_size: int, count: int, scatter: bool, seed: int):
    """Build either sequential or shuffled pool slot offsets."""
    chunk_size = pool_size // count
    if chunk_size <= 0:
        raise ValueError("pool-size must allow at least one chunk")
    slots = [offset * chunk_size for offset in range(count)]
    if not scatter:
        return chunk_size, slots
    rng = random.Random(seed)
    rng.shuffle(slots)
    return chunk_size, slots


def transferOnce(direction: str, hostPool: int, devicePool: int,
                 poolSize: int, chunks: int, scatter: bool, seed: int) -> None:
    """Do one full round-trip over the selected pool access pattern."""
    chunk_size, slots = makePoolPlan(poolSize, chunks, scatter, seed)
    for offset in slots:
        if direction == "h2d":
            cp.cuda.runtime.memcpyAsync(
                devicePool + offset,
                hostPool + offset,
                chunk_size,
                cp.cuda.runtime.memcpyHostToDevice,
                0,
            )
        else:
            cp.cuda.runtime.memcpyAsync(
                hostPool + offset,
                devicePool + offset,
                chunk_size,
                cp.cuda.runtime.memcpyDeviceToHost,
                0,
            )


def main() -> None:
    args = parse_args()
    if args.pool_size < args.chunks or args.pool_size // args.chunks == 0:
        raise SystemExit("pool-size must be larger than chunk count")

    cp.cuda.runtime.setDevice(args.gpu)
    host_pool = makePool(args.pool_size)
    device_pool = makeDevice(args.pool_size)
    chunk_size = args.pool_size // args.chunks
    pattern = "scatter" if args.scatter else "sequential"
    tag = f"{args.direction}_{pattern}"

    start = time.perf_counter()
    for _ in range(args.warmup):
        transferOnce(args.direction,
                     host_pool,
                     device_pool,
                     args.pool_size,
                     args.chunks,
                     args.scatter,
                     args.seed)
    cp.cuda.runtime.deviceSynchronize()
    start = time.perf_counter()
    for _ in range(args.iters):
        transferOnce(args.direction,
                     host_pool,
                     device_pool,
                     args.pool_size,
                     args.chunks,
                     args.scatter,
                     args.seed)
    cp.cuda.runtime.deviceSynchronize()
    elapsed = time.perf_counter() - start
    bandwidth = args.pool_size * args.iters / elapsed

    print(f"device=cuda:{args.gpu}")
    print(f"pool_size={args.pool_size}")
    print(f"chunks={args.chunks}")
    print(f"chunk_size={chunk_size}")
    print(f"direction={args.direction} pattern={pattern}")
    print(f"elapsed={elapsed:.6f}s")
    print(f"bandwidth_GiB_s={bandwidth / (1 << 30):.3f}")
    print(f"bandwidth_GBps={bandwidth / 1e9:.3f}")

    cp.cuda.runtime.free(device_pool)
    cp.cuda.runtime.freeHost(host_pool)


if __name__ == "__main__":
    main()
