#!/usr/bin/env python3
"""Benchmark host-memory to GPU transfers with raw CUDA memcpy calls."""

import argparse
import statistics
import time
import numpy as np
import cupy as cp

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--size", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    return parser.parse_args()


def makeHost(nbytes: int, pinned: bool):
    if pinned:
        view = cp.cuda.runtime.hostAlloc(nbytes, 0)
        return view, int(view)
    view = np.empty(nbytes, dtype=np.uint8, order="C")
    return view, view.ctypes.data


def bench(direction: str, pinned: bool, size: int, gpu: int, warmup: int,
          iters: int) -> float:
    cp.cuda.runtime.setDevice(gpu)
    device_ptr = cp.cuda.runtime.malloc(size)
    host_view, host_ptr = makeHost(size, pinned)
    if direction.startswith("h2d"):
        src_ptr = host_ptr
        dst_ptr = device_ptr
        kind = cp.cuda.runtime.memcpyHostToDevice
    else:
        src_ptr = device_ptr
        dst_ptr = host_ptr
        kind = cp.cuda.runtime.memcpyDeviceToHost

    for _ in range(warmup):
        cp.cuda.runtime.memcpyAsync(dst_ptr, src_ptr, size, kind, 0)
    cp.cuda.runtime.deviceSynchronize()
    start = time.perf_counter()
    for _ in range(iters):
        cp.cuda.runtime.memcpyAsync(dst_ptr, src_ptr, size, kind, 0)
    cp.cuda.runtime.deviceSynchronize()
    elapsed = time.perf_counter() - start
    bandwidth = size * iters / elapsed
    cp.cuda.runtime.free(device_ptr)
    if pinned:
        cp.cuda.runtime.freeHost(host_ptr)
    return bandwidth


def main() -> None:
    args = parse_args()
    methods = ("h2d_pinned", "h2d_pageable", "d2h_pinned", "d2h_pageable")
    samples = {}
    print(f"device=cuda:{args.gpu}")
    print("method            mean_GiB_s  std_GiB_s  per_copy_us")
    for method in methods:
        direction, pinned = method.split("_", 1)
        sample = bench(direction, pinned, args.size, args.gpu, args.warmup,
                       args.iters)
        samples[method] = sample
        print(f"{method:<18} {sample / (1 << 30):>10.3f}    0.000     "
              f"{args.size / sample * 1e6:>6.2f}")

    best = max(samples.items(), key=lambda item: item[1])
    print(f"best={best[0]} bandwidth_GiB_s={best[1] / (1 << 30):.3f}")

    for method in methods:
        if method in samples and statistics.pstdev([samples[method]]):
            print()


if __name__ == "__main__":
    main()
