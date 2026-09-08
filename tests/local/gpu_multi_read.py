#!/usr/bin/env python3

import argparse
import threading
import time

import torch
import triton
import triton.language as tl

BLOCK = 2048


@triton.jit
def read_kernel(src_ptr, out_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    value = tl.load(src_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + pid, tl.sum(value, dtype=tl.float32))


def worker(gpu, source, numel, warmup, iters, barrier, results):
    device = f"cuda:{gpu}"
    torch.cuda.set_device(device)
    grid = (numel + BLOCK - 1) // BLOCK
    out = torch.empty(grid, dtype=torch.float32, device=device)

    for _ in range(warmup):
        read_kernel[(grid,)](source, out, numel, BLOCK=BLOCK)
    torch.cuda.synchronize(device)
    barrier.wait()

    started = time.perf_counter()
    for _ in range(iters):
        read_kernel[(grid,)](source, out, numel, BLOCK=BLOCK)
    torch.cuda.synchronize(device)

    ended = time.perf_counter()
    barrier.wait()
    results[gpu] = ended - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--numel", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    if args.numel <= 0 or args.gpus <= 0 or args.warmup < 0 or args.iters <= 0:
        raise SystemExit("invalid arguments")

    source = torch.empty(args.numel, dtype=torch.float32, device="cpu").pin_memory()
    source.fill_(1.0)
    barrier = threading.Barrier(args.gpus + 1)
    threads = []
    results = {}
    for gpu in range(args.gpus):
        thread = threading.Thread(
            target=worker,
            args=(gpu, source, args.numel, args.warmup, args.iters, barrier, results),
        )
        thread.start()
        threads.append(thread)

    barrier.wait()
    started = time.perf_counter()
    barrier.wait()
    ended = time.perf_counter()
    for thread in threads:
        thread.join()

    elapsed = sum(results.values()) / args.gpus
    per_gpu = [
        (gpu, args.numel * args.iters * 4 / results[gpu])
        for gpu in range(args.gpus)
    ]
    aggregate = sum(bandwidth for _, bandwidth in per_gpu)

    print(
        f"numel={args.numel} gpus={args.gpus} warmup={args.warmup} "
        f"iters={args.iters} elapsed={elapsed:.6f}s"
    )
    for gpu, bandwidth in per_gpu:
        print(f"  cuda:{gpu} bandwidth={bandwidth / 1024**3:.3f} GiB/s")
    print(f"aggregate={aggregate / 1024**3:.3f} GiB/s")


if __name__ == "__main__":
    main()
