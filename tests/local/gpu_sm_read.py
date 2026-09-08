#!/usr/bin/env python3

import argparse
import time

import torch
import triton
import triton.language as tl

BLOCK = 2048


@triton.jit
def read_gpu_memory(src_ptr, out_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    stride = num_programs * BLOCK
    accumulator = tl.zeros([BLOCK], dtype=tl.float32)

    for start in range(0, numel, stride):
        offsets = start + pid * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(src_ptr + offsets, mask=offsets < numel, other=0.0)
        accumulator += value

    tl.store(out_ptr + pid, tl.sum(accumulator, dtype=tl.float32))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--size", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--sm", type=int, default=1, help="Triton program concurrency, not a physical SM pin")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    if args.size < 4 or args.size % 4 != 0:
        raise SystemExit("size must be a positive multiple of 4 bytes")
    if args.sm < 1 or args.warmup < 0 or args.iters < 1:
        raise SystemExit("invalid arguments")

    numel = args.size // 4

    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)
    source = torch.empty(numel, dtype=torch.float32, device=device)
    source.fill_(1.0)
    out = torch.empty(args.sm, dtype=torch.float32, device=device)
    grid = (args.sm,)

    for _ in range(args.warmup):
        read_gpu_memory[grid](source, out, numel, BLOCK=BLOCK)
    torch.cuda.synchronize()

    started = time.perf_counter()
    for _ in range(args.iters):
        read_gpu_memory[grid](source, out, numel, BLOCK=BLOCK)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    total_bytes = numel * 4 * args.iters
    aggregate = total_bytes / elapsed
    print(
        f"gpu={args.gpu} sm={args.sm} size_bytes={args.size} "
        f"warmup={args.warmup} iters={args.iters}"
    )
    print(f"aggregate={aggregate / 1024**3:.3f} GiB/s")


if __name__ == "__main__":
    main()
