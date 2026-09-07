#!/usr/bin/env python3
"""Benchmark Triton kernels with explicit device/host memory load and store."""

import argparse
import time

import torch
import triton
import triton.language as tl


@triton.jit
def read_f4_kernel(src_ptr, out_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    value = tl.load(src_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + pid, tl.sum(value, dtype=tl.float32))


@triton.jit
def write_f4_kernel(dst_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    value = tl.full([BLOCK], 1.0, dtype=tl.float32)
    tl.store(dst_ptr + offsets, value, mask=mask)


@triton.jit
def copy_f4_kernel(src_ptr, dst_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    value = tl.load(src_ptr + offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + offsets, value, mask=mask)


def parse_modes(raw):
    modes = []
    for mode in raw.split(","):
        if mode:
            modes.append(mode)
    return modes


def count_bytes(mode, size_bytes):
    if mode.endswith("_copy"):
        return size_bytes * 2
    return size_bytes


def allocate(mode, size_bytes, device):
    numel = size_bytes // 4
    if "device_" in mode:
        dtype = torch.float32
        source = torch.empty(numel, dtype=dtype, device=device)
        target = torch.empty(numel, dtype=dtype, device=device)
        source.fill_(1.0)
        target.zero_()
    else:
        dtype = torch.float32
        source = torch.empty(numel, dtype=dtype, device="cpu").pin_memory()
        target = torch.empty(numel, dtype=dtype, device="cpu").pin_memory()
        source.fill_(1.0)
        target.zero_()
    return source, target


def run_case(mode, source, target, size_bytes, warmup, iters):
    numel = size_bytes // 4
    BLOCK = 2048
    grid = (numel + BLOCK - 1) // BLOCK
    out = torch.empty(grid, dtype=torch.float32, device="cuda:0")
    for _ in range(warmup):
        if mode.endswith("_read"):
            read_f4_kernel[(grid,)](source, out, numel, BLOCK=BLOCK)
        elif mode.endswith("_write"):
            write_f4_kernel[(grid,)](target, numel, BLOCK=BLOCK)
        else:
            copy_f4_kernel[(grid,)](source, target, numel, BLOCK=BLOCK)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        if mode.endswith("_read"):
            read_f4_kernel[(grid,)](source, out, numel, BLOCK=BLOCK)
        elif mode.endswith("_write"):
            write_f4_kernel[(grid,)](target, numel, BLOCK=BLOCK)
        else:
            copy_f4_kernel[(grid,)](source, target, numel, BLOCK=BLOCK)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return count_bytes(mode, size_bytes) * iters / elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--size", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--mode",
        default="device_read,device_write,device_copy,host_pinned_read,host_pinned_write,host_pinned_copy",
    )
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)
    modes = parse_modes(args.mode)

    torch.tensor([123], device=device)
    print(
        f"device={device} name={torch.cuda.get_device_name()} "
        f"kernel=triton size_bytes={args.size} warmup={args.warmup} "
        f"iterations={args.iters}"
    )
    print("mode                  GiB_s")
    for mode in modes:
        source, target = allocate(mode, args.size, device)
        bandwidth = run_case(mode, source, target, args.size, args.warmup, args.iters)
        print(f"{mode:<21} {bandwidth / (1024**3):>9.3f}")
        del source, target
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
