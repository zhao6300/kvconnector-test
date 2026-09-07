#!/usr/bin/env python3
"""Mooncake TransferEngine benchmark for GPU0 <-> GPU1 cross-GPU transfer.

This mirrors the same-GPU Mooncake test but puts the source buffer on GPU0 and
the destination buffer on GPU1. It uses the same `TransferEngine` Mooncake
API and reports both sync-write and sync-read bandwidth.
"""

import argparse
import os
import socket
import time

import torch

from mooncake.engine import TransferEngine


def local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-gpu", type=int, default=0)
    parser.add_argument("--target-gpu", type=int, default=1)
    parser.add_argument("--protocol", type=str, default="nvlink_intra")
    parser.add_argument("--size", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    return parser.parse_args()


def bench_sync(fn, iterations: int) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        ret = fn()
        if ret != 0:
            raise RuntimeError(f"TransferEngine call failed: {ret}")
    torch.cuda.synchronize()
    return time.perf_counter() - start


def main() -> None:
    args = parse_args()
    device = torch.device(f"cuda:{args.source_gpu}")
    peer_device = torch.device(f"cuda:{args.target_gpu}")

    host = local_ip()

    src_engine = TransferEngine()
    if src_engine.initialize(host, "P2PHANDSHAKE", args.protocol, "") != 0:
        raise RuntimeError("Failed to initialize source TransferEngine")

    dst_engine = TransferEngine()
    if dst_engine.initialize(host, "P2PHANDSHAKE", args.protocol, "") != 0:
        raise RuntimeError("Failed to initialize target TransferEngine")

    torch.cuda.set_device(device)
    src = torch.empty(args.size, dtype=torch.uint8, device=device)

    torch.cuda.set_device(peer_device)
    dst = torch.empty(args.size, dtype=torch.uint8, device=peer_device)

    if src_engine.register_memory(src.data_ptr(), src.nbytes) != 0:
        raise RuntimeError("Failed to register source GPU memory")
    if dst_engine.register_memory(dst.data_ptr(), dst.nbytes) != 0:
        raise RuntimeError("Failed to register target GPU memory")

    target_name = f"{host}:{dst_engine.get_rpc_port()}"
    remote_addr = dst_engine.get_first_buffer_address(target_name)
    if remote_addr == 0:
        raise RuntimeError("Target GPU buffer address is missing")

    def write() -> int:
        return src_engine.transfer_sync_write(
            target_name, src.data_ptr(), remote_addr, args.size
        )

    def read() -> int:
        return src_engine.transfer_sync_read(
            target_name, src.data_ptr(), remote_addr, args.size
        )

    for _ in range(args.warmup):
        src.fill_(123)
        dst.zero_()
        if write() != 0:
            raise RuntimeError("write warmup failed")
        if read() != 0:
            raise RuntimeError("read warmup failed")

    src.fill_(91)
    dst.zero_()
    write_seconds = bench_sync(write, args.iters)
    torch.cuda.set_device(peer_device)
    torch.cuda.synchronize(peer_device)
    if not torch.all(dst.eq(91)):
        print(f"dst min={int(torch.min(dst))} max={int(torch.max(dst))}")
        raise AssertionError("Cross-GPU write did not verify")

    dst.fill_(71)
    src.zero_()
    torch.cuda.set_device(device)
    torch.cuda.synchronize(device)
    read_seconds = bench_sync(read, args.iters)
    torch.cuda.set_device(device)
    torch.cuda.synchronize(device)
    if not torch.all(src.eq(71)):
        print(f"src min={int(torch.min(src))} max={int(torch.max(src))}")
        raise AssertionError("Cross-GPU read did not verify")

    write_bw = args.size * args.iters / write_seconds
    read_bw = args.size * args.iters / read_seconds
    print(f"write_bw={write_bw / (1 << 30):.3f} GiB/s")
    print(f"read_bw={read_bw / (1 << 30):.3f} GiB/s")
    print(f"source_gpu={device.index} target_gpu={peer_device.index}")


if __name__ == "__main__":
    main()
