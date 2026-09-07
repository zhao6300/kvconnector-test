#!/usr/bin/env python3
from __future__ import annotations
import argparse
import multiprocessing as mp
import os
import queue
import socket
import sys
import time

os.environ.setdefault("MC_INTRANODE_NVLINK", "1")
os.environ.setdefault("MC_USE_NVLINK_IPC", "1")

def local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("127.0.0.1", 12345))
            return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"

def target_process(host, gpu, pool_size, result_queue, control_queue):
    import torch
    from mooncake.engine import TransferEngine
    torch.cuda.set_device(gpu)
    engine = TransferEngine()
    if engine.initialize(host, "P2PHANDSHAKE", "nvlink_intra", "") != 0:
        result_queue.put({"error": "target initialize failed"})
        return
    target = torch.empty(pool_size, dtype=torch.uint8, device=f"cuda:{gpu}")
    if engine.register_memory(target.data_ptr(), pool_size) != 0:
        result_queue.put({"error": "target memory registration failed"})
        return
    result_queue.put({"port": engine.get_rpc_port(), "buffer": target.data_ptr(), "size": pool_size, "gpu": gpu})
    while True:
        command = control_queue.get()
        if command == "fill-71":
            target.fill_(71)
            control_queue.put(0)
            control_queue.get()
        elif command == "verify-write":
            control_queue.put({"min": int(target.min().item()), "max": int(target.max().item())})
            control_queue.get()
        elif command == "stop":
            break
    control_queue.get()
    engine.unregister_memory(target.data_ptr())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-gpu", type=int, default=0)
    parser.add_argument("--target-gpu", type=int, default=1)
    parser.add_argument("--operations", default="write,read")
    parser.add_argument("--pool-size", type=int, default=1 << 30)
    parser.add_argument("--pieces", type=int, default=12)
    parser.add_argument("--runtime", type=float, default=2.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    mp.set_start_method("spawn", force=True)
    if args.source_gpu == args.target_gpu:
        raise SystemExit("source and target GPUs must differ")
    host = local_ip()
    control_queue = mp.Queue()
    result_queue = mp.Queue()
    target = mp.Process(
        target=target_process,
        args=(host, args.target_gpu, args.pool_size, result_queue, control_queue),
        daemon=True,
    )
    target.start()
    info = result_queue.get(timeout=10)
    if "error" in info:
        raise SystemExit(info["error"])
    import torch
    from mooncake.engine import TransferEngine
    torch.cuda.set_device(args.source_gpu)
    source_engine = TransferEngine()
    if source_engine.initialize(host, "P2PHANDSHAKE", "nvlink_intra", "") != 0:
        raise SystemExit("source initialize failed")
    source = torch.empty(args.pool_size, dtype=torch.uint8, device=f"cuda:{args.source_gpu}")
    if source_engine.register_memory(source.data_ptr(), args.pool_size) != 0:
        raise SystemExit("source memory registration failed")
    target_name = f"{host}:{info['port']}"
    target_pointer = info["buffer"]
    results: dict[str, dict[str, float | int | str]] = {}
    for operation in args.operations.split(","):
        source.fill_(91)
        torch.cuda.synchronize()
        if args.pieces == 1:
            chunks = [source.data_ptr()]
            remote_chunks = [target_pointer]
            lengths = [args.pool_size]
        else:
            section = args.pool_size // args.pieces
            chunks = [source.data_ptr() + section * usage for usage in range(args.pieces)]
            remote_chunks = [target_pointer + section * usage for usage in range(args.pieces)]
            lengths = [section] * args.pieces
        submit = source_engine.batch_transfer_sync_write if operation == "write" else source_engine.batch_transfer_sync_read
        rounds = []
        started = time.perf_counter()
        batch_throughputs = []
        while time.perf_counter() - started < args.runtime:
            round_started = time.perf_counter()
            if submit(target_name, chunks, remote_chunks, lengths) != 0:
                raise SystemExit(f"{operation} transfer failed")
            round_elapsed = time.perf_counter() - round_started
            rounds.append(round_elapsed)
            batch_throughputs.append(sum(lengths) / round_elapsed / 1e9)
        measured = time.perf_counter() - started
        throughput = sum(lengths) * len(rounds) / measured / 1e9
        results[operation] = {
            "bytes": sum(lengths) * len(rounds),
            "rounds": len(rounds),
            "wall": measured,
            "throughput_GBps": throughput,
            "throughput_GiBps": throughput * (1 << 30) / 1e9,
            "min_round_GBps": min(batch_throughputs),
            "max_round_GBps": max(batch_throughputs),
            "path": "Mooncake Python TransferEngine nvlink CUDA IPC/P2P",
        }
    if args.pieces == 1 and "write" in results:
        control_queue.put("fill-71")
        if control_queue.get(timeout=10) != 0:
            raise SystemExit("target fill command failed")
        source.zero_()
        torch.cuda.set_device(args.source_gpu)
        if source_engine.transfer_sync_read(target_name, source.data_ptr(), target_pointer, args.pool_size) != 0:
            raise SystemExit("read verification failed")
        if int(source[0].item()) != 71:
            raise SystemExit("read verification failed")
    control_queue.put("stop")
    target.join(timeout=5)
    if target.is_alive():
        target.terminate()
        target.join(timeout=2)
    summary = {
        "connector": "Mooncake Python TransferEngine",
        "source_gpu": args.source_gpu,
        "target_gpu": args.target_gpu,
        "pool_size": args.pool_size,
        "pieces": args.pieces,
        "operations": results,
    }
    if args.json:
        print(summary)
    else:
        print(f"source_gpu={args.source_gpu} target_gpu={args.target_gpu} pool={args.pool_size} pieces={args.pieces}")
        for operation, data in results.items():
            print(f"{operation}_bw={data['throughput_GBps']:.2f} GB/s ({data['throughput_GiBps']:.2f} GiB/s)")
            print(f"{operation}_rounds={data['rounds']} wall={data['wall']:.3f}s")

if __name__ == "__main__":
    main()
