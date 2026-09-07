#!/usr/bin/env python3
"\"Run Mooncake's official console benchmark on the CUDA IPC/P2P path.\""

import argparse
import os
import re
import signal
import subprocess
import sys
import tempfile
from pathlib import Path


TARGET_PORT_RE = re.compile(r"RPC using P2P handshake, listening on .*:(\d+)")
RESULT_RE = re.compile(
    r"Test completed: duration ([\d.]+), batch count (\d+), "
    r"throughput ([\d.]+) ([A-Za-z]+/s)"
)


def local_host() -> str:
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def start_target(binary: str, host: str, gpu: int, thread_count: int,
                 buffer_size: int, block_size: int, log_path: Path,
                 env: dict[str, str]) -> subprocess.Popen:
    command = [
        binary,
        "-mode", "target",
        "-protocol", "nvlink_intra",
        "-metadata_server", "P2PHANDSHAKE",
        "-gpu_id", str(gpu),
        "-local_server_name", host,
        "-threads", str(thread_count),
        "-buffer_size", str(buffer_size),
        "-block_size", str(block_size),
    ]
    with open(log_path, "w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log, env=env)
    for _ in range(400):
        if process.poll() is not None:
            log_path.write_text(log_path.read_text(encoding="utf-8") + "\n")
            raise RuntimeError("Mooncake target exited before registration")
        text = log_path.read_text(encoding="utf-8")
        if TARGET_PORT_RE.search(text):
            return process
        import time

        time.sleep(0.05)
    raise RuntimeError("Timed out waiting for Mooncake target RPC port")


def run_initiator(binary: str, host: str, port: int, operation: str,
                  gpu: int, thread_count: int, duration: int, buffer_size: int,
                  block_size: int, env: dict[str, str]) -> str:
    command = [
        binary,
        "-mode", "initiator",
        "-protocol", "nvlink_intra",
        "-metadata_server", "P2PHANDSHAKE",
        "-gpu_id", str(gpu),
        "-local_server_name", host,
        "-segment_id", f"{host}:{port}",
        "-operation", operation,
        "-threads", str(thread_count),
        "-duration", str(duration),
        "-buffer_size", str(buffer_size),
        "-block_size", str(block_size),
    ]
    result = subprocess.run(command, text=True, capture_output=True,
                            env=env, check=True)
    output = result.stdout + result.stderr
    match = RESULT_RE.search(output)
    if not match:
        raise RuntimeError(f"Mooncake {operation} result line missing:\n{output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default=os.environ.get(
        "MOONCAKE_BENCH_BIN",
        "/tmp/mooncake-build-minimal/mooncake-transfer-engine/example/"
        "transfer_engine_bench"))
    parser.add_argument("--source-gpu", type=int, default=0)
    parser.add_argument("--target-gpu", type=int, default=1)
    parser.add_argument("--host", default=None)
    parser.add_argument("--operations", default="write,read")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--duration", type=int, default=5)
    parser.add_argument("--buffer-size", type=int, default=1 << 30)
    parser.add_argument("--block-size", type=int, default=65536)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.source_gpu == args.target_gpu:
        raise SystemExit("source and target GPUs must differ for the P2P test")
    binary = Path(args.binary)
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise SystemExit(f"Mooncake benchmark binary not executable: {binary}")

    host = args.host or local_host()
    operations = [item.strip() for item in args.operations.split(",")
                  if item.strip()]
    if not all(operation in ("read", "write") for operation in operations):
        raise SystemExit("operations must contain only read and/or write")

    env = os.environ.copy()
    env["MC_INTRANODE_NVLINK"] = "1"
    env["MC_USE_NVLINK_IPC"] = "1"
    env["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + env.get(
        "LD_LIBRARY_PATH", "")
    env["GLOG_logtostderr"] = "1"

    with tempfile.TemporaryDirectory(prefix="mooncake-p2p-bench-") as tmpdir:
        target_log = Path(tmpdir) / "target.log"
        target = start_target(str(binary), host, args.target_gpu,
                              args.threads, args.buffer_size,
                              args.block_size, target_log, env)
        port = int(TARGET_PORT_RE.search(target_log.read_text()).group(1))
        print(f"target host={host} port={port} gpu={args.target_gpu}")
        try:
            for operation in operations:
                output = run_initiator(str(binary), host, port, operation,
                                       args.source_gpu, args.threads,
                                       args.duration, args.buffer_size,
                                       args.block_size, env)
                if args.verbose:
                    print(output, end="", flush=True)
                result = RESULT_RE.search(output)
                assert result
                duration, batch_count, throughput, unit = result.groups()
                print(f"{operation} duration={duration}s "
                      f"batch_count={batch_count} "
                      f"throughput={throughput} {unit}")
        finally:
            target.send_signal(signal.SIGTERM)
            try:
                target.wait(timeout=5)
            except subprocess.TimeoutExpired:
                target.kill()


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        print(error, file=sys.stderr)
        raise SystemExit(1)
