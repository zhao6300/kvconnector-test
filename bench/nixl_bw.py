#!/usr/bin/env python3
"""NIXL UCX P2P bandwidth benchmark between two GPUs."""
import argparse
import time

import torch

from nixl._api import nixl_agent, nixl_agent_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=55555)
    parser.add_argument("--size", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--mode", choices=["target", "initiator"], default="initiator")
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_default_device(f"cuda:{args.gpu}")
    if args.mode == "target":
        print(f"target_gpu={args.gpu}")
    tensor = torch.ones(args.size, dtype=torch.uint8) if args.mode == "target" else torch.zeros(args.size, dtype=torch.uint8)

    config = nixl_agent_config(True, True, args.port if args.mode == "target" else 0)
    agent = nixl_agent(args.mode, config)
    reg_descs = agent.register_memory(tensor)
    local_desc = agent.get_xfer_descs(tensor)

    if args.mode == "target":
        serialized = agent.get_serialized_descs(local_desc)
        while not agent.check_remote_metadata("initiator"):
            time.sleep(0.01)
        agent.send_notif("initiator", serialized)

        expected = args.warmup + args.iters
        received = 0
        while received < expected:
            notifs = agent.get_new_notifs()
            received += len(notifs.get("initiator", []))
            time.sleep(0.01)
        return

    agent.fetch_remote_metadata("target", args.ip, args.port)
    agent.send_local_metadata(args.ip, args.port)
    raw = None
    while raw is None:
        notifs = agent.get_new_notifs()
        raw = notifs.get("target", [None])[0]
        time.sleep(0.01)
    remote_desc = agent.deserialize_descs(raw)
    while not agent.check_remote_metadata("target"):
        time.sleep(0.01)

    local_prep = agent.prep_xfer_dlist("", local_desc, "VRAM", backends=["UCX"])
    remote_prep = agent.prep_xfer_dlist("target", remote_desc, "VRAM", backends=["UCX"])
    handle = agent.make_prepped_xfer("READ", local_prep, [0], remote_prep, [0], notif_msg=b"done")
    for _ in range(args.warmup):
        state = agent.transfer(handle)
        while state == "PROC":
            state = agent.check_xfer_state(handle)
        if state != "DONE":
            raise RuntimeError(f"Unexpected warmup state {state}")

    times = []
    torch.cuda.synchronize()
    for _ in range(args.iters):
        start = time.perf_counter()
        state = agent.transfer(handle)
        while state == "PROC":
            state = agent.check_xfer_state(handle)
        if state != "DONE":
            raise RuntimeError(f"Unexpected transfer state {state}")
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    torch.cuda.synchronize()
    us = sum(times) / len(times) * 1e6
    bw = args.size / sum(times) / 1e9
    print(f"# NIXL UCX P2P READ (world=2, initiator_gpu={args.gpu})")
    print("size(B)    time(us)   BW_GB/s")
    print(f"{args.size:<10d} {us:<10.2f} {bw:<10.2f}")

    agent.release_xfer_handle(handle)
    agent.invalidate_local_metadata(args.ip, args.port)
    agent.deregister_memory(reg_descs)


if __name__ == "__main__":
    main()
