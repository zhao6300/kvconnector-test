# comm

仓库内是小型、定向的 GPU 通信与 KV-transfer connector 测试脚本，集中在 `bench/` 目录下。主要说明见 [`bench/results.md`](bench/results.md)。

## 目录结构

- `bench/` : CUDA IPC、Mooncake、NIXL、CUDA P2P、NCCL、FlashInfer 相关基准脚本。
- `bench/results.md` : 汇总的基准结果和连接器语义说明。
- `run_logs/` : 少量代表性运行的 stdout/stderr 快照。

## 运行基准

脚本假设使用本机已有的 `/opt/venv` 环境。

```bash
/opt/venv/bin/python bench/gpu_ipc_bw.py
/opt/venv/bin/python -m pytest -q bench/mooncake_transfer_test.py
torchrun --nproc_per_node=2 bench/nccl_bw.py
```

NIXL 需要分别启动 target 和 initiator：

`cuda_ipc`/`cuda_copy` 是 UCX 的 VRAM 数据路径；`tcp` 只是控制路径。这台机器上 `UCX_CUDA_IPC_ENABLE_GET_ZCOPY=on` 是必要的，用来强制 `cuda_ipc` 的 `get_zcopy`，否则 UCX 会把无 NVLink 的机器视为 IPC 数据面不可用。
NIXL 示例使用 `sm,cuda_copy,cuda_ipc,tcp`：

```bash
UCX_TLS=sm,cuda_copy,cuda_ipc,tcp UCX_CUDA_IPC_ENABLE_GET_ZCOPY=on /opt/venv/bin/python bench/nixl_bw.py --mode target --gpu 1 --size 67108864 --warmup 2 --iters 10
UCX_TLS=sm,cuda_copy,cuda_ipc,tcp UCX_CUDA_IPC_ENABLE_GET_ZCOPY=on /opt/venv/bin/python bench/nixl_bw.py --mode initiator --gpu 0 --ip 127.0.0.1 --size 67108864 --warmup 2 --iters 10
```

Mooncake 传输测试示例如下：

```
/opt/venv/bin/python bench/mooncake_transfer_cross_gpu.py --source-gpu 0 --target-gpu 1 --size 67108864 --warmup 10 --iters 50
```

Mooncake 也可以在高带宽路径下直接测 P2P。推荐官方 `transfer_engine_bench`，并给 target/initiator 分别设置 GPU：

```bash
/opt/venv/bin/python bench/mooncake_p2p_bench.py \
  --binary /tmp/mooncake-build-minimal/mooncake-transfer-engine/example/transfer_engine_bench \
  --source-gpu 0 --target-gpu 1 --operations write,read --duration 5
```

这里的 `nvlink_intra` 名字里带 NVLink，但实现并不强制要求真的有 NVLink。本机没有 NVLink 时仍走 CUDA IPC + `cudaMemcpyBatchAsync` 的 PCIe P2P。需要 `MC_INTRANODE_NVLINK=1` 和 `MC_USE_NVLINK_IPC=1`，否则没有 RDMA 的机器会回落到低速 TCP。官方 benchmark 二进制需要源码构建并打开下面的选项：

```bash
cmake -S /tmp/mooncake-src -B /tmp/mooncake-build-minimal \
  -DUSE_CUDA=ON -DUSE_INTRA_NVLINK=ON \
  -DWITH_STORE=OFF -DWITH_EP=OFF -DBUILD_UNIT_TESTS=OFF
cmake --build /tmp/mooncake-build-minimal -j 64 \
  --target mooncake-transfer-engine/example/transfer_engine_bench
```

如果本机已经有构建产物，也可以用 `MOONCAKE_BENCH_BIN` 环境变量给脚本指定二进制路径。Python 快速测试里的 `protocol=tcp` 只是验证回退路径，不是这套高带宽测项。

Mooncake P2P 也可以完全用 Python 绑定来跑，不用依赖上面的 C++ benchmark 二进制：

```bash
MC_INTRANODE_NVLINK=1 MC_USE_NVLINK_IPC=1 \
/opt/venv/bin/python bench/mooncake_p2p_python_bench.py \
  --source-gpu 0 --target-gpu 1 \
  --pool-size 268435456 --pieces 1 --runtime 1.0
```

这版走的是 Mooncake Python `TransferEngine` 的 `nvlink_intra` 路径，再落到 CUDA IPC + `cudaMemcpyBatchAsync`/GPU P2P。它适合快速验证和重复采样；如果需要和官方 C++ benchmark 的提交线程模型逐项对齐，还是用上面的官方二进制。5 次统计结果显示 write 约为 `33.06±0.37 GB/s`，read 约为 `37.11±1.51 GB/s`。

## 阅读结果

`bench/results.md` 包含实测带宽和对应通信路径的简单解读，另外还给了 Nixl、Mooncake、LMCache、FlexKV 的连接器对比。

## 注意事项

- 仓库的重点是解释和测量通信路径，而不是打包成一个库。
- 运行基准会产生一些轻量的本地缓存和少量日志；这些默认走 `.gitignore`。
