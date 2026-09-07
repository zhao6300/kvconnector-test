# GPU Communication Benchmark Summary

Environment used for these runs: 4 GPUs, no RDMA/RoCE HCA, and `GPU0-GPU1` had a `NODE` connection rather than an NVLink tunnel. Bandwidth values are one-way source-to-target bandwidth unless stated otherwise.

| Test | Size | Result | Path | Test setup | Version |
|---|---:|---:|---|---|---|
| CUDA IPC, same GPU | 64 MiB | 1524.5 GiB/s | GPU0 device memory copy after cross-process CUDA IPC | Two processes both allocate on `cuda:0`; one exports memory to the other; timed destination `copy_(src)` on CUDA events | `bench/gpu_ipc_bw.py` |
| Mooncake, same GPU | 64 MiB | write 1.496 GiB/s; read 1.469 GiB/s | Same-GPU registered VRAM through Mooncake TransferEngine; no RDMA, so engine RPC/TCP fallback controls transfer | Source and destination buffers both on `cuda:0`; `transfer_sync_write/read`, 50 measured iterations | `bench/mooncake_transfer_cross_gpu.py --source-gpu 0 --target-gpu 0` |
| Mooncake, GPU0 to GPU1 | 64 MiB | write 1.385 GiB/s; read 1.415 GiB/s | Cross-GPU registered VRAM via Mooncake TransferEngine; no RDMA, so local TCP fallback controls transfer | Source on `cuda:0`, target on `cuda:1`; two TransferEngines; `transfer_sync_write/read`, 50 measured iterations | `bench/mooncake_transfer_cross_gpu.py --source-gpu 0 --target-gpu 1` |
| NIXL UCX, GPU0 to GPU1 | 64 MiB | 0.04 | UCX CUDA transport bound to `TLS=cuda_copy,tcp` | Target allocates on GPU1 and initiator on GPU0; synchronous READ, 10 measured iterations | `bench/nixl_bw.py --mode target/initiator --gpu 1/0 --size 67108864` |
| Raw CUDA P2P, GPU0 to GPU1 | 64 MiB | 33.09 | Direct CUDA `cudaMemcpyPeer` / CUDA P2P over PCIe | Native CUDA p2p benchmark; GPU0 copies directly to GPU1; 64 MiB row from native benchmark | `bench/p2p_bw.cu` |
| PyTorch NCCL P2P | 64 MiB | 34.56 | PyTorch NCCL `dist.send/recv`, NCCL chooses CUDA P2P/PCIe | Two `torchrun` ranks map rank0 to GPU0 and rank1 to GPU1; synchronous send/recv, 50 measured iterations | `run_logs/nccl_bw.log` |
| PyTorch NCCL ring AllReduce | 64 MiB | 29.16 effective | NCCL ring protocol over the same GPU0/GPU1 communication path | Two ranks on GPU0/GPU1; 20 measured iterations; effective BW excludes the two ring stream contributions | `run_logs/nccl_bw.log` |
| PyTorch NCCL ring AllReduce | 256 MiB | 29.80 effective | NCCL ring protocol over the same GPU0/GPU1 communication path | Same as above, largest logged message | `run_logs/nccl_bw.log` |
| FlashInfer PCIe IPC AllReduce | 16 MiB bf16 | 20.29 | SM-resident buffers shared through CUDA IPC/GPU shared memory | Two ranks on GPU0/GPU1 use FlashInfer `PcieIpcAllReduceWorkspace`; 50 measured iterations; log column is labeled `Gb/s` but the formula returns byte bandwidth | `bench/pcie_ipc_ar.py` and `run_logs/pcie_ipc_ar.log` |

## Interpretation

- `CUDA IPC same GPU` is the on-device copy/path bandwidth across two processes on GPU0 and is not comparable with GPU0-to-GPU1 rows.
- `Raw CUDA P2P`, NCCL P2P, and NCCL ring AllReduce are all limited by GPU0-to-GPU1 PCIe/system interconnect in this host.
- FlashInfer PCIe IPC AllReduce adds a fused GPU-side allreduce/proposal mechanism, so its 16 MiB result is below this host's basic P2P copy, though it avoids CPU-managed per-copy GPU-to-GPU synchronization.
- Mooncake same/cross GPU gives ~1.4-1.5 GiB/s because this environment has no RDMA HCA here, and Mooncake's local RPC control plane plus TCP fallback dominate the transfer.
- NIXL UCX, as tested here with `TLS=cuda_copy,tcp`, is far slower because the configured UCX engine/CUDA sequence offloads GPU transfers inefficiently on this host. This conclusion only applies to the tested NIXL/UCX path.

## Connector / P-D 形态汇总

| Connector | P/D 形态 | layerlevel save | metrics | 通信/存储路径 | 注意点 |
|---|---|---|---|---|---|
| **Nixl** | async / block-level | no-op | Prometheus | 分块异步 KV 传输，使用 RDMA/HMA 相关缓存元数据 | 把传输与存储的实现边界看清楚，不要当作同步一次性传输 |
| **Mooncake** | async / block-level P→D push | no-op | logs / accumulate | Prefill 通过 Mooncake 把 KV 推给 Decode，D 收到后继续解码 | P-push 记录成功，D 记录 failure；看指标时要明确从哪一侧观测 |
| **LMCache** | can be chunk / layerwise | true | logs | 可按 chunk 或 layerwise 两种模式工作，`use_layerwise` 决定是否层级别保存 | 兼具传输与存储能力；`use_layerwise` 会改变表中描述的实际含义 |
| **FlexKV** | async offload after finish | no-op | logs | 主要做请求结束后的异步 offload，不是逐层同步保存 | 更像 KV 存储/offload 连接器；未来接口和通用化能力仍在扩展 |

### 术语解释

- **P / D**：P 表示 Prefill，负责生成 KV；D 表示 Decode，负责继续生成 token。
- **async**：KV 传输不阻塞前向/调度主流程。
- **block-level**：KV 传输按 KV block 为单位进行，而不是一次性同步拷贝。
- **layerlevel save**：模型前向每层执行后，是否立刻保存对应层的 KV；`no-op` 表示不启用层级别同步保存。
- **P→D push**：由 Prefill 主动把 KV 推给 Decode。

### 关键判断

- **Nixl 更适合看传输路径本身**，它拥有较细的 Prometheus 指标，便于观察单次传输耗时、成功/失败和踩到的传输元数据。
- **Mooncake 是明确的 P → D push**；它的传输和观测路径都是 P/D 两端分离，因此不要把 P 侧成功和 D 侧失败混在一起。
- **LMCache 的行为最依赖配置**，`use_layerwise` 开或关会让它看起来像两种不同的路径。
- **FlexKV 当前更偏异步 offload**，不是四个中最典型的逐层同步保存连接器。
