# GPU 通信基准测试

测试环境：4 块 GPU，无 RDMA/RoCE HCA；`GPU0-GPU1` 是 `NODE` 连接，而不是 NVLink direct tunnel。除非另有说明，这里的带宽都是单路 source→target 带宽。

| 测试 | 数据量 | 结果 | 通信/存储路径 | 测试 setup | 引用 |
|---|---:|---:|---|---|---|
| CUDA IPC，同 GPU | 64 MiB | 1524.5 GiB/s | GPU0 跨进程 CUDA IPC 显存拷贝 | 两个进程都在 `cuda:0` 上分配内存；一个进程导出，另一个进程导入，再用 CUDA events 计时目标侧的 `copy_(src)` | `bench/gpu_ipc_bw.py` |
| Mooncake，同 GPU | 64 MiB | write 1.496 GiB/s；read 1.469 GiB/s | Mooncake TransferEngine 使用同 GPU 已注册 VRAM；无 RDMA，引擎 RPC/TCP fallback 控制实际速度 | source 和 destination 都在 `cuda:0`；`transfer_sync_write/read`，实测 50 次 | `bench/mooncake_transfer_cross_gpu.py --source-gpu 0 --target-gpu 0` |
| Mooncake，GPU0 → GPU1 | 64 MiB | write 1.385 GiB/s；read 1.415 GiB/s | Mooncake TransferEngine 使用跨 GPU 已注册 VRAM；无 RDMA，本机 TCP fallback 控制实际速度 | source 在 `cuda:0`，target 在 `cuda:1`；两个 TransferEngine 通信 | `bench/mooncake_transfer_cross_gpu.py --source-gpu 0 --target-gpu 1` |
| NIXL UCX，GPU0 → GPU1 | 64 MiB | 0.04 | UCX CUDA transport，绑定 `TLS=cuda_copy,tcp` | target 在 GPU1，initiator 在 GPU0；同步 READ，实测 10 次 | `bench/nixl_bw.py --mode target/initiator --gpu 1/0 --size 67108864` |
| 原生 CUDA P2P，GPU0 → GPU1 | 64 MiB | 33.09 | 直接 CUDA `cudaMemcpyPeer` / CUDA P2P over PCIe | 原生 CUDA p2p benchmark；GPU0 直拷到 GPU1 | `bench/p2p_bw.cu` |
| PyTorch NCCL P2P | 64 MiB | 34.56 | PyTorch NCCL `dist.send/recv`，NCCL 选择 CUDA P2P/PCIe | 两个 `torchrun` rank，rank0 → GPU0，rank1 → GPU1；同步 send/recv，50 次实测 | `run_logs/nccl_bw.log` |
| PyTorch NCCL ring AllReduce | 64 MiB | 29.16 effective | NCCL ring，GPU0/GPU1 之间 | 两个 rank；20 次实测；effective BW 扣除 ring 两次流贡献 | `run_logs/nccl_bw.log` |
| PyTorch NCCL ring AllReduce | 256 MiB | 29.80 effective | NCCL ring，GPU0/GPU1 之间 | 同上，这是本次测到的最大消息规模 | `run_logs/nccl_bw.log` |
| FlashInfer PCIe IPC AllReduce | 16 MiB bf16 | 20.29 | SM-resident buffer，通过 CUDA IPC/GPU 共享内存通信 | 两个 rank 在 GPU0/GPU1 上使用 FlashInfer `PcieIpcAllReduceWorkspace`；50 次实测；日志列名 `Gb/s` 但计算公式返回的是 byte bandwidth | `bench/pcie_ipc_ar.py` + `run_logs/pcie_ipc_ar.log` |

## 解释

- `CUDA IPC same GPU` 是跨两个进程但同一 GPU0 内部的显存拷贝路径，不应直接和 GPU0→GPU1 的跨 GPU 行对比。
- `Raw CUDA P2P`、PyTorch NCCL P2P、PyTorch NCCL ring AllReduce 都受这台机器的 GPU0↔GPU1 PCIe/system interconnect 限制。
- FlashInfer PCIe IPC AllReduce 使用融合的 GPU 侧 allreduce/proposal 机制，因此 16 MiB 的结果低于该机器的基本 P2P copy；它减少的是 CPU 侧每次 GPU→GPU 同步的干扰。
- Mooncake 同 GPU / 跨 GPU 都在 1.4-1.5 GiB/s 左右，因为这台机器没有 RDMA HCA，Mooncake 的本地 RPC 控制面加 TCP fallback 决定了主要瓶颈。
- NIXL UCX 在这里配置 `TLS=cuda_copy,tcp` 后明显较慢，因为对应 UCX engine/CUDA 序列在本机的 GPU offload 方式不够高效。这个结论只适用于本次测试的 NIXL/UCX 路径。

### 与官方测试的关系

这些测试不是逐字复刻官方 benchmark 脚本，而是使用官方 API 加上本机可复现的小型测试组合：

- Mooncake 使用的是官方 `TransferEngine` 的同步 API。
- NIXL 使用的是官方 `nixl_agent`/UCX 接口。
- CUDA P2P/IPC/NCCL/FlashInfer 都是本机直接调用的标准 API。

因此这里的结果定位是：**在同一台机器上验证各通信路径的真实走向和性能下限**，而不是说它和上游官方 benchmark 完全等价。

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
