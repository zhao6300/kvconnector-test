# GPU 通信基准测试

测试环境：4 块 GPU，无 RDMA/RoCE HCA；`GPU0-GPU1` 是 `NODE` 连接，而不是 NVLink direct tunnel。除非另有说明，这里的带宽都是单路 source→target 带宽。

| 测试 | 数据量 | 结果 | 通信/存储路径 | 测试 setup | 引用 |
|---|---:|---:|---|---|---|
| CUDA IPC，同 GPU | 64 MiB | 1524.5 GiB/s | GPU0 跨进程 CUDA IPC 显存拷贝 | 两个进程都在 `cuda:0` 上分配内存；一个进程导出，另一个进程导入，再用 CUDA events 计时目标侧的 `copy_(src)` | `bench/gpu_ipc_bw.py` |
| Mooncake `nvlink_intra`，GPU0 → GPU1 | 1 GiB pool / 64 KiB block | write 36.48 GB/s；read 37.33 GB/s | Mooncake 官方 C++ 引擎 + CUDA IPC + `cudaMemcpyBatchAsync`，实际数据面是 GPU0↔GPU1 CUDA P2P over PCIe，不是 TCP | target 在 GPU1，initiator 在 GPU0；`MC_INTRANODE_NVLINK=1`、`MC_USE_NVLINK_IPC=1`；12 个提交线程，5 秒实测；目标是独立进程 | `bench/mooncake_p2p_bench.py --binary /tmp/mooncake-build-minimal/mooncake-transfer-engine/example/transfer_engine_bench --source-gpu 0 --target-gpu 1 --operations write,read --duration 5` |
| Mooncake，同 GPU | 64 MiB | write 1.496 GiB/s；read 1.469 GiB/s | Mooncake TransferEngine 使用同 GPU 已注册 VRAM；无 RDMA，引擎 RPC/TCP fallback 控制实际速度 | source 和 destination 都在 `cuda:0`；`transfer_sync_write/read`，实测 50 次 | `bench/mooncake_transfer_cross_gpu.py --source-gpu 0 --target-gpu 0` |
| Mooncake，GPU0 → GPU1 | 64 MiB | write 1.385 GiB/s；read 1.415 GiB/s | Mooncake TransferEngine 使用跨 GPU 已注册 VRAM；无 RDMA，本机 TCP fallback 控制实际速度 | source 在 `cuda:0`，target 在 `cuda:1`；两个 TransferEngine 通信 | `bench/mooncake_transfer_cross_gpu.py --source-gpu 0 --target-gpu 1` |
| NIXL UCX，GPU0 → GPU1 | 256 MiB | 2.84–3.02 | `UCX_TLS=sm,cuda_copy,cuda_ipc,tcp`；`UCX_CUDA_IPC_ENABLE_GET_ZCOPY=on` 强制启用 CUDA IPC `get_zcopy` | target 在 GPU1，initiator 在 GPU0；显式注册 VRAM；同步 READ，实测 10 次；64/128/256 MiB 分别是 2.84/2.91/2.89–3.02 GB/s | `bench/nixl_bw.py --mode target/initiator --gpu 1/0 --size 268435456` |
| 原生 CUDA P2P，GPU0 → GPU1 | 64 MiB | 33.09 | 直接 CUDA `cudaMemcpyPeer` / CUDA P2P over PCIe | 原生 CUDA p2p benchmark；GPU0 直拷到 GPU1 | `bench/p2p_bw.cu` |
| PyTorch NCCL P2P | 64 MiB | 34.56 | PyTorch NCCL `dist.send/recv`，NCCL 选择 CUDA P2P/PCIe | 两个 `torchrun` rank，rank0 → GPU0，rank1 → GPU1；同步 send/recv，50 次实测 | `run_logs/nccl_bw.log` |
| PyTorch NCCL ring AllReduce | 64 MiB | 29.16 effective | NCCL ring，GPU0/GPU1 之间 | 两个 rank；20 次实测；effective BW 扣除 ring 两次流贡献 | `run_logs/nccl_bw.log` |
| PyTorch NCCL ring AllReduce | 256 MiB | 29.80 effective | NCCL ring，GPU0/GPU1 之间 | 同上，这是本次测到的最大消息规模 | `run_logs/nccl_bw.log` |
| FlashInfer PCIe IPC AllReduce | 16 MiB bf16 | 20.29 | SM-resident buffer，通过 CUDA IPC/GPU 共享内存通信 | 两个 rank 在 GPU0/GPU1 上使用 FlashInfer `PcieIpcAllReduceWorkspace`；50 次实测；日志列名 `Gb/s` 但计算公式返回的是 byte bandwidth | `bench/pcie_ipc_ar.py` + `run_logs/pcie_ipc_ar.log` |

### GPU kernel 直接访问 memory

这一组不使用 PyTorch `copy_`，也不使用 CUDA `memcpyAsync`；由 CUDA/Triton kernel 显式 load/store memory。数据块为 `float4` 在 CUDA 版本中，Triton 版本按相同字节总量使用连续 `float`；都在 `GPU0` 上测试；`--size 256MiB`，5 次预热，10 次实测，CUDA events/Python 同步计时所有 kernel 的总时间。`copy` 的延迟同时包含读和写，因此带宽按 `2 × bytes / time` 计算。

| 场景 | 结果 | 访问路径 | 说明 |
|---|---:|---|---|
| `device_read` | 1413.4 GiB/s | CUDA kernel 从 GPU DRAM 读 `float4`，每 block 只写一个很小的 reduction 结果 | 这是单 GPU 内部读带宽 |
| `device_write` | 1404.2 GiB/s | CUDA kernel 从每个 thread 向 GPU DRAM 写 `float4` | 这是单 GPU 内部写带宽 |
| `device_copy` | 1352.5 GiB/s | CUDA kernel 逐 `float4` 读 device buffer 后写另一个 device buffer | 这是 kernel 内部 D2D copy，结果按读+写流量计算 |
| `host_pinned_read` | 47.8 GiB/s | CUDA kernel 通过 pinned host memory 的 mapped device pointer 零拷贝读 host memory | 数据面是 PCIe，映射后不用单独 H2D copy |
| `host_pinned_write` | 48.9 GiB/s | CUDA kernel 直接向 pinned host memory 的 mapped device pointer 写入 | 结果反映 GPU store 到 host memory 的 PCIe 写路径 |
| `host_pinned_copy` | 72.1 GiB/s | CUDA kernel 读 pinned host memory，写另一段 pinned host memory | 结果按读+写流量计算；两个方向同时经过 PCIe，所以高于单向场景 |

### GPU kernel driver/compiler 对照

| Kernel 实现 | 结果 | 说明 |
|---|---:|---|
| raw CUDA C++ | device_read/write/copy: 1413.4 / 1404.2 / 1352.5 GiB/s；pinned host read/write/copy: 47.8 / 48.9 / 72.1 GiB/s | 这是上面给出的显式 CUDA kernel 路径 |
| Triton | device_read/write/copy: 1394.1 / 1408.8 / 1357.4 GiB/s；pinned host read/write/copy: 47.8 / 48.9 / 71.8 GiB/s | 同机同消息规模的 Triton 结果与 raw CUDA kernel 基本线重合，说明差异来自 kernel 写法，而不是换了另一条硬件路径 |

## 解释

- `CUDA IPC same GPU` 是跨两个进程但同一 GPU0 内部的显存拷贝路径，不应直接和 GPU0→GPU1 的跨 GPU 行对比。
- `Raw CUDA P2P`、PyTorch NCCL P2P、PyTorch NCCL ring AllReduce 都受这台机器的 GPU0↔GPU1 PCIe/system interconnect 限制。
- FlashInfer PCIe IPC AllReduce 使用融合的 GPU 侧 allreduce/proposal 机制，因此 16 MiB 的结果低于该机器的基本 P2P copy；它减少的是 CPU 侧每次 GPU→GPU 同步的干扰。
- Mooncake 的 Python TransferEngine 两个样例都在 1.4-1.5 GiB/s 左右，因为这台机器没有 RDMA HCA，Mooncake 的本地 RPC 控制面加 TCP fallback 决定了主要瓶颈。
- 上面 `Mooncake nvlink_intra` 这一行的结果用到官方 C++ 引擎和 `MC_INTRANODE_NVLINK=1`。它不要求真的有 NVLink，而是在没有 RDMA/HCA 的本机上把 CUDA IPC 作为数据面，随后由 CUDA `cudaMemcpyBatchAsync` 走 GPU0↔GPU1 PCIe P2P。IPC path 之所以比 64 MiB 的 Python 样例读到的 1.4 GiB/s 高出两个数量级，是因为这不再是 RPC 加 TCP fallback。
- Mooncake 中 `tcp`/`P2PHANDSHAKE` 只是控制/发现路径；`nvlink_intra` + CUDA IPC 才是这里测到的高速数据面。
- NIXL UCX 在这台机器上必须加 `UCX_CUDA_IPC_ENABLE_GET_ZCOPY=on`，否则 UCX 1.22 会因为没检测到 NVLink 而禁用 CUDA IPC `get_zcopy`，导致回落到低速路径。加上之后 NIXL UCX 能接近 Mooncake 的量级，但仍低于原生 CUDA P2P/NCCL。

## NIXL UCX CUDA IPC 修正说明

### 修正前的三个现象

| UCX 配置 | 现象 | 结论 |
|---|---|---|
| `sm,cuda_ipc,tcp` | NIXL 报 `VRAM memory is detected as host by UCX` | 在 NIXL/UCX 当前路径里，只给出 `cuda_ipc` 不够；UCX context 没有显式把 CUDA memory type 起来时，`VRAM` 注册会被拒 |
| `sm,cuda_copy,cuda_ipc,tcp` | NIXL 能注册并传输，但只有 `0.04 GB/s` | 这一步还是能看到 VRAM，但 UCX 1.22.0 默认没有把 CUDA IPC 当作真正数据面 |
| `sm,cuda_copy,cuda_ipc,tcp` + `UCX_CUDA_IPC_ENABLE_GET_ZCOPY=on` | NIXL 能稳定跑通 | 强制启用 CUDA IPC 的 `get_zcopy` 后，数据面可用 |

### 根因

UCX 1.22.0 对 `cuda_ipc` 的能力查询里，`get_zcopy` 并不是无条件开启。其 CUDA IPC transport 会在 `ENABLE_GET_ZCOPY=auto` 且系统检测不到 NVLink 时，把 `get_zcopy` 置为不可用。这台机器确实没有 NVLink direct tunnel，所以 UCX 默认推断成“禁用”，导致 NIXL 结束在 TCP/低效回退路径。

把 `UCX_CUDA_IPC_ENABLE_GET_ZCOPY` 改成 `on` 后，这个限制被绕过。此时 UCX 自己也把 `cuda_ipc` 的能力从 `get_zcopy <= 0` 变成 `get_zcopy unlimited`，NIXL 的 end-to-end 数据面就能用 CUDA IPC 了。

### 测试口径

- CRITICAL: 两个进程分别运行在 `GPU0` 和 `GPU1`，不是同一 GPU 的 IPC。
- 通信方向：`GPU0` 作为 initiator，`GPU1` 作为 target。
- 传输操作：NIXL `READ`，同步计时。
- 每个消息规模都做 2 次预热、10 次实测。
- 结果描述的是 NIXL/UCX end-to-end 路径，不是直接 `cudaMemcpyPeer`。

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
