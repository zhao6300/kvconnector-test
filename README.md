# comm

## 定位

仓库里存放的是小型、定向的 GPU 通信与 KV-transfer connector 测试脚本。这里的目的不是把所有逻辑打包成库，而是通过定向测试来回答几个问题：

- 某个 connector 究竟真的走了哪一条通信路径。
- API 的行为和实际使用的 DRAM / PCIe / RDMA / IPC 路径是否一致。
- 本地 GPU 显存、pinned host memory、H2D/D2H、framework copy 之间的带宽差距到底来自哪里。

结果与解释见 [`bench/results.md`](bench/results.md) 和 [`dvram_results.md`](dvram_results.md)。

## 目录结构

- `bench/` : GPU 通信、Mooncake、NIXL、CUDA P2P、NCCL、FlashInfer 等基准脚本与 connector 语义说明。
- `local_mem_tests/` : 本机 GPU/host memory 对照测试，包含 PyTorch、raw CUDA memcpy、CUDA pinned pool、raw CUDA kernel 和 Triton kernel。
- `bench/results.md` : 实测带宽、通信路径、关键限制和连接器语义。
- `dvram_results.md` : 本机 GPU 显存、pinned host memory、pinned pool、PyTorch copy 的带宽对照。
- `run_logs/` : 部分代表性运行的 stdout/stderr 快照。

## 运行约定

脚本假设本机使用 `/opt/venv`。仓库本身不提供一键式脚本，直接按场景运行对应命令更可靠。

## 本地内存测试

`local_mem_tests/` 用来把 API 层次和真实硬件路径分开。raw CUDA 与 Triton kernel 都不使用 PyTorch copy，也不走 `memcpyAsync`。

```bash
nvcc -O3 -std=c++17 -arch=native local_mem_tests/gpu_kernel_rw.cu \
     -o local_mem_tests/gpu_kernel_rw
./local_mem_tests/gpu_kernel_rw \
  --gpu 0 --size $((256*1024*1024)) --warmup 5 --iters 10

/opt/venv/bin/python local_mem_tests/gpu_kernel_rw_triton.py \
  --gpu 0 --size $((256*1024*1024)) --warmup 5 --iters 10

/opt/venv/bin/python local_mem_tests/gpu_mem_read.py \
  --gpu 0 --size $((256*1024*1024)) --warmup 5 --iters 10

/opt/venv/bin/python local_mem_tests/torch_mem_read.py \
  --gpu 0 --size $((256*1024*1024)) --warmup 5 --iters 10

/opt/venv/bin/python local_mem_tests/cpu_pinned_pool.py \
  --gpu 0 --pool-size $((256*1024*1024)) --chunks 64 \
  --warmup 5 --iters 10 --direction h2d

/opt/venv/bin/python local_mem_tests/cpu_pinned_pool.py \
  --gpu 0 --pool-size $((256*1024*1024)) --chunks 64 \
  --warmup 5 --iters 10 --direction d2h
```

## GPU 通信及 connector 测试

### 通用 connector / KV transfer

```bash
/opt/venv/bin/python bench/gpu_ipc_bw.py
/opt/venv/bin/python -m pytest -q bench/mooncake_transfer_test.py
torchrun --nproc_per_node=2 bench/nccl_bw.py
```

### Mooncake

`mooncake_transfer_cross_gpu.py` 走 Mooncake `TransferEngine`。Python 侧看到的 `protocol=tcp` 是回退路径；真正的高带宽路径是 `nvlink_intra` + CUDA IPC，再用 `cudaMemcpyBatchAsync` 走 GPU P2P。

需要:

- `MC_INTRANODE_NVLINK=1`
- `MC_USE_NVLINK_IPC=1`

没有这些时，没有 RDMA 的机器会回落到低速 TCP。

```bash
/opt/venv/bin/python bench/mooncake_transfer_cross_gpu.py \
  --source-gpu 0 --target-gpu 1 \
  --size 67108864 --warmup 10 --iters 50
```

官方 C++ benchmark 更完整。示例:

```bash
/opt/venv/bin/python bench/mooncake_p2p_bench.py \
  --binary /tmp/mooncake-build-minimal/mooncake-transfer-engine/example/transfer_engine_bench \
  --source-gpu 0 --target-gpu 1 \
  --operations write,read --duration 5
```

如果本机已有构建产物，可以用 `MOONCAKE_BENCH_BIN` 指定二进制。

### Mooncake Python 快速验证

```bash
MC_INTRANODE_NVLINK=1 MC_USE_NVLINK_IPC=1 \
/opt/venv/bin/python bench/mooncake_p2p_python_bench.py \
  --source-gpu 0 --target-gpu 1 \
  --pool-size 268435456 --pieces 1 --runtime 1.0
```

Python 版本适合快速验证；如果要与官方提交线程模型对齐，仍然使用上面的 C++ benchmark。

### NIXL

`cuda_ipc` 和 `cuda_copy` 是 UCX 的 VRAM 数据路径；`tcp` 只是控制路径。在这台机器上需要：

- `UCX_TLS=sm,cuda_copy,cuda_ipc,tcp`
- `UCX_CUDA_IPC_ENABLE_GET_ZCOPY=on`（否则 UCX 会因为未检测到 NVLink 而禁用 CUDA IPC `get_zcopy`）

```bash
UCX_TLS=sm,cuda_copy,cuda_ipc,tcp UCX_CUDA_IPC_ENABLE_GET_ZCOPY=on \
/opt/venv/bin/python bench/nixl_bw.py --mode target \
  --gpu 1 --size 67108864 --warmup 2 --iters 10

UCX_TLS=sm,cuda_copy,cuda_ipc,tcp UCX_CUDA_IPC_ENABLE_GET_ZCOPY=on \
/opt/venv/bin/python bench/nixl_bw.py --mode initiator \
  --gpu 0 --ip 127.0.0.1 --size 67108864 --warmup 2 --iters 10
```

## 阅读结果

- [`bench/results.md`](bench/results.md) 记录实测带宽、真实通信路径、限制条件与连接器语义。
- [`dvram_results.md`](dvram_results.md) 记录本机 GPU 显存、pinned host memory、pinned pool、PyTorch copy 的带宽对照。
- [`run_logs/`](run_logs/) 保存的是部分代表性 stdout/stderr，不是独立结果文件。

## 注意

- 仓库重点不是跨语言 packaging，而是解释和测量通信路径。
- 运行基准会产生少量缓存和日志，已由 `.gitignore` 排除。
- `bench/results.md` 与 `dvram_results.md` 都是“相关性较高”的结果记录，不是“通用性能库”。
