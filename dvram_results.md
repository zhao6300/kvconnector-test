# DVRAM / Local Memory Benchmark

## 一句话结论

- 单 GPU 显存 kernel 读/写大约在 `1.35–1.41 TB/s`。
- pinned host memory zero-copy 大约 `48 GiB/s`。
- 标准 pinned host pool H2D/D2H 大约 `51–53 GiB/s`。
- PyTorch framework copy 路径明显低于 raw DMA/kernel，尤其 pageable 只有约 `10 GiB/s`。

## 测试环境

- 机器：本地单机，无 RDMA/RoCE HCA。
- GPU：`RTX PRO 6000 Blackwell Server Edition`，97.9 GiB 显存。
- 本组测试固定使用 `GPU0`。
- CUDA：`13.2`
- Python：`3.12`
- PyTorch：`2.15.0a0+git04b971a`
- Triton：`3.7.1`
- 统一参数：`256 MiB` 数据块，5 次预热，10 次实测。
- 带宽单位：`GiB/s`（1 GiB = 2^30 bytes）。
- `device_copy` 与 `host_pinned_copy` 由同一个 kernel 同时做读和写，因此带宽按 `2 × bytes / time` 计算。

## 结果总览

### GPU DRAM direct kernel access

| 场景 | 实现 | 访问路径 | 结果 (GiB/s) | 说明 |
|---|---|---|---:|---|
| `device_read` | raw CUDA C++ kernel | GPU DRAM 读 | 1412.3 | kernel 显式 `__ldg` 读 `float4`，每 block 只写一个很小的 reduction 结果 |
| `device_write` | raw CUDA C++ kernel | GPU DRAM 写 | 1400.4 | kernel 显式向 GPU DRAM 写 `float4` |
| `device_copy` | raw CUDA C++ kernel | GPU DRAM 读+写 | 1353.2 | kernel 逐 `float4` 读 device buffer 后写另一个 device buffer |
| `device_read` | Triton kernel | GPU DRAM 读 | 1398.4 | Triton 显式 `tl.load`，语义上贴近 CUDA kernel |
| `device_write` | Triton kernel | GPU DRAM 写 | 1407.5 | Triton 显式 `tl.store` |
| `device_copy` | Triton kernel | GPU DRAM 读+写 | 1361.5 | Triton kernel 内部逐元素读 device buffer 写另一个 device buffer |

### pinned host memory zero-copy access

| 场景 | 实现 | 访问路径 | 结果 (GiB/s) | 说明 |
|---|---|---|---:|---|
| `host_pinned_read` | raw CUDA C++ kernel | pinned host memory zero-copy 读 | 47.8 | CUDA kernel 通过 mapped device pointer 直接读 pinned host memory |
| `host_pinned_write` | raw CUDA C++ kernel | pinned host memory zero-copy 写 | 48.9 | CUDA kernel 通过 mapped device pointer 直接写 pinned host memory |
| `host_pinned_copy` | raw CUDA C++ kernel | pinned host memory 读+写 | 72.6 | 一个 kernel 同时从一段 pinned host memory 读，并写另一段 |
| `host_pinned_read` | Triton kernel | pinned host memory zero-copy 读 | 47.8 | Triton kernel 直接从 pinned host memory 读 |
| `host_pinned_write` | Triton kernel | pinned host memory zero-copy 写 | 48.9 | Triton kernel 直接写 pinned host memory |
| `host_pinned_copy` | Triton kernel | pinned host memory 读+写 | 71.9 | 一个 Triton kernel 同时从一段 pinned host memory 读，并写另一段 |

### 标准 H2D/D2H 与 pinned host pool

| 场景 | 实现 | 访问路径 | 结果 (GiB/s) | 说明 |
|---|---|---|---:|---|
| `H2D pinned` | CuPy `memcpyAsync` | pinned host memory → GPU DRAM | 52.4 | 标准 CUDA H2D memcpy 路径 |
| `H2D pageable` | CuPy `memcpyAsync` | pageable host memory → GPU DRAM | 52.4 | 标准 CUDA H2D memcpy 路径，内存不是 pinned |
| `D2H pinned` | CuPy `memcpyAsync` | GPU DRAM → pinned host memory | 53.1 | 标准 CUDA D2H memcpy 路径 |
| `D2H pageable` | CuPy `memcpyAsync` | GPU DRAM → pageable host memory | 53.0 | 标准 CUDA D2H memcpy 路径，内存不是 pinned |
| pinned pool `H2D sequential` | CuPy `memcpyAsync` | pinned host pool → GPU DRAM | 51.3 | 256 MiB pool，64 个 4 MiB chunk 顺序搬 |
| pinned pool `H2D scatter` | CuPy `memcpyAsync` | pinned host pool → GPU DRAM | 51.3 | 256 MiB pool，64 个 4 MiB chunk 打散搬 |
| pinned pool `D2H sequential` | CuPy `memcpyAsync` | GPU DRAM → pinned host pool | 52.5 | 256 MiB pool，64 个 4 MiB chunk 顺序搬 |
| pinned pool `D2H scatter` | CuPy `memcpyAsync` | GPU DRAM → pinned host pool | 52.2 | 256 MiB pool，64 个 4 MiB chunk 打散搬 |

### PyTorch framework copy

| 场景 | 实现 | 访问路径 | 结果 (GiB/s) | 说明 |
|---|---|---|---:|---|
| `pageable_copy` | PyTorch | pageable host memory → GPU DRAM | 10.4 | `target.copy_(source)` 标准 torch copy 路径 |
| `pageable_to` | PyTorch | pageable host memory → GPU DRAM | 10.5 | `target.copy_(source.to(target.device))` torch copy/to 路径 |
| `pinned_copy` | PyTorch | pinned host memory → GPU DRAM | 24.9 | `target.copy_(pinned_source)` |
| `pinned_to` | PyTorch | pinned host memory → GPU DRAM | 24.6 | `target.copy_(pinned_source.to(target.device))` |

## 关键解读

1. **GPU 显存读写是另一条路径**

   `device_read/write/copy` 和 `host_pinned_read/write/copy` 的差距非常大。  
   说明两者不只是换 API，而是完全不同的硬件路径：

   - GPU 显存读写走本机 GPU DRAM。
   - pinned host zero-copy 读/写走 PCIe。
   - pinned host memory 的 mapped device pointer 不表示数据在 GPU 上，只是允许 GPU 通过地址直接访问 host memory。

2. **Triton 与 raw CUDA 基本线重合**

   这里没有看到“Triton 引入了一条不同硬件路径”的现象。  
   raw CUDA 与 Triton 都是让 GPU 本身发起 load/store。  
   差异更多来自 kernel 写法和编译器，不是通信路径。raw CUDA 版本更适合作为“更接近硬件上限”的参考。

3. **pinned host memory 本身有明显上限**

   `host_pinned_read/write` 都在 48 GiB/s 左右。  
   一个 kernel 同时读一段 pinned host memory 并写另一段时，结果升到约 72 GiB/s。  
   这说明不是简单把一个方向扩大，而是读/写两路 PCIe 流量同时被计入。

4. **pinned pool 与 H2D/D2H 接近**

   `pinned pool h2d/d2h` 与 `H2D/D2H memcpy` 都在 50 GiB/s 左右。  
   顺序和打散都非常接近；4 MiB chunk 已足够大，碎片化没有带来明显额外代价。  
   主要瓶颈还是 PCIe 物理 bus，而不是算法本身。

5. **PyTorch 结构化路径明显更慢**

   `pageable_copy/pageable_to` 约在 `10 GiB/s`，`pinned_copy/pinned_to` 提升到约 `25 GiB/s`。  
   这代表的是 **framework copy overhead**，不能等同于 raw loader/store 结果或峰值内存带宽。

## 运行方式

```bash
nvcc -O3 -std=c++17 -arch=native local_mem_tests/gpu_kernel_rw.cu \
     -o local_mem_tests/gpu_kernel_rw
./local_mem_tests/gpu_kernel_rw --gpu 0 --size $((256*1024*1024)) --warmup 5 --iters 10

/opt/venv/bin/python local_mem_tests/gpu_kernel_rw_triton.py \
  --gpu 0 --size $((256*1024*1024)) --warmup 5 --iters 10

/opt/venv/bin/python local_mem_tests/gpu_mem_read.py \
  --gpu 0 --size $((256*1024*1024)) --warmup 5 --iters 10

/opt/venv/bin/python local_mem_tests/torch_mem_read.py \
  --gpu 0 --size $((256*1024*1024)) --warmup 5 --iters 10

/opt/venv/bin/python local_mem_tests/cpu_pinned_pool.py \
  --gpu 0 --pool-size $((256*1024*1024)) --chunks 64 --warmup 5 --iters 10 --direction h2d

/opt/venv/bin/python local_mem_tests/cpu_pinned_pool.py \
  --gpu 0 --pool-size $((256*1024*1024)) --chunks 64 --warmup 5 --iters 10 --direction d2h
```

## 注意事项

- 这组结果主要用于本机 GPU/host memory 带宽对照，不适合直接和 RDMA/NIC/跨机通信结果混在同一语义下比较。
- `pinned host memory` 读数是 mapped device pointer 上的带宽，不是标准 `cudaMemcpy`。
- raw CUDA/Triton kernel 的 `device_copy` 使用了两个 device buffer，不等于同 GPU 两 peer 的 cross-GPU bandwidth。
- `H2D` 与 `D2H` 差异非常小，主要来自噪声、allocator/driver 细节和测试噪声。
- `GiB/s` 按 2^30 计算。
