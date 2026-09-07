# DVRAM / Local Memory Benchmark

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
- `device_copy` 与 `host_pinned_copy` 同一 kernel 同时做读和写，因此带宽按 `2 × bytes / time` 计算。

## 结果总览

| 场景 | 实现 | 访问路径 | 结果 | 说明 |
|---|---|---|---:|---|
| `device_read` | raw CUDA C++ kernel | GPU DRAM 读 | 1412.3 GiB/s | kernel 显式 `__ldg` 读 `float4`，每 block 只写一个很小的 reduction 结果 |
| `device_write` | raw CUDA C++ kernel | GPU DRAM 写 | 1400.4 GiB/s | kernel 显式向 GPU DRAM 写 `float4` |
| `device_copy` | raw CUDA C++ kernel | GPU DRAM 读+写 | 1353.2 GiB/s | kernel 逐 `float4` 读 device buffer 后写另一个 device buffer |
| `host_pinned_read` | raw CUDA C++ kernel | pinned host memory zero-copy 读 | 47.8 GiB/s | CUDA kernel 通过 mapped device pointer 直接读 pinned host memory |
| `host_pinned_write` | raw CUDA C++ kernel | pinned host memory zero-copy 写 | 48.9 GiB/s | CUDA kernel 通过 mapped device pointer 直接写 pinned host memory |
| `host_pinned_copy` | raw CUDA C++ kernel | pinned host memory 读+写 | 72.6 GiB/s | 一个 kernel 同时从一段 pinned host memory 读，并写另一段 |
| `device_read` | Triton kernel | GPU DRAM 读 | 1398.4 GiB/s | Triton 显式 `tl.load`，语义上贴近 CUDA kernel |
| `device_write` | Triton kernel | GPU DRAM 写 | 1407.5 GiB/s | Triton 显式 `tl.store` |
| `device_copy` | Triton kernel | GPU DRAM 读+写 | 1361.5 GiB/s | Triton kernel 内部逐元素读 device buffer 写另一个 device buffer |
| `host_pinned_read` | Triton kernel | pinned host memory zero-copy 读 | 47.8 GiB/s | Triton kernel 直接从 pinned host memory 读 |
| `host_pinned_write` | Triton kernel | pinned host memory zero-copy 写 | 48.9 GiB/s | Triton kernel 直接写 pinned host memory |
| `host_pinned_copy` | Triton kernel | pinned host memory 读+写 | 71.9 GiB/s | 一个 Triton kernel 同时从一段 pinned host memory 读，并写另一段 |
| `H2D pinned` | CuPy `memcpyAsync` | pinned host memory → GPU DRAM | 52.4 GiB/s | 标准 CUDA H2D memcpy 路径 |
| `H2D pageable` | CuPy `memcpyAsync` | pageable host memory → GPU DRAM | 52.4 GiB/s | 标准 CUDA H2D memcpy 路径，内存不是 pinned |
| `D2H pinned` | CuPy `memcpyAsync` | GPU DRAM → pinned host memory | 53.1 GiB/s | 标准 CUDA D2H memcpy 路径 |
| `D2H pageable` | CuPy `memcpyAsync` | GPU DRAM → pageable host memory | 53.0 GiB/s | 标准 CUDA D2H memcpy 路径，内存不是 pinned |
| `pageable_copy` | PyTorch | pageable host memory → GPU DRAM | 10.4 GiB/s | `target.copy_(source)` 标准 torch copy 路径 |
| `pageable_to` | PyTorch | pageable host memory → GPU DRAM | 10.5 GiB/s | `target.copy_(source.to(target.device))` torch copy/to 路径 |
| `pinned_copy` | PyTorch | pinned host memory → GPU DRAM | 24.9 GiB/s | `target.copy_(pinned_source)` |
| `pinned_to` | PyTorch | pinned host memory → GPU DRAM | 24.6 GiB/s | `target.copy_(pinned_source.to(target.device))` |
| pinned pool `H2D sequential` | CuPy `memcpyAsync` | pinned host pool → GPU DRAM | 51.3 GiB/s | 256 MiB pool，64 个 4 MiB chunk 顺序搬 |
| pinned pool `H2D scatter` | CuPy `memcpyAsync` | pinned host pool → GPU DRAM | 51.3 GiB/s | 256 MiB pool，64 个 4 MiB chunk 打散搬 |
| pinned pool `D2H sequential` | CuPy `memcpyAsync` | GPU DRAM → pinned host pool | 52.5 GiB/s | 256 MiB pool，64 个 4 MiB chunk 顺序搬 |
| pinned pool `D2H scatter` | CuPy `memcpyAsync` | GPU DRAM → pinned host pool | 52.2 GiB/s | 256 MiB pool，64 个 4 MiB chunk 打散搬 |

## 主要解释

1. **GPU 显存读写是另一条路径**

   `device_read/write/copy` 和 `host_pinned_read/write/copy` 的差距非常大。
   这说明两者之间不只是换了一个 API，而是完全不同的硬件路径：

   - GPU 显存读写走的是本机 GPU DRAM。
   - pinned host zero-copy 读/写走的是 PCIe。
   - pinned host memory 的 mapped device pointer 不表示数据在 GPU 上，只是允许 GPU 直接通过地址访问 host memory。

2. **Triton 与 raw CUDA 基本线重合**

   这里没有看到“Triton 引入了一条不同硬件路径”的现象。
   两者同规模下差距都在个位数，甚至 `device_write` 和 `host_pinned_copy` 有一边略高一点。
   因此可以认为：

   - raw CUDA 与 Triton 都是让 GPU 本身发起 load/store。
   - 差异更多来自 kernel 写法和编译器，不是通信路径。
   - raw CUDA 版本更适合作为“更接近硬件上限”的参考。

3. **pinned host memory 本身有明显上限**

   `host_pinned_read/write` 都在 48 GiB/s 左右。
   这接近这台机器 PCIe 单路径的典型表现。

   当一个 kernel 同时读一段 pinned host memory 并写另一段 pinned host memory 时，结果升到约 72 GiB/s。
   这说明这不是简单地把一个方向扩大，而是把读/写两路 PCIe 流量同时算进去的结果。

4. **pinned pool 与 H2D/D2H 相近**

   `pinned pool h2d/d2h` 与 `H2D/D2H memcpy` 都在 50 GiB/s 左右。
   这说明：

   - 顺序和打散都还非常接近。
   - 4 MiB chunk 已经足够大，碎片化没有带来明显的额外代价。
   - pinned host pool 的主要瓶颈还是 PCIe 物理 bus，不是算法本身。

5. **PyTorch 结构化路径明显更慢**

   `pageable_copy/pageable_to` 只有约 10 GiB/s。
   `pinned_copy/pinned_to` 提升到约 25 GiB/s。
   这是因为 PyTorch 这一层不只是 kernel 本身，还会带来更多框架层调度和路径组织开销。

   如果你想把结果解释得更细，可以把它当作 **framework copy overhead 的表现**，而不是峰值内存带宽的读数。

## 结论

- 单 GPU 显存本地 kernel 读/写大约在 `1.35–1.41 TB/s`。
- pinned host memory zero-copy大约 `48 GiB/s`。
- 标准 pinned host pool H2D/D2H 大约 `51–53 GiB/s`。
- PyTorch framework copy路径明显低于 raw DMA/kernel。尤其 `pageable` 路径只有 `10 GiB/s` 左右。

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

1. 这组结果主要是本机 GPU/host memory 的带宽对照，不适合直接和 RDMA、NIC、Node 间通信结果混在一张顺序表里比较含义。
2. `pinned host memory` 的 throughput 是在 mapped device pointer 上测出来的，不是通过标准 `cudaMemcpy`。
3. raw CUDA/Triton kernel 的 `device_copy` 是用了两个 device buffer，不等于 one GPU 两 peer 的 cross-GPU bandwidth.
4. `H2D` 与 `D2H` 的差异非常小，读数主要来自测试噪声和 allocator / driver detail。
5. `GiB/s` 按 2^30 计算。
