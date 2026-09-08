// Raw CUDA streaming read benchmark reading from mapped pinned host memory.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t status_ = (call);                                        \
        if (status_ != cudaSuccess) {                                        \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,      \
                         __LINE__, cudaGetErrorString(status_));            \
            std::exit(1);                                                    \
        }                                                                    \
    } while (false)

namespace {

constexpr int kThreads = 256;
constexpr float4 kFillValue = {1.0f, 2.0f, 3.0f, 4.0f};

__global__ void read_host_f4_stream_kernel(const float4* input, size_t count,
                                           float* output) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    const size_t first = static_cast<size_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
    float accumulator = 0.0f;
    for (size_t index = first; index < count; index += stride) {
        const float4 value = __ldcs(input + index);
        accumulator += value.x + value.y + value.z + value.w;
    }
    if (threadIdx.x == 0) {
        output[blockIdx.x] = accumulator;
    }
}

double gib_per_second(size_t bytes_per_launch, int launches,
                      float milliseconds) {
    const double total_bytes =
        static_cast<double>(bytes_per_launch) * launches;
    return (total_bytes / 1024.0 / 1024.0 / 1024.0) *
           (1000.0 / static_cast<double>(milliseconds));
}

size_t parse_size(const char* value) {
    return static_cast<size_t>(std::strtoull(value, nullptr, 0));
}

int parse_int(const char* value) {
    return static_cast<int>(std::atoi(value));
}

}  // namespace

int main(int argc, char** argv) {
    int gpu = 0;
    int warmup = 2;
    int iterations = 10;
    int blocks = 0;
    int threads = kThreads;
    size_t bytes = 256 * 1024 * 1024;

    for (int argument = 1; argument < argc; ++argument) {
        const std::string value = argv[argument];
        if (value == "--gpu" && argument + 1 < argc) {
            gpu = parse_int(argv[++argument]);
        } else if (value == "--warmup" && argument + 1 < argc) {
            warmup = parse_int(argv[++argument]);
        } else if (value == "--iters" && argument + 1 < argc) {
            iterations = parse_int(argv[++argument]);
        } else if (value == "--blocks" && argument + 1 < argc) {
            blocks = parse_int(argv[++argument]);
        } else if (value == "--threads" && argument + 1 < argc) {
            threads = parse_int(argv[++argument]);
        } else if (value == "--size" && argument + 1 < argc) {
            bytes = parse_size(argv[++argument]);
        } else if (value == "--help" || value == "-h") {
            std::printf(
                "usage: gpu_read_host_mem_cuda [--gpu N] [--warmup N] "
                "[--iters N] [--blocks N] [--threads N] [--size BYTES]\n");
            return 0;
        } else {
            std::fprintf(stderr, "unknown or incomplete argument: %s\n",
                         argv[argument]);
            return 2;
        }
    }

    bytes &= ~size_t{127};
    if (gpu < 0 || warmup <= 0 || iterations <= 0 || blocks < 0 ||
        threads <= 0 || bytes < sizeof(float4)) {
        std::fprintf(stderr, "invalid arguments\n");
        return 2;
    }

    CUDA_CHECK(cudaSetDevice(gpu));
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (gpu >= device_count) {
        std::fprintf(stderr, "cuda:%d is not a valid device\n", gpu);
        return 2;
    }

    cudaDeviceProp properties = {};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, gpu));
    if (properties.canMapHostMemory == 0) {
        std::fprintf(stderr, "device does not support mapped host memory\n");
        return 2;
    }

    if (blocks <= 0) {
        blocks = properties.multiProcessorCount;
    }

    const size_t count = bytes / sizeof(float4);
    const size_t stride = static_cast<size_t>(blocks) * threads;
    if (count < stride) {
        std::fprintf(
            stderr,
            "--size is too small; each launch should read at least "
            "blocks * threads unique addresses\n");
        return 2;
    }

    const size_t total_bytes = bytes * (warmup + iterations);
    float4* source_host = nullptr;
    float4* source_device = nullptr;
    float* output_device = nullptr;
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&source_host),
                             total_bytes,
                             cudaHostAllocMapped | cudaHostAllocPortable));
    CUDA_CHECK(cudaHostGetDevicePointer(
        reinterpret_cast<void**>(&source_device), source_host, 0));
    CUDA_CHECK(cudaMalloc(&output_device, blocks * sizeof(float)));

    const size_t fill_count = total_bytes / sizeof(float4);
    std::fill(source_host, source_host + fill_count, kFillValue);

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    int launch = 0;
    for (int iteration = 0; iteration < warmup; ++iteration, ++launch) {
        read_host_f4_stream_kernel<<<blocks, threads, 0, stream>>>(
            source_device + launch * count, count, output_device);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int iteration = 0; iteration < iterations; ++iteration, ++launch) {
        const size_t offset_index =
            static_cast<size_t>(launch % (warmup + iterations)) * count;
        read_host_f4_stream_kernel<<<blocks, threads, 0, stream>>>(
            source_device + offset_index, count, output_device);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    int residency = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &residency, read_host_f4_stream_kernel, threads, 0));

    std::printf("device=cuda:%d name=%s cc=%d.%d\n", gpu, properties.name,
                properties.major, properties.minor);
    std::printf(
        "source=%s hardware_sm=%d blocks=%d threads=%d "
        "resident_blocks_per_sm=%d\n",
        "mapped_pinned_host_memory", properties.multiProcessorCount, blocks,
        threads, residency);
    std::printf(
        "bytes_per_launch=%zu warmup=%d iters=%d unique_addresses_per_launch=%zu\n",
        bytes, warmup, iterations, std::min(stride, count));
    std::printf(
        "aggregate_bw=%.3f GiB/s per_launch_ms=%.6f\n",
        gib_per_second(bytes, iterations, milliseconds),
        milliseconds / static_cast<double>(iterations));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(output_device));
    CUDA_CHECK(cudaFreeHost(source_host));
    return 0;
}
