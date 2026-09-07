// Benchmark raw GPU kernel load and store bandwidth.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
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

constexpr int kBlock = 256;
constexpr int kGrid = 1984;
constexpr float4 kValue = {1.0f, 2.0f, 3.0f, 4.0f};

__global__ void write_f4_kernel(float4* output, size_t count) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
         index < count; index += stride) {
        output[index] = kValue;
    }
}

__global__ void read_f4_kernel(const float4* input, size_t count,
                               float* output) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    const size_t first = static_cast<size_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
    float result = 0.0f;
    for (size_t index = first; index < count; index += stride) {
        const float4 value = __ldg(input + index);
        result += value.x + value.y + value.z + value.w;
    }
    if (threadIdx.x == 0) {
        output[blockIdx.x] = result;
    }
}

__global__ void copy_f4_kernel(const float4* input, float4* output,
                               size_t count) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
         index < count; index += stride) {
        output[index] = __ldg(input + index);
    }
}

double gib_per_second(size_t bytes, int iterations, float milliseconds) {
    const double total_bytes = static_cast<double>(bytes) * iterations;
    return (total_bytes / 1024.0 / 1024.0 / 1024.0) *
           (1000.0 / static_cast<double>(milliseconds));
}

float run_case(const std::string& mode, const void* source,
               void* target, size_t count, float* output_gpu, int warmup,
               int iterations) {
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    for (int iteration = 0; iteration < warmup; ++iteration) {
        if (mode == "device_read" || mode == "host_pinned_read") {
            read_f4_kernel<<<kGrid, kBlock, 0, stream>>>(
                static_cast<const float4*>(source), count, output_gpu);
        } else if (mode == "device_write" || mode == "host_pinned_write") {
            write_f4_kernel<<<kGrid, kBlock, 0, stream>>>(
                static_cast<float4*>(target), count);
        } else {
            copy_f4_kernel<<<kGrid, kBlock, 0, stream>>>(
                static_cast<const float4*>(source),
                static_cast<float4*>(target), count);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int iteration = 0; iteration < iterations; ++iteration) {
        if (mode == "device_read" || mode == "host_pinned_read") {
            read_f4_kernel<<<kGrid, kBlock, 0, stream>>>(
                static_cast<const float4*>(source), count, output_gpu);
        } else if (mode == "device_write" || mode == "host_pinned_write") {
            write_f4_kernel<<<kGrid, kBlock, 0, stream>>>(
                static_cast<float4*>(target), count);
        } else {
            copy_f4_kernel<<<kGrid, kBlock, 0, stream>>>(
                static_cast<const float4*>(source),
                static_cast<float4*>(target), count);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return milliseconds;
}

std::vector<std::string> all_modes() {
    return {"device_read", "device_write", "device_copy",
            "host_pinned_read", "host_pinned_write", "host_pinned_copy"};
}

std::vector<std::string> parse_modes(const std::string& raw) {
    std::vector<std::string> modes;
    size_t begin = 0;
    while (begin <= raw.size()) {
        const size_t comma = raw.find(',', begin);
        const std::string mode = raw.substr(begin, comma - begin);
        if (!mode.empty()) {
            modes.push_back(mode);
        }
        if (comma == std::string::npos) {
            break;
        }
        begin = comma + 1;
    }
    return modes;
}

size_t counted_bytes(const std::string& mode, size_t bytes) {
    if (mode == "device_copy" || mode == "host_pinned_copy") {
        return bytes * 2;
    }
    return bytes;
}

}  // namespace

int main(int argc, char** argv) {
    int gpu = 0;
    int warmup = 5;
    int iterations = 20;
    size_t bytes = 256 * 1024 * 1024;
    std::vector<std::string> modes = all_modes();

    for (int argument = 1; argument < argc; ++argument) {
        const std::string value = argv[argument];
        if (value == "--gpu" && argument + 1 < argc) {
            gpu = std::atoi(argv[++argument]);
        } else if (value == "--warmup" && argument + 1 < argc) {
            warmup = std::atoi(argv[++argument]);
        } else if (value == "--iters" && argument + 1 < argc) {
            iterations = std::atoi(argv[++argument]);
        } else if (value == "--mode" && argument + 1 < argc) {
            modes = parse_modes(argv[++argument]);
        } else if (value == "--size" && argument + 1 < argc) {
            bytes = std::strtoull(argv[++argument], nullptr, 0);
        } else if (value == "--help" || value == "-h") {
            std::printf(
                "usage: gpu_kernel_rw [--gpu N] [--size BYTES] [--warmup N] "
                "[--iters N]\n"
                "                     [--mode a,b,...]\n"
                "modes: device_read,device_write,device_copy,"
                "host_pinned_read,host_pinned_write,host_pinned_copy\n");
            return 0;
        } else {
            std::fprintf(stderr, "unknown or incomplete argument: %s\n",
                         argv[argument]);
            return 2;
        }
    }

    bytes = bytes & ~size_t{255};
    if (bytes < 256 || warmup <= 0 || iterations <= 0) {
        std::fprintf(stderr, "invalid --size, --warmup, or --iters\n");
        return 2;
    }

    CUDA_CHECK(cudaSetDevice(gpu));
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (gpu < 0 || gpu >= device_count) {
        std::fprintf(stderr, "cuda:%d is not a valid device\n", gpu);
        return 2;
    }

    cudaDeviceProp properties = {};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, gpu));
    std::printf("device=cuda:%d name=%s cc=%d.%d\n", gpu, properties.name,
                properties.major, properties.minor);
    std::printf("kernel=%s size_bytes=%zu warmup=%d iterations=%d\n", "raw_cuda",
                bytes, warmup, iterations);
    std::printf("mode                  GiB_s     per_kernel_ms\n");

    const size_t count = bytes / sizeof(float4);
    float4* source_device = nullptr;
    float4* target_device = nullptr;
    float* output_device = nullptr;
    float4* source_host_pinned = nullptr;
    float4* target_host_pinned = nullptr;
    unsigned int host_flags = cudaHostAllocMapped | cudaHostAllocPortable;

    CUDA_CHECK(cudaMalloc(&source_device, bytes));
    CUDA_CHECK(cudaMalloc(&target_device, bytes));
    CUDA_CHECK(cudaMalloc(&output_device, kGrid * sizeof(float)));
    bool need_host = false;
    for (const std::string& mode : modes) {
        need_host = need_host ||
                    mode == "host_pinned_read" ||
                    mode == "host_pinned_write" ||
                    mode == "host_pinned_copy";
    }
    if (need_host) {
        CUDA_CHECK(cudaHostAlloc(&source_host_pinned, bytes, host_flags));
        CUDA_CHECK(cudaHostAlloc(&target_host_pinned, bytes, host_flags));
    }

    bool fill_device = false;
    bool fill_host = false;
    for (const std::string& mode : modes) {
        fill_device = fill_device ||
                      mode == "device_read" || mode == "device_copy";
        fill_host = fill_host ||
                    mode == "host_pinned_read" || mode == "host_pinned_copy";
    }
    if (fill_device) {
        write_f4_kernel<<<kGrid, kBlock>>>(source_device, count);
        CUDA_CHECK(cudaGetLastError());
    }
    if (fill_host) {
        void* host_pointer = source_host_pinned;
        void* device_pointer = nullptr;
        CUDA_CHECK(cudaHostGetDevicePointer(&device_pointer, host_pointer, 0));
        write_f4_kernel<<<kGrid, kBlock>>>(
            static_cast<float4*>(device_pointer), count);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    for (const std::string& mode : modes) {
        void* source = nullptr;
        void* target = nullptr;
        if (mode == "device_read" || mode == "device_write" ||
                mode == "device_copy") {
            source = source_device;
            target = target_device;
        } else if (mode == "host_pinned_read" ||
                   mode == "host_pinned_write" ||
                   mode == "host_pinned_copy") {
            void* source_pointer = source_host_pinned;
            void* target_pointer = target_host_pinned;
            CUDA_CHECK(cudaHostGetDevicePointer(&source_pointer,
                                                source_pointer, 0));
            CUDA_CHECK(cudaHostGetDevicePointer(&target_pointer,
                                                target_pointer, 0));
            source = source_pointer;
            target = target_pointer;
        }
        const float milliseconds =
            run_case(mode, source, target, count, output_device, warmup,
                     iterations);
        const double bandwidth = gib_per_second(counted_bytes(mode, bytes),
                                                iterations, milliseconds);
        bool is_read = mode == "device_read" || mode == "host_pinned_read";
        float checksum = 0.0f;
        if (is_read) {
            std::vector<float> summary(kGrid, 0.0f);
            CUDA_CHECK(cudaMemcpy(summary.data(), output_device,
                                  kGrid * sizeof(float),
                                  cudaMemcpyDeviceToHost));
            for (float value : summary) {
                checksum += value;
            }
        }
        std::printf("%-21s %9.3f %17.6f", mode.c_str(), bandwidth,
                    milliseconds / iterations);
        if (is_read) {
            std::printf(" checksum=%.1f", checksum);
        }
        std::putchar('\n');
        if (mode == "device_read" || mode == "host_pinned_read") {
            std::vector<float> summary(kGrid, 0.0f);
            (void)summary;
        }
    }

    CUDA_CHECK(cudaFree(output_device));
    CUDA_CHECK(cudaFree(target_device));
    CUDA_CHECK(cudaFree(source_device));
    if (source_host_pinned != nullptr) {
        CUDA_CHECK(cudaFreeHost(source_host_pinned));
    }
    if (target_host_pinned != nullptr) {
        CUDA_CHECK(cudaFreeHost(target_host_pinned));
    }
    return 0;
}
