// CUDA P2P bandwidth benchmark between two GPUs using cudaMemcpyPeerAsync
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define SAFE_CUDART(call)                                                     \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                 \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

__global__ void fill_kernel(float* ptr, size_t n, float val) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) ptr[idx] = val;
}

int main(int argc, char** argv) {
    int dev_a = 0, dev_b = 1;
    if (argc >= 3) { dev_a = atoi(argv[1]); dev_b = atoi(argv[2]); }

    int n_devices = 0;
    SAFE_CUDART(cudaGetDeviceCount(&n_devices));
    if (dev_a >= n_devices || dev_b >= n_devices) {
        printf("Invalid device ids\n");
        return 1;
    }

    // Test P2P capability
    int can_access = -1;
    SAFE_CUDART(cudaDeviceCanAccessPeer(&can_access, dev_a, dev_b));
    printf("P2P: dev%d -> dev%d %s\n", dev_a, dev_b,
           can_access ? "ENABLED" : "DISABLED");
    if (!can_access) {
        printf("P2P not supported between %d and %d, falling back to host copy path.\n", dev_a, dev_b);
    }

    if (can_access) {
        SAFE_CUDART(cudaDeviceEnablePeerAccess(dev_b, 0));
        // Avoid cudaDeviceEnablePeerAccess failure if already enabled.
    }
    SAFE_CUDART(cudaSetDevice(dev_b));
    if (can_access) {
        SAFE_CUDART(cudaDeviceEnablePeerAccess(dev_a, 0));
    }
    SAFE_CUDART(cudaSetDevice(dev_a));

    // Allocate buffers
    size_t max_bytes = 1UL << 30;  // 1 GiB per side for DtoD
    size_t sizes[] = {1UL << 12, 1UL << 16, 1UL << 20, 1UL << 22, 1UL << 24, 1UL << 26, 1UL << 28, 1UL << 30};
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);

    float *buf_a, *buf_b;
    SAFE_CUDART(cudaSetDevice(dev_a));
    SAFE_CUDART(cudaMalloc(&buf_a, max_bytes));
    SAFE_CUDART(cudaSetDevice(dev_b));
    SAFE_CUDART(cudaMalloc(&buf_b, max_bytes));

    // Fill on source
    SAFE_CUDART(cudaSetDevice(dev_a));
    fill_kernel<<<(int)((max_bytes / 4 + 255) / 256), 256>>>(buf_a, max_bytes / 4, 1.0f);
    SAFE_CUDART(cudaGetLastError());

    cudaStream_t stream = nullptr;
    SAFE_CUDART(cudaSetDevice(dev_a));
    SAFE_CUDART(cudaStreamCreate(&stream));

    // Warmup
    for (int i = 0; i < 10; i++) {
        SAFE_CUDART(cudaMemcpyPeerAsync(buf_b, dev_b, buf_a, dev_a, 1UL << 20, stream));
    }
    SAFE_CUDART(cudaStreamSynchronize(stream));

    printf("%-12s  %-12s  %-12s\n", "Size_Bytes", "Time_ms", "BW_GB/s");
    for (int i = 0; i < n_sizes; i++) {
        size_t bytes = sizes[i];
        const int iters = (bytes < 1UL << 22) ? 100 : 20;

        cudaEvent_t start, stop;
        SAFE_CUDART(cudaEventCreate(&start));
        SAFE_CUDART(cudaEventCreate(&stop));
        SAFE_CUDART(cudaEventRecord(start, stream));
        for (int j = 0; j < iters; j++) {
            SAFE_CUDART(cudaMemcpyPeerAsync(buf_b, dev_b, buf_a, dev_a, bytes, stream));
        }
        SAFE_CUDART(cudaEventRecord(stop, stream));
        SAFE_CUDART(cudaEventSynchronize(stop));
        float ms = 0.0f;
        SAFE_CUDART(cudaEventElapsedTime(&ms, start, stop));

        float bw = (float)bytes / 1024.0f / 1024.0f / 1024.0f / (ms / 1000.0f / iters);
        printf("%-12zu  %-12.4f  %-12.2f\n", bytes, ms / iters, bw);

        SAFE_CUDART(cudaEventDestroy(start));
        SAFE_CUDART(cudaEventDestroy(stop));
    }

    cudaFree(buf_a);
    cudaFree(buf_b);
    return 0;
}
