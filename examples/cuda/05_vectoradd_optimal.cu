/**
 * 05_vectoradd_optimal.cu - Well-optimized vector addition (baseline).
 *
 * Expected Tachyon findings:
 *   - Memory-bound (high DRAM throughput, low SM throughput)
 *   - Good coalescing efficiency
 *   - Useful as "before" report for diff comparison
 */
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define N (1 << 24)
#define BLOCK_SIZE 256

#define CHECK_CUDA(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

__global__ void vector_add(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    size_t bytes = N * sizeof(float);
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);
    for (int i = 0; i < N; ++i) { h_a[i] = (float)i; h_b[i] = (float)(N - i); }

    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_c, bytes));
    CHECK_CUDA(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vector_add<<<grid, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    vector_add<<<grid, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    float gb = 3.0f * bytes / 1e9;
    printf("vector_add (%d elements): %.3f ms, %.1f GB/s\n", N, ms, gb / (ms / 1000.0f));

    CHECK_CUDA(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    printf("c[0]=%f, c[N-1]=%f (all should be %d)\n", h_c[0], h_c[N - 1], N);

    CHECK_CUDA(cudaFree(d_a)); CHECK_CUDA(cudaFree(d_b)); CHECK_CUDA(cudaFree(d_c));
    free(h_a); free(h_b); free(h_c);
    CHECK_CUDA(cudaEventDestroy(start)); CHECK_CUDA(cudaEventDestroy(stop));
    return 0;
}
