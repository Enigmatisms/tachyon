/**
 * 04_histogram_atomic.cu - Histogram via global atomicAdd (high contention).
 *
 * Expected Tachyon findings:
 *   - Latency-bound due to atomic contention
 *   - High warp stall counts
 *   - Suggestion: use shared memory privatization
 */
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define N (1 << 24)
#define NUM_BINS 256
#define BLOCK_SIZE 256

#define CHECK_CUDA(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

__global__ void histogram_atomic(const unsigned char *data, unsigned int *bins, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride) {
        atomicAdd(&bins[data[i]], 1);
    }
}

int main() {
    unsigned char *h_data = (unsigned char *)malloc(N);
    for (int i = 0; i < N; ++i) h_data[i] = (unsigned char)(rand() % 32);

    unsigned int h_bins[NUM_BINS] = {0};
    unsigned char *d_data; unsigned int *d_bins;
    CHECK_CUDA(cudaMalloc(&d_data, N));
    CHECK_CUDA(cudaMalloc(&d_bins, NUM_BINS * sizeof(unsigned int)));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, N, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_bins, 0, NUM_BINS * sizeof(unsigned int)));

    int grid = 1024;

    histogram_atomic<<<grid, BLOCK_SIZE>>>(d_data, d_bins, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemset(d_bins, 0, NUM_BINS * sizeof(unsigned int)));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    histogram_atomic<<<grid, BLOCK_SIZE>>>(d_data, d_bins, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("histogram_atomic (%d elements, %d bins): %.3f ms\n", N, NUM_BINS, ms);

    CHECK_CUDA(cudaMemcpy(h_bins, d_bins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    unsigned long total = 0;
    for (int i = 0; i < NUM_BINS; ++i) total += h_bins[i];
    printf("Total count = %lu (expected %d)\n", total, N);

    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_bins));
    free(h_data);
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return 0;
}
