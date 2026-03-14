/**
 * 02_reduce_uncoalesced.cu - Reduction with deliberately uncoalesced access.
 *
 * Expected Tachyon findings:
 *   - Memory-bound with poor coalescing efficiency
 *   - High L1 sector-to-request ratio
 *   - Suggestion: use sequential addressing for coalesced access
 */
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define N (1 << 22)
#define BLOCK_SIZE 256

#define CHECK_CUDA(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

__global__ void reduce_uncoalesced(float *data, float *result, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? data[i] : 0.0f;
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;
        if (index < blockDim.x) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    if (tid == 0) atomicAdd(result, sdata[0]);
}

int main() {
    size_t bytes = N * sizeof(float);
    float *h_data = (float *)malloc(bytes);
    for (int i = 0; i < N; ++i) h_data[i] = 1.0f;

    float *d_data, *d_result;
    CHECK_CUDA(cudaMalloc(&d_data, bytes));
    CHECK_CUDA(cudaMalloc(&d_result, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_result, 0, sizeof(float)));

    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    reduce_uncoalesced<<<grid, BLOCK_SIZE>>>(d_data, d_result, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemset(d_result, 0, sizeof(float)));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    reduce_uncoalesced<<<grid, BLOCK_SIZE>>>(d_data, d_result, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("reduce_uncoalesced (%d elements): %.3f ms\n", N, ms);

    float h_result = 0;
    CHECK_CUDA(cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    printf("Sum = %.0f (expected %d)\n", h_result, N);

    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_result));
    free(h_data);
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return 0;
}
