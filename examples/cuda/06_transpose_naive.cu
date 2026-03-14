/**
 * 06_transpose_naive.cu - Naive matrix transpose (uncoalesced writes + bank conflicts).
 *
 * Expected Tachyon findings:
 *   - Memory-bound with poor write coalescing
 *   - High shared memory bank conflicts
 *   - Suggestion: shared memory with padding
 */
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define N 2048
#define BLOCK_SIZE 16

#define CHECK_CUDA(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

__global__ void transpose_naive(const float *in, float *out, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < n && y < n) out[x * n + y] = in[y * n + x];
}

__global__ void transpose_shared_conflicts(const float *in, float *out, int n) {
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE];
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    if (x < n && y < n) tile[threadIdx.y][threadIdx.x] = in[y * n + x];
    __syncthreads();
    x = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    y = blockIdx.x * BLOCK_SIZE + threadIdx.y;
    if (x < n && y < n) out[y * n + x] = tile[threadIdx.x][threadIdx.y];
}

int main() {
    size_t bytes = N * N * sizeof(float);
    float *h_in = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    for (int i = 0; i < N * N; ++i) h_in[i] = (float)i;

    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, bytes));
    CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    transpose_naive<<<grid, block>>>(d_in, d_out, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    transpose_naive<<<grid, block>>>(d_in, d_out, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms_naive = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_naive, start, stop));
    float gb = 2.0f * bytes / 1e9;
    printf("transpose_naive (%dx%d): %.3f ms, %.1f GB/s\n", N, N, ms_naive, gb / (ms_naive / 1000.0f));

    CHECK_CUDA(cudaEventRecord(start));
    transpose_shared_conflicts<<<grid, block>>>(d_in, d_out, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms_shared = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_shared, start, stop));
    printf("transpose_shared_conflicts (%dx%d): %.3f ms, %.1f GB/s\n", N, N, ms_shared, gb / (ms_shared / 1000.0f));

    CHECK_CUDA(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    int errors = 0;
    for (int y = 0; y < N && errors < 5; ++y)
        for (int x = 0; x < N && errors < 5; ++x)
            if (h_out[y * N + x] != h_in[x * N + y]) ++errors;
    printf("Verification: %s\n", errors == 0 ? "PASS" : "FAIL");

    CHECK_CUDA(cudaFree(d_in)); CHECK_CUDA(cudaFree(d_out));
    free(h_in); free(h_out);
    CHECK_CUDA(cudaEventDestroy(start)); CHECK_CUDA(cudaEventDestroy(stop));
    return 0;
}
