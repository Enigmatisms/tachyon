/**
 * 03_stencil_2d.cu - 2D 5-point stencil computation.
 *
 * Expected Tachyon findings:
 *   - Mixed compute+memory characteristics
 *   - Interesting roofline position
 *   - Potential for shared memory optimization (halo region)
 */
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define WIDTH 1024
#define HEIGHT 1024
#define BLOCK_X 16
#define BLOCK_Y 16

#define CHECK_CUDA(call) do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

__global__ void stencil_2d(const float *in, float *out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= 1 && x < w - 1 && y >= 1 && y < h - 1) {
        int idx = y * w + x;
        out[idx] = 0.2f * (in[idx] + in[idx - 1] + in[idx + 1] + in[idx - w] + in[idx + w]);
    }
}

int main() {
    size_t bytes = WIDTH * HEIGHT * sizeof(float);
    float *h_in = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);
    for (int i = 0; i < WIDTH * HEIGHT; ++i) h_in[i] = (float)(rand() % 1000) / 1000.0f;

    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, bytes));
    CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((WIDTH + BLOCK_X - 1) / BLOCK_X, (HEIGHT + BLOCK_Y - 1) / BLOCK_Y);

    stencil_2d<<<grid, block>>>(d_in, d_out, WIDTH, HEIGHT);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int iter = 0; iter < 10; ++iter) {
        stencil_2d<<<grid, block>>>(d_in, d_out, WIDTH, HEIGHT);
        float *tmp = d_in; d_in = d_out; d_out = tmp;
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("stencil_2d (%dx%d, 10 iters): %.3f ms (%.3f ms/iter)\n", WIDTH, HEIGHT, ms, ms / 10.0f);

    CHECK_CUDA(cudaMemcpy(h_out, d_in, bytes, cudaMemcpyDeviceToHost));
    printf("out[512][512] = %.6f\n", h_out[512 * WIDTH + 512]);

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    free(h_in); free(h_out);
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return 0;
}
