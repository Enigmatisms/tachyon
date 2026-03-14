# CUDA 示例程序

6 个 CUDA 小程序，涵盖常见的性能模式和反模式。每个都是独立的单文件，拿来给 Tachyon 跑 profiling 刚好能看到有意思的分析结果。但我个人觉得目前这些例子还是太 toy 了，生产环境的例子比这个难上百倍。

## 编译

```bash
make all

# 指定 GPU 架构
make all ARCH="-arch=sm_80"
```

需要 `nvcc`，Makefile 会自动检测路径。

## 程序列表

| 文件 | 作用 | 预期瓶颈 |
|------|---------|----------|
| `01_matmul_naive.cu` | 512x512 朴素矩阵乘 | 计算瓶颈，大量冗余 global load |
| `02_reduce_uncoalesced.cu` | 交叉寻址的 reduction | 访存不连续，sector/request 比高 |
| `03_stencil_2d.cu` | 1024x1024 五点 stencil | 计算+访存混合，roofline 分析 |
| `04_histogram_atomic.cu` | 全局 atomicAdd 直方图 | 延迟瓶颈，atomic 竞争严重 |
| `05_vectoradd_optimal.cu` | 16M 向量加法（优化版） | 纯访存瓶颈，coalescing 良好 |
| `06_transpose_naive.cu` | 朴素转置 + shared memory 版 | 写入不连续，bank conflict |

## 配合 Tachyon 使用

```bash
# profile 单个程序
tachyon profile ./01_matmul_naive

# 前后对比
tachyon profile ./06_transpose_naive --output ./before
# （改完代码后）
tachyon profile ./06_transpose_optimized --output ./after
tachyon diff ./before/report.ncu-rep ./after/report.ncu-rep

# AI 交互分析
tachyon chat --report ./report.ncu-rep

# 纯规则分析（不需要 LLM）
tachyon profile ./02_reduce_uncoalesced --no-ai
```
