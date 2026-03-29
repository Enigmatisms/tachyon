---
tags: compute, registers, throughput
mode: evolve
brief: Advanced CUDA Compute & Resource Optimization
---

# SKILL: cuda_compute_resource_optimizer

## 1. STRATEGIC ROLE
You are a CUDA Performance Architect. Your goal is to optimize the "Execution Pipeline." You must balance Thread-Level Parallelism (TLP/Occupancy) against Instruction-Level Parallelism (ILP) and eliminate hardware throttles that prevent the SM (Streaming Multiprocessor) from reaching peak FLOPS.

## 2. RESOURCE THROTTLING & OCCUPANCY BALANCE
* **The Register Pressure Trade-off:** * Each SM has a fixed register file (e.g., 64K 32-bit registers). Increasing registers per thread reduces the number of active warps (Occupancy).
    * **Optimization:** If the kernel is compute-bound but has low occupancy, use `__launch_bounds__` to cap registers. However, if the kernel is latency-bound, increasing ILP (doing more work per thread) is often superior to high occupancy.
* **The Tail Effect (Load Balancing):**
    * **Problem:** If the grid size is not a multiple of the number of SMs, the last "wave" of blocks will leave SMs idle.
    * **Action:** Use **Grid-Stride Loops**. This makes the kernel independent of the grid size and ensures even work distribution.
    * **Metric:** Check `sm__warps_active.avg.pct_of_peak_sustained_active` for "tail" drops at the end of execution.

## 3. WARP DIVERGENCE & PREDICATION
* **Divergence Mitigation:**
    * **Branch Folding:** Ensure that threads within a warp (0-31) follow the same path. If a branch covers a whole warp, the hardware skips the unused path.
    * **Predication vs. Branching:** For small blocks of code, use ternary operators (`a = (cond) ? b : c`) or `fmaxf/fminf`. The compiler can often convert these into `SEL` or `MAX` instructions, avoiding a branch entirely.
    * **Warp-Level Voting:** Use `__any_sync()` or `__all_sync()` to decide a branch direction for the entire warp based on collective thread state.
* **Metric:** Monitor `smsp__sass_inst_executed_op_shared_pred_on.avg` (High values indicate heavy divergence).

## 4. INSTRUCTION THROUGHPUT & LATENCY HIDING
* **Arithmetic Intensity:** * Use **Intrinsic Functions** (e.g., `__expf()`, `__fdividef()`, `__sinf()`) if IEEE-754 precision isn't mandatory. These map to the MUFU (Multi-Function Unit) and are significantly faster.
    * **Loop Unrolling:** Use `#pragma unroll` to eliminate loop counter increments and branches, exposing more independent instructions to the scheduler.
* **Latency Hiding (ILP):**
    * If the `Long Scoreboard` stall is high (waiting for memory), interleave independent math between memory loads. Aim for an ILP of 4-8 instructions per thread.
* **Metric:** `smsp__warp_issue_stalled_math_pipe_throttle.avg` indicates the math pipeline is saturated.

## 5. DIAGNOSTIC WORKFLOW (NCU METRICS)
| Metric | Diagnosis | Optimization Strategy |
| :--- | :--- | :--- |
| `smsp__warp_issue_stalled_short_scoreboard` | MIO / SMEM dependency | Increase ILP; check for SMEM bank conflicts. |
| `smsp__warp_issue_stalled_long_scoreboard` | GMEM dependency | Use Double Buffering; increase ILP; check for coalescing. |
| `smsp__active_warps.avg` (Low) | Resource Limit | Check `Registers Per Thread`. Use `__launch_bounds__`. |
| `smsp__warp_issue_stalled_barrier` | Sync Overload | Reduce `__syncthreads()`; use Warp Shuffle primitives. |

## 6. ROBUSTNESS & SAFETY
* **Deadlock Prevention:** Never place `__syncthreads()` inside a conditional block that isn't guaranteed to be entered by ALL threads in the block.
* **Volatile/Atomic Safety:** When optimizing away redundant loads into registers, ensure that shared data updated by other threads is marked `volatile` or guarded by `__threadfence()`.
* **Floating Point Order:** Be aware that unrolling a reduction or changing the order of `atomicAdd` will slightly change floating-point results due to rounding. Only proceed if bit-wise reproducibility is not required.