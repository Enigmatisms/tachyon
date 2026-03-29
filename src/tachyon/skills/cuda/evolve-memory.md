---
tags: memory, bandwidth, latency
mode: evolve
brief: Advanced CUDA Memory & Traffic Optimization
---

# SKILL: cuda_memory_traffic_optimizer

## 1. STRATEGIC ROLE
You are a CUDA Memory Architect. Your mission is to maximize "Effective Bandwidth" and minimize "Memory Wall" stalls. You must ensure that every memory transaction is fully utilized (coalesced) and that data movement is overlapped with computation using advanced tiling and pipelining techniques.

## 2. THE GOLDEN RULES OF ACCESS PATTERNS
* **Global Memory Coalescing:** * **Mechanism:** The GPU fetches memory in 32-byte or 128-byte sectors. If threads in a warp access a contiguous 128-byte range, the LSU (Load/Store Unit) issues a single transaction.
    * **Optimization:** Ensure thread `i` accesses `base_ptr + i`. Switch from **AoS** (Array of Structures) to **SoA** (Structure of Arrays) to align identical fields in memory.
* **Shared Memory (SMEM) Bank Conflicts:**
    * **Mechanism:** SMEM is divided into 32 banks (4 bytes each). If multiple threads in a warp access different addresses in the same bank, the hardware serializes the access.
    * **Action:** Use **Padding**. For a tile of `[32][32]`, declare it as `[32][33]` to shift the bank index of each row. Use **Swizzling** (XOR indexing) for patterns like Transpose to ensure uniform bank distribution.

## 3. TILING & HIERARCHY MANAGEMENT
* **Shared Memory Tiling:** * Break large Global Memory structures into smaller "tiles" that fit into SMEM. 
    * **Pattern:** Collaborative Load -> `__syncthreads()` -> Compute -> `__syncthreads()`. This maximizes L2 cache hits and reduces GMEM pressure.
* **L2 Cache Residency:** * Use `__restrict__` and `const` keywords to hint that data is read-only, allowing the compiler to use the `LDG` (Read-Only Data Cache) path, which bypasses L1 and reduces cache pollution.

## 4. LATENCY HIDING: DOUBLE BUFFERING & PIPELINING
* **Double Buffering:** * Overlap `Tile N+1` loading with `Tile N` computation. This requires two SMEM buffers (or register sets).
    * **Logic:** While the GPU is waiting for the high-latency Global Memory load for the next iteration, the Warp Schedulers can keep the ALUs busy with the current data.
* **Asynchronous Copy (Ampere/Hopper):**
    * Use `cuda::memcpy_async` or `cp.async` PTX instructions. These move data directly from GMEM to SMEM without consuming warp registers, allowing the warp to continue compute immediately.
* **Tensor Memory Accelerator (TMA - sm_90a):**
    * For Hopper, offload multidimensional tiling and address generation to the hardware TMA engine. This completely decouples data movement from the execution pipe.

## 5. DIAGNOSTIC WORKFLOW (NCU METRICS)
| Metric | Diagnosis | Optimization Strategy |
| :--- | :--- | :--- |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.avg.pct_of_peak` | Uncoalesced Loads | Align memory; Refactor to SoA. |
| `l1tex__data_bank_conflicts.avg` | SMEM Bank Conflicts | Add padding `[N+1]`; Use Swizzling. |
| `gpu__compute_memory_throughput.avg` | Bandwidth Bound | Optimize for reuse (Tiling); Use Vectorized Loads (`float4`). |
| `l1tex__t_bytes_pipe_lsu_mem_global_op_st.avg` | Redundant Stores | Cache intermediate results in Registers or SMEM. |

## 6. VECTORIZED MEMORY TRANSACTIONS
* **Optimization:** Use `float4`, `uint4`, or `int4` to perform 128-bit loads/stores.
* **Impact:** This reduces the number of instructions the LSU must track and maximizes the utilization of the memory bus.
* **Constraint:** Pointers MUST be 16-byte aligned. Use `__builtin_assume_aligned` or manual alignment checks to avoid `Illegal Address` errors.

## 7. ROBUSTNESS & SAFETY
* **Sync Hazards:** When using `cp.async`, you must use a barrier (e.g., `asm volatile("cp.async.wait_all");` or `cuda::pipeline`) to ensure data is present before use.
* **Alignment:** Always handle the "tail" of the data (where `N % 4 != 0`) when using vectorized loads to prevent out-of-bounds (OOB) memory access.
* **Volatile Consistency:** If using SMEM for flag-based synchronization, mark flags as `volatile` to prevent the compiler from caching them in registers.