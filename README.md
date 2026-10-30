# CUDA / GPU 编程学习

24 周 GPU 编程学习记录，目标方向：**AI 芯片 / GPU 架构**。

## 环境

- GPU：NVIDIA RTX 5060 Ti 16GB（Blackwell，sm_120）
- 平台：Windows + WSL2 Ubuntu
- CUDA：13.x

## 目录结构

按周组织，每周一个目录：

```
week01-cuda-basics/   # CUDA 基础与线程模型
week02-cuda-basics/   # 内存层次、shared memory、GEMM（含 W03 tiling 与 bank conflict）
gemm-benchmark.html   # W03 GEMM 三版本性能对比复盘
```

## 第一周：CUDA 基础与线程模型

| 程序 | 主题 | 核心知识点 |
|---|---|---|
| `vectorAdd.cu` | 向量加法 | 异构计算、host/device、5 步骨架 |
| `saxpy.cu` | SAXPY | kernel 结构、参数传递 |
| `thread_info.cu` | 线程坐标 | grid/block/thread 三层模型、内置变量 |
| `matrix_add.cu` | 矩阵加法 | 二维 grid/block、行主序映射、边界检查 |
| `const_transform.cu` | 常量变换 | constant memory、`__constant__` |

**核心概念：**

- 异构编程 5 步：分配 device 内存 → 拷 H2D → 启动 kernel → 拷 D2H → 释放
- 线程层次：grid → block → thread
- 四个内置变量：`gridDim`、`blockDim`（尺寸）；`blockIdx`、`threadIdx`（坐标）
- 一维全局编号：`i = blockIdx.x * blockDim.x + threadIdx.x`
- 二维映射：
  - `col = blockIdx.x * blockDim.x + threadIdx.x`
  - `row = blockIdx.y * blockDim.y + threadIdx.y`
  - `idx = row * width + col`（行主序）
- 内存类型：
  - **global**：读写、容量大、相对慢
  - **constant**：只读、带缓存、64KB、支持广播
  - **shared**：片上、block 内共享（第二周）

## 第二周：内存层次、shared memory 与 GEMM

| 程序 | 主题 | 核心知识点 |
|---|---|---|
| `shared_reverse.cu` | 块内数组反转 | shared block 私有、块内局部下标、`__syncthreads` |
| `timed_vectoradd.cu` | 事件计时 | `cudaEvent`、CUDA_CHECK 宏 |
| `matmul_naive.cu` / `matmul_bench.cu` | naive GEMM | 每线程一元素、内层 K 累加、Roofline |

**核心概念：**

- 内存层次（快→慢）：寄存器 → local/shared → L2 → global；
  - 寄存器：单线程私有，最快；
  - shared：block 共享、片上 SRAM；
  - local：线程私有、在显存（被动溢出），不是独立器件；
  - global：片外 GDDR/HBM，容量大、慢；
  - constant：数据在显存但有片上缓存，只读、支持广播；
- GEMM 总 FLOP = `2MNK`（一条 FMA 算乘+加 = 2 FLOP）；
- naive 2048³ 实测 1257 GFLOPS（5.3%），完全 memory-bound。

## 第三周：GEMM Tiling 与 Bank Conflict

| 程序 | 主题 | 核心知识点 |
|---|---|---|
| `matmul_tiled_1d.cu` | 1D tiling | block 包一个 tile、沿 K 分阶段、两次 `__syncthreads` |
| `matmul_tiled_2d.cu` | 2D register blocking | 每线程 2×2、寄存器累加、线性 tid 搬运 |
| `padding_conflict.cu` | 冲突版（行宽 32） | 同列全挤 bank0，32-way 冲突 |
| `padding_noconflict.cu` | padding 版（行宽 33） | 同列错开，冲突归零 |

**性能对比（2048³，理论 23685 GFLOPS）：**

| 版本 | 耗时 | GFLOPS | 占理论算力 |
|---|---|---|---|
| naive | 13.667 ms | 1257 | 5.3% |
| 1D tiling | 10.611 ms | 1619 | 6.8% |
| 2D register blocking | 3.713 ms | 4627 | 19.5% |

**核心概念：**

- **tiling**：block 负责 C 的一个 tile，沿 K 分阶段，把 global 数据搬进 shared 复用；
- **算术强度** = `BM×BN×2 / [(BM+BN)×4]`，TK 会被约掉（只影响阶段数、同步次数、shared 占用）；
- **register blocking**：每线程算 2×2 输出、用寄存器累加，shared 数据被反复复用，是提升最大的一步；
- **bank conflict**：shared 分 32 个 bank（bank = 地址%32），同 bank 不同地址串行；
- **padding**：行宽 32→33（只多 128 字节），让同列错开，ncu 冲突 31→0；
- ncu 指标：`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum`（写）/ `_ld.sum`（读）。

详见 `gemm-benchmark.html`。

## 编译运行

```bash
nvcc -o vectorAdd vectorAdd.cu
./vectorAdd
```
