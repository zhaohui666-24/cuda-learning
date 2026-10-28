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
...
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

## 编译运行

```bash
nvcc -o vectorAdd vectorAdd.cu
./vectorAdd
```
