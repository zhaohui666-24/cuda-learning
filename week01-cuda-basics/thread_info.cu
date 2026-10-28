#include <cstdio>

__global__ void printInfo() {
    printf("blockIdx=%d, threadIdx=%d, globalId=%d\n",
           blockIdx.x, threadIdx.x,
           blockIdx.x * blockDim.x + threadIdx.x);
}

int main() {
    printInfo<<<2, 16>>>();
    cudaDeviceSynchronize();   // 必须等 GPU 打印完，否则程序就退出了
    return 0;
}
