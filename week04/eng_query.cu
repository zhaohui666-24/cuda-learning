#include <cstdio>
int main(){
    int dev; cudaGetDevice(&dev);
    cudaDeviceProp p; cudaGetDeviceProperties(&p,dev);
    printf("Device: %s\n", p.name);
    printf("asyncEngineCount: %d\n", p.asyncEngineCount);
    printf("concurrentKernels: %d\n", p.concurrentKernels);
    return 0;
}
