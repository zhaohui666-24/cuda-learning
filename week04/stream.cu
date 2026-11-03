#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do { \
    cudaError_t e=(call); \
    if(e!=cudaSuccess){printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} \
}while(0)

__global__ void saxpy(float a, const float* x, float* y, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i<n) y[i] = a*x[i] + y[i];
}

int main() {
    int N = 1<<22;                 // 4M 元素
    int CHUNKS = 4;
    int chunk = N/CHUNKS;
    size_t full = N*sizeof(float);
    float a = 2.0f;

    // pinned host 内存
    float *hX, *hY, *hYref;
    CUDA_CHECK(cudaMallocHost(&hX,full));
    CUDA_CHECK(cudaMallocHost(&hY,full));
    CUDA_CHECK(cudaMallocHost(&hYref,full));
    for (int i=0;i<N;i++){hX[i]=(float)(i%100); hY[i]=1.0f; hYref[i]=1.0f;}

    float *dX,*dY;
    CUDA_CHECK(cudaMalloc(&dX,full));
    CUDA_CHECK(cudaMalloc(&dY,full));

    cudaEvent_t s,e;
    cudaEventCreate(&s);cudaEventCreate(&e);

    // —— 单流（默认流，全串行）——
    cudaEventRecord(s);
    cudaMemcpy(dX,hX,full,cudaMemcpyHostToDevice);
    cudaMemcpy(dY,hY,full,cudaMemcpyHostToDevice);
    saxpy<<<(N+255)/256,256>>>(a,dX,dY,N);
    cudaMemcpy(hYref,dY,full,cudaMemcpyDeviceToHost);
    cudaEventRecord(e);cudaEventSynchronize(e);
    float ms1;cudaEventElapsedTime(&ms1,s,e);
    // —— 多流（4 个 stream，异步重叠）——
    cudaStream_t streams[CHUNKS];
    for (int i=0;i<CHUNKS;i++) cudaStreamCreate(&streams[i]);

    cudaEventRecord(s);
    for (int c=0;c<CHUNKS;c++){
        int off=c*chunk;
        size_t b=chunk*sizeof(float);
        cudaMemcpyAsync(dX+off,hX+off,b,cudaMemcpyHostToDevice,streams[c]);
        cudaMemcpyAsync(dY+off,hY+off,b,cudaMemcpyHostToDevice,streams[c]);
        saxpy<<<(chunk+255)/256,256,0,streams[c]>>>(a,dX+off,dY+off,chunk);
        cudaMemcpyAsync(hY+off,dY+off,b,cudaMemcpyDeviceToHost,streams[c]);
    }
    for (int i=0;i<CHUNKS;i++) cudaStreamSynchronize(streams[i]);
    cudaEventRecord(e);cudaEventSynchronize(e);
    float ms2;cudaEventElapsedTime(&ms2,s,e);

    // 验证多流结果
    bool ok=true;
    for (int i=0;i<N;i++) if(hY[i]!=hYref[i]){ok=false;break;}

    printf("Single stream : %.3f ms\n", ms1);
    printf("Multi stream  : %.3f ms  (speedup %.2fx)  %s\n", ms2, ms1/ms2, ok?"OK":"WRONG");

    for (int i=0;i<CHUNKS;i++) cudaStreamDestroy(streams[i]);
    cudaFreeHost(hX);cudaFreeHost(hY);cudaFreeHost(hYref);
    cudaFree(dX);cudaFree(dY);
    return 0;
}
