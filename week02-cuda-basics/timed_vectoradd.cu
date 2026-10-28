#include <cstdio>
#include <cmath>
#include <cstdlib>

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i<n) c[i] = a[i]+b[i];
}

int main() {
    int n = 10 << 20;              // 10M 元素
    size_t s = n*sizeof(float);

    float *ha=(float*)malloc(s), *hb=(float*)malloc(s), *hc=(float*)malloc(s);
    for (int i=0;i<n;i++){ha[i]=(float)i; hb[i]=(float)(i*2);}

    float *da,*db,*dc;
    CUDA_CHECK(cudaMalloc(&da,s));
    CUDA_CHECK(cudaMalloc(&db,s));
    CUDA_CHECK(cudaMalloc(&dc,s));
    CUDA_CHECK(cudaMemcpy(da,ha,s,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db,hb,s,cudaMemcpyHostToDevice));

    int threads=256, blocks=(n+threads-1)/threads;

    cudaEvent_t start,stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    vectorAdd<<<blocks,threads>>>(da,db,dc,n);
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(stop);
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms=0;
    cudaEventElapsedTime(&ms,start,stop);
    printf("Kernel time: %.3f ms\n", ms);

    CUDA_CHECK(cudaMemcpy(hc,dc,s,cudaMemcpyDeviceToHost));
    bool ok=true;
    for (int i=0;i<n;i++){
        if (fabs(hc[i]-(ha[i]+hb[i]))>1e-2){ok=false; printf("Error %d\n",i); break;}
    }
    printf(ok?"PASSED\n":"FAILED\n");

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(da); cudaFree(db); cudaFree(dc);
    free(ha); free(hb); free(hc);
    return 0;
}
