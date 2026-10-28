#include <cstdio>
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // 线程的全局编号
    if (i < n) c[i] = a[i] + b[i];
}
int main() {
    int n = 1024; size_t s = n * sizeof(float);
    float *ha=(float*)malloc(s), *hb=(float*)malloc(s), *hc=(float*)malloc(s);
    for (int i=0;i<n;i++){ha[i]=i; hb[i]=i*2;}
    float *da,*db,*dc;
    cudaMalloc(&da,s); cudaMalloc(&db,s); cudaMalloc(&dc,s);
    cudaMemcpy(da,ha,s,cudaMemcpyHostToDevice);
    cudaMemcpy(db,hb,s,cudaMemcpyHostToDevice);
    int threads=256, blocks=(n+threads-1)/threads;
    vectorAdd<<<blocks,threads>>>(da,db,dc,n);
    cudaMemcpy(hc,dc,s,cudaMemcpyDeviceToHost);
    bool ok=true;
    for (int i=0;i<n;i++) if(hc[i]!=ha[i]+hb[i]){ok=false;break;}
    printf(ok ? "PASSED\n" : "FAILED\n");
    cudaFree(da); cudaFree(db); cudaFree(dc);
    free(ha); free(hb); free(hc);
    return 0;
}
