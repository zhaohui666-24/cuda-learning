#include <cstdio>
__global__ void saxpy(float a, const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
int main() {
    int n = 1024; size_t s = n * sizeof(float);
    float a = 2.0;
    float *ha=(float*)malloc(s), *hb=(float*)malloc(s);
    float *hta=(float*)malloc(s), *htb=(float*)malloc(s);
    for (int i=0;i<n;i++){ha[i]=i; hb[i]=i*2; hta[i]=ha[i]; htb[i]=hb[i];}
    float *da,*db;
    cudaMalloc(&da,s); cudaMalloc(&db,s);
    cudaMemcpy(da,ha,s,cudaMemcpyHostToDevice);
    cudaMemcpy(db,hb,s,cudaMemcpyHostToDevice);
    int threads=256, blocks=(n+threads-1)/threads;
    saxpy<<<blocks,threads>>>(a,da,db,n);
    cudaMemcpy(hb,db,s,cudaMemcpyDeviceToHost);
    bool ok=true;
    for (int i=0;i<n;i++) if(hb[i]!= a * hta[i]+htb[i]){printf("Error at index %d\n", i); ok=false;break;}
    printf(ok ? "PASSED\n" : "FAILED\n");
    cudaFree(da); cudaFree(db);
    free(ha); free(hb); free(hta); free(htb);
    return 0;
}