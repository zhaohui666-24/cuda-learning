#include <cstdio>
#include <cmath>
#include <cstdlib>

__constant__ float c_gain;
__constant__ float c_bias;

__global__ void transform(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = c_gain * x[i] + c_bias;   // 直接读常量
}

int main() {
    int n = 1024;
    size_t s = n * sizeof(float);
    float *hx=(float*)malloc(s), *hy=(float*)malloc(s);
    for (int i=0;i<n;i++) hx[i]=(float)i;

    float *dx,*dy;
    cudaMalloc(&dx,s); cudaMalloc(&dy,s);
    cudaMemcpy(dx,hx,s,cudaMemcpyHostToDevice);

    float gain=2.0f, bias=3.0f;
    cudaMemcpyToSymbol(c_gain, &gain, sizeof(float));
    cudaMemcpyToSymbol(c_bias, &bias, sizeof(float));

    int threads=256, blocks=(n+threads-1)/threads;
    transform<<<blocks,threads>>>(dx,dy,n);
    cudaMemcpy(hy,dy,s,cudaMemcpyDeviceToHost);

    bool ok=true;
    for (int i=0;i<n;i++){
        if (fabs(hy[i]-(gain*hx[i]+bias))>1e-3){
            ok=false; printf("Error at %d: got %f\n",i,hy[i]); break;
        }
    }
    printf(ok?"PASSED\n":"FAILED\n");

    cudaFree(dx); cudaFree(dy);
    free(hx); free(hy);
    return 0;
}
