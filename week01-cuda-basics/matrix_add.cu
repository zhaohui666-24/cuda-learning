#include <stdio.h>
#include <cstdio>
__global__ void matrixAdd(const float* A, const float* B, float* C,
                          int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < height && col < width) {   // 两个方向都要做边界检查
        int idx = row * width + col;
        C[idx] = A[idx] + B[idx];
    }
}
int main() {
    int width= 5; 
    int height= 8;
    int n = width * height;
    size_t s = n * sizeof(float);
    float *ha=(float*)malloc(s), *hb=(float*)malloc(s), *hc=(float*)malloc(s);
    for (int i=0;i<n;i++){ha[i]=i; hb[i]=i*2; hc[i]=0;}
    float *da,*db,*dc;
    cudaMalloc(&da,s); cudaMalloc(&db,s); cudaMalloc(&dc,s);
    cudaMemcpy(da,ha,s,cudaMemcpyHostToDevice);
    cudaMemcpy(db,hb,s,cudaMemcpyHostToDevice);
    cudaMemcpy(dc,hc,s,cudaMemcpyHostToDevice);
    dim3 block(2, 2);
    dim3 grid((width + block.x - 1)/block.x, (height + block.y - 1)/block.y);
    matrixAdd<<<grid, block>>>(da, db, dc, width, height);
    cudaMemcpy(hc,dc,s,cudaMemcpyDeviceToHost);
    bool ok=true;
    for (int r=0;r<height;r++){
        for (int c=0;c<width;c++){
            int idx = r*width+c;
            if (fabs(hc[idx] - (ha[idx]+hb[idx])) > 1e-5) {
                ok = false;  // 这行才是"检查"
                printf("Error at row %d, col %d: %.0f + %.0f != %.0f\n", r, c, ha[idx], hb[idx], hc[idx]);
                break;
            }
            printf("%.0f + %.0f = %.0f\n", ha[idx], hb[idx], hc[idx]);
        }
    }

    cudaFree(da); cudaFree(db); cudaFree(dc);
    free(ha); free(hb); free(hc);
    printf(ok ? "PASSED\n" : "FAILED\n");
    return 0;
}



