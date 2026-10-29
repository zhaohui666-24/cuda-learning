#include <cstdio>
#include <cmath>
#include <cstdlib>

#define CUDA_CHECK(call) do { \
    cudaError_t e=(call); \
    if(e!=cudaSuccess){printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} \
}while(0)

#define TILE 16

__global__ void matmulTiled1D(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y*TILE + ty;
    int col = blockIdx.x*TILE + tx;

    float sum = 0.0f;
    for (int p=0; p<(K+TILE-1)/TILE; p++) {
        // ① 搬 A tile：A[row][p*TILE+tx]，越界填 0
        sA[ty][tx] = (row<M && p*TILE+tx<K) ? A[row*K + p*TILE+tx] : 0.0f;
        // ① 搬 B tile：B[p*TILE+ty][col]，越界填 0
        sB[ty][tx] = (p*TILE+ty<K && col<N) ? B[(p*TILE+ty)*N + col] : 0.0f;
        __syncthreads();              // 第①次

        for (int k=0;k<TILE;k++) sum += sA[ty][k]*sB[k][tx];
        __syncthreads();              // 第②次
    }
    if (row<M && col<N) C[row*N+col] = sum;
}

int main() {
    int M=2048,N=2048,K=2048;
    size_t sA=M*K*4,sB=K*N*4,sC=M*N*4;

    float *hA=(float*)malloc(sA),*hB=(float*)malloc(sB),*hC=(float*)malloc(sC);
    for (int i=0;i<M*K;i++) hA[i]=1.0f;
    for (int i=0;i<K*N;i++) hB[i]=1.0f;

    float *dA,*dB,*dC;
    CUDA_CHECK(cudaMalloc(&dA,sA));
    CUDA_CHECK(cudaMalloc(&dB,sB));
    CUDA_CHECK(cudaMalloc(&dC,sC));
    CUDA_CHECK(cudaMemcpy(dA,hA,sA,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB,hB,sB,cudaMemcpyHostToDevice));

    dim3 block(16,16);
    dim3 grid((N+block.x-1)/block.x,(M+block.y-1)/block.y);

    cudaEvent_t start,stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    matmulTiled1D<<<grid,block>>>(dA,dB,dC,M,N,K);
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(stop);
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms=0;
    cudaEventElapsedTime(&ms,start,stop);
    double gflops = 2.0*M*N*K / (ms/1000.0) / 1e9;
    printf("%.3f ms, %.2f GFLOPS\n", ms, gflops);

    CUDA_CHECK(cudaMemcpy(hC,dC,sC,cudaMemcpyDeviceToHost));
    printf("C[0]=%.0f, C[last]=%.0f (expect %d)\n", hC[0], hC[M*N-1], K);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
    return 0;
}
