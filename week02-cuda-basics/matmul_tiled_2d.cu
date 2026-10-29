#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do { \
    cudaError_t e=(call); \
    if(e!=cudaSuccess){printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} \
}while(0)

#define BM 32          // block 算 C 的行数
#define BN 32          // block 算 C 的列数
#define TK 16          // 每阶段 K 的宽度
#define RM 2           // 每线程算 2 行
#define RN 2           // 每线程算 2 列

__global__ void matmulTiled2D(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    __shared__ float sA[BM][TK];   // 32×16
    __shared__ float sB[TK][BN];   // 16×32

    int ty = threadIdx.y, tx = threadIdx.x;   // 16×16 = 256 线程
    int tid = ty*blockDim.x + tx;

    int row0 = blockIdx.y*BM + 2*ty;          // 输出块行起始
    int col0 = blockIdx.x*BN + 2*tx;          // 输出块列起始

    float acc[RM][RN] = {{0,0},{0,0}};

    for (int p=0; p<(K+TK-1)/TK; p++) {
        // —— 加载 sA：32×16=512 元素，256 线程，每线程 2 个 ——
        #pragma unroll
        for (int i=0;i<2;i++){
            int id = i*256 + tid;
            int ar = id/16, ak = id%16;
            int gr = blockIdx.y*BM+ar, gc = p*TK+ak;
            sA[ar][ak] = (gr<M && gc<K) ? A[gr*K+gc] : 0.0f;
        }
        // —— 加载 sB：16×32=512 元素，每线程 2 个 ——
        #pragma unroll
        for (int i=0;i<2;i++){
            int id = i*256 + tid;
            int bk = id/32, bc = id%32;
            int gr = p*TK+bk, gc = blockIdx.x*BN+bc;
            sB[bk][bc] = (gr<K && gc<N) ? B[gr*N+gc] : 0.0f;
        }
        __syncthreads();

        // —— 每线程算 2×2 输出块 ——
        #pragma unroll
        for (int i=0;i<RM;i++)
            #pragma unroll
            for (int j=0;j<RN;j++)
                for (int k=0;k<TK;k++)
                    acc[i][j] += sA[2*ty+i][k] * sB[k][2*tx+j];
        __syncthreads();
    }

    #pragma unroll
    for (int i=0;i<RM;i++)
        #pragma unroll
        for (int j=0;j<RN;j++){
            int r=row0+i, c=col0+j;
            if (r<M && c<N) C[r*N+c]=acc[i][j];
        }
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
    dim3 grid((N+BN-1)/BN, (M+BM-1)/BM);

    cudaEvent_t start,stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    matmulTiled2D<<<grid,block>>>(dA,dB,dC,M,N,K);
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
