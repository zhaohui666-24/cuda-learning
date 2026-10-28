#include <cstdio>
#include <cmath>
#include <cstdlib>

#define CUDA_CHECK(call) do { \
    cudaError_t e=(call); \
    if(e!=cudaSuccess){printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} \
}while(0)

__global__ void matmulNaive(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    if (row<M && col<N) {
        float sum=0.0f;
        for (int k=0;k<K;k++) sum += A[row*K+k]*B[k*N+col];
        C[row*N+col] = sum;
    }
}

int main() {
    int M=256,N=256,K=256;
    size_t sA=M*K*4,sB=K*N*4,sC=M*N*4;

    float *hA=(float*)malloc(sA),*hB=(float*)malloc(sB),
          *hC=(float*)malloc(sC),*hRef=(float*)malloc(sC);
    srand(0);
    for (int i=0;i<M*K;i++) hA[i]=(float)(rand()%100)/10.0f;
    for (int i=0;i<K*N;i++) hB[i]=(float)(rand()%100)/10.0f;

    // CPU 参考
    for (int r=0;r<M;r++)
        for (int c=0;c<N;c++){
            float sum=0;
            for (int k=0;k<K;k++) sum += hA[r*K+k]*hB[k*N+c];
            hRef[r*N+c]=sum;
        }

    float *dA,*dB,*dC;
    CUDA_CHECK(cudaMalloc(&dA,sA));
    CUDA_CHECK(cudaMalloc(&dB,sB));
    CUDA_CHECK(cudaMalloc(&dC,sC));
    CUDA_CHECK(cudaMemcpy(dA,hA,sA,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB,hB,sB,cudaMemcpyHostToDevice));

    dim3 block(16,16);
    dim3 grid((N+block.x-1)/block.x,(M+block.y-1)/block.y);
    matmulNaive<<<grid,block>>>(dA,dB,dC,M,N,K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hC,dC,sC,cudaMemcpyDeviceToHost));

    bool ok=true;
    for (int i=0;i<M*N;i++){
        if (fabs(hC[i]-hRef[i])>1e-2){
            ok=false; printf("Mismatch %d: got %f ref %f\n",i,hC[i],hRef[i]); break;
        }
    }
    printf(ok?"PASSED\n":"FAILED\n");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC); free(hRef);
    return 0;
}
