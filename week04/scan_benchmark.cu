#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call) do { \
    cudaError_t e=(call); \
    if(e!=cudaSuccess){printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} \
}while(0)

#define BLOCK 256

// Hillis-Steele exclusive scan（block 内）
__global__ void hsScan(const float* input, float* output, int n) {
    __shared__ float s[BLOCK];
    int tid=threadIdx.x;
    int gid=blockIdx.x*BLOCK+tid;
    if(tid==0) s[0]=0.0f;
    else       s[tid]=(gid-1<n)?input[gid-1]:0.0f;
    __syncthreads();
    for(int d=1;d<BLOCK;d<<=1){
        float v=(tid>=d)?s[tid-d]:0.0f;
        __syncthreads();
        s[tid]+=v;
        __syncthreads();
    }
    if(gid<n) output[gid]=s[tid];
}

// Brent-Kung exclusive scan（block 内，用 BLOCK 而非 n）
__global__ void bkScan(const float* input, float* output, int n) {
    __shared__ float x[BLOCK];
    int tid=threadIdx.x;
    int gid=blockIdx.x*BLOCK+tid;
    x[tid]=(gid<n)?input[gid]:0.0f;
    __syncthreads();
    for(int d=1;d<BLOCK;d*=2){
        if(tid%(2*d)==0) x[tid+2*d-1]+=x[tid+d-1];
        __syncthreads();
    }
    if(tid==0) x[BLOCK-1]=0.0f;
    __syncthreads();
    for(int d=BLOCK/2;d>=1;d/=2){
        if(tid%(2*d)==0){
            float t=x[tid+d-1];
            x[tid+d-1]=x[tid+2*d-1];
            x[tid+2*d-1]=t+x[tid+2*d-1];
        }
        __syncthreads();
    }
    if(gid<n) output[gid]=x[tid];
}

float timeIt(void(*launch)(const float*,float*,int),const float*dA,float*dOut,int n,int iters){
    cudaEvent_t s,e;
    cudaEventCreate(&s);cudaEventCreate(&e);
    for(int i=0;i<3;i++) launch(dA,dOut,n);
    cudaEventRecord(s);
    for(int i=0;i<iters;i++) launch(dA,dOut,n);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms;cudaEventElapsedTime(&ms,s,e);
    cudaEventDestroy(s);cudaEventDestroy(e);
    return ms/iters;
}

// 用函数指针不方便传 kernel，改成宏
#define TIME(kernel,dA,dOut,n,iters) ({ \
    cudaEvent_t _s,_e; cudaEventCreate(&_s);cudaEventCreate(&_e); \
    for(int i=0;i<3;i++) kernel<<<(n+BLOCK-1)/BLOCK,BLOCK>>>(dA,dOut,n); \
    cudaEventRecord(_s); \
    for(int i=0;i<iters;i++) kernel<<<(n+BLOCK-1)/BLOCK,BLOCK>>>(dA,dOut,n); \
    cudaEventRecord(_e); cudaEventSynchronize(_e); \
    float _ms;cudaEventElapsedTime(&_ms,_s,_e); \
    cudaEventDestroy(_s);cudaEventDestroy(_e); _ms/iters; })

int main(){
    int sizes[]={256,4096,1048576,16777216};
    int iters[]={1000,500,50,5};
    for(int si=0;si<4;si++){
        int n=sizes[si];
        size_t bytes=n*sizeof(float);
        float*hA=(float*)malloc(bytes);
        for(int i=0;i<n;i++)hA[i]=(float)(i%7);
        float*dA,*dOut;
        cudaMalloc(&dA,bytes);cudaMalloc(&dOut,bytes);
        cudaMemcpy(dA,hA,bytes,cudaMemcpyHostToDevice);
        int blocks=(n+BLOCK-1)/BLOCK;
        float hs=TIME(hsScan,dA,dOut,n,iters[si]);
        float bk=TIME(bkScan,dA,dOut,n,iters[si]);
        printf("n=%9d blocks=%6d | Hillis-Steele %.4f ms | Brent-Kung %.4f ms | ratio BK/HS %.2f\n",
               n,blocks,hs,bk,bk/hs);
        cudaFree(dA);cudaFree(dOut);free(hA);
    }
    return 0;
}
