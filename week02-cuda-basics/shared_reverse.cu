#include <cstdio>
#include <cmath>
#include <cstdlib>

__global__ void blockReverse(const float* input, float* output) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    sdata[tid] = input[gid];    // 1. 搬进 shared
    __syncthreads();            // 2. 等大家都搬完

    int rev = blockDim.x - 1 - tid;
    output[gid] = sdata[rev];   // 3. 读"对称线程"写的数据
}


int main() {
    int perBlock = 256, numBlocks = 2;
    int n = perBlock * numBlocks;
    size_t s = n * sizeof(float);

    float *hin=(float*)malloc(s), *hout=(float*)malloc(s);
    for (int i=0;i<n;i++) hin[i]=(float)i;

    float *din,*dout;
    cudaMalloc(&din,s); cudaMalloc(&dout,s);
    cudaMemcpy(din,hin,s,cudaMemcpyHostToDevice);

    blockReverse<<<numBlocks, perBlock>>>(din, dout);
    cudaMemcpy(hout,dout,s,cudaMemcpyDeviceToHost);

    bool ok=true;
    for (int b=0;b<numBlocks;b++){
        for (int t=0;t<perBlock;t++){
            int gid=b*perBlock+t;
            float expect = (float)(b*perBlock + (perBlock-1-t));  // 块内对称位置
            if (fabs(hout[gid]-expect)>1e-3){
                ok=false; printf("Error gid=%d got %f expect %f\n",gid,hout[gid],expect); break;
            }
        }
    }
    printf(ok?"PASSED\n":"FAILED\n");

    cudaFree(din); cudaFree(dout);
    free(hin); free(hout);
    return 0;
}
