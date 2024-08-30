#include <stdio.h>
#include "newsz.h"
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <cub/cub.cuh>
// #include "cuCompactor.cuh"

#include "nvcomp/lz4.hpp"
#include "nvcomp.hpp"
#include "nvcomp/nvcompManagerFactory.hpp"

#define BLKS 40
#define THDS 128
#define FULL_MASK 0xffffffff

__device__ int g_ints;

struct int_predicate
{
    
	__host__ __device__
	bool operator()(const int x)
	{
		return x>0;
	}
};

struct to_copy
{
  __host__ __device__
  bool operator()(const uint8_t x)
  {
    return x==1;
  }
};




__global__ void compress(float *data, float *scales, float *zeropts, int8_t *out){
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float scratchpad[];
    __shared__ float min;
    __shared__ float max;

    typedef cub::BlockReduce<float, THDS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage1;

    float item = data[blockIdx.x*blockDim.x+threadIdx.x];

    float tmax = BlockReduce(temp_storage1).Reduce(item, cub::Max());
    float tmin = BlockReduce(temp_storage1).Reduce(item, cub::Min());
    
    if (threadIdx.x==0)
    {
        max = tmax;
        min = tmin;
    }

    __syncthreads();

    float vrange = max - min;
    float scale = vrange/((2^8) - 1);
    int zeropt = -1*lrintf(min*scale) - (2^7);

    int q_item = lrintf(item/scale) + zeropt;

    // Clamp quantized value
    if(q_item>127)q_item = 127;
    if(q_item <-128)q_item = -128;
    int8_t q_val = (int8_t)(0xff & q_item);
    out[blockIdx.x*blockDim.x+threadIdx.x] = q_val;
    if (threadIdx.x==0)
    {
        scales[blockIdx.x] = scale;
        zeropts[blockIdx.x]= zeropt;
    }
    
}

__global__ void decompress(int8_t *q_data, float *scales, float *zeropts, float *out){
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float scratchpad[];
    __shared__ float min;
    __shared__ float max;

    typedef cub::BlockReduce<float, THDS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage1;

    int8_t q_val = q_data[blockIdx.x*blockDim.x+threadIdx.x];

    out[blockIdx.x*blockDim.x+threadIdx.x] = (q_val - zeropts[bid])*scales[bid];
}

__global__ void p_ints(){
	printf("codebook entries used: %d\n", g_ints);
}

unsigned char* SZ_device_compress(float *data, size_t num_elements, int blocksize, size_t *outsize){
    float *scales, *zeropts;
    int8_t *q_out;
    unsigned char *cmpbytes;
    int num_blocks = num_elements/blocksize;

    cudaMalloc(&scales, sizeof(float)*num_blocks);
    cudaMalloc(&zeropts,sizeof(float)*num_blocks);
    cudaMalloc(&q_out, num_elements);

    using namespace nvcomp;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    const int chunk_size = 1 << 16;
    nvcompType_t data_type = NVCOMP_TYPE_CHAR;

     

    compress<<<num_blocks, blocksize>>>(data, scales, zeropts, q_out);
    cudaDeviceSynchronize();

    LZ4Manager nvcomp_manager{chunk_size, data_type, stream};
    CompressionConfig comp_config = nvcomp_manager.configure_compression(num_elements);

    uint8_t* comp_buffer;
    cudaMalloc(&comp_buffer, comp_config.max_compressed_buffer_size);
    
    nvcomp_manager.compress((const uint8_t *)q_out, comp_buffer, comp_config);

    size_t c_size = nvcomp_manager.get_compressed_output_size(comp_buffer);
    cudaFree(q_out);

    *outsize = sizeof(float)*(num_blocks+num_blocks)+c_size;
    cudaMalloc(&cmpbytes, *outsize);

    cudaMemcpy(cmpbytes, (unsigned char *)scales, sizeof(float)*num_blocks, cudaMemcpyDeviceToDevice);
    cudaMemcpy(cmpbytes+sizeof(float)*num_blocks, (unsigned char *)zeropts, sizeof(float)*num_blocks, cudaMemcpyDeviceToDevice);
    cudaMemcpy(cmpbytes+sizeof(float)*num_blocks+sizeof(float)*num_blocks, comp_buffer, c_size, cudaMemcpyDeviceToDevice);

    float h_firstscale;
    cudaMemcpy(&h_firstscale, cmpbytes, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(scales);
    cudaFree(zeropts);
    cudaFree(comp_buffer);
    return cmpbytes;
}

float* SZ_device_decompress(unsigned char *cmpbytes, size_t num_elements, int blocksize, size_t *cmpsize){
    float *scales, *zeropts;
    uint8_t *q_cmp;
    int8_t *q_vals;
    float *out;
    int num_blocks = num_elements/blocksize;
    size_t c_size = *cmpsize-(2*sizeof(float)*num_blocks);

    float first_val, *d_first;

    cudaMalloc(&d_first, sizeof(float));
    cudaMemcpy((unsigned char *)&first_val, cmpbytes, sizeof(float), cudaMemcpyDeviceToHost);



    cudaMalloc((void **)&scales, sizeof(float)*num_blocks);
    cudaMalloc((void **)&zeropts,sizeof(float)*num_blocks);
    cudaMalloc((void **)&q_cmp, c_size);
    cudaMemcpy((unsigned char *)scales, cmpbytes, sizeof(float)*num_blocks, cudaMemcpyDeviceToDevice);
    
    cudaMemcpy((unsigned char *)zeropts, cmpbytes+sizeof(float)*num_blocks, sizeof(float)*num_blocks, cudaMemcpyDeviceToDevice);
    
    cudaMemcpy(q_cmp, cmpbytes+sizeof(float)*num_blocks+sizeof(float)*num_blocks, c_size, cudaMemcpyDeviceToDevice);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    const int chunk_size = 1 << 16;
    
    
    nvcompType_t data_type = NVCOMP_TYPE_CHAR;

    auto decomp_manager = nvcomp::create_manager(q_cmp, stream);

    nvcomp::DecompressionConfig decomp_config = decomp_manager->configure_decompression((uint8_t *)q_cmp);
    cudaMalloc(&q_vals, num_elements);

    decomp_manager->decompress((uint8_t*)q_vals, (uint8_t*)q_cmp, decomp_config);
    cudaFree(q_cmp);

    cudaMalloc(&out, sizeof(float)*num_elements);

    decompress<<<num_blocks, blocksize>>>(q_vals, scales, zeropts, out);
    cudaDeviceSynchronize();
    
    cudaFree(scales);
    cudaFree(zeropts);
    cudaFree(q_vals);

    return out;
}

int main(int argc, char** argv){
    char oriFilePath[640], outputFilePath[645];
    float* data;
    size_t nbEle;
    if(argc < 3)
    {
		printf("Usage: testfloat_compress_fastmode2 [srcFilePath] [block size] [err bound] [--cuda]\n");
		printf("Example: testfloat_compress_fastmode2 testfloat_8_8_128.dat 64 1E-3 --cuda\n");
		exit(0);
    }

    sprintf(oriFilePath, "%s", argv[1]);
    int blockSize = atoi(argv[2]);
    float errBound = atof(argv[3]);
    nbEle = atoi(argv[4]);

    data = (float*)malloc(sizeof(float)*nbEle);
    sprintf(outputFilePath, "%s.sznew", oriFilePath);

    FILE *in_file;
    in_file = fopen(oriFilePath, "rb");
    
    fread(data, sizeof(float), nbEle, in_file);
    fclose(in_file);
    
    float max = data[0];
    float min = data[0];
    for(int i=0;i<nbEle;i++){
	if(data[i]>=max){
		max = data[i];
	}
	if(data[i]<=min){
		min = data[i];
	}
    }
    errBound = errBound*(max-min);

    // Move to device
    float *d_data;
    unsigned char *cmpbytes;
    size_t outsize;
    cudaMalloc(&d_data, sizeof(float)*nbEle);
    cudaMemcpy(d_data, data, sizeof(float)*nbEle, cudaMemcpyHostToDevice);
    //SZ_device_compress(d_data, nbEle, errBound, blockSize, cmpbytes, &outsize);

    cudaFree(d_data);
    
}
