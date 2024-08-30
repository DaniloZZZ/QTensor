#include "cuSZp_entry.h"
#include "cuSZp_timer.h"
#include "cuSZp_utility.h"
#include "cuSZp.h"


extern "C"{
    /** Before entering SZp_compress, must allocate on device:
     * - d_cmpBytes
    */
    unsigned char* cuSZp_device_compress(float *oriData, size_t *outSize, float absErrBound, size_t nbEle){
        unsigned char *d_cmpBytes, *d_finalCmpBytes;
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        cudaMalloc((void**)&d_cmpBytes, sizeof(float)*nbEle);
        SZp_compress_deviceptr(oriData, d_cmpBytes, nbEle, outSize, absErrBound, stream);
        cudaMalloc((void**)&d_finalCmpBytes, *outSize);
        cudaMemcpy(d_finalCmpBytes, d_cmpBytes, *outSize, cudaMemcpyDeviceToDevice);
        cudaFree(d_cmpBytes);
	//cudaFree(oriData);
        return d_finalCmpBytes;
    }

    /** Before entering SZp_decompress, must allocate on device:
     * - d_decData
    */
    float* cuSZp_device_decompress(size_t nbEle, unsigned char* cmpBytes, size_t cmpSize, float errorBound){
        float *d_decData;
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        cudaMalloc((void**)&d_decData, sizeof(float)*nbEle);
        SZp_decompress_deviceptr(d_decData, cmpBytes, nbEle, cmpSize, errorBound, stream);
        cudaFree(cmpBytes);
	return d_decData;
    }
    
}
