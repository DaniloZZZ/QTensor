#include "newsz.h"
#include <stdio.h>

extern "C"{
    
    unsigned char* newSZ_device_compress(float *oriData, size_t *outSize, size_t nbEle, int blockSize){
        //unsigned char* cmpbytes;
        return SZ_device_compress(oriData, nbEle, blockSize, outSize);
        //printf("in wrap cmpbytes: %p\n", cmpbytes);
	//return cmpbytes;
    }

    float* newSZ_device_decompress(size_t nbEle, unsigned char* cmpBytes, int blocksize, size_t cmpsize){
        size_t *cmpsize_ptr;
        *cmpsize_ptr = cmpsize;

        float *res = SZ_device_decompress(cmpBytes, nbEle, blocksize, cmpsize_ptr);
	return res;
    }
    
}
