#ifndef CUSZP_INCLUDE_CUSZP_H
#define CUSZP_INCLUDE_CUSZP_H

static const int cmp_tblock_size = 32; // 32 should be the best, not need to modify.
static const int dec_tblock_size = 32; // 32 should be the best, not need to modify.
static const int cmp_chunk = 8192;
static const int dec_chunk = 8192;

__global__ void SZp_compress_kernel(const float* const __restrict__ oriData, unsigned char* const __restrict__ cmpData, volatile unsigned int* const __restrict__ cmpOffset, volatile int* const __restrict__ flag, const float eb, const size_t nbEle);
__global__ void SZp_decompress_kernel(float* const __restrict__ decData, const unsigned char* const __restrict__ cmpData, volatile unsigned int* const __restrict__ cmpOffset, volatile int* const __restrict__ flag, const float eb, const size_t nbEle);

#endif // CUSZP_INCLUDE_CUSZP_H