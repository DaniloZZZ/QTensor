#include "cuSZp.h"

__device__ inline int quantization(float data, float recipPrecision)
{
    float dataRecip = data*recipPrecision;
    int s = dataRecip>=-0.5f?0:1;
    return (int)(dataRecip+0.5f) - s;
}


__device__ inline int get_bit_num(unsigned int x)
{
    return (sizeof(unsigned int)*8) - __clz(x);
}


__global__ void SZp_compress_kernel(const float* const __restrict__ oriData, unsigned char* const __restrict__ cmpData, volatile unsigned int* const __restrict__ cmpOffset, volatile int* const __restrict__ flag, const float eb, const size_t nbEle)
{
    __shared__ unsigned int base_idx;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * blockDim.x + tid;
    const int lane = idx & 31;
    const int warp = idx >> 5;
    const int block_num = cmp_chunk/32;
    const int rate_ofs = (nbEle+31)/32;
    const float recipPrecision = 0.5f/eb;

    int base_start_idx;
    int base_block_start_idx, base_block_end_idx;
    int quant_chunk_idx;
    int block_idx;
    int currQuant, lorenQuant, prevQuant, maxQuant;
    int absQuant[cmp_chunk];
    unsigned int sign_flag[block_num];
    int sign_ofs;
    int fixed_rate[block_num];
    unsigned int thread_ofs = 0;

    // Prequantization + Lorenzo Prediction + Fixed-length encoding + store fixed-length to global memory.
    base_start_idx = warp * cmp_chunk * 32;
    for(int j=0; j<block_num; j++)
    {
        // Block initilization.
        base_block_start_idx = base_start_idx + j * 1024 + lane * 32;
        base_block_end_idx = base_block_start_idx + 32;
        sign_flag[j] = 0;
        block_idx = base_block_start_idx/32;
        prevQuant = 0;
        maxQuant = 0;

        // Operation for each block
        for(int i=base_block_start_idx; i<base_block_end_idx; i++)
        {
            // Get quantization and Lorenzo prediction
            quant_chunk_idx = j * 32 + i % 32;
            currQuant = quantization(oriData[i], recipPrecision);
            lorenQuant = currQuant - prevQuant;
            prevQuant = currQuant;
            // Get and combine sign info.
            sign_ofs = i % 32;
            sign_flag[j] |= (lorenQuant < 0) << (31 - sign_ofs);
            // Get absolute quant.
            absQuant[quant_chunk_idx] = abs(lorenQuant);
            // Update max quant.
            maxQuant = maxQuant > absQuant[quant_chunk_idx] ? maxQuant : absQuant[quant_chunk_idx];
        }

        // Record block info.
        fixed_rate[j] = get_bit_num(maxQuant);
        thread_ofs += (fixed_rate[j]) ? (32+fixed_rate[j]*32) : 0;
        // Write block fixed rate to compressed data.
        if(block_idx<rate_ofs) cmpData[block_idx] = (unsigned char)fixed_rate[j];
        __syncthreads();
    }

    // Warp-level prefix-sum (inclusive), also thread-block-level.
    for(int i=1; i<32; i<<=1)
    {
        int tmp = __shfl_up_sync(0xffffffff, thread_ofs, i);
        if(lane >= i) thread_ofs += tmp;
    }
    __syncthreads();

    // Write warp(i.e. thread-block)-level prefix-sum to global-memory.
    if(lane==31) 
    {
        cmpOffset[warp+1] = (thread_ofs+7)/8;
        if(warp==0)
            flag[1] = 2;
        else
            flag[warp+1] = 1;
    }
    __syncthreads();

    // Global-level prefix-sum (exclusive).
    if(warp>0)
    {
        if(!lane)
        {
            int temp_flag = 1;
            while(temp_flag!=2) temp_flag = flag[warp];
            __threadfence();
            cmpOffset[warp] += cmpOffset[warp-1];
            __threadfence();
            flag[warp+1] = 2;
        }
    }
    else
    {
        if(!lane) cmpOffset[0] = 0;
    }
    __syncthreads();
    
    // Assigning compression bytes by given prefix-sum results.
    if(!lane) base_idx = cmpOffset[warp] + rate_ofs;
    __syncthreads();

    // Bit shuffle for each index, also storing data to global memory.
    unsigned int base_cmp_byte_ofs = base_idx;
    unsigned int cmp_byte_ofs;
    unsigned int tmp_byte_ofs = 0;
    unsigned int cur_byte_ofs = 0;
    for(int j=0; j<block_num; j++)
    {
        int chunk_idx_start = j*32;

        // Restore index for j-th iteration.
        tmp_byte_ofs = (fixed_rate[j]) ? (4+fixed_rate[j]*4) : 0;
        for(int i=1; i<32; i<<=1)
        {
            int tmp = __shfl_up_sync(0xffffffff, tmp_byte_ofs, i);
            if(lane >= i) tmp_byte_ofs += tmp;
        }
        unsigned int prev_thread = __shfl_up_sync(0xffffffff, tmp_byte_ofs, 1);
        if(!lane) cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs;
        else cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs + prev_thread;

        // Operation for each block, if zero block then do nothing.
        if(fixed_rate[j])
        {
            // Assign sign information for one block.
            cmpData[cmp_byte_ofs++] = 0xff & (sign_flag[j] >> 24);
            cmpData[cmp_byte_ofs++] = 0xff & (sign_flag[j] >> 16);
            cmpData[cmp_byte_ofs++] = 0xff & (sign_flag[j] >> 8);
            cmpData[cmp_byte_ofs++] = 0xff & sign_flag[j];

            // Assign quant bit information for one block by bit-shuffle.
            unsigned char tmp_char0, tmp_char1, tmp_char2, tmp_char3;
            int mask = 1;
            for(int i=0; i<fixed_rate[j]; i++)
            {
                // Initialization.
                tmp_char0 = 0;
                tmp_char1 = 0;
                tmp_char2 = 0;
                tmp_char3 = 0;

                // Get ith bit in 0~7 quant, and store to tmp_char0.
                tmp_char0 = (((absQuant[chunk_idx_start+0] & mask) >> i) << 7) |
                            (((absQuant[chunk_idx_start+1] & mask) >> i) << 6) |
                            (((absQuant[chunk_idx_start+2] & mask) >> i) << 5) |
                            (((absQuant[chunk_idx_start+3] & mask) >> i) << 4) |
                            (((absQuant[chunk_idx_start+4] & mask) >> i) << 3) |
                            (((absQuant[chunk_idx_start+5] & mask) >> i) << 2) |
                            (((absQuant[chunk_idx_start+6] & mask) >> i) << 1) |
                            (((absQuant[chunk_idx_start+7] & mask) >> i) << 0);

                // Get ith bit in 8~15 quant, and store to tmp_char1.
                tmp_char1 = (((absQuant[chunk_idx_start+8] & mask) >> i) << 7) |
                            (((absQuant[chunk_idx_start+9] & mask) >> i) << 6) |
                            (((absQuant[chunk_idx_start+10] & mask) >> i) << 5) |
                            (((absQuant[chunk_idx_start+11] & mask) >> i) << 4) |
                            (((absQuant[chunk_idx_start+12] & mask) >> i) << 3) |
                            (((absQuant[chunk_idx_start+13] & mask) >> i) << 2) |
                            (((absQuant[chunk_idx_start+14] & mask) >> i) << 1) |
                            (((absQuant[chunk_idx_start+15] & mask) >> i) << 0);

                // Get ith bit in 16~23 quant, and store to tmp_char2.
                tmp_char2 = (((absQuant[chunk_idx_start+16] & mask) >> i) << 7) |
                            (((absQuant[chunk_idx_start+17] & mask) >> i) << 6) |
                            (((absQuant[chunk_idx_start+18] & mask) >> i) << 5) |
                            (((absQuant[chunk_idx_start+19] & mask) >> i) << 4) |
                            (((absQuant[chunk_idx_start+20] & mask) >> i) << 3) |
                            (((absQuant[chunk_idx_start+21] & mask) >> i) << 2) |
                            (((absQuant[chunk_idx_start+22] & mask) >> i) << 1) |
                            (((absQuant[chunk_idx_start+23] & mask) >> i) << 0);
                
                // Get ith bit in 24-31 quant, and store to tmp_char3.
                tmp_char3 = (((absQuant[chunk_idx_start+24] & mask) >> i) << 7) |
                            (((absQuant[chunk_idx_start+25] & mask) >> i) << 6) |
                            (((absQuant[chunk_idx_start+26] & mask) >> i) << 5) |
                            (((absQuant[chunk_idx_start+27] & mask) >> i) << 4) |
                            (((absQuant[chunk_idx_start+28] & mask) >> i) << 3) |
                            (((absQuant[chunk_idx_start+29] & mask) >> i) << 2) |
                            (((absQuant[chunk_idx_start+30] & mask) >> i) << 1) |
                            (((absQuant[chunk_idx_start+31] & mask) >> i) << 0);

                // Move data to global memory.
                cmpData[cmp_byte_ofs++] = tmp_char0;
                cmpData[cmp_byte_ofs++] = tmp_char1;
                cmpData[cmp_byte_ofs++] = tmp_char2;
                cmpData[cmp_byte_ofs++] = tmp_char3;
                mask <<= 1;
            }
        }

        // Index updating across different iterations.
        cur_byte_ofs += __shfl_sync(0xffffffff, tmp_byte_ofs, 31);
    }
}



__global__ void SZp_decompress_kernel(float* const __restrict__ decData, const unsigned char* const __restrict__ cmpData, volatile unsigned int* const __restrict__ cmpOffset, volatile int* const __restrict__ flag, const float eb, const size_t nbEle)
{
    __shared__ unsigned int base_idx;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * blockDim.x + tid;
    const int lane = idx & 31;
    const int warp = idx >> 5;
    const int block_num = dec_chunk/32;
    const int rate_ofs = (nbEle+31)/32;

    int base_start_idx;
    int base_block_start_idx;
    int block_idx;    
    int absQuant[32];
    int currQuant, lorenQuant, prevQuant;
    int sign_ofs;
    int fixed_rate[block_num];
    unsigned int thread_ofs = 0;

    // Obtain fixed rate information for each block.
    for(int j=0; j<block_num; j++)
    {
        block_idx = warp * dec_chunk + j * 32 + lane;
        if(block_idx<rate_ofs) 
        {
            fixed_rate[j] = (int)cmpData[block_idx];
            thread_ofs += (fixed_rate[j]) ? (32+fixed_rate[j]*32) : 0;
        }
        __syncthreads();
    }

    // Warp-level prefix-sum (inclusive), also thread-block-level.
    for(int i=1; i<32; i<<=1)
    {
        int tmp = __shfl_up_sync(0xffffffff, thread_ofs, i);
        if(lane >= i) thread_ofs += tmp;
    }
    __syncthreads();

    // Write warp(i.e. thread-block)-level prefix-sum to global-memory.
    if(lane==31) 
    {
        cmpOffset[warp+1] = (thread_ofs+7)/8;
        if(warp==0)
            flag[1] = 2;
        else
            flag[warp+1] = 1;
    }
    __syncthreads();

    // Global-level prefix-sum (exclusive).
    if(warp>0)
    {
        if(!lane)
        {
            int temp_flag = 1;
            while(temp_flag!=2) temp_flag = flag[warp];
            __threadfence();
            cmpOffset[warp] += cmpOffset[warp-1];
            __threadfence();
            flag[warp+1] = 2;
        }
    }
    else
    {
        if(!lane) cmpOffset[0] = 0;
    }
    __syncthreads();

    // Retrieving compression bytes and reconstruct decompression data.
    if(!lane) base_idx = cmpOffset[warp] + rate_ofs;
    __syncthreads();

    // Restore bit-shuffle for each block.
    unsigned int base_cmp_byte_ofs = base_idx;
    unsigned int cmp_byte_ofs;
    unsigned int tmp_byte_ofs = 0;
    unsigned int cur_byte_ofs = 0;
    base_start_idx = warp * dec_chunk * 32;
    for(int j=0; j<block_num; j++)
    {
        // Block initialization.
        base_block_start_idx = base_start_idx + j * 1024 + lane * 32;
        unsigned int sign_flag = 0;

        // Restore index for j-th iteration.
        tmp_byte_ofs = (fixed_rate[j]) ? (4+fixed_rate[j]*4) : 0;
        for(int i=1; i<32; i<<=1)
        {
            int tmp = __shfl_up_sync(0xffffffff, tmp_byte_ofs, i);
            if(lane >= i) tmp_byte_ofs += tmp;
        }
        unsigned int prev_thread = __shfl_up_sync(0xffffffff, tmp_byte_ofs, 1);
        if(!lane) cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs;
        else cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs + prev_thread;

        // Operation for each block, if zero block then do nothing.
        if(fixed_rate[j])
        {
            // Retrieve sign information for one block.
            sign_flag = (0xff000000 & (cmpData[cmp_byte_ofs++] << 24)) |
                        (0x00ff0000 & (cmpData[cmp_byte_ofs++] << 16)) |
                        (0x0000ff00 & (cmpData[cmp_byte_ofs++] << 8))  |
                        (0x000000ff & cmpData[cmp_byte_ofs++]);
            
            // Retrieve quant data for one block.
            unsigned char tmp_char0, tmp_char1, tmp_char2, tmp_char3;
            for(int i=0; i<32; i++) absQuant[i] = 0;
            for(int i=0; i<fixed_rate[j]; i++)
            {
                // Initialization.
                tmp_char0 = cmpData[cmp_byte_ofs++];
                tmp_char1 = cmpData[cmp_byte_ofs++];
                tmp_char2 = cmpData[cmp_byte_ofs++];
                tmp_char3 = cmpData[cmp_byte_ofs++];

                // Get ith bit in 0~7 abs quant from global memory.
                absQuant[0] |= ((tmp_char0 >> 7) & 0x00000001) << i;
                absQuant[1] |= ((tmp_char0 >> 6) & 0x00000001) << i;
                absQuant[2] |= ((tmp_char0 >> 5) & 0x00000001) << i;
                absQuant[3] |= ((tmp_char0 >> 4) & 0x00000001) << i;
                absQuant[4] |= ((tmp_char0 >> 3) & 0x00000001) << i;
                absQuant[5] |= ((tmp_char0 >> 2) & 0x00000001) << i;
                absQuant[6] |= ((tmp_char0 >> 1) & 0x00000001) << i;
                absQuant[7] |= ((tmp_char0 >> 0) & 0x00000001) << i;

                // Get ith bit in 8~15 abs quant from global memory.
                absQuant[8] |= ((tmp_char1 >> 7) & 0x00000001) << i;
                absQuant[9] |= ((tmp_char1 >> 6) & 0x00000001) << i;
                absQuant[10] |= ((tmp_char1 >> 5) & 0x00000001) << i;
                absQuant[11] |= ((tmp_char1 >> 4) & 0x00000001) << i;
                absQuant[12] |= ((tmp_char1 >> 3) & 0x00000001) << i;
                absQuant[13] |= ((tmp_char1 >> 2) & 0x00000001) << i;
                absQuant[14] |= ((tmp_char1 >> 1) & 0x00000001) << i;
                absQuant[15] |= ((tmp_char1 >> 0) & 0x00000001) << i;

                // Get ith bit in 16-23 abs quant from global memory.
                absQuant[16] |= ((tmp_char2 >> 7) & 0x00000001) << i;
                absQuant[17] |= ((tmp_char2 >> 6) & 0x00000001) << i;
                absQuant[18] |= ((tmp_char2 >> 5) & 0x00000001) << i;
                absQuant[19] |= ((tmp_char2 >> 4) & 0x00000001) << i;
                absQuant[20] |= ((tmp_char2 >> 3) & 0x00000001) << i;
                absQuant[21] |= ((tmp_char2 >> 2) & 0x00000001) << i;
                absQuant[22] |= ((tmp_char2 >> 1) & 0x00000001) << i;
                absQuant[23] |= ((tmp_char2 >> 0) & 0x00000001) << i;

                // // Get ith bit in 24-31 abs quant from global memory.
                absQuant[24] |= ((tmp_char3 >> 7) & 0x00000001) << i;
                absQuant[25] |= ((tmp_char3 >> 6) & 0x00000001) << i;
                absQuant[26] |= ((tmp_char3 >> 5) & 0x00000001) << i;
                absQuant[27] |= ((tmp_char3 >> 4) & 0x00000001) << i;
                absQuant[28] |= ((tmp_char3 >> 3) & 0x00000001) << i;
                absQuant[29] |= ((tmp_char3 >> 2) & 0x00000001) << i;
                absQuant[30] |= ((tmp_char3 >> 1) & 0x00000001) << i;
                absQuant[31] |= ((tmp_char3 >> 0) & 0x00000001) << i;
            }
            
            // Delorenzo and store data back to decompression data.
            prevQuant = 0;
            for(int i=0; i<32; i++)
            {
                sign_ofs = i % 32;
                if(sign_flag & (1 << (31 - sign_ofs)))
                    lorenQuant = absQuant[i] * -1;
                else
                    lorenQuant = absQuant[i];
                currQuant = lorenQuant + prevQuant;
                decData[base_block_start_idx+i] = currQuant * eb * 2;
                prevQuant = currQuant;
            }
        }

        // Index updating across different iterations.
        cur_byte_ofs += __shfl_sync(0xffffffff, tmp_byte_ofs, 31);
    }
}