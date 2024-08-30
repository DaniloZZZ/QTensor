//#include "cuszx_entry.h"
//#include "szx_defines.h"
//#include "szx_BytesToolkit.h"
//#include "szx_TypeManager.h"
//#include "timingGPU.h"

#include "cusz.h"
#include "cli/quality_viewer.hh"
#include "cli/timerecord_viewer.hh"
#include "utils/io.hh"
#include "utils/print_gpu.hh"

// template <typename T>
extern "C"{
unsigned char* cusz_device_compress(float *data, float r2r_error,size_t len,size_t *outSize)
{
    /* For demo, we use 3600x1800 CESM data. */

    cusz_header header;
    uint8_t*    exposed_compressed;
    uint8_t*    compressed;
    size_t      compressed_len;

    float *d_uncompressed, *h_uncompressed;
    float *d_decompressed, *h_decompressed;

    d_uncompressed = data;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // using default
    // cusz_framework* framework = cusz_default_framework();
    // alternatively
    cusz_framework fw = cusz_framework{
        .pipeline     = Auto,
        .predictor    = cusz_custom_predictor{.type = LorenzoI},
        .quantization = cusz_custom_quantization{.radius = 512},
        .codec        = cusz_custom_codec{.type = Huffman}};
    cusz_framework* framework = &fw;

    // Brace initializing a struct pointer is not supported by all host compilers
    // when nvcc forwards.
    // cusz_framework* framework = new cusz_framework{
    //     .pipeline     = Auto,
    //     .predictor    = cusz_custom_predictor{.type = LorenzoI},
    //     .quantization = cusz_custom_quantization{.radius = 512},
    //     .codec        = cusz_custom_codec{.type = Huffman}};


    cusz_compressor* comp       = cusz_create(framework, FP32);
    cusz_config*     config     = new cusz_config{.eb = r2r_error, .mode = Rel};
    cusz_len         uncomp_len = cusz_len{len, 1, 1, 1};  // x, y, z, w
    cusz_len         decomp_len = uncomp_len;

    cusz::TimeRecord compress_timerecord;
    

    {
        cusz_compress(
            comp, config, d_uncompressed, uncomp_len, &exposed_compressed, &compressed_len, &header,
            (void*)&compress_timerecord, stream);

        /* User can interpret the collected time information in other ways. */
        cusz::TimeRecordViewer::view_compression(&compress_timerecord, len * sizeof(float), compressed_len);

        /* verify header */
        printf("header.%-*s : %x\n", 12, "(addr)", &header);
        printf("header.%-*s : %lu, %lu, %lu\n", 12, "{x,y,z}", header.x, header.y, header.z);
        printf("header.%-*s : %lu\n", 12, "filesize", ConfigHelper::get_filesize(&header));
    }

    /* If needed, User should perform a memcopy to transfer `exposed_compressed` before `compressor` is destroyed. */
    cudaMalloc(&compressed, compressed_len);
    cudaMemcpy(compressed, exposed_compressed, compressed_len, cudaMemcpyDeviceToDevice);
    cudaFree(exposed_compressed);
    cudaStreamDestroy(stream);
    *outSize = compressed_len;
    return compressed;
}

float* cusz_device_decompress(uint8_t* cmpbytes, size_t len, size_t compressed_len, float r2r_error){
    cusz::TimeRecord decompress_timerecord;
    cudaStream_t stream;
    cusz_header header;
    float* d_decompressed;
    cudaMalloc(&d_decompressed, sizeof(float) * len);

    cusz_framework fw = cusz_framework{
        .pipeline     = Auto,
        .predictor    = cusz_custom_predictor{.type = LorenzoI},
        .quantization = cusz_custom_quantization{.radius = 512},
        .codec        = cusz_custom_codec{.type = Huffman}};
    cusz_framework* framework = &fw;

    cusz_compressor* comp       = cusz_create(framework, FP32);
    cusz_config*     config     = new cusz_config{.eb = r2r_error, .mode = Rel};
    cusz_len         uncomp_len = cusz_len{len, 1, 1, 1};  // x, y, z, w
    cusz_len         decomp_len = uncomp_len;


    cudaStreamCreate(&stream);
    {
        cusz_decompress(
            comp, &header, cmpbytes, compressed_len, d_decompressed, decomp_len,
            (void*)&decompress_timerecord, stream);

        cusz::TimeRecordViewer::view_decompression(&decompress_timerecord, len * sizeof(float));
    }


    cusz_release(comp);

    // cudaFree(cmpbytes);
    cudaStreamDestroy(stream);
    return d_decompressed;
}


    // unsigned char* cuSZx_integrated_compress(float *data, float r2r_threshold, float r2r_err, size_t nbEle, int blockSize, size_t *outSize){
    //     float max,min;
    //     unsigned char* bytes;
    //     max = data[0];
    //     min = data[0];
    //     for (size_t i = 0; i < nbEle; i++)
    //     {
    //         if(data[i] > max) max = data[i];
    //         if(data[i] < min) min = data[i];
    //     }
        
    //     float threshold = r2r_threshold*(max-min);
    //     float errBound = r2r_err*(max-min);
    //     bytes = cuSZx_fast_compress_args_unpredictable_blocked_float(data, outSize, errBound, nbEle, blockSize, threshold);
   	//     // printf("outSize %p\n", bytes);
    //     return bytes;
    // }

    // float* cuSZx_integrated_decompress(unsigned char *bytes, size_t nbEle){
    //     // printf("test\n");
    //     float**data;
	//     cuSZx_fast_decompress_args_unpredictable_blocked_float(data, nbEle, bytes);
    //     return *data;
    // }

    // unsigned char* cuSZx_device_compress(float *oriData, size_t *outSize, float absErrBound, size_t nbEle, int blockSize, float threshold){
    //     return device_ptr_cuSZx_compress_float(oriData, outSize, absErrBound, nbEle, blockSize, threshold);
    // }

    // float* cuSZx_device_decompress(size_t nbEle, unsigned char* cmpBytes){
    //     return device_ptr_cuSZx_decompress_float(nbEle, cmpBytes);
    // }
    
    
}
