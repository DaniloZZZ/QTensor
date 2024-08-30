
unsigned char* SZ_device_compress(float *data, size_t num_elements, int blocksize, size_t *outsize);
float* SZ_device_decompress(unsigned char *cmpbytes, size_t num_elements, int blocksize, size_t *cmpsize);
