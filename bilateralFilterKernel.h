// bilateralFilterKernel.h
#ifndef BILATERALFILTERKERNEL_H
#define BILATERALFILTERKERNEL_H

#include <cuda_runtime.h>

__global__ void bilateralFilterKernel(
    const float* input, 
    float* output, 
    int width, 
    int height, 
    int kernel_size, 
    float sigma_s, 
    float sigma_r
);

#endif // BILATERALFILTERKERNEL_H
