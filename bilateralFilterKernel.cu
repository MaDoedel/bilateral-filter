#include "bilateralFilterKernel.h"
#include <math.h>


__global__ void bilateralFilterKernel(
    const unsigned char *input, unsigned char *output,
    int width, int height, int kernel_size, 
    float sigma_s, float sigma_r) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float center_intensity = input[y * width + x];
        float sum = 0.0f;
        float norm_factor = 0.0f;
        int half_kernel = kernel_size / 2;

        for (int ky = -half_kernel; ky <= half_kernel; ++ky) {
            for (int kx = -half_kernel; kx <= half_kernel; ++kx) {
                int nx = min(max(x + kx, 0), width - 1);
                int ny = min(max(y + ky, 0), height - 1);
                float neighbor_intensity = input[ny * width + nx];

                float spatial_dist = kx * kx + ky * ky;
                float intensity_dist = (neighbor_intensity - center_intensity) * (neighbor_intensity - center_intensity);

                float weight = expf(-(spatial_dist / (2 * sigma_s * sigma_s) + intensity_dist / (2 * sigma_r * sigma_r)));
                sum += weight * neighbor_intensity;
                norm_factor += weight;
            }
        }

        output[y * width + x] = static_cast<unsigned char>(sum / norm_factor);
    }
}
