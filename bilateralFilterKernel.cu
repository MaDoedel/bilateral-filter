#include "bilateralFilterKernel.h"
#include <math.h>


__global__ void bilateralFilterKernel(
    const float *input, float *output,
    int width, int height, int kernel_size, 
    float sigma_s, float sigma_r) {
    
    // get coordinates of the current thread aka pixel
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // check if the thread/pixel is within the image
    if (x < width && y < height) {
        float center_intensity = input[y * width + x]; // maybe height
        int half_kernel = kernel_size / 2;

        // for now, this is the wp part
        float sum_weights = 0;
        for (int kx = -half_kernel; kx <= half_kernel; kx++) {
            int nx = x + kx;
            if (nx < 0 || nx >= width) continue;

            for (int ky = -half_kernel; ky <= half_kernel; ky++) {
                int ny = y + ky;
                if (ny < 0 || ny >= height) continue;

                float neighbor_intensity = input[ny * width + nx]; // also here

                float spatial_dist = std::sqrt(static_cast<float>((kx) * (kx) + (ky) * (ky)));
                float spatial_dist_gf = std::exp(-(spatial_dist * spatial_dist) / (2 * sigma_s * sigma_s));
                float intensity_dist = std::sqrt(static_cast<float>((neighbor_intensity - center_intensity) * (neighbor_intensity - center_intensity)));
                float intensity_dist_gf = std::exp(-(intensity_dist * intensity_dist) / (2 * sigma_r * sigma_r));

                sum_weights += spatial_dist_gf * intensity_dist_gf;
            }
        }

        float norm_factor = 0;
        for (int kx = -half_kernel; kx < half_kernel +1; kx++) {
            int nx = x + kx;
            if (nx < 0 || nx >= width) continue;

            for (int ky = -half_kernel; ky < half_kernel +1; ky++) {
                int ny = y + ky;
                if (ny < 0 || ny >= height) continue;

                float neighbor_intensity = input[ny * width + nx]; // also here

                float spatial_dist = std::sqrt(static_cast<float>((kx) * (kx) + (ky) * (ky)));
                float spatial_dist_gf = std::exp(-(spatial_dist * spatial_dist) / (2 * sigma_s * sigma_s));
                float intensity_dist = std::sqrt(static_cast<float>((neighbor_intensity - center_intensity) * (neighbor_intensity - center_intensity)));
                float intensity_dist_gf = std::exp(-(intensity_dist * intensity_dist) / (2 * sigma_r * sigma_r));

                norm_factor += spatial_dist_gf * intensity_dist_gf * neighbor_intensity;
            }
        }

        output[y * width + x] = norm_factor / sum_weights;

    }
}
