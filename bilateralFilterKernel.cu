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

                float spatial_dist = sqrtf(powf(kx - x, 2) + powf(ky -y, 2)); // this is not a guassian function
                float spatial_dist_gf = expf(-(powf(spatial_dist,2)/(2*powf(sigma_s,2))));
                float intensity_dist = sqrtf(powf(neighbor_intensity - center_intensity, 2));
                float intensity_dist_gf = expf(-(powf(intensity_dist,2)/(2*powf(sigma_r,2))));

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

                float spatial_dist = sqrtf(powf(kx - x, 2) + powf(ky -y, 2));
                float spatial_dist_gf = expf(-(powf(spatial_dist,2)/(2*powf(sigma_s,2))));
                float intensity_dist = sqrtf(powf(neighbor_intensity - center_intensity, 2));
                float intensity_dist_gf = expf(-(powf(intensity_dist,2)/(2*powf(sigma_r,2))));

                norm_factor += spatial_dist_gf * intensity_dist_gf * neighbor_intensity;
            }
        }

        output[y * width + x] = norm_factor / sum_weights;

    }
}
