#include "bilateralFilterKernel.h"
#include <opencv2/opencv.hpp>
#include <iostream>

__global__ void bilateralFilterKernel(
    const unsigned char *input, unsigned char *output, 
    int width, int height, int kernel_size, 
    float sigma_s, float sigma_r);

void applyBilateralFilter(const cv::Mat &input, cv::Mat &output, int kernel_size, float sigma_s, float sigma_r) {
    int width = input.cols;
    int height = input.rows;
    size_t image_size = width * height * sizeof(unsigned char);

    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, image_size);
    cudaMalloc(&d_output, image_size);

    cudaMemcpy(d_input, input.data, image_size, cudaMemcpyHostToDevice);

    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);

    bilateralFilterKernel<<<grid_size, block_size>>>(d_input, d_output, width, height, kernel_size, sigma_s, sigma_r);
    cudaDeviceSynchronize();

    cudaMemcpy(output.data, d_output, image_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    cv::Mat img = cv::imread("../test.jpg", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error: Unable to load image!" << std::endl;
        return -1;
    }

    cv::Mat result(img.size(), img.type());

    int kernel_size = 9;
    float sigma_s = 3.0;
    float sigma_r = 50.0;

    applyBilateralFilter(img, result, kernel_size, sigma_s, sigma_r);

    cv::imshow("Original Image", img);
    cv::imshow("Filtered Image", result);
    cv::imwrite("filtered_image.jpg", result);
    cv::waitKey(0);

    return 0;
}
