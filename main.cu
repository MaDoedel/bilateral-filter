#include "bilateralFilterKernel.h"
#include <opencv2/opencv.hpp>
#include <iostream>


// Error checking macro
// Error checking macro
void cudaCheckError() {                                          
    cudaError_t e = cudaGetLastError();                             
    if (e != cudaSuccess) {                                         
        std::cout << "CUDA error: " << cudaGetErrorString(e) << " at " << __FILE__ << ":" << __LINE__ << std::endl; 
        exit(EXIT_FAILURE);                                         
    }                                                               
}

__global__ void bilateralFilterKernel(
    const float *input, float *output, 
    int width, int height, int kernel_size, 
    float sigma_s, float sigma_r);

void applyBilateralFilter(const cv::Mat &input, cv::Mat &output, int kernel_size, float sigma_s, float sigma_r) {
    int width = input.cols;
    int height = input.rows;
    size_t image_size = width * height * sizeof(float);

    float *d_input, *d_output;

    cudaMalloc(&d_input, image_size);
    cudaCheckError();
    cudaMalloc(&d_output, image_size);
    cudaCheckError();

    float *input_floats = (float*)malloc(image_size);
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            input_floats[y * width + x] = static_cast<float>(input.at<unsigned char>(y, x));
        }
    }

    cudaMemcpy(d_input, input_floats, image_size, cudaMemcpyHostToDevice);
    cudaCheckError();

    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);

    bilateralFilterKernel<<<grid_size, block_size>>>(d_input, d_output, width, height, kernel_size, sigma_s, sigma_r);
    cudaDeviceSynchronize();
    cudaCheckError();

    float *output_floats = (float*)malloc(image_size);
    cudaMemcpy(output_floats, d_output, image_size, cudaMemcpyDeviceToHost);
    cudaCheckError();
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            std::cout << output_floats[y * width + x] << std::endl;
        }
    }

    cudaFree(d_input);
    cudaCheckError();
    cudaFree(d_output);
    cudaCheckError();
}

int main() {
    cv::Mat img = cv::imread("../test.jpg", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error: Unable to load image!" << std::endl;
        return -1;
    }

    cv::imshow("Original Image", img);
    cv::waitKey(0);
    // return 0;

    

    cv::Mat result(img.size(), img.type());

    int kernel_size = 9;
    float sigma_s = 3.0;
    float sigma_r = 50.0;

    applyBilateralFilter(img, result, kernel_size, sigma_s, sigma_r);

    cv::imshow("Filtered Image", result);
    cv::imwrite("filtered_image.jpg", result);
    cv::waitKey(0);

    return 0;
}
