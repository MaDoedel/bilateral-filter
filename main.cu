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
    //cudaCheckError();
    cudaMalloc(&d_output, image_size);
    //cudaCheckError();

    float *input_floats = (float*)malloc(image_size);
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            input_floats[y * width + x] = static_cast<float>(input.at<unsigned char>(y, x));
        }
    }

    cudaMemcpy(d_input, input_floats, image_size, cudaMemcpyHostToDevice);
    //cudaCheckError();

    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);

    bilateralFilterKernel<<<grid_size, block_size>>>(d_input, d_output, width, height, kernel_size, sigma_s, sigma_r);
    cudaDeviceSynchronize();
    //cudaCheckError();

    float *output_floats = (float*)malloc(image_size);
    cudaMemcpy(output_floats, d_output, image_size, cudaMemcpyDeviceToHost);
    //cudaCheckError();

    output = cv::Mat(height, width, CV_32F, output_floats);
    output.convertTo(output, CV_8U);

    cudaFree(d_input);
    //cudaCheckError();
    cudaFree(d_output);
    //cudaCheckError();
}

int main() {
    cv::Mat img = cv::imread("../test.jpg", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Error: Unable to load image!" << std::endl;
        return -1;
    }

    // Define 10 different values for kernel sizes, sigma_s, and sigma_r
    std::vector<int> kernel_sizes = {3, 5, 7, 9, 11, 13, 15, 17, 19, 21};
    std::vector<float> sigma_s_values = {1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5};
    std::vector<float> sigma_r_values = {10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0};

    // Single loop index
    for (size_t i = 0; i < kernel_sizes.size(); ++i) {
        int kernel_size = kernel_sizes[i];
        float sigma_s = sigma_s_values[i];
        float sigma_r = sigma_r_values[i];

        cv::Mat result(img.size(), img.type());

        // Apply the bilateral filter
        applyBilateralFilter(img, result, kernel_size, sigma_s, sigma_r);

        // Create a filename for the output image
        std::stringstream filename;
        filename << "../cuda_"
                 << "n:" << kernel_size 
                 << ",sigma_s:" << sigma_s 
                 << ",sigma_r:" << sigma_r 
                 << ".jpg";

        // Save the result
        if (!cv::imwrite(filename.str(), result)) {
            std::cerr << "Error: Unable to save image " << filename.str() << std::endl;
        }
    }

    return 0;
}
