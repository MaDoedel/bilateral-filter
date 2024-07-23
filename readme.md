# Bilateral Filter Image Processing

Bilateral filtering is a non-linear, edge-preserving, and noise-reducing smoothing filter for images. This repository contains both a Python script and a C++/CUDA implementation of bilateral filtering using OpenCV and NumPy/CUDA.

## Prerequisites

### Python

- Python 3.x
- OpenCV 
- NumPy

### C++/CUDA

- CMake
- CUDA Toolkit
- OpenCV

## Installation and Usage

1. **Clone the repository:**

    ```bash
    git clone git@github.com:MaDoedel/bilateral-filter.git
    cd bilateral-filter
    ```

    Ensure you have an image named `test.jpg` in the same directory as the script.


### Python

2. **Make sure to install opencv, numpy and run python3**

    ```bash
    python3 bifilter.py
    ```

### C++/CUDA

2. **Make sure to install the right driver and cuda toolkit, clone the repo and build the project**

    ```bash
    git clone git@github.com:MaDoedel/bilateral-filter.git
    cd bilateral-filtering
    mkdir build
    cd build
    cmake ..
    make
    ./bilateralFilter
    ```

### View Results
The script will apply the bilateral filter with different parameter sets and display the filtered images along with the original image. It will also save the filtered images in the same directory.

## Parameters

The script tries different sets of parameters for the bilateral filter:

- **Kernel Size (`n`)**: Size of the filter kernel.
- **Spatial Sigma (`sigma_s`)**: Standard deviation for the spatial distance.
- **Range Sigma (`sigma_r`)**: Standard deviation for the intensity difference.

### Example Parameter Sets

- Set 1: `n = 9`, `sigma_s = 3`, `sigma_r = 50`
- Set 2: `n = 11`, `sigma_s = 5`, `sigma_r = 75`
- Set 3: `n = 7`, `sigma_s = 2`, `sigma_r = 25`

## Example Visualization

### Input Image

The following image is a noisy grayscale image with dimensions 325x325 pixels. The image can be downloaded from [this link](https://boofcv.org/index.php?title=File:Kodim17_noisy.jpg).

![Noisy Image](/test.jpg)

### Filter Parameters and Results

The bilateral filter was applied to the noisy image with various parameter sets to demonstrate its effectiveness in noise reduction while preserving edges. Below are the results for different parameter configurations:

#### Parameter Set 1

- **Kernel Size (`n`)**: 9
- **Spatial Sigma (`sigma_s`)**: 3
- **Range Sigma (`sigma_r`)**: 50

![Filtered Image Set 1](/n:9,sigma_s:3,sigma_r:50.jpg)

#### Parameter Set 2

- **Kernel Size (`n`)**: 11
- **Spatial Sigma (`sigma_s`)**: 5
- **Range Sigma (`sigma_r`)**: 75

![Filtered Image Set 2](/n:11,sigma_s:5,sigma_r:75.jpg)

#### Parameter Set 3

- **Kernel Size (`n`)**: 7
- **Spatial Sigma (`sigma_s`)**: 2
- **Range Sigma (`sigma_r`)**: 25

![Filtered Image Set 3](/n:7,sigma_s:2,sigma_r:25.jpg)

#### Parameter Set 4

- **Kernel Size (`n`)**: 3
- **Spatial Sigma (`sigma_s`)**: 1
- **Range Sigma (`sigma_r`)**: 10

![Filtered Image Set 4](/cuda_n:3,sigma_s:1,sigma_r:10.jpg)

#### Parameter Set 5

- **Kernel Size (`n`)**: 5
- **Spatial Sigma (`sigma_s`)**: 1
- **Range Sigma (`sigma_r`)**: 15

![Filtered Image Set 5](/cuda_n:5,sigma_s:1.5,sigma_r:15.jpg)

#### Parameter Set 6

- **Kernel Size (`n`)**: 7
- **Spatial Sigma (`sigma_s`)**: 2
- **Range Sigma (`sigma_r`)**: 20

![Filtered Image Set 6](/cuda_n:7,sigma_s:2,sigma_r:20.jpg)

#### Parameter Set 7

- **Kernel Size (`n`)**: 9
- **Spatial Sigma (`sigma_s`)**: 2.5
- **Range Sigma (`sigma_r`)**: 25

![Filtered Image Set 7](/cuda_n:9,sigma_s:2.5,sigma_r:25.jpg)

#### Parameter Set 8

- **Kernel Size (`n`)**: 11
- **Spatial Sigma (`sigma_s`)**: 3
- **Range Sigma (`sigma_r`)**: 30

![Filtered Image Set 8](/cuda_n:11,sigma_s:3,sigma_r:30.jpg)

#### Parameter Set 9

- **Kernel Size (`n`)**: 13
- **Spatial Sigma (`sigma_s`)**: 3.5
- **Range Sigma (`sigma_r`)**: 35

![Filtered Image Set 9](/cuda_n:13,sigma_s:3.5,sigma_r:35.jpg)

#### Parameter Set 10

- **Kernel Size (`n`)**: 15
- **Spatial Sigma (`sigma_s`)**: 4
- **Range Sigma (`sigma_r`)**: 40

![Filtered Image Set 10](/cuda_n:15,sigma_s:4,sigma_r:40.jpg)

#### Parameter Set 11

- **Kernel Size (`n`)**: 17
- **Spatial Sigma (`sigma_s`)**: 4.5
- **Range Sigma (`sigma_r`)**: 45

![Filtered Image Set 11](/cuda_n:17,sigma_s:4.5,sigma_r:45.jpg)

#### Parameter Set 12

- **Kernel Size (`n`)**: 19
- **Spatial Sigma (`sigma_s`)**: 5
- **Range Sigma (`sigma_r`)**: 50

![Filtered Image Set 12](/cuda_n:19,sigma_s:5,sigma_r:50.jpg)

#### Parameter Set 13

- **Kernel Size (`n`)**: 21
- **Spatial Sigma (`sigma_s`)**: 5.5
- **Range Sigma (`sigma_r`)**: 55

![Filtered Image Set 13](/cuda_n:21,sigma_s:5.5,sigma_r:55.jpg)

## CUDA Acceleration

The C++/CUDA implementation provides a significantly faster approach than using Python, especially for large images and multiple parameter sets. The use of CUDA allows the bilateral filter to leverage GPU acceleration, resulting in reduced computation time and improved performance.

## Code Structure

### Python

- **`main()`**: Reads the image, converts it to grayscale, applies bilateral filtering with different parameters, and displays/saves the results.
- **`gaussian()`**: Calculates the Gaussian function.
- **`bilateralFilter()`**: Applies the bilateral filter to the image.
- **Helper functions**: Includes padding, distance calculation, etc.

### C++/CUDA

- **`applyBilateralFilter()`**: Implements the bilateral filter using CUDA for acceleration.
- **Main function**: Loads the image, applies the filter with different parameters, and saves the results.

## Notes

- Ensure your image file (`test.jpg`) exists in the working directory
- Adjust parameters as needed to get the best results for your specific image