# Bilateral Filter Image Processing

Bilateral filtering is a non-linear, edge-preserving, and noise-reducing smoothing filter for images. This repository contains a Python script that implements bilateral filtering using OpenCV and NumPy.

## Prerequisites

- Python 3.x
- OpenCV (cv2)
- NumPy

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/bilateral-filtering.git
    cd bilateral-filtering
    ```

2. **Install required libraries:**

    You can install the required libraries using `pip`:

    ```bash
    pip install opencv-python-headless numpy
    ```

## Usage

1. **Place your image:**

    Ensure you have an image named `test.jpg` in the same directory as the script.

2. **Run the script:**

    ```bash
    python bilateral_filter.py
    ```

3. **View results:**

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

## Code Structure

- **`main()`**: Reads the image, converts it to grayscale, applies bilateral filtering with different parameters, and displays/saves the results.
- **`gaussian()`**: Calculates the Gaussian function.
- **`bilateralFilter()`**: Applies the bilateral filter to the image.
- **Helper functions**: Includes padding, distance calculation, etc.

## Notes

- Ensure your image file (`test.jpeg`) exists in the working directory.
- Adjust parameters as needed to get the best results for your specific image.

## License

This project is licensed under the MIT License.
