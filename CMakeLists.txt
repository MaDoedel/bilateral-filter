cmake_minimum_required(VERSION 3.10)
project(BilateralFilterCUDA LANGUAGES CXX CUDA)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)



# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Enable CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Set the CUDA compilation flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_60")

# Add the CUDA kernel source file
set(CUDA_SOURCES bilateralFilterKernel.cu)

# Add the CUDA source files
set(CUDA_SRCS main.cu ${CUDA_SOURCES})

# Define the executable
add_executable(bilateralFilter ${CUDA_SRCS})

# Set the CUDA and C++ properties for the executable
set_target_properties(bilateralFilter PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 75
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED True
)

# Link against OpenCV and CUDA
target_link_libraries(bilateralFilter ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
