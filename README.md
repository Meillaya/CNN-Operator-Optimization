# Custom Neural Network Operator Optimization

This project is a hands-on exercise to implement and optimize a custom deep learning operator from scratch.

## Project Goals

- Implement custom operators (e.g., convolution, attention) in CUDA.
- Integrate these operators with PyTorch using C++ extensions.
- Profile and optimize the CUDA kernels using tools like NVIDIA Nsight.
- (Advanced) Compare performance against vendor-optimized libraries like cuDNN.
- (Advanced) Explore distributed execution with NCCL.

## Structure

- `csrc/`: C++/CUDA source code for the custom operators.
- `python/`: Python wrapper code for PyTorch integration.
- `tests/`: Tests for the custom operators. 