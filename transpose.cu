/**
* File: transpose.cu
* Description: This program transposes a matrix.
Approach:
- We will use a 2D grid of threads to transpose the matrix.
- Each thread is responsible for one element of the output matrix.
- Assume a square matrix for simplicity.
*/
#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

__global__ void transpose(float* input, float* output, int n) {
    int row = blockIdx.x;
    int col = blockIdx.y;
    int idx = threadIdx.x;
    int idy = threadIdx.y;
    if (row < n && col < n) {
        output[col * n + row] = input[row * n + col];
    }
}

void test_transpose(float* input, float* output, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            assert(output[j * n + i] == input[i * n + j]);
        }
    }
}

int main() {
    int n = 100;
    float input[100][100];
    float output[100][100];

    for (int i = 0; i < 100; i++) {
        for (int j = 0; j < 100; j++) {
            input[i][j] = i * 100 + j;
        }
    }

    float *d_input = nullptr;
    float *d_output = nullptr;

    cudaMalloc(&d_input, 100 * 100 * sizeof(float));
    cudaMalloc(&d_output, 100 * 100 * sizeof(float));
    
    cudaMemcpy(d_input, input, 100 * 100 * sizeof(float), cudaMemcpyHostToDevice);
}
