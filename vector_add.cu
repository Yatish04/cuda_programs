// 
/*
* File: vector_add.cu
* Description: This program adds two vectors together.
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>



__global__ void vector_add(float* a, float* b, float* c, int n) {
   int idx = threadIdx.x;
   if (idx < n) {
       c[idx] = a[idx] + b[idx];
   }
}

void test_vector_add(float* a, float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        assert(c[i] == a[i] + b[i]);
    }
}

int main() {
    int n = 10;

    float *a = new float[n];
    float *b = new float[n];
    float *c = new float[n];

    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = 2 * i;
    }

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;

    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));

    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    std::cout<<"Running vector addition with "<<blocks<<" blocks and "<<threads<<" threads"<<std::endl;

    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    cudaMemcpy(c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    test_vector_add(a, b, c, n);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    std::cout << "Vector addition completed successfully!" << std::endl;
    return 0;
}