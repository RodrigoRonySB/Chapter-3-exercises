#include <cuda_runtime.h>

// Kernel: A = B * C
// B: matriz N x N (row-major), C: vector N, A: vector N
__global__ void matVecKernel(const float* __restrict__ B,
                             const float* __restrict__ C,
                             float* __restrict__ A,
                             int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x; // hilo -> fila i
    if (row < N) {
        float acc = 0.0f;
        int base = row * N;                          // inicio de la fila i en row-major
        #pragma unroll 4
        for (int j = 0; j < N; ++j) {
            acc += B[base + j] * C[j];               // producto escalar de la fila i con C
        }
        A[row] = acc;
    }
}

void matVecHost(float* h_A,          // salida: A (N)
                const float* h_B,    // entrada: B (N x N)
                const float* h_C,    // entrada: C (N)
                int N)
{
    size_t bytesMat = (size_t)N * (size_t)N * sizeof(float);
    size_t bytesVec = (size_t)N * sizeof(float);

    float *d_B = nullptr, *d_C = nullptr, *d_A = nullptr;

    
    cudaMalloc(&d_B, bytesMat);
    cudaMalloc(&d_C, bytesVec);
    cudaMalloc(&d_A, bytesVec);

    
    cudaMemcpy(d_B, h_B, bytesMat, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, bytesVec, cudaMemcpyHostToDevice);

    // configurar y lanzar el kernel (1D: 1 hilo por fila)
    int threadsPerBlock = 256;                              // múltiplo de 32 (warp)
    int numBlocks       = (N + threadsPerBlock - 1) / threadsPerBlock; // ceil(N/TPB)
    matVecKernel<<<numBlocks, threadsPerBlock>>>(d_B, d_C, d_A, N);

    
    cudaMemcpy(h_A, d_A, bytesVec, cudaMemcpyDeviceToHost);

   
    cudaFree(d_B); cudaFree(d_C); cudaFree(d_A);
}
