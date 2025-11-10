//A    Función auxiliar del host
#include <cuda_runtime.h>

// Prototipos de kernels (se implementan en B, C y D)
__global__ void matAddElem (const float* A, const float* B, float* C, int N);
__global__ void matAddRow  (const float* A, const float* B, float* C, int N);
__global__ void matAddCol  (const float* A, const float* B, float* C, int N);

void matAddHost(float* h_C, const float* h_A, const float* h_B, int N)
{
    size_t bytes = (size_t)N * (size_t)N * sizeof(float);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // -------- Lanzamiento del kernel (se completa en B/C/D) --------
    // matAddElem<<<grid, block>>>(d_A, d_B, d_C, N);
    // matAddRow <<<grid, block>>>(d_A, d_B, d_C, N);
    // matAddCol <<<grid, block>>>(d_A, d_B, d_C, N);
    // ---------------------------------------------------------------

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}






//B    Kernel “1 hilo = 1 elemento” (2D).

__global__ void matAddElem(const float* A, const float* B, float* C, int N)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // i
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // j
    if (row < N && col < N) {
        int idx = row * N + col;                      // row-major
        C[idx] = A[idx] + B[idx];
    }
}

void matAddHost(float* h_C, const float* h_A, const float* h_B, int N)
{
    size_t bytes = (size_t)N * (size_t)N * sizeof(float);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // -------- Lanzamiento del kernel (B) --------
    dim3 block(16, 16);
    dim3 grid( (N + block.x - 1) / block.x,
               (N + block.y - 1) / block.y );
    matAddElem<<<grid, block>>>(d_A, d_B, d_C, N);
    // ---------------------------------------------------------------

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}




//C    Kernel “1 hilo = 1 fila” (1D).

__global__ void matAddRow(const float* A, const float* B, float* C, int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        int base = row * N;            // inicio de la fila
        for (int col = 0; col < N; ++col) {
            int idx = base + col;
            C[idx] = A[idx] + B[idx];
        }
    }
}

void matAddHost(float* h_C, const float* h_A, const float* h_B, int N)
{
    size_t bytes = (size_t)N * (size_t)N * sizeof(float);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // -------- Lanzamiento del kernel (C) --------
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    matAddRow<<<blocks, threads>>>(d_A, d_B, d_C, N);
    // ---------------------------------------------------------------

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}




//D  Kernel “1 hilo = 1 columna” (1D).

__global__ void matAddCol(const float* A, const float* B, float* C, int N)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < N) {
        for (int row = 0; row < N; ++row) {
            int idx = row * N + col;   // acceso por columnas (salto de N)
            C[idx] = A[idx] + B[idx];
        }
    }
}

void matAddHost(float* h_C, const float* h_A, const float* h_B, int N)
{
    size_t bytes = (size_t)N * (size_t)N * sizeof(float);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // -------- Lanzamiento del kernel (D) --------
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    matAddCol<<<blocks, threads>>>(d_A, d_B, d_C, N);
    // ---------------------------------------------------------------

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}




/*
E) Ventajas y desventajas de cada diseño
| Diseño                    | Paralelismo         | Acceso a memoria                                           | Pros                                                                 | Contras                                                                                   |
|--------------------------|---------------------|-------------------------------------------------------------|----------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| 1 hilo = 1 elemento (B)  | Máximo (N² hilos)   | Coalescente natural en 2D (con 16×16)                      | Mejor uso de la GPU, alta ocupación, mapea directo a datos 2D, sencillo | Más hilos (pero es justamente lo que queremos en GPU)                                     |
| 1 hilo = 1 fila (C)      | Solo N hilos        | Dentro del hilo es contiguo; entre hilos hay stride de N   | Código simple; buena localidad intra-hilo                           | Poca paralelización si N no es muy grande; coalescencia pobre entre hilos → menor ancho de banda efectivo |
| 1 hilo = 1 columna (D)   | Solo N hilos        | Acceso con stride de N dentro del hilo y entre hilos       | Implementación trivial                                               | Muy mala eficiencia de memoria global y muy poco paralelismo; frecuentemente el peor       |

Conclusión: para suma de matrices (y la mayoría de operaciones elemento–a–elemento en 2D), el diseño B: “1 hilo = 1 elemento” con grid/bloque 2D es el más eficiente y escalable.
*/



