#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <algorithm>

namespace cg = cooperative_groups;

// --------------------------------------------------------------------------- // 
//                     Error-checking macro for CUDA calls                     //
// --------------------------------------------------------------------------- //
#define CHECK_CUDA_ERR(x)                                      \
    do {                                                       \
        cudaError_t err = x;                                   \
        if (err != cudaSuccess) {                              \
            fprintf(stderr,                                     \
                    "CUDA Error at %s:%d -> %s\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                \
        }                                                      \
    } while (0)


// Maksymalne rozmiary danych
const int MAX_ALPHABET_SIZE = 100;
const int MAX_WORD_SIZE = 1024;

__device__ void sleep_milliseconds(unsigned int ms) {
    // Get the GPU clock frequency (in kHz)
    unsigned long long int start = clock64();
    unsigned long long int wait_cycles = ms * (1000000); // Convert ms to nanoseconds

    // Busy-wait loop
    while ((clock64() - start) < wait_cycles) {
        // Do nothing, just loop
    }
}

/* ========================= EXAMPLE FUNCTIONS ============================= */
__global__ void shfl_up_example() {
    int x = threadIdx.x; // Each thread initializes its own value
    int y = __shfl_up_sync(0xFFFFFFFF, x, 1); // Shift up by 1 with full mask

    int laneId = threadIdx.x % warpSize; // Calculate lane ID

    if (laneId >= 1) { // Only valid for threads where laneId >= 1
        printf("Thread %d: x = %d, y (from thread %d) = %d\n", threadIdx.x, x, threadIdx.x - 1, y);
    } else {
        printf("Thread %d: x = %d, y = Undefined\n", threadIdx.x, x);
    }
}

__global__ void example_warp_shuffle_kernel() {
    cg::grid_group grid = cg::this_grid();

    const int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int warpIndex = threadIdx.x / warpSize;
    const int laneIndex = threadIdx.x % warpSize;
    sleep_milliseconds(threadIndex * 100);
    printf("Thread Index: %d\n", threadIndex);
    //__syncthreads();
    grid.sync();
    sleep_milliseconds(threadIndex * 100);
    printf("Warp Index: %d\n", warpIndex);
    //__syncthreads();
    grid.sync();
    sleep_milliseconds(threadIndex * 100);
    printf("Lane Index: %d\n", laneIndex);
    //__syncthreads();
    grid.sync();
    printf("Warp Size: %d\n", warpSize);
}
/* ======================================================================== */

/* ============================ CALCULATE X MATRIX KERNEL ============================= */
__global__ void calculateXMatrix(int *X, char *Q, char *T, int row_number, int col_number) {
    /* PREPARATION OF NEEDED DATA */
    const int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int rowIndex = threadIndex;
    const int baseFlattenIndex = rowIndex * col_number;
    const int i = rowIndex;

    /* FOR LOOP */
    if(threadIndex < row_number) {
        for(int j = 0; j < col_number; j++) {
            if(j == 0) { X[baseFlattenIndex + j] = 0; }
            else if(T[j - 1] == Q[i]) { X[baseFlattenIndex + j] = j; }
            else { X[baseFlattenIndex + j] = X[baseFlattenIndex + j - 1]; }
        }
    }
}
/* ==================================================================================== */

/* ==================== HELPER FUNCTIONS =====================*/
__device__ int min_of_three(int a, int b, int c) {
    return min(min(a, b), c);
}
/* ============================================================*/

/* ============================ CALCULATE D MATRIX KERNEL ============================= */
__global__ void calculateDMatrixNaive(int* D, int *X, char *Q, char *T, char *P, int rows_D, int cols_D, int rows_X, int cols_X) {
    /* PREPARATION OF NEEDED DATA */
    cg::grid_group grid = cg::this_grid();
    const int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    // kolumna ktora bedziemy przetwarzac
    const int j = threadIndex;
    // const int i = 0;

    // printf("Hello form thread %d, rows: %d\n", j, rows_D);
    // grid.sync();
    //__syncthreads();
    // printf("Let's start calculations! Rows: %d\n", rows_D);
    if (j < cols_D) {
        for(int i = 0; i < rows_D; i++) {
            // calculating l
            // int l = -1;  // Initialize `l` to an invalid value
            // char P_prev = (i > 0) ? P[i - 1] : -1; // Example: Get P[i-1] from T

            // // Find P[i-1] in Q
            // for (int k = 0; k < rows_X; k++) {
            //     if (Q[k] == P_prev) {
            //         l = k;  // Found the index
            //         break;
            //     }
            // }

            int l = (i > 0 && P[i - 1] >= 'A' && P[i - 1] <= 'Z') ? (P[i - 1] - 'A') : -1;

            // THIS CALCULATES ASM MATRIX
            // if(i == 0) { D[i * cols_D + j] = 0; }

            // THIS CALCULATES LEVENSHTEIN DISTANCE
            if(i == 0) { D[i * cols_D + j] = j; }
            else if(j == 0) { D[i * cols_D + j] = i; }
            else if(T[j - 1] == P[i - 1]) { D[i * cols_D + j] = D[(i - 1) * cols_D + (j - 1)]; }
            else if(X[l * cols_X + j] == 0) { 
                D[i * cols_D + j] = 
                1 + min_of_three(
                    D[(i - 1) * cols_D + j],
                    D[(i - 1) * cols_D + (j - 1)],
                    i + j - 1
                ); }
            else { 
                D[i * cols_D + j] = 
                1 + min_of_three(
                    D[(i - 1) * cols_D + j],
                    D[(i - 1) * cols_D + (j - 1)],
                    D[(i - 1) * cols_D + X[l * cols_X + j] - 1] + (j - 1 - X[l * cols_X + j])
                ); 
            }
            grid.sync();
            //__syncthreads();
            //printf("Finished row %d from thread %d!\n", i, j);

            // if(j == 0) {
            //     printf("After %d-th iteration:\n", i);
            //     for (int k = 0; k < rows_D; k++) {
            //         for (int l = 0; l < cols_D; l++) {
            //             printf("%d |", D[k * cols_D + l]);
            //         }
            //         printf("\n");
            //     }
            // }  
        }
    }
}
/* ==================================================================================== */

/* ======================= CALCULATE D MATRIX KERNEL ADVANCED ========================= */
__global__ void calculateDMatrixAdvanced(int* D, int *X, char *Q, char *T, char *P, int rows_D, int cols_D, int rows_X, int cols_X) {
    /* PREPARATION OF NEEDED DATA */
    cg::grid_group grid = cg::this_grid();
    const int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    // kolumna ktora bedziemy przetwarzac
    const int j = threadIndex;

    // numer wątku w warpie
    int laneId = threadIdx.x % warpSize; // Calculate lane ID
    
    int w = laneId;

    // numer wątku w bloku
    int threadId = threadIndex % blockDim.x;

    int AVar;
    int BVar;
    int CVar;
    int DVar;
    if (j < cols_D) {
        /* Deklaracja AVar, BVar, CVar, DVar */
        
        for(int i = 0; i < rows_D; i++) {
            
            // inicjalizacja pamięci wspoldzielonej miedzy watkami
            // __shared__ int sharedData[blockDim.x]; // Adjust size as needed
            
            // if (i > 0)
            //     sharedData[threadId] = 

            // Calculate l directly using ASCII values
            int l = (i > 0 && P[i - 1] >= 'A' && P[i - 1] <= 'Z') ? (P[i - 1] - 'A') : -1;

            // THIS CALCULATES LEVENSHTEIN DISTANCE
            if(i == 0) { 
                D[i * cols_D + j] = j;
                DVar = j; 
            }
            else {

                if(j % warpSize == 0) {
                    AVar = __shfl_up_sync(0xFFFFFFFF, DVar, 1);
                    AVar = D[(i - 1) * cols_D + (j - 1)];
                }
                else {
                    AVar = __shfl_up_sync(0xFFFFFFFF, DVar, 1);
                }
                
                BVar = DVar;
                CVar = D[(i - 1) * cols_D + X[l * cols_X + j] - 1];
                
                if(j == 0) {
                    DVar = i;
                }
                else if (T[j - 1] == P[i - 1]) {
                    DVar = AVar;
                }
                else if (X[l * cols_X + j] == 0) {
                    DVar = 1 + min_of_three(AVar, BVar, i + j - 1);
                }
                else {
                    DVar = 1 + min_of_three(AVar, BVar, CVar + (j - 1 - X[l * cols_X + j]));
                }
                D[i * cols_D + j] = DVar;
            } 
            
            // synchronizujemy wszystkie watki w obrebie gridu
            grid.sync();

            // if(j == 0) {
            //     printf("After %d-th iteration:\n", i);
            //     for (int k = 0; k < rows_D; k++) {
            //         for (int l = 0; l < cols_D; l++) {
            //             printf("%d |", D[k * cols_D + l]);
            //         }
            //         printf("\n");
            //     }
            // }  
        }
    }
}
/* ==================================================================================== */

/* ====================================== MAIN ======================================== */
int main(int argc, char *argv[]) {
    // Sprawdzanie liczby argumentów
    if (argc != 2) {
        std::cerr << "Użycie: " << argv[0] << " <nazwa_pliku>" << std::endl;
        return 1;
    }

    /* MEMORY CHECK */
    int device_id = 0;
    cudaDeviceProp device_prop;

    // Get properties of the device
    cudaGetDeviceProperties(&device_prop, device_id);

    // Print total global memory in MB
    std::cout << "Device Name: " << device_prop.name << std::endl;
    std::cout << "Total Global Memory: " << device_prop.totalGlobalMem / (1024.0 * 1024.0) << " MB" << std::endl;

    // Pobieranie nazwy pliku z argumentów programu
    const char *filename = argv[1];

    // Deklaracja zmiennych
    int a_len;
    char A_read[MAX_ALPHABET_SIZE];
    int n, m;
    char T_read[MAX_WORD_SIZE];
    char P_read[MAX_WORD_SIZE];

    // Otwieranie pliku
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Nie można otworzyć pliku: " << filename << std::endl;
        return 1;
    }

    // Wczytywanie danych
    file >> a_len;
    file >> A_read;
    file >> n;
    file >> T_read;
    file >> m;
    file >> P_read;

    file.close(); // Zamknięcie pliku
    
    if (std::strlen(T_read) != n || std::strlen(P_read) != m) {
        std::cerr << "Niezgodność długości słów!" << std::endl;
        return 1;
    }

    /* stworzenie tablic do kopiowania */
    int X[a_len * (n+1)] = {0};
    char Q[a_len];
    for(int i = 0; i < a_len; i++) Q[i] = A_read[i]; 
    char T[n];
    for(int i = 0; i < n; i++) T[i] = T_read[i];
    char P[m];
    for(int i = 0; i < m; i++) P[i] = P_read[i];

    /* WSKAZNIKI DO ZALOKOWANIA PAMIECI NA GPU */
    int* d_X;
    char* d_Q;
    char* d_T;
    char* d_P;

    /* TABLICA X */
    // Allocate memory on the device
    size_t size_X = a_len * (n + 1) * sizeof(int);
    cudaMalloc((void**)&d_X, size_X);
    // Copy the array from host to device
    cudaMemcpy(d_X, X, size_X, cudaMemcpyHostToDevice);

    /* ALFABET Q */
    size_t size_Q = a_len * sizeof(char);
    cudaMalloc((void**)&d_Q, size_Q);
    // Copy the array from host to device
    cudaMemcpy(d_Q, Q, size_Q, cudaMemcpyHostToDevice);

    /* SLOWO T */
    size_t size_T = n * sizeof(char);
    cudaMalloc((void**)&d_T, size_T);
    // Copy the array from host to device
    cudaMemcpy(d_T, T, size_T, cudaMemcpyHostToDevice);

    /* SLOWO P */
    size_t size_P = m * sizeof(char);
    cudaMalloc((void**)&d_P, size_P);
    // Copy the array from host to device
    cudaMemcpy(d_P, P, size_P, cudaMemcpyHostToDevice);

    /* WYWOLUJEMY KERNEL DO OBLICZENIA MACIERZY X */
    calculateXMatrix<<<1, a_len>>>(d_X, d_Q, d_T, a_len, n + 1);

    /* SYNCHRONIZACJA */
    cudaDeviceSynchronize();

    /* KOPIUJEMY PAMIEC */
    cudaMemcpy(X, d_X, size_X, cudaMemcpyDeviceToHost);

    /* TESTOWE WYPISANIE X */
    // std::cout << std::endl << "Tablica X po obliczeniu:" << std::endl;
    // for (int i = 0; i < a_len; i++) {
    //     for (int j = 0; j < n + 1; j++) {
    //         std::cout << X[i * (n + 1) + j] << "|";
    //     }
    //     std::cout << std::endl;
    // }

    /* === TERAZ MACIERZ D === */
    // HOST
    int D[(m + 1) * (n + 1)] = {0};

    // DEVICE POINTER
    int* d_D;

    // ALOKUJEMY
    size_t size_D = (m + 1) * (n + 1) * sizeof(int);
    cudaMalloc((void**)&d_D, size_D);
    // Copy the array from host to device
    cudaMemcpy(d_D, D, size_D, cudaMemcpyHostToDevice);

    // (int* D, int *X, char *Q, char *T, char *P, int rows_D, int cols_D, int rows_X, int cols_X)
    /* LAUNCHING NORMAL KERNEL */
    // calculateDMatrixNaive<<<1, n+1>>>(d_D, d_X, d_Q, d_T, d_P, m+1, n+1, a_len, n+1);

    /* ====================== LAUNCHING COOPERATIVE KERNEL ======================== */
    // Define grid and block dimensions
    int threadsPerBlock = n + 1; // As in your original kernel call
    int numBlocks = 1; // Single block, adjust if needed
    dim3 grid(numBlocks);
    dim3 block(threadsPerBlock);

    // Define shared memory size (if required, set appropriately)
    size_t sharedMemSize = 0;

    int rows_D = m + 1;
    int cols_D = n + 1;
    int rows_X = a_len;
    int cols_X = n + 1;

    // Kernel arguments
    void* kernelArgs[] = {
        &d_D,
        &d_X,
        &d_Q,
        &d_T,
        &d_P,
        &rows_D,
        &cols_D,
        &rows_X,
        &cols_X
    };

    /* Launch the kernel using cudaLaunchCooperativeKernel */
    // cudaError_t err = cudaLaunchCooperativeKernel(
    //     (void*)calculateDMatrixNaive,
    //     grid,
    //     block,
    //     kernelArgs,
    //     sharedMemSize
    //);

    printf("================= OUTPUT FROM ADVANCED KERNEL ===================\n");

    cudaError_t err0 = cudaLaunchCooperativeKernel(
        (void*)calculateDMatrixAdvanced,
        grid,
        block,
        kernelArgs,
        sharedMemSize
    );

    CHECK_CUDA_ERR(cudaDeviceSynchronize());

    /* KOPIUJEMY PAMIEC */
    CHECK_CUDA_ERR(cudaMemcpy(D, d_D, size_D, cudaMemcpyDeviceToHost));

    // Wyświetlanie wyników
    std::cout << "Odległość Levenshteina: " << D[(m + 1) * (n + 1) - 1] << std::endl;

    /* WYZEROWANIE TABLICY D */
    for (int i = 0; i < m + 1; i++) {
        for (int j = 0; j < n + 1; j++) {
            D[i * (n + 1) + j] = 0;
        }
    }


    printf("================= OUTPUT FROM NAIVE KERNEL ===================\n");
    /* Launch the kernel using cudaLaunchCooperativeKernel */
    cudaError_t err = cudaLaunchCooperativeKernel(
        (void*)calculateDMatrixNaive,
        grid,
        block,
        kernelArgs,
        sharedMemSize
    );

    // Check for launch errors
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }

    /* SYNCHRONIZACJA I SPRAWDZENIE BLEDOW */
    CHECK_CUDA_ERR(cudaGetLastError());
    CHECK_CUDA_ERR(cudaDeviceSynchronize());

    /* KOPIUJEMY PAMIEC */
    CHECK_CUDA_ERR(cudaMemcpy(D, d_D, size_D, cudaMemcpyDeviceToHost));

    /* TESTOWE WYPISANIE D */
    // std::cout << std::endl << "Po obliczeniu tablicy D:" << std::endl;
    // for (int i = 0; i < m + 1; i++) {
    //     for (int j = 0; j < n + 1; j++) {
    //         std::cout << D[i * (n + 1) + j] << "|";
    //     }
    //     std::cout << std::endl;
    // }

    //example_warp_shuffle_kernel<<<2, 64>>>();

    // void* kernel_args[] = {};
    // cudaLaunchCooperativeKernel((void*)example_warp_shuffle_kernel, 1, 32, kernel_args);

    // cudaDeviceSynchronize();

    // Launch kernel
    // shfl_up_example<<<1, 32>>>();

    // // Wait for GPU to finish
    // cudaDeviceSynchronize();

    // Wyświetlanie wyników
    std::cout << "\nAlfabet: " << A_read << " (dlugość: " << a_len << ")" << std::endl;
    std::cout << "Słowo 1: " << T_read << " (dlugość: " << n << ")" << std::endl;
    std::cout << "Słowo 2: " << P_read << " (dlugość: " << m << ")" << std::endl;
    std::cout << "Odległość Levenshteina: " << D[(m + 1) * (n + 1) - 1] << std::endl;

    return 0;
}