#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <chrono> 

namespace cg = cooperative_groups;

/* STALE DO ODZYSKANIA SCIEZKI */
#define MATCH   0
#define REPLACE 1
#define INSERT  2
#define DELETE  3
#define JUMP    4 // Multi-character jump from X[l, j]

// Maksymalne rozmiary danych
const int MAX_ALPHABET_SIZE = 100;
const int MAX_WORD_SIZE = 25200;
const int THREADS_PER_BLOCK = 1024;

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
// GPU
__device__ int min_of_three(int a, int b, int c) {
    return min(min(a, b), c);
}
// CPU
static inline int min_of_three_cpu(int a, int b, int c) {
    return std::min(a, std::min(b, c));
}
/* ============================================================*/

/* ========================= LEVENSHTEIN DISTANCE ON CPU ============================= */
int levenshteinNaiveCPU(
    const char* T, int n,
    const char* P, int m,
    const int* X, int a_len, 
    int colsX  // (n+1)
)
{
    // We'll create a local 2D array D of size (m+1) x (n+1).
    // For large n, m, consider using new[] or std::vector to avoid large stack usage.
    int* D = new int[(m + 1) * (n + 1)];

    // Initialize D with 0s
    std::memset(D, 0, sizeof(int) * (m+1) * (n+1));

    // Fill in row 0: D[0, j] = j
    for(int j = 0; j <= n; j++) {
        D[0 * (n+1) + j] = j;
    }
    // Fill in col 0: D[i, 0] = i
    for(int i = 0; i <= m; i++) {
        D[i * (n+1) + 0] = i;
    }

    // Now compute the Levenshtein distance with the same logic as your naive CUDA kernel
    for(int i = 1; i <= m; i++) {
        // Convert P[i-1] to index l if it is [A-Z], else -1
        char pChar = P[i-1];
        int l = -1;
        if(pChar >= 'A' && pChar <= 'Z') {
            l = pChar - 'A';  // e.g. 'A'->0, 'B'->1, ...
        }

        for(int j = 1; j <= n; j++) {
            if(T[j - 1] == P[i - 1]) {
                // Same character => no additional cost
                D[i * (n+1) + j] = D[(i - 1) * (n+1) + (j - 1)];
            }
            else {
                // If P[i-1] is outside 'A'..'Z', then l = -1 => skip X usage
                if(l < 0) {
                    // Then we can't use X for last occurrence => fallback
                    D[i * (n+1) + j] = 
                        1 + min_of_three_cpu(
                              D[(i - 1)*(n+1) + j],     // deletion
                              D[(i - 1)*(n+1) + (j-1)], // substitution
                              D[i*(n+1) + (j-1)]        // insertion
                          ); 
                }
                else {
                    // We do have l in [0 .. a_len-1], so we can check X[l, j].
                    int lastOcc = X[l * colsX + j];  // X[l,j]
                    if(lastOcc == 0) {
                        // then use the 'i+j-1' fallback
                        D[i * (n+1) + j] = 
                            1 + min_of_three_cpu(
                                  D[(i - 1)*(n+1) + j],
                                  D[(i - 1)*(n+1) + (j - 1)],
                                  (i + j - 1)
                              );
                    }
                    else {
                        // Note the same formula from the kernel
                        int bigTerm = D[(i - 1)*(n+1) + (lastOcc - 1)]
                                      + ((j - 1) - lastOcc);
                        D[i * (n+1) + j] = 
                            1 + min_of_three_cpu(
                                  D[(i - 1)*(n+1) + j],
                                  D[(i - 1)*(n+1) + (j - 1)],
                                  bigTerm
                              );
                    }
                }
            }
        }
    }

    // The Levenshtein distance is in the bottom-right corner of D
    int distance = D[m * (n+1) + n];

    // Clean up
    delete[] D;
    return distance;
}
/* =================================================================================== */

/* ============================ CALCULATE D MATRIX KERNEL ============================= */
__global__ void calculateDMatrixNaive(int* D, int *X, char *Q, char *T, char *P, int rows_D, int cols_D, int rows_X, int cols_X) {
    /* PREPARATION OF NEEDED DATA */
    cg::grid_group grid = cg::this_grid();
    const int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    // kolumna ktora bedziemy przetwarzac
    const int j = threadIndex;

    if (j < cols_D) {
        for(int i = 0; i < rows_D; i++) {
            // ASSUMING THAT ALPHABET IS A-Z CAPITALS
            int l = (i > 0 && P[i - 1] >= 'A' && P[i - 1] <= 'Z') ? (P[i - 1] - 'A') : -1;

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

     /* Deklaracja AVar, BVar, CVar, DVar */
    int AVar;
    int BVar;
    int CVar;
    int DVar;
    if (j < cols_D) {
        for(int i = 0; i < rows_D; i++) {
            // Calculate l directly using ASCII values
            int P_prev = P[i - 1];
            int l = (i > 0 && P_prev >= 'A' && P_prev <= 'Z') ? (P_prev - 'A') : -1;

            // THIS CALCULATES LEVENSHTEIN DISTANCE
            if(i == 0) { 
                D[i * cols_D + j] = j;
                DVar = j; 
            }
            else {
                if(j == 0 || j % warpSize != 0) {
                    AVar = __shfl_up_sync(0xFFFFFFFF, DVar, 1);
                }   
                else if(j % warpSize == 0) {
                    AVar = __shfl_up_sync(0xFFFFFFFF, DVar, 1);
                    AVar = D[(i - 1) * cols_D + (j - 1)];
                }
                // else {
                //     AVar = __shfl_up_sync(0xFFFFFFFF, DVar, 1);
                // }

                BVar = DVar;
                int X_l_j = X[l * cols_X + j];
                CVar = D[(i - 1) * cols_D + X_l_j - 1];
                
                if(j == 0) {
                    DVar = i;
                } 
                else if (T[j - 1] == P_prev) {
                    DVar = AVar;
                }
                else if (X_l_j == 0) {
                    DVar = 1 + min_of_three(AVar, BVar, i + j - 1);
                }
                else {
                    DVar = 1 + min_of_three(AVar, BVar, CVar + (j - 1 - X_l_j));
                }

                D[i * cols_D + j] = DVar;
            } 
            
            // synchronizujemy wszystkie watki w obrebie gridu
            grid.sync();
        }
    }
}
/* ==================================================================================== */

/* ======================= CALCULATE D MATRIX KERNEL WITH TRANSFORMATIONS ========================= */
__global__ void calculateDMatrixWithTransformations(int* D, int *X, char *Q, char *T, char *P, int rows_D, int cols_D, int rows_X, int cols_X, int* Op, int *JumpLen) {
    /* PREPARATION OF NEEDED DATA */
    cg::grid_group grid = cg::this_grid();
    const int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    // kolumna ktora bedziemy przetwarzac
    const int j = threadIndex;

     /* Deklaracja AVar, BVar, CVar, DVar */
    int AVar, BVar, CVar, DVar;

    if (j < cols_D) {
        for(int i = 0; i < rows_D; i++) {
            // Calculate l directly using ASCII values
            int P_prev = P[i - 1];
            int l = (i > 0 && P_prev >= 'A' && P_prev <= 'Z') ? (P_prev - 'A') : -1;

            // THIS CALCULATES LEVENSHTEIN DISTANCE
            if(i == 0) { 
                D[i * cols_D + j] = j;
                DVar = j;
                // Operation is Insert if j>0, else Match if j=0
                Op[i*cols_D + j] = (j > 0) ? INSERT : MATCH;
                JumpLen[i*cols_D + j] = 0;
            }
            else {
                AVar = __shfl_up_sync(0xFFFFFFFF, DVar, 1);
                if(j % warpSize == 0) {
                    AVar = D[(i - 1) * cols_D + (j - 1)];
                }

                BVar = DVar;
                // int X_l_j = X[l * cols_X + j];
                // CVar = D[(i - 1) * cols_D + X_l_j - 1];
                
                int X_l_j = X[l * cols_X + j];
                CVar = INT_MAX;

                if (X_l_j > 0) {
                    CVar = D[(i - 1) * cols_D + (X_l_j - 1)];
                }

                if(j == 0) {
                    DVar = i;
                    Op[i*cols_D + j] = DELETE;
                    JumpLen[i*cols_D + j] = 0;
                } 
                else if (T[j - 1] == P_prev) {
                    // Match
                    DVar = AVar;
                    Op[i*cols_D + j] = MATCH;
                    JumpLen[i*cols_D + j] = 0;
                }
                else if (X_l_j == 0) {
                    // We pick the min of (AVar, BVar, i + j - 1) + 1
                    // If i+j-1 is chosen => JUMP
                    // else if AVar => REPLACE
                    // else if BVar => ???

                    int costReplace = AVar + 1;
                    int costDelete  = BVar + 1;
                    int costJump    = (i + j - 1) + 1; // multi-step skip

                    DVar = min_of_three(costReplace, costDelete, costJump);
                    if (DVar == costReplace) {
                        Op[i*cols_D + j] = REPLACE;
                        JumpLen[i*cols_D + j] = 0;
                    } else if (DVar == costDelete) {
                        Op[i*cols_D + j] = DELETE;
                        JumpLen[i*cols_D + j] = 0;
                    } else {
                        // The JUMP path
                        Op[i*cols_D + j] = JUMP;
                        // how many columns are we skipping in T?
                        // Typically: jumpLen = (j - 1) - (some index?), 
                        // but if the formula is i + j -1, we might store that as well.
                        JumpLen[i*cols_D + j] = (i + j - 1); // Example
                    }
                }
                else {
                    // We pick min( AVar, BVar, CVar + (j-1 - X_l_j ) ) + 1
                    int costReplace = AVar + 1;
                    int costDelete  = BVar + 1;
                    int costJump    = CVar + (j - 1 - X_l_j) + 1;

                    DVar = min_of_three(costReplace, costDelete, costJump);
                    if (DVar == costReplace) {
                        Op[i*cols_D + j] = REPLACE;
                        JumpLen[i*cols_D + j] = 0;
                    } else if (DVar == costDelete) {
                        Op[i*cols_D + j] = DELETE;
                        JumpLen[i*cols_D + j] = 0;
                    } else {
                        Op[i*cols_D + j] = JUMP;
                        JumpLen[i*cols_D + j] = (j - 1 - X_l_j);
                    }
                }

                D[i * cols_D + j] = DVar;
            } 
            
            // synchronizujemy wszystkie watki w obrebie gridu
            grid.sync();
        }
    }
}
/* ==================================================================================== */

/* ======================= CALCULATE D MATRIX WITH SHARED MEMORY ========================= */
__global__ void calculateDMatrixShared(int* D, int *X, char *Q, char *T, char *P, int rows_D, int cols_D, int rows_X, int cols_X) {
    // For dynamic shared memory, we must specify 'extern __shared__':
    __shared__ char sT[THREADS_PER_BLOCK];
    
    int threadIdInBlock = threadIdx.x;

    extern __shared__ char s[];

    // // We'll store T in sT and P in sP, laid out back-to-back
    char* sP = &s[0];      // covers indices [0..n-1]

    // 1) Copy T and P from global memory into shared memory
    //    We'll do it in a loop so multiple threads help copy.
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Copy T into sT
    if(tid < cols_D - 1) {
        sT[threadIdInBlock] = T[tid]; 
    }

    if(tid < rows_D - 1) {
        sP[tid] = P[tid];
    }

    // Sync so that sT and sP are fully loaded before use
    __syncthreads();

    /* PREPARATION OF NEEDED DATA */
    cg::grid_group grid = cg::this_grid();
    const int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    // kolumna ktora bedziemy przetwarzac
    const int j = threadIndex;

     /* Deklaracja AVar, BVar, CVar, DVar */
    int AVar;
    int BVar;
    int CVar;
    int DVar;
    if (j < cols_D) {
        for(int i = 0; i < rows_D; i++) {
            // Calculate l directly using ASCII values
            int l = -1;
            int P_prev;
            if(i > 0) {
                P_prev = sP[i - 1];
                l = (i > 0 && P_prev >= 'A' && P_prev <= 'Z') ? (P_prev - 'A') : -1;
            }

            // THIS CALCULATES LEVENSHTEIN DISTANCE
            if(i == 0) { 
                D[i * cols_D + j] = j;
                DVar = j; 
            }
            else {
                AVar = __shfl_up_sync(0xFFFFFFFF, DVar, 1);
                // if(j == 0 || j % warpSize != 0) {
                //     AVar = __shfl_up_sync(0xFFFFFFFF, DVar, 1);
                // }   
                // else
                if(j % warpSize == 0) {
                    //AVar = __shfl_up_sync(0xFFFFFFFF, DVar, 1);
                    AVar = D[(i - 1) * cols_D + (j - 1)];
                }
                // else {
                //     AVar = __shfl_up_sync(0xFFFFFFFF, DVar, 1);
                // }

                BVar = DVar;
                int X_l_j = X[l * cols_X + j];
                CVar = D[(i - 1) * cols_D + X_l_j - 1];
                
                if(j == 0) {
                    DVar = i;
                } 
                else if (sT[threadIdInBlock - 1] == P_prev) {
                    DVar = AVar;
                }
                else if (X_l_j == 0) {
                    DVar = 1 + min_of_three(AVar, BVar, i + j - 1);
                }
                else {
                    DVar = 1 + min_of_three(AVar, BVar, CVar + (j - 1 - X_l_j));
                }

                D[i * cols_D + j] = DVar;
            } 
            
            // synchronizujemy wszystkie watki w obrebie gridu
            grid.sync();
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

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("Free memory: %zu MB, Total memory: %zu MB\n", free_mem / (1024 * 1024), total_mem / (1024 * 1024));

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
    int* X = new int[a_len * (n + 1)];
    std::memset(X, 0, (a_len*(n+1)) * sizeof(int));
    char* Q = new char[a_len];
    for(int i = 0; i < a_len; i++) Q[i] = A_read[i]; 
    char* T = new char[n];
    for(int i = 0; i < n; i++) T[i] = T_read[i];
    char* P = new char[m];
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

    //printf("[MEMCHECK] WORKS\n");

    /* WYWOLUJEMY KERNEL DO OBLICZENIA MACIERZY X */
    calculateXMatrix<<<1, a_len>>>(d_X, d_Q, d_T, a_len, n + 1);

    /* SYNCHRONIZACJA */
    cudaDeviceSynchronize();

    /* KOPIUJEMY PAMIEC */
    cudaMemcpy(X, d_X, size_X, cudaMemcpyDeviceToHost);


    printf("Macierz X:\n");
    for(int i = 0; i < a_len; i++){
        for(int j = 0; j < n + 1; j++) {
            printf("%d |", X[i * (n + 1) + j]);
        }
        printf("\n");
    }

    /* === TERAZ MACIERZ D === */
    // HOST
    int* D = new int[(m + 1)*(n + 1)];
    std::memset(D, 0, (m + 1)*(n + 1)*sizeof(int));


    // DEVICE POINTER
    int* d_D;

    cudaMemGetInfo(&free_mem, &total_mem);

    // ALOKUJEMY
    size_t size_D = (m + 1) * (n + 1) * sizeof(int);
    cudaError_t err1 = cudaMalloc((void**)&d_D, size_D);
    if (err1 != cudaSuccess) {
        std::cerr << "Failed to allocate memory: " << cudaGetErrorString(err1) << std::endl;
        return 1;
    }
    // Copy the array from host to device
    CHECK_CUDA_ERR(cudaMemcpy(d_D, D, size_D, cudaMemcpyHostToDevice));

    /* === TERAZ MACIERZ Op === */
    // HOST
    int* Op = new int[(m + 1)*(n + 1)];
    std::memset(Op, 0, (m + 1)*(n + 1)*sizeof(int));


    // DEVICE POINTER
    int* d_Op;

    cudaMemGetInfo(&free_mem, &total_mem);

    // ALOKUJEMY
    size_t size_Op = (m + 1) * (n + 1) * sizeof(int);
    cudaError_t err3 = cudaMalloc((void**)&d_Op, size_Op);
    if (err1 != cudaSuccess) {
        std::cerr << "Failed to allocate memory: " << cudaGetErrorString(err3) << std::endl;
        return 1;
    }
    // Copy the array from host to device
    CHECK_CUDA_ERR(cudaMemcpy(d_Op, Op, size_Op, cudaMemcpyHostToDevice));

    /* === TERAZ MACIERZ JumpLen === */
    // HOST
    int* JumpLen = new int[(m + 1)*(n + 1)];
    std::memset(JumpLen, 0, (m + 1)*(n + 1)*sizeof(int));


    // DEVICE POINTER
    int* d_JumpLen;

    cudaMemGetInfo(&free_mem, &total_mem);

    // ALOKUJEMY
    size_t size_JumpLen = (m + 1) * (n + 1) * sizeof(int);
    cudaError_t err4 = cudaMalloc((void**)&d_JumpLen, size_JumpLen);
    if (err4 != cudaSuccess) {
        std::cerr << "Failed to allocate memory: " << cudaGetErrorString(err4) << std::endl;
        return 1;
    }

    // Copy the array from host to device
    CHECK_CUDA_ERR(cudaMemcpy(d_JumpLen, JumpLen, size_JumpLen, cudaMemcpyHostToDevice));

    /* ====================== LAUNCHING COOPERATIVE KERNEL ======================== */
    // We'll assume we need (n + 1) total threads:
    int totalThreads = n + 1;

    // Typical maximum block size on many GPUs is 1024
    int maxBlockSize = 1024;

    // threadsPerBlock is the smaller of (totalThreads) and maxBlockSize
    int threadsPerBlock = (totalThreads < maxBlockSize) ? totalThreads : maxBlockSize;

    // Compute how many blocks we need so that
    // (blocks * threadsPerBlock) >= totalThreads
    int numBlocks = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

    printf("Blocks: %d, Threads per block: %d\n", numBlocks, threadsPerBlock);

    dim3 grid(numBlocks);
    dim3 block(threadsPerBlock);

    // Define shared memory size (if required, set appropriately)
    size_t sharedMemSize = 48000;

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

    // Kernel arguments
    void* kernelArgsWithTransformations[] = {
        &d_D,
        &d_X,
        &d_Q,
        &d_T,
        &d_P,
        &rows_D,
        &cols_D,
        &rows_X,
        &cols_X,
        &d_Op,
        &d_JumpLen
    };

    printf("================= OUTPUT FROM CPU ===================\n");

    // auto cpuStart = std::chrono::high_resolution_clock::now();

    // int distance = levenshteinNaiveCPU(T, n, P, m, X, a_len, n+1);

    // auto cpuEnd = std::chrono::high_resolution_clock::now();

    // auto cpuDurationMs = std::chrono::duration_cast<std::chrono::milliseconds>(cpuEnd - cpuStart).count();

    // std::cout << "CPU Levenshtein distance = " << distance << "  (";
    // std::cout << "Time: " << cpuDurationMs << " ms)" << std::endl;

    printf("================= OUTPUT FROM ADVANCED KERNEL ===================\n");
    
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("[MEMCHECK] Free memory: %zu MB, Total memory: %zu MB\n", free_mem / (1024 * 1024), total_mem / (1024 * 1024));

    // Create CUDA events
    cudaEvent_t startAdv, stopAdv;
    cudaEventCreate(&startAdv);
    cudaEventCreate(&stopAdv);

    // Record the start event
    cudaEventRecord(startAdv);

    cudaError_t err0 = cudaLaunchCooperativeKernel(
        (void*)calculateDMatrixAdvanced,
        grid,
        block,
        kernelArgs,
        sharedMemSize
    );

    // Synchronize to make sure the kernel finishes
    cudaEventRecord(stopAdv);
    cudaEventSynchronize(stopAdv);

    CHECK_CUDA_ERR(cudaGetLastError());
    CHECK_CUDA_ERR(cudaDeviceSynchronize());

    float advancedKernelTimeMs = 0.0f;
    cudaEventElapsedTime(&advancedKernelTimeMs, startAdv, stopAdv);

    /* KOPIUJEMY PAMIEC */
    CHECK_CUDA_ERR(cudaMemcpy(D, d_D, size_D, cudaMemcpyDeviceToHost));

    // Wyświetlanie wyników
    std::cout << "Odległość Levenshteina: " << D[(m + 1) * (n + 1) - 1] << std::endl;
    std::cout << "   (Time: " << advancedKernelTimeMs << " ms)" << std::endl;

    // Cleanup events
    cudaEventDestroy(startAdv);
    cudaEventDestroy(stopAdv);

    /* WYZEROWANIE TABLICY D */
    for (int i = 0; i < m + 1; i++) {
        for (int j = 0; j < n + 1; j++) {
            D[i * (n + 1) + j] = 0;
        }
    }

    printf("================= OUTPUT FROM KERNEL WITH SHARED MEMORY ===================\n");
    
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("[MEMCHECK] Free memory: %zu MB, Total memory: %zu MB\n", free_mem / (1024 * 1024), total_mem / (1024 * 1024));

    // Create CUDA events
    cudaEvent_t startSh, stopSh;
    cudaEventCreate(&startSh);
    cudaEventCreate(&stopSh);

    // Record the start event
    cudaEventRecord(startSh);

    cudaError_t err2 = cudaLaunchCooperativeKernel(
        (void*)calculateDMatrixAdvanced,
        grid,
        block,
        kernelArgs,
        sharedMemSize
    );

    // Synchronize to make sure the kernel finishes
    cudaEventRecord(stopSh);
    cudaEventSynchronize(stopSh);

    CHECK_CUDA_ERR(cudaGetLastError());
    CHECK_CUDA_ERR(cudaDeviceSynchronize());

    float sharedKernelTimeMs = 0.0f;
    cudaEventElapsedTime(&sharedKernelTimeMs, startSh, stopSh);

    /* KOPIUJEMY PAMIEC */
    CHECK_CUDA_ERR(cudaMemcpy(D, d_D, size_D, cudaMemcpyDeviceToHost));

    // Wyświetlanie wyników
    std::cout << "Odległość Levenshteina: " << D[(m + 1) * (n + 1) - 1] << std::endl;
    std::cout << "   (Time: " << sharedKernelTimeMs << " ms)" << std::endl;

    // Cleanup events
    cudaEventDestroy(startSh);
    cudaEventDestroy(stopSh);

    /* WYZEROWANIE TABLICY D */
    for (int i = 0; i < m + 1; i++) {
        for (int j = 0; j < n + 1; j++) {
            D[i * (n + 1) + j] = 0;
        }
    }

    printf("================= OUTPUT FROM KERNEL WITH TRANSFORMATIONS ===================\n");
    
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("[MEMCHECK] Free memory: %zu MB, Total memory: %zu MB\n", free_mem / (1024 * 1024), total_mem / (1024 * 1024));

    // Create CUDA events
    cudaEvent_t startOps, stopOps;
    cudaEventCreate(&startOps);
    cudaEventCreate(&stopOps);

    // Record the start event
    cudaEventRecord(startOps);

    cudaError_t err5 = cudaLaunchCooperativeKernel(
        (void*)calculateDMatrixWithTransformations,
        grid,
        block,
        kernelArgsWithTransformations,
        sharedMemSize
    );

    // Synchronize to make sure the kernel finishes
    cudaEventRecord(stopOps);
    cudaEventSynchronize(stopOps);

    CHECK_CUDA_ERR(cudaGetLastError());
    CHECK_CUDA_ERR(cudaDeviceSynchronize());

    float transformationsKernelTimeMs = 0.0f;
    cudaEventElapsedTime(&transformationsKernelTimeMs, startOps, stopOps);

    /* KOPIUJEMY PAMIEC */
    CHECK_CUDA_ERR(cudaMemcpy(D, d_D, size_D, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(Op, d_Op, size_Op, cudaMemcpyDeviceToHost));

    // Wyświetlanie wyników
    std::cout << "Odległość Levenshteina: " << D[(m + 1) * (n + 1) - 1] << std::endl;
    std::cout << "   (Time: " << transformationsKernelTimeMs << " ms)" << std::endl;

    // Cleanup events
    cudaEventDestroy(startOps);
    cudaEventDestroy(stopOps);

    /* WYZEROWANIE TABLICY D */
    for (int i = 0; i < m + 1; i++) {
        for (int j = 0; j < n + 1; j++) {
            D[i * (n + 1) + j] = 0;
        }
    }

    /* Wyswietlenie tablicy skoków JumpLen */
    printf("TABLICA OPERACJI OP:\n");
    for (int i = 0; i < m + 1; i++) {
        for (int j = 0; j < n + 1; j++) {
            printf(" %d |", Op[i * (n + 1) + j]);
        }
        printf("\n");
    }

    /* Wyswietlenie tablicy skoków JumpLen */
    printf("TABLICA skokow JumpLen:\n");
    for (int i = 0; i < m + 1; i++) {
        for (int j = 0; j < n + 1; j++) {
            printf(" %d |", JumpLen[i * (n + 1) + j]);
        }
        printf("\n");
    }

    printf("================= OUTPUT FROM NAIVE KERNEL ===================\n");

    cudaEvent_t startNaive, stopNaive;
    cudaEventCreate(&startNaive);
    cudaEventCreate(&stopNaive);

    cudaEventRecord(startNaive);

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

    cudaEventRecord(stopNaive);
    cudaEventSynchronize(stopNaive);

    /* SYNCHRONIZACJA I SPRAWDZENIE BLEDOW */
    CHECK_CUDA_ERR(cudaGetLastError());
    CHECK_CUDA_ERR(cudaDeviceSynchronize());

    float naiveKernelTimeMs = 0.0f;
    cudaEventElapsedTime(&naiveKernelTimeMs, startNaive, stopNaive);

    /* KOPIUJEMY PAMIEC */
    CHECK_CUDA_ERR(cudaMemcpy(D, d_D, size_D, cudaMemcpyDeviceToHost));

    // Wyświetlanie wyników
    // std::cout << "\nAlfabet: " << A_read << " (dlugość: " << a_len << ")" << std::endl;
    // std::cout << "Słowo 1: " << T_read << " (dlugość: " << n << ")" << std::endl;
    // std::cout << "Słowo 2: " << P_read << " (dlugość: " << m << ")" << std::endl;
    std::cout << "Odległość Levenshteina: " << D[(m + 1) * (n + 1) - 1] << std::endl;
    std::cout << "   (Time: " << naiveKernelTimeMs << " ms)" << std::endl;

    cudaEventDestroy(startNaive);
    cudaEventDestroy(stopNaive);

    // === Zwolnienie pamięci na GPU
    cudaFree(d_X);
    cudaFree(d_Q);
    cudaFree(d_T);
    cudaFree(d_P);
    cudaFree(d_D);

    // === Zwolnienie pamięci na host (heap)
    delete[] X;
    delete[] Q;
    delete[] T;
    delete[] P;
    delete[] D;

    return 0;
}