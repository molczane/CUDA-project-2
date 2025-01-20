#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <iostream>

namespace cg = cooperative_groups;

// ---------------------------------------------------------------------------
// Error-checking macro for CUDA calls
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Device helper to do a warp shuffle up (for compute >= 3.0)
// For compute 7.x or later, you'd typically use __shfl_up_sync(mask, var, delta).
// ---------------------------------------------------------------------------
__device__ __forceinline__
int warpShuffleUp(int value, int delta, int width = 32)
{
#if __CUDACC_VER_MAJOR__ >= 9
    // On modern GPUs (cc 7.x+), we must specify a mask. 0xFFFFFFFF is common for 32-thread warps
    return __shfl_up_sync(0xFFFFFFFF, value, delta, width);
#else
    // On older GPUs (cc 3.0+), no mask argument is required
    return __shfl_up(value, delta, width);
#endif
}

// ---------------------------------------------------------------------------
// Kernel: Compute Levenshtein distance matrix D for strings T and P
//   T: text of length n
//   P: pattern of length m
//   D: a matrix of size (m+1) x (n+1) stored in row-major form
//
// We assume a single block large enough to hold n threads (i.e., blockDim.x >= n+1).
// For large n, you need a multi-block approach, plus additional boundary handling.
// ---------------------------------------------------------------------------
__global__
void levenshteinKernelShuffle(const char* T, int n,
                              const char* P, int m,
                              int* D)
{
    // Shared memory usage:  
    extern __shared__ char s_mem[];
    
    cg::grid_group grid = cg::this_grid();

    // We store T in the first (n+1) chars, then P in the next (m+1) chars
    char* s_T = s_mem;           
    char* s_P = s_mem + (n+1);   

    // Copy T into shared memory
    for(int idx = threadIdx.x; idx < (n+1); idx += blockDim.x)
    {
        if (idx < n) {
            s_T[idx] = T[idx];
        }
    }
    // Copy P into shared memory
    for(int idx = threadIdx.x; idx < (m+1); idx += blockDim.x)
    {
        if (idx < m) {
            s_P[idx] = P[idx];
        }
    }
    __syncthreads();  // ensure T and P are ready

    // Let thread j = threadIdx.x handle column j in each row's DP
    // BUT we do NOT return if j > n. We simply skip code if j>n.
    int j = threadIdx.x;

    // Initialize D[0][j] = j for all j <= n
    // Even if j>n, we skip this, but still do the barrier below
    if (j <= n)
    {
        D[0*(n+1) + j] = j;
    }
    __syncthreads();  // ensure row 0 is written

    // For each row i in [1..m], do the Levenshtein update
    for(int i = 1; i <= m; i++)
    {
        // Initialize D[i][0] = i (by a single thread, e.g., thread 0)
        if (threadIdx.x == 0)
        {
            D[i*(n+1) + 0] = i;
        }
        __syncthreads();  // ensure D[i][0] is set before we read it

        // Only threads with j <= n do the actual DP steps
        if (j <= n)
        {
            // Bvar = D[i-1, j]
            int Bvar = D[(i-1)*(n+1) + j];

            // Avar = D[i-1, j-1], but we get it from the neighbor (j-1) if possible
            // We'll do the warp shuffle approach for better performance
            int Avar = 0;
            if (j > 0)
            {
                int laneId = threadIdx.x & 0x1f; // within-warp lane
                if (laneId != 0)
                {
                    // get Bvar from the thread whose lane is (laneId-1)
                    int neighborVal = __shfl_up_sync(0xFFFFFFFF, Bvar, 1);
                    Avar = neighborVal;
                }
                else
                {
                    // boundary fallback if j>0 but laneId==0
                    Avar = D[(i-1)*(n+1) + (j-1)];
                }
            }
            // else if j=0 => Avar = D[i-1, -1] which is conceptually 0 or a boundary value

            // For the standard Levenshtein cost:
            // cost = 0 if s_T[j-1] == s_P[i-1], else 1
            int t_jm1 = (j > 0) ? s_T[j-1] : -1; 
            int p_im1 = s_P[i-1];
            int cost  = (t_jm1 == p_im1) ? 0 : 1;

            // D[i, j-1] is the left neighbor in the *current* row i
            // but we've not computed it yet in local memory. We'll fetch from global D which we 
            // store after each j iteration. So we do them in "lockstep" style:
            // We'll do a partial synchronization or store to shared memory. 
            // For simplicity, let's read from global D[i, j-1].
            int leftVal = (j > 0) ? D[i*(n+1) + (j-1)] : i;

            // up = D[i-1, j] + 1 = Bvar + 1
            int up   = Bvar + 1;
            // left = D[i, j-1] + 1
            int left = leftVal + 1;
            // diag = Avar + cost
            int diag = Avar + cost;

            int curDist = min(up, min(left, diag));
            // Write back D[i, j]
            D[i*(n+1) + j] = curDist;
        }
        __syncthreads();  
    } // end for i
}

// ---------------------------------------------------------------------------
// Host code to run the kernel
// ---------------------------------------------------------------------------
void runWarpShuffleLevenshtein(const std::string &hostT,
                               const std::string &hostP)
{
    int n = static_cast<int>(hostT.size());
    int m = static_cast<int>(hostP.size());

    // Allocate host memory for DP matrix:
    // We'll store D in row-major order, size = (m+1)*(n+1).
    size_t sizeD = (m+1)*(n+1)*sizeof(int);
    int* hostD = (int*)malloc(sizeD);

    // Allocate device memory
    char *devT = nullptr, *devP = nullptr;
    int  *devD = nullptr;

    CHECK_CUDA_ERR(cudaMalloc((void**)&devT, n * sizeof(char)));
    CHECK_CUDA_ERR(cudaMalloc((void**)&devP, m * sizeof(char)));
    CHECK_CUDA_ERR(cudaMalloc((void**)&devD, sizeD));

    // Copy T and P to device
    CHECK_CUDA_ERR(cudaMemcpy(devT, hostT.data(), n*sizeof(char), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(devP, hostP.data(), m*sizeof(char), cudaMemcpyHostToDevice));

    // For simplicity, zero out devD
    CHECK_CUDA_ERR(cudaMemset(devD, 0, sizeD));

    // Launch kernel
    // We want (n+1) threads in x-dimension if possible, so that each j is handled by exactly one thread.
    // blockDim.x >= (n+1). For large n, you must break this up or run multiple blocks.
    // Enough threads for j in [0..n] (or a bit more)
    dim3 blockDim( (n+1) <= 1024 ? (n+1) : 1024 );
    dim3 gridDim(1);

    // Shared memory size: (n+1 + m+1)*sizeof(char)
    size_t sharedMemSize = (n+1 + m+1)*sizeof(char);

    levenshteinKernelShuffle<<<gridDim, blockDim, sharedMemSize>>>(devT, n, devP, m, devD);

    CHECK_CUDA_ERR(cudaGetLastError());
    CHECK_CUDA_ERR(cudaDeviceSynchronize());

    // Copy D back to host
    CHECK_CUDA_ERR(cudaMemcpy(hostD, devD, sizeD, cudaMemcpyDeviceToHost));

    // Print final distance = D[m, n]
    int distance = hostD[m*(n+1) + n];
    std::cout << "Levenshtein distance D(" << hostT << "," << hostP
              << ") = " << distance << std::endl;

    // Cleanup
    free(hostD);
    CHECK_CUDA_ERR(cudaFree(devT));
    CHECK_CUDA_ERR(cudaFree(devP));
    CHECK_CUDA_ERR(cudaFree(devD));
}

// ---------------------------------------------------------------------------
// main()
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    // Example usage
    // Strings T, P
    std::string T = "CATGACTG";
    std::string P = "TACTG";

    runWarpShuffleLevenshtein(T, P);

    return 0;
}