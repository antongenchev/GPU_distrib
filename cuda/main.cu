#include <iostream>
#include <vector>
#include <string>
#include <curand_kernel.h>
#include <iomanip>
#include <fstream>
#include <chrono>

using namespace std;

// Thread count info
#define NUM_THREADS 128 // Number of threads per block
#define NUM_BLOCKS 16 // Number of blocks in the grid
#define TOTAL_THREADS (NUM_THREADS * NUM_BLOCKS)  // Total number of threads

#define LEN_RESULT 100 // adjust
#define LEN_SHARED_MEM 100 // adjust
#define LEN_InputKernelMainFunc 100 // adjust

// Global variable to hold the ranges
unsigned long long range[2];
unsigned long long threadRanges[TOTAL_THREADS][2]; // To store start and end of range for each thread

__constant__ int shared_memory[LEN_SHARED_MEM];

// Function to parse command-line arguments for range values
void parseCommandLineArguments(int argc, char* argv[]) {
    // Default values
    range[0] = 0;  // range_from default
    range[1] = 0;  // range_to default
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        // Parse --range_from=
        if (arg.find("--range_from=") == 0) {
            range[0] = std::stoull(arg.substr(13).c_str());
        }
        // Parse --range_to=
        else if (arg.find("--range_to=") == 0) {
            range[1] = std::stoull(arg.substr(11).c_str());
        }
    }
}

void calculateThreadRanges(unsigned long long rangeStart, unsigned long long rangeEnd, unsigned long long threadRanges[TOTAL_THREADS][2]) {
    /* Divide the range into multiple ranges, one for each thread */
    unsigned long long totalRangeSize = rangeEnd - rangeStart + 1;
    unsigned long long chunkSize = totalRangeSize / TOTAL_THREADS; // Size for each thread
    unsigned long long remainder = totalRangeSize % TOTAL_THREADS; // Extra elements to distribute
    unsigned long long currentStart = rangeStart;
    for (int i = 0; i < TOTAL_THREADS; ++i) {
        // Each thread gets at least 'chunkSize' elements, plus one more if there are remainders
        unsigned long long currentEnd = currentStart + chunkSize - 1;
        if (i < remainder) {
            currentEnd += 1; // Distribute the remainder to the first few threads
        }
        // Store the calculated range for this thread
        threadRanges[i][0] = currentStart;
        threadRanges[i][1] = currentEnd;
        // Update the starting point for the next thread
        currentStart = currentEnd + 1;
    }
}

__device__ void getInputKernelMainFunc(unsigned long long n, int* InputKernelMainFunc) {
    /* Given a number n return the corresponding input needed for the KernelMainFunc */
    // TODO
}

__device__ void KernelMainFunc(int InpputKernelMainFunc[], int* result) {
    // Run the main kernel funcion
}

__global__ void CUDA_kernel(unsigned long long* threadRanges, int* result) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= TOTAL_THREADS) return; // Safety check

    int InputKernelMainFunc[LEN_InputKernelMainFunc];

    // Each thread uses its assigned range
    unsigned long long rangeStart = threadRanges[tid * 2];
    unsigned long long rangeEnd = threadRanges[tid * 2 + 1];

    for (unsigned long long n = rangeStart; n <= rangeEnd; ++n) {
        getInputKernelMainFunc(n, InputKernelMainFunc);
        KernelMainFunc(InputKernelMainFunc, result);
    }
}

int main(int argc, char* argv[]) {
    // Parse the command-line arguments for range
    parseCommandLineArguments(argc, argv);

    // Get the range for each thread
    calculateThreadRanges(range[0], range[1], threadRanges);
    unsigned long long flattenedThreadRanges[TOTAL_THREADS * 2];
    for (int i = 0; i < TOTAL_THREADS; ++i) {
        flattenedThreadRanges[i * 2] = threadRanges[i][0];
        flattenedThreadRanges[i * 2 + 1] = threadRanges[i][1];
    }
    // Allocate memory on the GPU for flattened threadRanges
    unsigned long long* d_threadRanges;
    cudaMalloc((void**)&d_threadRanges, sizeof(unsigned long long) * 2 * TOTAL_THREADS);
    // Copy flattened threadRanges from host to device
    cudaMemcpy(d_threadRanges, flattenedThreadRanges, sizeof(unsigned long long) * 2 * TOTAL_THREADS, cudaMemcpyHostToDevice);

    // Allocate memory for storing results
    int* devResults;
	int* hostResults = new int[LEN_RESULT];
    for (int i = 0; i < LEN_RESULT; ++ i) {
        hostResults[i] = 0;
    }
    cudaMalloc((void**)&devResults, LEN_RESULT * sizeof(int));

    // Run the CUDA kernel
    auto start = std::chrono::high_resolution_clock::now();
    CUDA_kernel<<<NUM_BLOCKS, NUM_THREADS>>>(d_threadRanges, devResults);
    cudaDeviceSynchronize();
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = stop - start;
    // Print the time taken by the kernel
    std::cout << "Time taken by kernel: " << duration.count() << " ms" << std::endl;
    // Print any CUDA errors
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    // Copy results back to host from device
    cudaMemcpy(hostResults, devResults, LEN_RESULT * sizeof(int), cudaMemcpyDeviceToHost);

    // // Write the results to a file:
    std::ofstream outFile("results.txt"); // Create and open file
    // Check if file is opened successfully
    if (outFile.is_open()) {
        // Write results to the file
        for (int i = 0; i < LEN_RESULT; ++i) {
            outFile << "result[" << i << "]: " << hostResults[i] << "\n";
        }
        outFile.close();  // Close the file after writing
        std::cout << "Results successfully written to results.txt\n";
    } else {
        std::cerr << "Error: Could not open file for writing.\n";
    }

    // Free alocated memory on the CPU
    delete[] hostResults;
    // Free alocated memory on the GPU
	cudaFree(devResults);

    return 0;
}