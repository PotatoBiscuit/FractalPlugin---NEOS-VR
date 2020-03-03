
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void computeCoordsMandelbrot(double* coordX, double* coordY, int width, int height, double minX, double minY, double rangeX, double rangeY)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int y = i / width;
    int x = i % width;
    coordX[i] = minX + (rangeX / width) * x;
    coordY[i] = minY + (rangeY / height) * y;
}

__global__ void findIterationsMandelbrot(int* iterArray, double* coordX, double* coordY)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    double realZ = coordX[i];
    double imagZ = coordY[i];
    double realZ2 = 0;
    double imagZ2 = 0;
    while (iterArray[i] < 255)
    {
        iterArray[i]++;
        realZ2 = realZ * realZ;
        imagZ2 = imagZ * imagZ;
        if (realZ2 + imagZ2 > 4)
        {
            break;
        }
        imagZ = 2 * realZ * imagZ + coordY[i];
        realZ = realZ2 - imagZ2 + coordX[i];
    }
}

// Helper function for using CUDA to add vectors in parallel.
extern "C" __declspec(dllexport) void computeMandelbrot(int* iterArray, int width, int height, double minX, double minY, double rangeX, double rangeY)
{
    cudaError_t cudaStatus;
    double* coordX;
    double* coordY;
    int* iterArrayGpu;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaMalloc(&coordX, width * height * sizeof(double));
    cudaMalloc(&coordY, width * height * sizeof(double));
    cudaMalloc(&iterArrayGpu, width * height * sizeof(int));


    // Launch a kernel on the GPU with one thread for each element.
    computeCoordsMandelbrot <<<(width*height)/256, 256>>>(coordX, coordY, width, height, minX, minY, rangeX, rangeY);
    findIterationsMandelbrot <<<(width * height) / 256, 256 >>>(iterArrayGpu, coordX, coordY);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(iterArray, iterArrayGpu, width * height * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(coordX);
    cudaFree(coordY);
    cudaFree(iterArrayGpu);
}
