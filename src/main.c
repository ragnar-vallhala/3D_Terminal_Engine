#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

#define MATRIX_SIZE 4  // Change as needed

void checkError(cl_int error, const char* message) {
    if (error != CL_SUCCESS) {
        fprintf(stderr, "Error: %s (%d)\n", message, error);
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Initialize data
    float A[MATRIX_SIZE * MATRIX_SIZE] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    float B[MATRIX_SIZE * MATRIX_SIZE] = {
        16, 15, 14, 13,
        12, 11, 10, 9,
        8, 7, 6, 5,
        4, 3, 2, 1
    };
    float C[MATRIX_SIZE * MATRIX_SIZE] = {0};

    // Set up OpenCL
    cl_int error;
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);
    
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &error);
    checkError(error, "Failed to create context");
    
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &error);
    checkError(error, "Failed to create command queue");

    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                 sizeof(A), A, &error);
    checkError(error, "Failed to create buffer A");

    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                 sizeof(B), B, &error);
    checkError(error, "Failed to create buffer B");

    cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(C), NULL, &error);
    checkError(error, "Failed to create buffer C");

    // Load and build the kernel
    const char *kernelSource = "kernels/matrix_mult.cl"; // Path to kernel source file
    FILE *kernelFile = fopen(kernelSource, "r");
    if (!kernelFile) {
        perror("Failed to open kernel file");
        exit(EXIT_FAILURE);
    }
    fseek(kernelFile, 0, SEEK_END);
    size_t kernelSize = ftell(kernelFile);
    fseek(kernelFile, 0, SEEK_SET);
    char *kernelCode = (char*)malloc(kernelSize + 1);
    fread(kernelCode, 1, kernelSize, kernelFile);
    kernelCode[kernelSize] = '\0';
    fclose(kernelFile);

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernelCode, 
                                                    &kernelSize, &error);
    checkError(error, "Failed to create program");

    error = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (error != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        char *log = (char*)malloc(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
        fprintf(stderr, "Build error: %s\n", log);
        free(log);
        exit(EXIT_FAILURE);
    }

    cl_kernel kernel = clCreateKernel(program, "matrix_mult", &error);
    checkError(error, "Failed to create kernel");
    int size = MATRIX_SIZE; 
    error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    error |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    error |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
    error |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &size);
    checkError(error, "Failed to set kernel arguments");

    size_t globalWorkSize[2] = {MATRIX_SIZE, MATRIX_SIZE};
    error = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    checkError(error, "Failed to enqueue kernel");

    clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, sizeof(C), C, 0, NULL, NULL);

    // Print the result
    printf("Result matrix C:\n");
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            printf("%6.2f ", C[i * MATRIX_SIZE + j]);
        }
        printf("\n");
    }

    // Clean up
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(kernelCode);

    return 0;
}

