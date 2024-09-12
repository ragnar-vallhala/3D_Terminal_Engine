__kernel void matrix_mult(
    __global const float *A, 
    __global const float *B, 
    __global float *C, 
    const unsigned int N) 
{
    int row = get_global_id(0);
    int col = get_global_id(1);

    float sum = 0.0f;
    for (int k = 0; k < N; ++k) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

