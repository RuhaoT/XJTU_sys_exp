
#ifndef __MATMUL_IJK_MULTIPLY_H__
#define __MATMUL_IJK_MULTIPLY_H__

void multiply(double** A, double** B, double** C, int size)
{
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < size; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

#endif // __MATMUL_IJK_MULTIPLY_H__
