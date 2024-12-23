
#ifndef __MATMUL_IKJ_MULTIPLY_H__
#define __MATMUL_IKJ_MULTIPLY_H__

void multiply(double** A, double** B, double** C, int size)
{
    for (int i = 0; i < size; i++) {
        for (int k = 0; k < size; k++) {
            for (int j = 0; j < size; j++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

#endif // __MATMUL_IKJ_MULTIPLY_H__
