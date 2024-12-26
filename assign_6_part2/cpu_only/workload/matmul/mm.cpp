#include <chrono>
#include <iostream>
#include <random>

#ifdef IJK
#include "ijk_multiply.h"
#endif

#ifdef IKJ
#include "ikj_multiply.h"
#endif

/* #ifdef IJ
#include "block_ij_multiply.h"
#endif

#ifdef IK
#include "block_ik_multiply.h"
#endif

#ifdef KJ
#include "block_kj_multiply.h"
#endif */

#ifdef GEM5
#include "gem5/m5ops.h"
#endif

int main(int argc, char** argv)
{

#if defined(IJK) || defined(IKJ)
    int matrix_size;
    if (argc != 2) {
        std::cout << "Invalid count for command line inputs. Please refer to "
        "the usage example below for more information." << std::endl;
        std::cout << "[matrix_size: int]" << std::endl;
    } else {
        matrix_size = std::atoi(argv[1]);
    }
#endif

#if defined(IJ) || defined(IK) || defined(KJ)
    int matrix_size;
    int block_size;
    if (argc != 3) {
        std::cout << "Invalid count for command line inputs. Please refer to "
        "the usage example below for more information." << std::endl;
        std::cout << "[matrix_size: int] [block_size: int]" << std::endl;
    } else {
        matrix_size = std::atoi(argv[1]);
        block_size = std::atoi(argv[2]);
    }
#endif

#if defined(IJK) || defined(IKJ) || defined(IJ) || defined(IK) || defined(KJ)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    short int *dataA = new short int [matrix_size * matrix_size];
    short int *dataB = new short int [matrix_size * matrix_size];
    short int *dataC = new short int [matrix_size * matrix_size];

    short int **A = new short int* [matrix_size];
    short int **B = new short int* [matrix_size];
    short int **C = new short int* [matrix_size];

    for (int i = 0; i < matrix_size; i++)    {
        A[i] = &dataA [matrix_size * i];
        B[i] = &dataB [matrix_size * i];
        C[i] = &dataC [matrix_size * i];
        for (int j = 0; j < matrix_size; j++) {
            A[i][j] = static_cast<short int>(dis(gen) * 32767);
            B[i][j] = static_cast<short int>(dis(gen) * 32767);
            C[i][j] = 0;
        }
    }
#endif

    std::cout << "Beginning matrix multiply ..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

#ifdef GEM5
    m5_work_begin(0,0);
#endif

#if defined(IJK) || defined(IKJ)
    multiply(A, B, C, matrix_size);
#endif

#if defined(IJ) || defined(IK) || defined(KJ)
    multiply(A, B, C, matrix_size, block_size);
#endif

#ifdef GEM5
    m5_work_end(0,0);
#endif

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Finished matrix multiply." << std::endl;
    std::cout << "Execution time: " << (double) (end - start).count() / 1e6 << " ms" << std::endl;

#if defined(IJK) || defined(IKJ) || defined(IJ) || defined(IK) || defined(KJ)
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] dataA;
    delete[] dataB;
    delete[] dataC;
#endif

    return 0;
}
