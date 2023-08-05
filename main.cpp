#include <iostream>
#include <cstdio>
#include <cstdint>
#include <stdlib.h>
#include <cmath>
#include <sys/time.h>
#include <arm_neon.h>
#include <pthread.h>
#include <fstream>

#define NUM_THREADS 8
#define BLOCK_SIZE 512
int step_i;
timeval start{}, single_start{}, single_end{}, end{}, traditional_start{}, traditional_end{};
const int n = 2 * BLOCK_SIZE;
const int m = 2 * BLOCK_SIZE;
const int k = 2 * BLOCK_SIZE;

//float32_t A[n][k];
//float32_t B[k][m];
//float32_t C[n][m];
//float32_t D[n][m];
//float32_t E[n][m];

float32_t **A, **B, **C, **D, **E;

void matrix_multiply_c(float32_t **A, float32_t **B, float32_t **C, uint32_t n, uint32_t m, uint32_t k) {
    for (int i_idx = 0; i_idx < n; i_idx++) {
        for (int k_idx = 0; k_idx < k; k_idx++) {
            for (int j_idx = 0; j_idx < m; j_idx++) {
                C[j_idx][i_idx] += A[k_idx][i_idx] * B[j_idx][k_idx];
            }
        }
    }
}
//SIMD单线程
void *matrix_multiply_neon_single(void *args) {


    float32x4_t A0;
    float32x4_t A1;
    float32x4_t A2;
    float32x4_t A3;

    float32x4_t B0;
    float32x4_t B1;
    float32x4_t B2;
    float32x4_t B3;

    float32x4_t C0;
    float32x4_t C1;
    float32x4_t C2;
    float32x4_t C3;
    for (int i_idx = 0; i_idx < n; i_idx += 4) {
        for (int j_idx = 0; j_idx < m; j_idx += 4) {
            C0 = vmovq_n_f32(0);
            C1 = vmovq_n_f32(0);
            C2 = vmovq_n_f32(0);
            C3 = vmovq_n_f32(0);
            for (int k_idx = 0; k_idx < k; k_idx += 4) {

                A0 = vld1q_f32(A[k_idx] + i_idx);
                A1 = vld1q_f32(A[k_idx + 1] + i_idx);
                A2 = vld1q_f32(A[k_idx + 2] + i_idx);
                A3 = vld1q_f32(A[k_idx + 3] + i_idx);

                B0 = vld1q_f32(B[j_idx] + k_idx);
                B1 = vld1q_f32(B[j_idx + 1] + k_idx);
                B2 = vld1q_f32(B[j_idx + 2] + k_idx);
                B3 = vld1q_f32(B[j_idx + 3] + k_idx);

                C0 = vfmaq_laneq_f32(C0, A0, B0, 0);
                C0 = vfmaq_laneq_f32(C0, A1, B0, 1);
                C0 = vfmaq_laneq_f32(C0, A2, B0, 2);
                C0 = vfmaq_laneq_f32(C0, A3, B0, 3);

                C1 = vfmaq_laneq_f32(C1, A0, B1, 0);
                C1 = vfmaq_laneq_f32(C1, A1, B1, 1);
                C1 = vfmaq_laneq_f32(C1, A2, B1, 2);
                C1 = vfmaq_laneq_f32(C1, A3, B1, 3);

                C2 = vfmaq_laneq_f32(C2, A0, B2, 0);
                C2 = vfmaq_laneq_f32(C2, A1, B2, 1);
                C2 = vfmaq_laneq_f32(C2, A2, B2, 2);
                C2 = vfmaq_laneq_f32(C2, A3, B2, 3);

                C3 = vfmaq_laneq_f32(C3, A0, B3, 0);
                C3 = vfmaq_laneq_f32(C3, A1, B3, 1);
                C3 = vfmaq_laneq_f32(C3, A2, B3, 2);
                C3 = vfmaq_laneq_f32(C3, A3, B3, 3);
            }
            vst1q_f32(D[j_idx] + i_idx, C0);
            vst1q_f32(D[j_idx + 1] + i_idx, C1);
            vst1q_f32(D[j_idx + 2] + i_idx, C2);
            vst1q_f32(D[j_idx + 3] + i_idx, C3);
        }
    }
    return nullptr;
}

//SIMD优化算法
void *matrix_multiply_neon(void *args) {


    float32x4_t A0;
    float32x4_t A1;
    float32x4_t A2;
    float32x4_t A3;

    float32x4_t B0;
    float32x4_t B1;
    float32x4_t B2;
    float32x4_t B3;

    float32x4_t C0;
    float32x4_t C1;
    float32x4_t C2;
    float32x4_t C3;
    int core = step_i++;
    for (int i_idx = core * n / NUM_THREADS; i_idx < (core + 1) * n / NUM_THREADS; i_idx += 4) {
        for (int j_idx = 0; j_idx < m; j_idx += 4) {
            //初始化结果子矩阵
            C0 = vmovq_n_f32(0);
            C1 = vmovq_n_f32(0);
            C2 = vmovq_n_f32(0);
            C3 = vmovq_n_f32(0);
            for (int k_idx = 0; k_idx < k; k_idx += 4) {

                //加载A矩阵
                A0 = vld1q_f32(A[k_idx] + i_idx);
                A1 = vld1q_f32(A[k_idx + 1] + i_idx);
                A2 = vld1q_f32(A[k_idx + 2] + i_idx);
                A3 = vld1q_f32(A[k_idx + 3] + i_idx);
                //加载B矩阵
                B0 = vld1q_f32(B[j_idx] + k_idx);
                B1 = vld1q_f32(B[j_idx + 1] + k_idx);
                B2 = vld1q_f32(B[j_idx + 2] + k_idx);
                B3 = vld1q_f32(B[j_idx + 3] + k_idx);
                //计算
                C0 = vfmaq_laneq_f32(C0, A0, B0, 0);
                C0 = vfmaq_laneq_f32(C0, A1, B0, 1);
                C0 = vfmaq_laneq_f32(C0, A2, B0, 2);
                C0 = vfmaq_laneq_f32(C0, A3, B0, 3);

                C1 = vfmaq_laneq_f32(C1, A0, B1, 0);
                C1 = vfmaq_laneq_f32(C1, A1, B1, 1);
                C1 = vfmaq_laneq_f32(C1, A2, B1, 2);
                C1 = vfmaq_laneq_f32(C1, A3, B1, 3);

                C2 = vfmaq_laneq_f32(C2, A0, B2, 0);
                C2 = vfmaq_laneq_f32(C2, A1, B2, 1);
                C2 = vfmaq_laneq_f32(C2, A2, B2, 2);
                C2 = vfmaq_laneq_f32(C2, A3, B2, 3);

                C3 = vfmaq_laneq_f32(C3, A0, B3, 0);
                C3 = vfmaq_laneq_f32(C3, A1, B3, 1);
                C3 = vfmaq_laneq_f32(C3, A2, B3, 2);
                C3 = vfmaq_laneq_f32(C3, A3, B3, 3);
            }
            //存储结果
            vst1q_f32(C[j_idx] + i_idx, C0);
            vst1q_f32(C[j_idx + 1] + i_idx, C1);
            vst1q_f32(C[j_idx + 2] + i_idx, C2);
            vst1q_f32(C[j_idx + 3] + i_idx, C3);
        }
    }
    return nullptr;
}

//初始化矩阵
void matrix_init_rand(float32_t **M, uint32_t numvals) {
    for (int i = 0; i < numvals; i++) {
        for (int j = 0; j < numvals; j++) {
            M[i][j] = (float) rand() / (float) (RAND_MAX);
        }
    }
}

void matrix_init(float32_t **M, uint32_t cols, uint32_t rows, float32_t val) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            M[i][j] = val;
        }
    }
}

bool f32comp_noteq(float32_t a, float32_t b) {
    if (fabs(a - b) < 0.000001) {
        return false;
    }
    return true;
}

bool matrix_comp(float32_t **A, float32_t **B, uint32_t rows, uint32_t cols) {
    float32_t a;
    float32_t b;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            a = A[i][j];
            b = B[i][j];
            if (f32comp_noteq(a, b)) {
                printf("i=%d, j=%d, A=%f, B=%f\n", i, j, a, b);
                return false;
            }
        }
    }
    return true;
}

int main() {
    A = new float32_t *[n];
    B = new float32_t *[n];
    C = new float32_t *[n];
    D = new float32_t *[n];
    E = new float32_t *[n];
    for (int i = 0; i < n; i++) {
        A[i] = new float32_t[n];
        B[i] = new float32_t[n];
        C[i] = new float32_t[n];
        D[i] = new float32_t[n];
        E[i] = new float32_t[n];
    }

    bool c_eq_asm;
    bool c_eq_neon;
    std::ofstream inputFileA, inputFileB, resultFile1, resultFile2, resultFile3;
    matrix_init_rand(A, n);
    matrix_init_rand(B, n);
    //outPutA
    inputFileA.open("inputA.txt", std::ios::trunc);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            inputFileA << A[i][j] << " ";
        }
        inputFileA << "\n";
    }
    inputFileA.close();
    //outPutB
    inputFileB.open("inputB.txt", std::ios::trunc);
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < m; j++) {
            inputFileB << B[i][j] << " ";
        }
        inputFileB << "\n";
    }
    inputFileB.close();
    matrix_init(C, n, m, 0);

//    print_matrix(A, k, n);
//    print_matrix(B, m, k);
    //print_matrix(C, n, m);

//    printf("C\n");
//    print_matrix(E, n, m);
    printf("===============================\n");
    gettimeofday(&start, nullptr);
    pthread_t pthreads[NUM_THREADS];
    for (auto &pthread: pthreads) {
        void *p = nullptr;
        pthread_create(&pthread, nullptr, matrix_multiply_neon, p);
    }
    for (auto &pthread: pthreads) {
        pthread_join(pthread, nullptr);
    }

    gettimeofday(&end, nullptr);
    std::cout << "time=" << (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) / 1000 << "ms"
              << std::endl;

    //outputC
    resultFile1.open("result1.txt", std::ios::trunc);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            resultFile1 << C[i][j] << " ";
        }
        resultFile1 << "\n";
    }
    resultFile1.close();
    printf("Neon\n");
    gettimeofday(&single_start, nullptr);
    matrix_multiply_neon_single(nullptr);
    gettimeofday(&single_end, nullptr);
    //outputD
    resultFile2.open("result2.txt", std::ios::trunc);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            resultFile2 << D[i][j] << " ";
        }
        resultFile2 << "\n";
    }
    resultFile2.close();
    std::cout << "single time="
              << (single_end.tv_sec - single_start.tv_sec) * 1000 + (single_end.tv_usec - single_start.tv_usec) / 1000
              << "ms" << std::endl;
//    print_matrix(D, n, m);
    gettimeofday(&traditional_start, nullptr);
    matrix_multiply_c(A, B, E, n, m, k);
    gettimeofday(&traditional_end, nullptr);
    std::cout << "traditional time="
              << (traditional_end.tv_sec - traditional_start.tv_sec) * 1000 +
                 (traditional_end.tv_usec - traditional_start.tv_usec) / 1000
              << "ms" << std::endl;
    //outputE
    resultFile3.open("result3.txt", std::ios::trunc);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            resultFile3 << E[i][j] << " ";
        }
        resultFile3 << "\n";
    }
    resultFile3.close();
    std::cout << "multi_thread" << std::endl;
    std::cout << system("diff result1.txt result3.txt > check1.txt") << std::endl;
    std::cout << "single_thread" << std::endl;
    std::cout << system("diff result2.txt result3.txt > check2.txt") << std::endl;
//    c_eq_neon = matrix_comp(E, C, n, m);
//    printf("Neon equal to C? %d\n", c_eq_neon);
    printf("===============================\n");
}