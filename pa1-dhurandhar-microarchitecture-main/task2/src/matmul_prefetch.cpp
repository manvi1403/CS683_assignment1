// matmul_prefetch.cpp  STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING

#include <immintrin.h>
#include <algorithm>
#include "matmul.h"

void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your cache-blocked SIMD + prefetch
    // implementation.
    // matmul_naive(A, B, C, M, N, K, lda, ldb, ldc);
    int block_m = 32;
    int block_n = 32;
    int Prefetch_Distance = 128;
    for(int i=0;i<M;i+=block_m){
        // const float* a = A + static_cast<long>(i) * lda;
        // _mm_prefetch(reinterpret_cast<const char*>(A + static_cast<long>(i+8)*lda),_MM_HINT_T0);
        int i_end = std::min(i + block_m, M);
        for(int j = 0;j < N;j+=block_n){
            int j_end = std::min(j + block_n, N);
            for(int ii = i;ii < i_end;ii++){
                const float* a = A + static_cast<long>(ii) * lda;
                for(int jj = j;jj < j_end;jj++){
                    float acc = 0.0f;
                    const float* b = B + static_cast<long>(jj) * ldb;
            // _mm_prefetch(reinterpret_cast<const char*>(B + static_cast<long>(j+8)*ldb),_MM_HINT_T0);
                    for(int p =0;p <K;p++){
                        if((p%16 == 0) && p+ Prefetch_Distance < K){
                            _mm_prefetch(reinterpret_cast<const char*>(a + p + Prefetch_Distance),_MM_HINT_T0);
                            _mm_prefetch(reinterpret_cast<const char*>(b + p + Prefetch_Distance),_MM_HINT_T0);
                        }
                        acc += a[p]*b[p];
                    }
                    C[static_cast<long>(ii)*ldc+jj] = acc;
                }
            }
        }
    }
}