// matmul_optimized.cpp  STAGE 3: PUT IT ALL TOGETHER
//
// This is the graded function AND the kernel that gets injected into llama.cpp. Combine
// everything you have learned across the whole assignment  loop reordering, register
// blocking and unrolling (Task 1 / Stage 1 here), cache tiling and software prefetch
// (Stage 2)  and TUNE it to be as fast as you can. Your speedup over matmul_naive determines
// your score (see the tier table the harness prints), and this same function will power a
// real LLM inference via `make llama-demo`.

#include <immintrin.h>
#include <algorithm>
#include "matmul.h"
#include <vector>
#include <cstdlib>
#include <cstddef>

const int MC = 128;
const int NC = 256;
const int KC = 256;
const int MR = 8;
const int NR = 8;
const int PREFETCH_DISTANCE = 32;
struct Buffer{
    float* ptr = nullptr;
    Buffer(size_t n_floats){
        ptr = static_cast<float*>(std::aligned_alloc(32, ((n_floats*sizeof(float) + 31)/32)*32));
    }
    ~Buffer(){
        std::free(ptr);
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};
inline void pack_A(const float* __restrict A, float* __restrict A_pack, int mc, int kc, int lda){
    for(int i=0;i<mc;i++){
        const float*src = A + static_cast<long>(i) * lda;
        float* dst = A_pack + static_cast<long>(i) * kc;
        if(i+1<mc){
            _mm_prefetch(reinterpret_cast<const char*>(src + lda), _MM_HINT_T0);
        }
        int p = 0;
        for(;p+7<kc;p+=8){
            _mm256_storeu_ps(dst + p, _mm256_loadu_ps(src + p));
        }
        for(;p<kc;p++){
            dst[p] = src[p];
        }
    }
}
inline void pack_B(const float* __restrict B, float* __restrict B_pack, int nc, int kc, int ldb){
    for(int i=0;i<kc;i++){
        const float*src = B + i;
        float* dst = B_pack + static_cast<long>(i) * nc;
        for(int j=0; j< nc;j++){
            dst[j] = src[static_cast<long>(j) * ldb];
        }
    }
}


inline void kernel(const float* __restrict A,const float* __restrict B,float* __restrict C,int kc,int ldb,int ldc,bool f){
    __m256 c0,c1,c2,c3,c4,c5,c6,c7;
    if(f){
        c0 = _mm256_loadu_ps(C + 0*ldc);
        c1 = _mm256_loadu_ps(C + 1*ldc);
        c2 = _mm256_loadu_ps(C + 2*ldc);
        c3 = _mm256_loadu_ps(C + 3*ldc);
        c4 = _mm256_loadu_ps(C + 4*ldc);
        c5 = _mm256_loadu_ps(C + 5*ldc);
        c6 = _mm256_loadu_ps(C + 6*ldc);
        c7 = _mm256_loadu_ps(C + 7*ldc);
    }
    else{
        c0 = _mm256_setzero_ps();
        c1 = _mm256_setzero_ps();
        c2 = _mm256_setzero_ps();
        c3 = _mm256_setzero_ps();
        c4 = _mm256_setzero_ps();
        c5 = _mm256_setzero_ps();
        c6 = _mm256_setzero_ps();
        c7 = _mm256_setzero_ps();
    }
    int k =0;
    for(;k + 3 < kc;k+=4){
        if(k + PREFETCH_DISTANCE < kc){
            _mm_prefetch(reinterpret_cast<const char*>(A + k + PREFETCH_DISTANCE),_MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(B + static_cast<long>(k + PREFETCH_DISTANCE) * ldb),_MM_HINT_T0);
        }
        __m256 b = _mm256_loadu_ps(B + static_cast<long>(k)*ldb);
        c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k),b,c0);
        c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k + kc),b,c1);
        c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k + 2*kc),b,c2);
        c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k + 3*kc),b,c3);
        c4 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k + 4*kc),b,c4);
        c5 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k + 5*kc),b,c5);
        c6 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k + 6*kc),b,c6);
        c7 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k + 7*kc),b,c7);


        b = _mm256_loadu_ps(B + static_cast<long>(k + 1)*ldb);
        c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k + 1),b,c0);
        c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k + kc + 1),b,c1);
        c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k + 2*kc + 1),b,c2);
        c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k + 3*kc + 1),b,c3);
        c4 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k + 4*kc+ 1),b,c4);
        c5 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k + 5*kc+ 1),b,c5);
        c6 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k + 6*kc + 1),b,c6);
        c7 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k + 7*kc+1),b,c7);

        b = _mm256_loadu_ps(B + static_cast<long>(k + 2)*ldb);
        c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k + 2),b,c0);
        c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k + kc+2),b,c1);
        c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k + 2*kc+2),b,c2);
        c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k + 3*kc + 2),b,c3);
        c4 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k + 4*kc + 2),b,c4);
        c5 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k + 5*kc + 2),b,c5);
        c6 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k + 6*kc + 2),b,c6);
        c7 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k + 7*kc + 2),b,c7);


        b = _mm256_loadu_ps(B + static_cast<long>(k + 3)*ldb);
        c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k + 3),b,c0);
        c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k + kc + 3),b,c1);
        c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k + 2*kc + 3),b,c2);
        c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k + 3*kc + 3),b,c3);
        c4 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k + 4*kc + 3),b,c4);
        c5 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k + 5*kc  +3),b,c5);
        c6 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k + 6*kc + 3),b,c6);
        c7 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k + 7*kc + 3),b,c7);

    }
    for(;k < kc;k++){
        const __m256 b = _mm256_loadu_ps(B + static_cast<long>(k) * ldb);
        c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k),b,c0);
        c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k + kc),b,c1);
        c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k + 2*kc),b,c2);
        c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k + 3*kc),b,c3);
        c4 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k + 4*kc),b,c4);
        c5 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k + 5*kc),b,c5);
        c6 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k + 6*kc),b,c6);
        c7 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + k + 7*kc),b,c7);
    }

    _mm256_storeu_ps(C + 0*ldc,c0);
    _mm256_storeu_ps(C + 1*ldc,c1);
    _mm256_storeu_ps(C + 2*ldc,c2);
    _mm256_storeu_ps(C + 3*ldc,c3);
    _mm256_storeu_ps(C + 4*ldc,c4);
    _mm256_storeu_ps(C + 5*ldc,c5);
    _mm256_storeu_ps(C + 6*ldc,c6);
    _mm256_storeu_ps(C + 7*ldc,c7);

}


inline void edge_kernel(const float* __restrict A, const float* __restrict B, float* __restrict C, int rows, int cols, int kc, int ldb, int ldc, bool f){
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            float sum = f ? C[static_cast<long>(i)*ldc + j] : 0.0f;
            const float* Arow = A + static_cast<long>(i)*kc;
            for(int k=0;k<kc;k++){
                sum += Arow[k] * B[static_cast<long>(k)* ldb + j];
            }
            C[static_cast<long>(i)*ldc + j] = sum;
        }
    }
}


void matmul_optimized(const float* A, const float* B, float* C,
                      int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your best combined implementation.
    Buffer Apack_buf(static_cast<size_t>(MC)*KC);
    Buffer Bpack_buf(static_cast<size_t>(NC)*KC);
    float* Apack = Apack_buf.ptr;
    float* Bpack = Bpack_buf.ptr;
    for(int i=0;i<N;i+=NC){
        int nc = std::min(NC, N - i);
        for(int p = 0;p < K; p+=KC){
            int kc = std::min(KC, K - p);
            bool f = p != 0;
            pack_B(B + static_cast<long>(i)*ldb + p, Bpack, nc, kc, ldb);
            for(int j = 0;j < M; j+=MC){
                int mc = std::min(MC, M - j);
                pack_A(A + static_cast<long>(j) * lda + p, Apack, mc,kc,lda);
                for(int k = 0; k < NC; k += NR){
                    int cols = std::min(NR, nc - k);
                    for(int l = 0; l < MC; l += MR){
                        int rows = std::min(MR, mc - l);
                        const float* Ablk = Apack + static_cast<long>(l) * kc;
                        const float* Bblk = Bpack + static_cast<long>(k);
                        float* Cblk = C + static_cast<long>(j + l)*ldc + (i + k);
                        int next = l + 16;
                        if(next < mc){
                            _mm_prefetch(reinterpret_cast<const char*>(Apack + static_cast<long>(next)*kc), _MM_HINT_T0);
                        } 
                        if(rows == 8 && cols == 8){
                            kernel(Ablk, Bblk, Cblk, kc, nc, ldc, f);
                        }
                        else{
                            edge_kernel(Ablk, Bblk, Cblk, rows,cols, kc,nc,ldc,f);
                        }
                    }
                }
            }
        }
    }
    // matmul_naive(A, B, C, M, N, K, lda, ldb, ldc);
}
