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

const int MC = 128;
const int NC = 256;
const int KC = 256;
const int MR = 4;
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
inline void pack_A(const float* A, float* A_pack, int mc, int kc, int lda){
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
inline void pack_B(const float* B, float* B_pack, int nc, int kc, int ldb){
    for(int i=0;i<nc;i++){
        const float*src = B + static_cast<long>(i) * ldb;
        float* dst = B_pack + static_cast<long>(i) * kc;
        if(i+1<nc){
            _mm_prefetch(reinterpret_cast<const char*>(src + ldb), _MM_HINT_T0);
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
template <int ROWS,int COLS>
inline void micro_kernel_fix(const float* A,const float* B,float* C,int kc,int lda,int ldb,int ldc,bool f){
    static_assert(ROWS > 0 && COLS > 0,"ROWS/COLS must be positive");

    __m256 acc[ROWS][COLS];
    #pragma GCC unroll 16
    for(int i=0;i<ROWS;i++)
        #pragma GCC unroll 16
        for(int j=0;j<COLS;j++)
            acc[i][j] = _mm256_setzero_ps();

    #pragma GCC unroll 16
    for(int i=0;i<ROWS;i++){
        _mm_prefetch(reinterpret_cast<const char*>(C + static_cast<long>(i)*ldc),_MM_HINT_T0);
    }

    int p = 0;
    for(;p+7<kc;p+=8){
        if(p + PREFETCH_DISTANCE < kc){
            #pragma GCC unroll 16
            for(int i=0;i<ROWS;i++){
                _mm_prefetch(reinterpret_cast<const char*>(A + static_cast<long>(i) * lda + p + PREFETCH_DISTANCE),_MM_HINT_T0);
            }

            #pragma GCC unroll 16
            for(int j=0;j<COLS;j++){
                _mm_prefetch(reinterpret_cast<const char*>(B + static_cast<long>(j) * ldb + p + PREFETCH_DISTANCE),_MM_HINT_T0);
            }
        }
        __m256 va[ROWS];
        __m256 vb[COLS];

        #pragma GCC unroll 16
        for(int i=0;i<ROWS;i++){
            va[i] = _mm256_loadu_ps(A  + static_cast<long>(i)*lda + p);
        }
        #pragma GCC unroll 16
        for(int i=0;i<COLS;i++){
            vb[i] = _mm256_loadu_ps(B  + static_cast<long>(i)*ldb + p);
        }

        #pragma GCC unroll 16
        for(int i=0;i<ROWS;i++)
            #pragma GCC unroll 16;
            for(int j=0;j<COLS;j++)
                acc[i][j]= _mm256_fmadd_ps(va[i],vb[j],acc[i][j]);
    }
    #pragma GCC unroll 16
    for(int i=0;i<ROWS;i++){
        #pragma GCC unroll 16
        for(int j=0;j<COLS;j++){
            __m128 low = _mm256_castps256_ps128(acc[i][j]);
            __m128 high = _mm256_extractf128_ps(acc[i][j],1);
            __m128 sum4 = _mm_add_ps(low,high);
            __m128 shuffle = _mm_movehdup_ps(sum4);
            __m128 sums = _mm_add_ps(sum4 , shuffle);
            shuffle = _mm_movehl_ps(shuffle,sums);
            sums = _mm_add_ss(sums,shuffle);
            float sum = _mm_cvtss_f32(sums);

            for(int k=p;k<kc;k++){
                sum+= A[static_cast<long>(i) *lda + k] * B[static_cast<long>(j)*ldb + k];
            }
            float* c = C + static_cast<long>(i) * ldc + j;
            *c = f ? (*c + sum):sum;
        }
    }
}

inline void micro_kernel_scalar(const float* A, const float* B, float * C, int rows, int cols, int kc, int lda, int ldb, int ldc, bool f){
    #pragma GCC unroll 16
    for(int i=0;i<rows;i++){
        #pragma GCC unroll 16
        for(int j=0;j<cols;j++){
            float sum = 0.0f;
            for(int k = 0;k<kc;k++){
                sum+=A[static_cast<long>(i)*lda + k]*B[static_cast<long>(j)*ldb+k];
            }
            float* c = C + static_cast<long>(i)*ldc + j;
            *c = f ? (*c + sum) : sum;
        }
    }
}
inline void run_tile(const float* A, const float* B, float* C, int rows, int cols, int kc, int lda, int ldb, int ldc, bool f){
    if(rows == MR && cols == NR){
        micro_kernel_fix<MR,NR>(A,B,C,kc,lda,ldb,ldc,f);
    }
    else if(rows == MR){
        #pragma GCC unroll 16
        for(int j=0;j<cols;j++){
            micro_kernel_fix<MR,1>(A,B+static_cast<long>(j)*ldb,C + static_cast<long>(j), kc,lda,ldb,ldc,f);
        }
    }
    else if(cols == NR){
        #pragma GCC unroll 16
        for(int i=0;i<rows;i++){
            micro_kernel_fix<1,NR>(A + static_cast<long>(i)*lda, B, C + static_cast<long>(i)*ldc, kc, lda,ldb,ldc,f);
        }
    }
    else{
        micro_kernel_scalar(A,B,C,rows,cols,kc,lda,ldb,ldc,f);
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
                        const float* Bblk = Bpack + static_cast<long>(k) * kc;
                        float* Cblk = C + static_cast<long>(j + l)*ldc + (i + k);
                        int next = l + MR;
                        if(next < mc){
                            _mm_prefetch(reinterpret_cast<const char*>(Apack + static_cast<long>(next)*kc), _MM_HINT_T0);
                        } 
                        run_tile(Ablk, Bblk, Cblk, rows, cols, kc, kc, kc, ldc, f);
                    }
                }
            }
        }
    }
    // matmul_naive(A, B, C, M, N, K, lda, ldb, ldc);
}
