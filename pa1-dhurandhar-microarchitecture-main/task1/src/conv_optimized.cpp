// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
// Hint: measure after every change. Not every "optimization" helps  let the numbers,
// not intuition, decide.

#include <immintrin.h>

#include "convolution.h"

void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {
    // TODO(student): replace this placeholder with your best combined implementation.
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride

    for(int i=0;i<H;i+=H){
        for(int j=0;j<W;j+=W){
        for (int oy = i; oy < H; ++oy) {
            int ox=j;
            for (; ox+63 < W; ox+=64) {
                
                __m256 vacc=_mm256_setzero_ps();
                __m256 vacc1=_mm256_setzero_ps();
                __m256 vacc2=_mm256_setzero_ps();
                __m256 vacc3=_mm256_setzero_ps();
                __m256 vacc4=_mm256_setzero_ps();
                __m256 vacc5=_mm256_setzero_ps();
                __m256 vacc6=_mm256_setzero_ps();
                __m256 vacc7=_mm256_setzero_ps();
                
                for (int ky = 0; ky < K; ++ky) {
                    int indi=((oy+ky)*in_stride)+ox;
                    for (int kx = 0; kx < K; ++kx) {
                        __m256 vker = _mm256_set1_ps(ker[ky * K + kx]);
                        __m256 vin  = _mm256_loadu_ps(&in[indi+kx]);
                        
                        __m256 vin1  = _mm256_loadu_ps(&in[indi+kx+8]);
                        __m256 vin2  = _mm256_loadu_ps(&in[indi+kx+16]);
                        __m256 vin3  = _mm256_loadu_ps(&in[indi+kx+24]);
                        __m256 vin4  = _mm256_loadu_ps(&in[indi+kx+32]);
                        
                        __m256 vin5 = _mm256_loadu_ps(&in[indi+kx+40]);
                        __m256 vin6  = _mm256_loadu_ps(&in[indi+kx+48]);
                        __m256 vin7  = _mm256_loadu_ps(&in[indi+kx+56]);
                        vacc=_mm256_fmadd_ps(vin,vker,vacc);
                        vacc1=_mm256_fmadd_ps(vin1,vker,vacc1);
                        vacc2=_mm256_fmadd_ps(vin2,vker,vacc2);
                        vacc3=_mm256_fmadd_ps(vin3,vker,vacc3);
                        vacc4=_mm256_fmadd_ps(vin4,vker,vacc4);
                        vacc5=_mm256_fmadd_ps(vin5,vker,vacc5);
                        vacc6=_mm256_fmadd_ps(vin6,vker,vacc6);
                        vacc7=_mm256_fmadd_ps(vin7,vker,vacc7);
                    }
                }
                _mm256_stream_ps(&out[oy * W + ox], vacc);
                _mm256_stream_ps(&out[oy * W + ox + 8], vacc1);
                _mm256_stream_ps(&out[oy * W + ox + 16], vacc2);
                _mm256_stream_ps(&out[oy * W + ox + 24], vacc3);
                _mm256_stream_ps(&out[oy * W + ox + 32], vacc4);
                _mm256_stream_ps(&out[oy * W + ox + 40], vacc5);
                _mm256_stream_ps(&out[oy * W + ox + 48], vacc6);
                _mm256_stream_ps(&out[oy * W + ox + 56], vacc7);
            }
            for (; ox < W; ++ox) {
                float acc = 0.0f;
                for (int ky = 0; ky < K; ++ky) {
                    for (int kx = 0; kx < K; ++kx) {
                        acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                    }
                }
                out[oy * W + ox] = acc;
            }
        }
    }   }
    }