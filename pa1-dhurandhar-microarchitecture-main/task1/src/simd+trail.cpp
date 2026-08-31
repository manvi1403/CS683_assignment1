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

    // for (int oy = 0; oy < H; ++oy) {
    //     for (int ox = 0; ox < W; ++ox) {
    //         float acc = 0.0f;
    //         for (int ky = 0; ky < K; ++ky) {
    //             for (int kx = 0; kx < K; ++kx) {
    //                 acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
    //             }
    //         }
    //         out[oy * W + ox] = acc;
    //     }
    // }
    int tile_size=64;
    for(int i=0;i<H;i+=tile_size){
        for(int j=0;j<W;j+=tile_size){
            int i_end=(i+tile_size < H)?i+tile_size :H;
            int j_end=(j+tile_size< W)? j+tile_size : W;
    
    for (int oy = i; oy < i_end; ++oy) {
        int ox=j;
        for (; ox+3 < j_end; ox+=4) {
            
            __m256 vacc=_mm256_setzero_ps();
            for (int ky = 0; ky < K; ++ky) {
                int indi=((oy+ky)*in_stride)+ox;
                for (int kx = 0; kx < K; ++kx) {
                    __m256 vin  = _mm256_loadu_ps(&in[indi+kx]);
                    __m256 vker = _mm256_set1_ps(ker[ky * K + kx]);
                    vacc=_mm256_fmadd_ps(vin,vker,vacc);
                }
                
            }
             _mm256_storeu_ps(&out[oy * W + ox], vacc);
           
        }
        for (; ox < j_end; ++ox) {
            float acc = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                }
            }
            out[oy * W + ox] = acc;
        }
    }
        }
    }
  
}
