// conv_reorder.cpp  STAGE 1: LOOP REORDERING
// Hint: loops from outermost to innermost -> ky, kx, oy, ox.

#include "convolution.h"
#include<iostream>

void conv_reorder(const float* in, float* out, const float* ker,
                  int H, int W, int K) {
   
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
    float acc = 0.0f;
    int ad2=0;
    for(int oy=0;oy<H;oy++){
        ad2=(oy*W);
            for(int ox=0;ox<W;ox++){
                out[ad2 + ox]=acc;
            }
        }
    ad2=0;
    int addi=0;
    int indi=0;
    int addi2=0;
    int addi3=0;
    int indi2=0;
    for(int ky=0;ky<K;++ky){
        
        for(int kx=0;kx<K;++kx){     
            addi2=addi+kx;
            indi2=indi+kx;
            addi3=addi2;
            ad2=0;
            for(int oy=0;oy<H;oy++){

                for(int ox=0;ox<W;ox++){
                    out[ad2 + ox]+=in[addi3+ox] * ker[indi2];
                }
                addi3+=in_stride;
                ad2+=W;
            }
        }
        addi+=in_stride;
        indi+=K;
    }
}
