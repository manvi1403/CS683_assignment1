// conv_reorder.cpp  STAGE 1: LOOP REORDERING
// Hint: loops from outermost to innermost -> ky, kx, oy, ox.

#include "convolution.h"

void conv_optimized(const float* in, float* out, const float* ker,
                  int H, int W, int K) {
    // TODO(student): replace this placeholder with your reordered implementation.
      const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride

 for(int oy=0;oy<H;++oy){
                for(int ox=0;ox<W;++ox){
                   out[oy * W + ox] =  0.0f;
                }
            }
             int tile_size=64;
    for(int i=0;i<H;i+=tile_size){
        for(int j=0;j<W;j+=tile_size){
            int h=(i+tile_size < H)?i+tile_size :H;
            int w=(j+tile_size< W)? j+tile_size : W;
    for(int ky=0;ky<K;++ky){
        for(int kx=0;kx<K;++kx){
            const float kval = ker[ky * K + kx];

            for(int oy=i;oy<h;++oy){
                for(int ox=j;ox<w;++ox){
                   out[oy * W + ox]+=  in[(oy + ky) * in_stride + (ox + kx)] *kval;
                }
            }
        }
    }
   
}}}