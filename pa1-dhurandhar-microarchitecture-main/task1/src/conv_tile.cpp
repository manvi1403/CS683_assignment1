// conv_tile.cpp  STAGE 3: CACHE TILING

#include "convolution.h"

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
                int p=K/2;
                int in_stride=W+2*p;
    // TODO(student): replace this placeholder with your tiled/blocked implementation.
    int tile_size=124;
    for(int i=0;i<H;i+=tile_size){
        for(int j=0;j<W;j+=tile_size){
            int h=(i+tile_size < H)?i+tile_size :H;
            int w=(j+tile_size< W)? j+tile_size : W;
    for (int oy = i; oy < h; ++oy) {
        for (int ox = j; ox < w; ++ox) {
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
