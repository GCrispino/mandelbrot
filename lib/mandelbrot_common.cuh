#ifndef MANDELBROT_COMMON_H
#   define MANDELBROT_COMMON_H
#endif

#ifdef __CUDACC__
#   define __DEVICE__ __device__
#   define __HOST__ __host__
#else
#   define __DEVICE__
#   define __HOST__
#endif

namespace mandelbrot{
    using COMPLEX::complex;

    /**
     * Computes z for max iteration number m, and returns the j
     * 	norm(z) > 2
     */
    unsigned __DEVICE__ __HOST__ mandelbrot_c(complex<REAL> c, unsigned m){
        
        complex<REAL> z(0,0);
        REAL norm_z;

        for (unsigned j = 1;j < m;++j){
            z = pow(z,2) + c;
            norm_z = norm(z);
            if (norm_z > 2){
                return j;
            }
        }

        
        return 0;
    }

}


    
