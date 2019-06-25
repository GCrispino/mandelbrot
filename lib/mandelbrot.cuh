#ifndef MANDELBROT_COMMON_H
#   include "mandelbrot_common.cuh"
#endif

#ifdef __CUDACC__
#   define __DEVICE__ __device__
#   define __HOST__ __host__
#   define MANDELBROT_FN mandelbrot_gpu
#   include "mandelbrot_gpu.cuh"
#else
#   define __DEVICE__
#   define __HOST__
#   define MANDELBROT_FN mandelbrot_cpu
#   define COMPLEX std 
#endif

#include <iostream>
#include <omp.h>

namespace mandelbrot{
    using COMPLEX::complex;

    enum exec_mode{
        CPU,
        GPU
    };

    void mandelbrot_cpu(
        unsigned n_threads,
        complex<REAL> c0, complex<REAL> c1,
        REAL delta_x, REAL delta_y,
        unsigned w, unsigned h, unsigned m,
        unsigned *table
    ){

        omp_set_num_threads(n_threads);

        #pragma omp parallel for
        for (unsigned i = 0; i < w * h; ++i){
            unsigned pixel_y = i / w;
            REAL y = c0.imag() + pixel_y * delta_y;

            unsigned pixel_x = i % w;
            REAL x = c0.real() + pixel_x * delta_x;
            table[i] = mandelbrot_c(complex<REAL>(x,y),m);
        }

    }

    void mandelbrot(
        exec_mode ex, unsigned n_threads,
        complex<REAL> c0, complex<REAL> c1,
        REAL delta_x, REAL delta_y,
        unsigned w, unsigned h, unsigned m,
        unsigned *table
    ){

        std::cout << "Delta x: " << delta_x << std::endl;
        std::cout << "Delta y: " << delta_y << std::endl;

        std::cout << "c0: (" << c0.real() << ',' << c0.imag() << ")" << std::endl;
        std::cout << "c1: (" << c1.real() << ',' << c1.imag() << ")" << std::endl;
        std::cout << "Exec mode: " << ex << std::endl;

        switch(ex){
            case exec_mode::CPU:
                mandelbrot_cpu(
                    n_threads,
                    c0, c1,
                    delta_x, delta_y,
                    w, h, m,
                    table
                );
                break;
            case exec_mode::GPU:
                MANDELBROT_FN(
                    n_threads,
                    c0, c1,
                    delta_x, delta_y,
                    w, h, m,
                    table
                );
                break;
        }

           

    }
}
