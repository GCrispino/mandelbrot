#define QUOTEME(x) QUOTEME_1(x)
#define QUOTEME_1(x) #x
#ifdef __CUDACC__
#   define __DEVICE__ __device__
#   define __HOST__ __host__
#   define MANDELBROT_FN mandelbrot_gpu
#   define INCLUDE_FILE(x) QUOTEME(thrust/complex.h)
#   define COMPLEX thrust
#else
#   define __DEVICE__
#   define __HOST__
#   define MANDELBROT_FN mandelbrot_cpu
#   define INCLUDE_FILE(x) QUOTEME(complex)
#   define COMPLEX std 
#endif

#include <iostream>
#include <omp.h>

namespace mandelbrot{
    using COMPLEX::complex;

    enum exec_mode{
        CPU = 0,
        GPU = 1
    };
    

    /**
     * Computes z for max iteration number m, and returns the j
     * 	norm(z) > 2
     */
    unsigned __DEVICE__ __HOST__ mandelbrot_c(complex<float> c, unsigned m){
        
        complex<float> z(0,0);
        float norm_z;

        for (unsigned j = 1;j < m;++j){
            z = pow(z,2) + c;
            norm_z = norm(z);
            if (norm_z > 2){
                return j;
            }
        }

        
        return 0;
    }

    void mandelbrot_cpu(
        unsigned n_threads,
        complex<float> c0, complex<float> c1,
        float delta_x, float delta_y,
        unsigned w, unsigned h, unsigned m,
        unsigned *table
    ){

        omp_set_num_threads(n_threads);

        #pragma omp parallel for
        for (unsigned i = 0; i < w * h; ++i){
            unsigned pixel_y = i / w;
            float y = c0.imag() + pixel_y * delta_y;

            unsigned pixel_x = i % w;
            float x = c0.real() + pixel_x * delta_x;
            table[pixel_y * w + pixel_x] = mandelbrot_c(complex<float>(x,y),m);
        }

    }

    void mandelbrot(
        exec_mode ex, unsigned n_threads,
        complex<float> c0, complex<float> c1,
        float delta_x, float delta_y,
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
