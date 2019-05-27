#define QUOTEME(x) QUOTEME_1(x)
#define QUOTEME_1(x) #x
#ifdef __CUDACC__
#define INCLUDE_FILE(x) QUOTEME(thurst/complex.h)
#define COMPLEX thurst
#else
#define INCLUDE_FILE(x) QUOTEME(complex)
#define COMPLEX std 
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
    unsigned mandelbrot_c(complex<float> c, unsigned m){
        
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

    unsigned ** mandelbrot_cpu(
        unsigned n_threads,
        complex<float> c0, complex<float> c1,
        float delta_x, float delta_y,
        unsigned w, unsigned h, unsigned m,
        unsigned **table
    ){

        omp_set_num_threads(n_threads);

        #pragma omp parallel for
        for (unsigned i = 0; i < w * h; ++i){
            unsigned pixel_y = i / w;
            float y = c0.imag() + pixel_y * delta_y;

            unsigned pixel_x = i % w;
            float x = c0.real() + pixel_x * delta_x;
            table[pixel_y][pixel_x] = mandelbrot_c(complex<float>(x,y),m);
        }

        return table;
    }

    unsigned ** mandelbrot_gpu(
        complex<float> c0, complex<float> c1,
        float delta_x, float delta_y,
        unsigned w, unsigned h, unsigned m,
        unsigned **table
    ){
        // usar threads como threads por bloco

        return table;
    }


    void mandelbrot(
        exec_mode ex, unsigned n_threads,
        complex<float> c0, complex<float> c1,
        float delta_x, float delta_y,
        unsigned w, unsigned h, unsigned m,
        unsigned **table
    ){

        std::cout << "Delta x: " << delta_x << std::endl;
        std::cout << "Delta y: " << delta_y << std::endl;

        std::cout << "c0: (" << c0.real() << ',' << c0.imag() << ")" << std::endl;
        std::cout << "c1: (" << c1.real() << ',' << c1.imag() << ")" << std::endl;

        switch(ex){
            case exec_mode::CPU:
                table = mandelbrot_cpu(
                    n_threads,
                    c0, c1,
                    delta_x, delta_y,
                    w, h, m,
                    table
                );
                break;
            case exec_mode::GPU:
                table = mandelbrot_gpu(
                    c0, c1,
                    delta_x, delta_y,
                    w, h, m,
                    table
                );
                break;
        }

           

    }
}
