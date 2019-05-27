#define QUOTEME(x) QUOTEME_1(x)
#define QUOTEME_1(x) #x
#ifdef __CUDACC__
#define INCLUDE_FILE(x) QUOTEME(thrust/complex.h)
#define COMPLEX thrust
#else
#define INCLUDE_FILE(x) QUOTEME(complex)
#define COMPLEX std 
#endif

#include <iostream>
#include <omp.h>

void cudaWrap(cudaError_t err){
    if (err){
        std::cerr << "Erro! " << cudaGetErrorString(err) << std::endl;
        exit(err);
    }
}


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
//            res += a[index_y * n + i] * b[index_x + i * n];
            table[pixel_x * h + pixel_y] = mandelbrot_c(complex<float>(x,y),m);
        }

    }


    __global__ void mbrot_gpu(
        complex<float> *c0, complex<float> *c1,
        float *delta_x, float *delta_y,
        unsigned *w, unsigned *h, unsigned *m,
        unsigned *table
    ){
        
    }

    void mandelbrot_gpu(
        complex<float> c0, complex<float> c1,
        float delta_x, float delta_y,
        unsigned w, unsigned h, unsigned m,
        unsigned *table
    ){
        // usar threads como threads por bloco
        complex<float> *d_c0, *d_c1;
        unsigned *d_table;
        unsigned *d_w, *d_h, *d_m;


        // allocate memory for variables

        // alloc table
        // =========================================================
        cudaWrap(cudaMalloc((void **) &d_table, sizeof(unsigned) * w * h));
        // =========================================================

        cudaWrap(cudaMalloc(&d_c0, sizeof(complex<float>)));
        cudaWrap(cudaMalloc(&d_c1, sizeof(complex<float>)));
        cudaWrap(cudaMalloc(&d_w, sizeof(unsigned)));
        cudaWrap(cudaMalloc(&d_h, sizeof(unsigned)));
        cudaWrap(cudaMalloc(&d_m, sizeof(unsigned)));
        // =========================================================

        // Memcpying
        // =========================================================
        cudaWrap(cudaMemcpy(d_c0, &c0, sizeof(complex<float>), cudaMemcpyHostToDevice));
        cudaWrap(cudaMemcpy(d_c1,&c1, sizeof(complex<float>), cudaMemcpyHostToDevice));
        cudaWrap(cudaMemcpy(d_w, &w, sizeof(unsigned), cudaMemcpyHostToDevice));
        cudaWrap(cudaMemcpy(d_h, &h, sizeof(unsigned), cudaMemcpyHostToDevice));
        cudaWrap(cudaMemcpy(d_m, &m, sizeof(unsigned), cudaMemcpyHostToDevice));
        // =========================================================

    
        cudaWrap(cudaMemcpy(table, d_table, sizeof(unsigned) * w * h, cudaMemcpyDeviceToHost));

        cudaWrap(cudaFree(d_c0));
        cudaWrap(cudaFree(d_c1));
        cudaWrap(cudaFree(d_w));
        cudaWrap(cudaFree(d_h));
        cudaWrap(cudaFree(d_m));

        cudaWrap(cudaFree(d_table));
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
                mandelbrot_gpu(
                    c0, c1,
                    delta_x, delta_y,
                    w, h, m,
                    table
                );
                break;
        }

           

    }
}
