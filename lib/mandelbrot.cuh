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

void cudaWrapError(cudaError_t err){
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
    unsigned __device__ __host__ mandelbrot_c(complex<float> c, unsigned m){
        
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


    __global__ void mbrot_gpu(
        complex<float> *c0, complex<float> *c1,
        float *delta_x, float *delta_y,
        unsigned *w, unsigned *h, unsigned *m,
        unsigned *table
    ){
       // Maybe try this after with two-dimension-indexing

        unsigned index = blockIdx.x * blockDim.x + threadIdx.x;

        if (index > ((*w) * (*h)) - 1){
            return ;
        }

        unsigned pixel_y = index / (*w);
        float y = c0->imag() + pixel_y * (*delta_y);

        unsigned pixel_x = index % (*w);
        float x = c0->real() + pixel_x * (*delta_x);

        table[pixel_y * (*w) + pixel_x] = mandelbrot_c(complex<float>(x,y),*m);

    }

    void mandelbrot_gpu(
        unsigned n_threads,
        complex<float> c0, complex<float> c1,
        float delta_x, float delta_y,
        unsigned w, unsigned h, unsigned m,
        unsigned *table
    ){
        // usar threads como threads por bloco
        complex<float> *d_c0, *d_c1;
        float *d_delta_x, *d_delta_y;
        unsigned *d_table;
        unsigned *d_w, *d_h, *d_m;


        // allocate memory for variables

        // alloc table
        // =========================================================
        cudaWrapError(cudaMalloc((void **) &d_table, sizeof(unsigned) * w * h));
        // =========================================================

        cudaWrapError(cudaMalloc(&d_c0, sizeof(complex<float>)));
        cudaWrapError(cudaMalloc(&d_c1, sizeof(complex<float>)));
        cudaWrapError(cudaMalloc(&d_w, sizeof(unsigned)));
        cudaWrapError(cudaMalloc(&d_delta_x, sizeof(float)));
        cudaWrapError(cudaMalloc(&d_delta_y, sizeof(float)));
        cudaWrapError(cudaMalloc(&d_h, sizeof(unsigned)));
        cudaWrapError(cudaMalloc(&d_m, sizeof(unsigned)));
        // =========================================================

        // Memcpying
        // =========================================================
        cudaWrapError(cudaMemcpy(d_c0, &c0, sizeof(complex<float>), cudaMemcpyHostToDevice));
        cudaWrapError(cudaMemcpy(d_c1,&c1, sizeof(complex<float>), cudaMemcpyHostToDevice));
        cudaWrapError(cudaMemcpy(d_delta_x, &delta_x, sizeof(float), cudaMemcpyHostToDevice));
        cudaWrapError(cudaMemcpy(d_delta_y, &delta_y, sizeof(float), cudaMemcpyHostToDevice));
        cudaWrapError(cudaMemcpy(d_w, &w, sizeof(unsigned), cudaMemcpyHostToDevice));
        cudaWrapError(cudaMemcpy(d_h, &h, sizeof(unsigned), cudaMemcpyHostToDevice));
        cudaWrapError(cudaMemcpy(d_m, &m, sizeof(unsigned), cudaMemcpyHostToDevice));
        // =========================================================

    
        unsigned blocks_per_grid = ceil((w * h) / n_threads);
        mbrot_gpu<<< blocks_per_grid , n_threads >>>(
           d_c0, d_c1,
           d_delta_x, d_delta_y,
           d_w, d_h, d_m,
           d_table
        );

        cudaWrapError(cudaDeviceSynchronize());

        cudaWrapError(cudaMemcpy(table, d_table, sizeof(unsigned) * w * h, cudaMemcpyDeviceToHost));

        cudaWrapError(cudaFree(d_c0));
        cudaWrapError(cudaFree(d_c1));
        cudaWrapError(cudaFree(d_delta_x));
        cudaWrapError(cudaFree(d_delta_y));
        cudaWrapError(cudaFree(d_w));
        cudaWrapError(cudaFree(d_h));
        cudaWrapError(cudaFree(d_m));

        cudaWrapError(cudaFree(d_table));
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
