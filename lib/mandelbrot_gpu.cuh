#ifndef MANDELBROT_COMMON_H
#   include "mandelbrot_common.cuh"
#endif

namespace mandelbrot{
    using COMPLEX::complex;


    void cudaWrapError(cudaError_t err){
        if (err){
            std::cerr << "Erro! " << cudaGetErrorString(err) << std::endl;
            exit(err);
        }
    }

    __global__ void mbrot_gpu(
            complex<REAL> *c0, complex<REAL> *c1,
            REAL *delta_x, REAL *delta_y,
            unsigned *w, unsigned *h, unsigned *m,
            unsigned *table
        ){
           // Maybe try this after with two-dimension-indexing

            unsigned index = blockIdx.x * blockDim.x + threadIdx.x;

            if (index > ((*w) * (*h)) - 1){
                return ;
            }

            unsigned pixel_y = index / (*w);
            REAL y = c0->imag() + pixel_y * (*delta_y);

            unsigned pixel_x = index % (*w);
            REAL x = c0->real() + pixel_x * (*delta_x);

            table[pixel_y * (*w) + pixel_x] = mandelbrot_c(complex<REAL>(x,y),*m);

        }

    void mandelbrot_gpu(
            unsigned n_threads,
            complex<REAL> c0, complex<REAL> c1,
            REAL delta_x, REAL delta_y,
            unsigned w, unsigned h, unsigned m,
            unsigned *table
        ){
            // usar threads como threads por bloco
            complex<REAL> *d_c0, *d_c1;
            REAL *d_delta_x, *d_delta_y;
            unsigned *d_table;
            unsigned *d_w, *d_h, *d_m;


            // allocate memory for variables

            // alloc table
            // =========================================================
            cudaWrapError(cudaMalloc((void **) &d_table, sizeof(unsigned) * w * h));
            // =========================================================

            cudaWrapError(cudaMalloc(&d_c0, sizeof(complex<REAL>)));
            cudaWrapError(cudaMalloc(&d_c1, sizeof(complex<REAL>)));
            cudaWrapError(cudaMalloc(&d_w, sizeof(unsigned)));
            cudaWrapError(cudaMalloc(&d_delta_x, sizeof(REAL)));
            cudaWrapError(cudaMalloc(&d_delta_y, sizeof(REAL)));
            cudaWrapError(cudaMalloc(&d_h, sizeof(unsigned)));
            cudaWrapError(cudaMalloc(&d_m, sizeof(unsigned)));
            // =========================================================

            // Memcpying
            // =========================================================
            cudaWrapError(cudaMemcpy(d_c0, &c0, sizeof(complex<REAL>), cudaMemcpyHostToDevice));
            cudaWrapError(cudaMemcpy(d_c1,&c1, sizeof(complex<REAL>), cudaMemcpyHostToDevice));
            cudaWrapError(cudaMemcpy(d_delta_x, &delta_x, sizeof(REAL), cudaMemcpyHostToDevice));
            cudaWrapError(cudaMemcpy(d_delta_y, &delta_y, sizeof(REAL), cudaMemcpyHostToDevice));
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

    }


