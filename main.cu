#include <iostream>
#include <string>
#include <png++/image.hpp>

#define QUOTEME(x) QUOTEME_1(x)
#define QUOTEME_1(x) #x
#ifdef __CUDACC__
#   define  __CUDA__ 1
#   define INCLUDE_FILE(x) QUOTEME(thrust/complex.h)
#   define COMPLEX thrust
#else
#   define  __CUDA__ 0
#   define INCLUDE_FILE(x) QUOTEME(complex)
#   define COMPLEX std 
#endif

#ifndef REAL
#   define REAL float
#endif

#include INCLUDE_FILE()

#include "mandelbrot.cuh"


void print_table(unsigned w, unsigned h, unsigned ** table){
    for (unsigned i = 0;i < h; ++i){
        for (unsigned j = 0;j < w; ++j){
            std::cout << table[i * h + j]  << ' '; 
        }

        std::cout << std::endl; 
    }
}

png::image<png::rgb_pixel> create_image(unsigned w, unsigned h, unsigned *table){

    printf("w = %d, h = %d\n",w,h);
    png::image< png::rgb_pixel > image(w, h);

    #pragma omp parallel for
    for (png::uint_32 y = 0; y < image.get_height(); ++y)
    {
        for (png::uint_32 x = 0; x < image.get_width(); ++x)
        {
            if (table[y * w + x] == 0){
                image[y][x] = png::rgb_pixel(30, 30, 30);
            }
            else{
                image[y][x] = png::rgb_pixel(table[y * w + x] * 2, table[y * w + x] * 2, 170 + table[y * w + x] * 2);
            }
        }
    }

    return image;
}


struct params {
    COMPLEX::complex<REAL> c0,c1;
    unsigned w,h,n_threads;
    mandelbrot::exec_mode ex;
    std::string output_path;

    params(
        const COMPLEX::complex<REAL> &c0, const COMPLEX::complex<REAL> &c1,
        unsigned w, unsigned h, unsigned n_threads,
        mandelbrot::exec_mode ex, const std::string &output_path
    ): c0(c0), c1(c1), w(w), h(h), n_threads(n_threads), ex(ex), output_path(output_path)
    {}
};

mandelbrot::exec_mode get_exec_mode(const char * mode){
    mandelbrot::exec_mode ex;
    std::string mode_upper(mode);
    for (auto & c: mode_upper) c = toupper(c);

    if (!mode_upper.compare("CPU")){
        ex = mandelbrot::exec_mode::CPU;
    }
    else if (!mode_upper.compare("GPU")){
        if (!__CUDA__){
            std::cerr << "WARNING! You chose to use GPU execution without using nvcc" << std::endl;
            std::cerr << "\tDefaulting to CPU execution..." << std::endl;
            return mandelbrot::exec_mode::CPU;
        }
        ex = mandelbrot::exec_mode::GPU;
    }
    else{
        std::cerr << "Invalid execution mode (\"cpu\" or \"gpu\" is allowed)!" << std::endl;
        exit(1);
    }

    return ex;
}

struct params parse_args(int argc, char **argv){

    using COMPLEX::complex;


    std::string 
        usage("USAGE: mbrot <C0_REAL> <C0_IMAG> <C1_REAL> <C1_IMAG> <W> <H> <CPU/GPU> <THREADS> <OUTPUT>");

    if (argc != 10){
        std::cerr << usage << std::endl;
        exit(1);
    }

    REAL 
        c0_real = atof(argv[1]), c0_imag = atof(argv[2]),
        c1_real = atof(argv[3]), c1_imag = atof(argv[4]);

    unsigned w = atoi(argv[5]), h = atoi(argv[6]), n_threads = atoi(argv[8]);

    const complex<REAL> c0(c0_real, c0_imag), c1(c1_real, c1_imag);

    std::cout << c0 << ' ' << c1 << ' ' << w << ' ' << h << std::endl;

    return params(
        c0,c1,
        w, h, n_threads, get_exec_mode(argv[7]),argv[9]
    );
}

int main(int argc, char **argv){
    using COMPLEX::complex;

    params args = parse_args(argc, argv);

    const mandelbrot::exec_mode ex = args.ex;
    const unsigned w = args.w, h = args.h, m = 250;

    unsigned *table = new unsigned[w * h];

	complex<REAL> c0(args.c0),c1(args.c1);

	const REAL delta_x = (c1.real() - c0.real()) / w;
	const REAL delta_y = (c1.imag() - c0.imag()) / h;

    mandelbrot::mandelbrot(ex, args.n_threads, c0,c1,delta_x,delta_y,w,h,m,table);
     
    png::image< png::rgb_pixel > image = create_image(w,h,table);
    image.write(args.output_path);

    delete[] table;

	return 0;
}
