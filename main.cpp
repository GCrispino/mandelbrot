#include <iostream>
#include <string>
#include <png++/image.hpp>

#define QUOTEME(x) QUOTEME_1(x)
#define QUOTEME_1(x) #x
#ifdef __CUDACC__
#   define INCLUDE_FILE(x) QUOTEME(thurst/complex.h)
#   define COMPLEX thurst
#else
#   define INCLUDE_FILE(x) QUOTEME(complex)
#   define COMPLEX std 
#endif

#include INCLUDE_FILE(HEADER)

#include "mandelbrot.hpp"


void print_table(unsigned w, unsigned h, unsigned ** table){
    for (unsigned i = 0;i < h; ++i){
        for (unsigned j = 0;j < w; ++j){
            std::cout << table[i][j]  << ' '; 
        }

        std::cout << std::endl; 
    }
}

png::image<png::rgb_pixel> create_image(unsigned w, unsigned h, unsigned **table){

    png::image< png::rgb_pixel > image(w, h);

    #pragma omp parallel for
    for (png::uint_32 y = 0; y < image.get_height(); ++y)
    {
        for (png::uint_32 x = 0; x < image.get_width(); ++x)
        {
            if (table[y][x] == 0){
                image[y][x] = png::rgb_pixel(30, 30, 30);
            }
            else{
                image[y][x] = png::rgb_pixel(table[y][x] * 2, table[y][x] * 2, 170 + table[y][x] * 2);
            }
        }
    }

    return image;
}


struct params {
    COMPLEX::complex<float> c0,c1;
    unsigned w,h,n_threads;
    mandelbrot::exec_mode ex;
    std::string output_path;

    params(
        const COMPLEX::complex<float> &c0, const COMPLEX::complex<float> &c1,
        unsigned w, unsigned h, unsigned n_threads,
        mandelbrot::exec_mode ex, const std::string &output_path
    ): c0(c0), c1(c1), w(w), h(h), n_threads(n_threads), ex(ex), output_path(output_path)
    {}
};

mandelbrot::exec_mode get_exec_mode(const std::string &s){
    return mandelbrot::exec_mode::CPU;
}

struct params parse_args(int argc, char **argv){

    using COMPLEX::complex;


    std::string 
        usage("USAGE: mbrot <C0_REAL> <C0_IMAG> <C1_REAL> <C1_IMAG> <W> <H> <CPU/GPU> <THREADS> <OUTPUT>");

    if (argc != 10){
        std::cerr << usage << std::endl;
        exit(1);
    }

    float 
        c0_real = atof(argv[1]), c0_imag = atof(argv[2]),
        c1_real = atof(argv[3]), c1_imag = atof(argv[4]);

    unsigned w = atoi(argv[5]), h = atoi(argv[6]), n_threads = atoi(argv[8]);

    const complex<float> c0(c0_real, c0_imag), c1(c1_real, c1_imag);

    std::cout << c0 << ' ' << c1 << ' ' << w << ' ' << h << std::endl;

    return params(
        c0,c1,
        w, h, n_threads, get_exec_mode(std::string(argv[7])),argv[9]
    );
}

int main(int argc, char **argv){
    using COMPLEX::complex;

    params args = parse_args(argc, argv);


    const mandelbrot::exec_mode ex = args.ex;
    const unsigned w = args.w, h = args.h, m = 250;

    unsigned **table = new unsigned *[h];
    for (unsigned i = 0;i < h;++i)
        table[i] = new unsigned[w];

	complex<float> c0(args.c0),c1(args.c1);

	const float delta_x = (c1.real() - c0.real()) / w;
	const float delta_y = (c1.imag() - c0.imag()) / h;

    mandelbrot::mandelbrot(ex, args.n_threads, c0,c1,delta_x,delta_y,w,h,m,table);
    
    png::image< png::rgb_pixel > image = create_image(w,h,table);
    image.write(args.output_path);

    for (unsigned i = 0;i < h;++i){
        delete[] table[i];
    }
    delete[] table;

	return 0;
}
