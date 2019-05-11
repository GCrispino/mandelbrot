#define QUOTEME(x) QUOTEME_1(x)
#define QUOTEME_1(x) #x
#ifdef __CUDACC__
#define INCLUDE_FILE(x) QUOTEME(thurst/complex.h)
#define COMPLEX thurst
#else
#define INCLUDE_FILE(x) QUOTEME(complex)
#define COMPLEX std 
#endif


namespace mandelbrot{
    using COMPLEX::complex;

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

    void mandelbrot(
        complex<float> c0, complex<float> c1,
        float delta_x, float delta_y,
        unsigned w, unsigned h, unsigned m,unsigned **table
    ){

        std::cout << "Delta x: " << delta_x << std::endl;
        std::cout << "Delta y: " << delta_y << std::endl;

        std::cout << "c0: (" << c0.real() << ',' << c0.imag() << ")" << std::endl;
        std::cout << "c1: (" << c1.real() << ',' << c1.imag() << ")" << std::endl;

        float y = c0.imag(),x = c0.real();
        for (unsigned pixel_y = 0; pixel_y < h; y += delta_y, ++pixel_y){
            for (unsigned pixel_x = 0; pixel_x < w; x += delta_x, ++pixel_x){
                table[pixel_y][pixel_x] = mandelbrot_c(complex<float>(x,y),m);
            }
            x = c0.real();
        }

    }
}
