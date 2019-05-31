# MAC 5742 - EP2 - Mandelbrot set
Mandelbrot set visualization using parallel computation made for MAC5742 course at IME-USP

## Overview

This repository contains the code for the ssecond program exercise at **MAC 5742 - Introduction to Concurrent, Parallel and Distributed Programming** at IME-USP.

The exercise solution should contain code that calculates the mandelbrot set image given a few parameters. This should be done to execute this in parallel using either OpenMP or CUDA.

## Building

The build process uses the `nvcc` compiler as default, to do GPU processing. If you don't have it installed, you can change the compiler with the env variable `ALT_CC` (tested only with nvcc and g++). 
Ex: 
```bash
$ make clean
$ ALT_CC=g++ make
```

If you do, just go to the `src` folder and run `make`:

```bash
$ cd src
$ make
```

This will generate an executable file called `mbrot` under the `src` folder

## Running

To run, follow this pattern:

```bash
$ ./mbrot <C0_REAL> <C0_IMAG> <C1_REAL> <C1_IMAG> <W> <H> <EXEC_MODE> <N_THREADS> <OUTPUT_PATH>
```

Where:
* `CO_REAL`: real part of the C0 complex number;
* `CO_IMAG`: imaginary part of the C0 complex number;
* `C1_REAL`: real part of the C1 complex number;
* `C1_IMAG`: imaginary part of the C1 complex number;
* `W`: output image's width;
* `H`: output image's height;
* `EXEC_MODE`: 0 for executing on CPU, 1 for GPU;
* `N_THREADS`: number of threads for parallelization. If executing on GPU, number of threads for each block;
* `OUTPUT_PATH`: output path for generated image.

Running this will generate a PNG image visualization of the mandelbrot set size with dimensionns WxH, limited by C0 and C1 complex numbers regions.

### Example
Running the following command will generate this image:

```bash
$ ./mbrot 0.03 -0.65 0.1 -0.6 1920 1080 1 256 examples/0.03_-0.65_0.1_-0.6.png
```

![Example image](https://github.com/GCrispino/mandelbrot/raw/master/examples/0.03_-0.65_0.1_-0.6.png)

More examples are located in the `examples` folder.
