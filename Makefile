CC=nvcc
CFLAGS=-I$(IDIR) --std=c++11 -Xcompiler -fopenmp
IDIR=lib
LIBS=
_DEPS=mandelbrot.hpp
DEPS=$(patsubst %,$(IDIR)/%,$(_DEPS))
MACROS=

ODIR=obj

_OBJ = main.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS) `libpng-config --cflags`

all: makedir mbrot

mbrot: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS) `libpng-config --ldflags`

#openmp: CC := g++
#openmp: CFLAGS += -fopenmp
#openmp: mbrot

#cuda: CC := nvcc
#openmp: MACROS += __CUDACC__
#cuda: mbrot


debug: CFLAGS += -DDEBUG -g
debug: mbrot

makedir:
	mkdir -p obj
.PHONY: clean 
clean:
	rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~ 
