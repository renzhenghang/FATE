# ifdef GMP_HOME
#   INC := -I$(GMP_HOME)/include
#   LIB := -L$(GMP_HOME)/lib
# endif
# ifndef GMP_HOME
#   INC :=
#   LIB :=
# endif

ifdef CGBN_HOME
	CGBN_INC := -I$(CGBN_HOME)/include -I$(CGBN_HOME)
    CGBN_LIB := -L$(CGBN_HOME)/lib
endif

ifndef CGBN_HOME
	CGBN_INC :=
	CGBN_LIB := 
endif

volta: 
	nvcc -x cu $(CGBN_INC) $(CGBN_LIB) --shared -arch=sm_70 operator.cpp -o paillier_gpu.so \
	 --compiler-options '-fPIC' -lgmp && mv paillier_gpu.so ../../../../lib

clean:
	rm -f ../../../../lib/paillier_gpu.so
