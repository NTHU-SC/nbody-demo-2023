ifeq ($(ARCH),kokkos)
	CXX = icpc
	CXXFLAGS = -qopenmp -DKOKKOS# compiler specific
	KOKKOS_DIR=/home/pvelesko/local/tarballs/kokkos/build_latest/install
	INCLUDES = -I$(KOKKOS_DIR)/include -I./
	LIBS = -L$(KOKKOS_DIR)/lib -lkokkos 
	SOURCES = ./programming_models/kokkos/*.cpp 
else ($(ARCH),kokkos_cuda)
	CXX = nvcc_wrapper
	export NVCC_WRAPPER_DEFAULT_COMPILER=icpc
	CXXFLAGS = -qopenmp --expt-extended-lambda -DKOKKOS -DKOKKOS_CUDA# compiler specific
	KOKKOS_DIR=/home/pvelesko/local/tarballs/kokkos/build_cuda_sm70/install
	INCLUDES = -I$(KOKKOS_DIR)/include -I./
	LIBS = -L$(KOKKOS_DIR)/lib -lkokkos 
	SOURCES = ./programming_models/kokkos/*.cpp 
else
	CXX=icpc	
	CXXFLAGS = -xhost -qopenmp # compiler specific
	KOKKOS_DIR=/home/pvelesko/local/tarballs/kokkos/build_latest/install
	SOURCES = GSimulation.cpp main.cpp
endif


COMPFLAGS = -g -std=c++11 -O2 -DASALIGN 
SOURCES += main.cpp

.SUFFIXES: .o .cpp

##########################################
OBJSC = $(SOURCES:.cpp=.o)
##########################################

EXEC=nbody.x

all: cpu

%.o: %.cpp
	$(info )
	$(info Compiling the object file for CPU: )
	$(CXX) $(CXXFLAGS) $(COMPFLAGS) $(INCLUDES) -c $< -o $@ 

cpu: $(OBJSC)
	$(info )
	$(info Linking the CPU executable:)
	$(CXX) $(CXXFLAGS) $(COMPFLAGS) $(INCLUDES) -o $(EXEC) $(OBJSC) $(LIBS)
	
run: 
	$(info )
	$(info Run the default test case on CPU: )
	./nbody.x 
	
clean: 
	rm -f $(OBJSC) nbody.x *.optrpt 


