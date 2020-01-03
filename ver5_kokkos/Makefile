ifeq ($(ARCH),kokkos)
	CXX = icpc
	CXXFLAGS = -qopenmp -DKOKKOS# compiler specific
	KOKKOS_DIR=/home/pvelesko/local/tarballs/kokkos/build_latest/install
	INCLUDES = -I$(KOKKOS_DIR)/include -I./programming_models/kokkos -I./
	LIBS = -L$(KOKKOS_DIR)/lib -lkokkos 
	SOURCES = ./programming_models/kokkos/GSimulation.cpp 

else ifeq ($(ARCH),kokkos_cuda)
	CXX = nvcc_wrapper
	export NVCC_WRAPPER_DEFAULT_COMPILER=icpc
	CXXFLAGS = -qopenmp --expt-extended-lambda -DKOKKOS -DKOKKOS_CUDA# compiler specific
	KOKKOS_DIR=/home/pvelesko/local/tarballs/kokkos/build_cuda_sm70/install
	INCLUDES = -I$(KOKKOS_DIR)/include -I./programming_models/kokkos -I./
	LIBS = -L$(KOKKOS_DIR)/lib -lkokkos 
	SOURCES = ./programming_models/kokkos/GSimulation.cpp 

else ifeq ($(ARCH),openmptarget_icx)
	CXX = icpx
	CXXFLAGS = -fiopenmp -fopenmp-targets=spir64 -D__STRICT_ANSI__ # compiler specific
	INCLUDES = -I./programming_models/openmp -I./
	LIBS = 
	SOURCES = ./programming_models/openmp/GSimulation.cpp 

else ifeq ($(ARCH),openmptarget_clang-ykt)
	CXX = clang++
	CXXFLAGS = -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_70 -ffast-math -Wdeprecated # compiler specific
	INCLUDES = -I./programming_models/openmp -I./
	LIBS = 
	SOURCES = ./programming_models/openmp/GSimulation.cpp 

else ifeq ($(ARCH),openmptarget_xl)
	CXX = xlc++
	CXXFLAGS = -qsmp=omp -qoffload
	INCLUDES = -I./programming_models/openmp -I./
	LIBS = 
	SOURCES = ./programming_models/openmp/GSimulation.cpp 

else ifeq ($(ARCH),cpu)
	CXX=icpc	
	CXXFLAGS = -xhost -qopenmp # compiler specific
	KOKKOS_DIR=/home/pvelesko/local/tarballs/kokkos/build_latest/install
	INCLUDES = -I./programming_models/cpu -I./
	SOURCES = GSimulation.cpp main.cpp
	SOURCES = ./programming_models/cpu/GSimulation.cpp  
endif


COMPFLAGS = -g -std=c++11 -O2 -DASALIGN 
SOURCES += main.cpp

.SUFFIXES: .o .cpp

##########################################
OBJSC = $(SOURCES:.cpp=.o)
##########################################

EXEC=nbody.x

all: cpu

check_arch:
ifndef ARCH
	$(info ) 
	$(info please set ARCH env var: )
	$(info cpu, kokkos, kokkos_cuda, openmptarget_icx, openmptarget_clang-ykt)
	$(error copmilation aborted.)
endif

%.o: %.cpp
	$(info )
	$(info Compiling the object file for CPU: )
	$(CXX) $(CXXFLAGS) $(COMPFLAGS) $(INCLUDES) -c $< -o $@

cpu: check_arch $(OBJSC)
	$(info )
	$(info Linking the CPU executable:)
	$(CXX) $(CXXFLAGS) $(COMPFLAGS) $(INCLUDES) -o $(EXEC) $(OBJSC) $(LIBS)
	
run: 
	$(info )
	$(info Run the default test case on CPU: )
	./nbody.x 
	
clean: 
	rm -f $(OBJSC) nbody.x *.optrpt 


