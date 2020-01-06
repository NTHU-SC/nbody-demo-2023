ifeq ($(ARCH),kokkos)
	CXX = icpc
	CXXFLAGS = -qopenmp -DKOKKOS -DASALIGN
	KOKKOS_DIR=/home/pvelesko/local/tarballs/kokkos/build_latest/install
	INCLUDES = -I$(KOKKOS_DIR)/include -I./programming_models/kokkos -I./
	LIBS = -L$(KOKKOS_DIR)/lib -lkokkos 
	SOURCES = ./programming_models/kokkos/GSimulation.cpp 

else ifeq ($(ARCH),kokkos_cuda)
	CXX = nvcc_wrapper
	export NVCC_WRAPPER_DEFAULT_COMPILER=icpc
	CXXFLAGS = -qopenmp --expt-extended-lambda -DKOKKOS -DKOKKOS_CUDA
	KOKKOS_DIR=/home/pvelesko/local/tarballs/kokkos/build_cuda_sm70/install
	INCLUDES = -I$(KOKKOS_DIR)/include -I./programming_models/kokkos -I./
	LIBS = -L$(KOKKOS_DIR)/lib -lkokkos 
	SOURCES = ./programming_models/kokkos/GSimulation.cpp 

else ifeq ($(ARCH),openmptarget_icx)
	CXX = icpx
	CXXFLAGS = -fiopenmp -fopenmp-targets=spir64 -D__STRICT_ANSI__
	INCLUDES = -I./programming_models/openmp -I./
	LIBS = 
	SOURCES = ./programming_models/openmp/GSimulation.cpp 

else ifeq ($(ARCH),openmptarget_clang-ykt)
	CXX = clang++
	CXXFLAGS = -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_70 -ffast-math -Wdeprecated
	INCLUDES = -I./programming_models/openmp -I./
	LIBS = 
	SOURCES = ./programming_models/openmp/GSimulation.cpp 

else ifeq ($(ARCH),openmptarget_xl)
	CXX = xlc++
	CXXFLAGS = -qsmp=omp -qoffload
	INCLUDES = -I./programming_models/openmp -I./
	LIBS = 
	SOURCES = ./programming_models/openmp/GSimulation.cpp 

else ifeq ($(ARCH),opencl)
	CXX=icpc
	CXXFLAGS = -lOpenCL
	INCLUDES = -I./programming_models/opencl -I./
	SOURCES = ./programming_models/opencl/Compute.cpp  

else ifeq ($(ARCH),sycl-intel)
	CXX=clang++
	CXXFLAGS = -std=c++11 -O2 -fsycl
	INCLUDES = -I./programming_models/sycl -I./
	SOURCES = ./programming_models/sycl/Compute.cpp  

else ifeq ($(ARCH),sycl-codeplay)
	CXX=compute++
	CXXFLAGS = -std=c++11 -O2 -lComputeCpp -sycl-driver
	INCLUDES = -I./programming_models/sycl -I./
	SOURCES = ./programming_models/sycl/Compute.cpp  

else ifeq ($(ARCH),cpu)
	CXX=icpc
	CXXFLAGS = -xhost -qopenmp -DASALIGN
	INCLUDES = -I./programming_models/cpu -I./
	SOURCES = ./programming_models/cpu/Compute.cpp  
endif


COMPFLAGS = -g -std=c++11 -O2
SOURCES += main.cpp GSimulation.cpp

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
	$(info cpu, opencl, kokkos, kokkos_cuda, openmptarget_icx, openmptarget_clang-ykt, sycl-intel, sycl-codeplay)
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


