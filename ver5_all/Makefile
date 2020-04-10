ifeq ($(ARCH),cuda)
	CXX = nvcc
	CXXFLAGS = -gencode arch=compute_70,code=sm_70  -use_fast_math 
	INCLUDES = -I./programming_models/cuda -I./
	SOURCES = ./programming_models/cuda/Compute.cu

else ifeq ($(ARCH),kokkos)
	CXX = icpc
	CXXFLAGS = -qopenmp -DKOKKOS -DASALIGN
	KOKKOS_DIR=/home/pvelesko/local/tarballs/kokkos/build_latest/install
	INCLUDES = -I$(KOKKOS_DIR)/include -I./programming_models/kokkos -I./
	LIBS = -L$(KOKKOS_DIR)/lib -lkokkos 
	SOURCES = ./programming_models/kokkos/Compute.cpp 

else ifeq ($(ARCH),kokkos_cuda)
	CXX = nvcc_wrapper
	export NVCC_WRAPPER_DEFAULT_COMPILER=icpc
	CXXFLAGS = -qopenmp --expt-extended-lambda -DKOKKOS -DKOKKOS_CUDA
	KOKKOS_DIR=/home/pvelesko/local/tarballs/kokkos/build_cuda_sm70/install
	INCLUDES = -I$(KOKKOS_DIR)/include -I./programming_models/kokkos -I./
	LIBS = -L$(KOKKOS_DIR)/lib -lkokkos 
	SOURCES = ./programming_models/kokkos/Compute.cpp 

else ifeq ($(ARCH),openmptarget_icx)
	CXX = icpx
	CXXFLAGS = -fiopenmp -fopenmp-targets=spir64 -D__STRICT_ANSI__ -ffast-math
	INCLUDES = -I./programming_models/openmp -I./
	LIBS = 
	SOURCES = ./programming_models/openmp/Compute.cpp 

else ifeq ($(ARCH),openmptarget_clang-ykt)
	CXX = clang++
	CXXFLAGS = -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_70 -Wdeprecated -ffast-math
	INCLUDES = -I./programming_models/openmp -I./
	LIBS = 
	SOURCES = ./programming_models/openmp/Compute.cpp 

else ifeq ($(ARCH),openmptarget_xl)
	CXX = xlc++
	CXXFLAGS = -qsmp=omp -qoffload
	INCLUDES = -I./programming_models/openmp -I./
	LIBS = 
	SOURCES = ./programming_models/openmp/Compute.cpp 

else ifeq ($(ARCH),opencl)
	CXX=icpc
	CXXFLAGS = -lOpenCL
	INCLUDES = -I./programming_models/opencl -I./
	SOURCES = ./programming_models/opencl/Compute.cpp  

else ifeq ($(ARCH),sycl-4cuda)
	CXX=clang++
	CXXFLAGS = -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -DSYCL4CUDA
	INCLUDES = -I./programming_models/sycl -I./
	SOURCES = ./programming_models/sycl/Compute.cpp  

else ifeq ($(ARCH),sycl-4cuda-usm)
	CXX=clang++
	CXXFLAGS = -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -DSYCL4CUDA 
	INCLUDES = -I./programming_models/sycl_usm -I./
	SOURCES = ./programming_models/sycl_usm/Compute.cpp  

else ifeq ($(ARCH),sycl-intel)
	CXX=dpcpp
	CXXFLAGS = -fsycl -fdebug-info-for-profiling -gline-tables-only
	INCLUDES = -I./programming_models/sycl -I./
	SOURCES = ./programming_models/sycl/Compute.cpp  

else ifeq ($(ARCH),sycl-intel-usm)
	CXX=dpcpp
	CXXFLAGS = -fsycl -DUSM -fdebug-info-for-profiling -gline-tables-only  -ffast-math
	INCLUDES = -I./programming_models/sycl_usm -I./
	SOURCES = ./programming_models/sycl_usm/Compute.cpp  


else ifeq ($(ARCH),sycl-codeplay)
	CXX=compute++
	CXXFLAGS = -lComputeCpp -sycl-driver 
	INCLUDES = -I./programming_models/sycl -I./
	SOURCES = ./programming_models/sycl/Compute.cpp  

else ifeq ($(ARCH),sycl-hipsycl)
	CXX=syclcc-clang-wrapper #wrapper just sets -gcc-toolchain
	CXXFLAGS = -ffast-math
	INCLUDES = -I./programming_models/sycl -I./
	SOURCES = ./programming_models/sycl/Compute.cpp  

else ifeq ($(ARCH),hip)
	CXX=hipcc
	CXXFLAGS = -ffast-math
	INCLUDES = -I./programming_models/hip -I./
	SOURCES = ./programming_models/hip/Compute.cpp  


else ifeq ($(ARCH),cpu)
	CXX=icpc
	CXXFLAGS = -xhost -qopenmp -DASALIGN
	INCLUDES = -I./programming_models/cpu -I./
	SOURCES = ./programming_models/cpu/Compute.cpp  
endif


COMPFLAGS = -std=c++14 -O2 -DMAXTHREADS
SOURCES += main.cpp GSimulation.cpp

.SUFFIXES: .o .cpp

##########################################
OBJSC = $(addsuffix .o, $(basename ${SOURCES}))
##########################################

EXEC=nbody.x

all: cpu

check_arch:
ifeq ($(ARCH), cpu)
	$(info Building Nbody for $(ARCH) )
else ifeq ($(ARCH), opencl)
	$(info Building Nbody for $(ARCH) )
else ifeq ($(ARCH), kokkos)
	$(info Building Nbody for $(ARCH) )
else ifeq ($(ARCH), kokkos_cuda)
	$(info Building Nbody for $(ARCH) )
else ifeq ($(ARCH), openmptarget_icx)
	$(info Building Nbody for $(ARCH) )
else ifeq ($(ARCH), openmptarget_clang-ykt)
	$(info Building Nbody for $(ARCH) )
else ifeq ($(ARCH), sycl-intel)
	$(info Building Nbody for $(ARCH) )
else ifeq ($(ARCH), sycl-4cuda-usm)
	$(info Building Nbody for $(ARCH) )
else ifeq ($(ARCH), sycl-4cuda)
	$(info Building Nbody for $(ARCH) )
else ifeq ($(ARCH), sycl-intel-usm)
	$(info Building Nbody for $(ARCH) )
else ifeq ($(ARCH), sycl-codeplay)
	$(info Building Nbody for $(ARCH) )
else ifeq ($(ARCH), sycl-hipsycl)
	$(info Building Nbody for $(ARCH) )
else ifeq ($(ARCH), cuda)
	$(info Building Nbody for $(ARCH) )
else ifeq ($(ARCH), hip)
	$(info Building Nbody for $(ARCH) )
else
	$(info ARCH = ${ARCH} ) 
	$(info please set ARCH env var: )
	$(info cpu, opencl, kokkos, kokkos_cuda, openmptarget_icx, openmptarget_clang-ykt, sycl-intel, sycl-intel-usm, sycl-4cuda, sycl-4cuda-usm, sycl-codeplay, cuda, hip)
	$(error copmilation aborted.)
endif

%.o: %.cu
	$(info )
	$(info Compiling the object file for CUDA: )
	$(CXX) $(CXXFLAGS) $(COMPFLAGS) $(INCLUDES) -c $< -o $@

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
	rm -f $(OBJSC) nbody.x *.optrpt   ./programming_models/*/*.o


