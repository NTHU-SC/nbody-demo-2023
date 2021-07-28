ifeq ($(ARCH),cuda)
	CXX = nvcc
	CXXFLAGS = -gencode arch=compute_70,code=sm_70  -use_fast_math 
	INCLUDES = -I./programming_models/cuda -I./
	SOURCES = ./programming_models/cuda/Compute.cu

else ifeq ($(ARCH),kokkos)
	CXX = icpx
CXXFLAGS = -fsycl -fsycl-unnamed-lambda -fiopenmp -DKOKKOS -std=c++17 -DKOKKOS_SYCL
	KOKKOS_DIR=${Kokkos_ROOT}
	INCLUDES = -I$(KOKKOS_DIR)/include -I./programming_models/kokkos -I./
	LIBS = -L$(KOKKOS_DIR)/lib -lkokkoscore
	SOURCES = ./programming_models/kokkos/Compute.cpp 

else ifeq ($(ARCH),kokkos_cuda)
	CXX = nvcc_wrapper
	CXXFLAGS = -fopenmp -DKOKKOS -DKOKKOS_CUDA --expt-extended-lambda -arch=sm_75
	KOKKOS_DIR=${KOKKOS_ROOT}
	INCLUDES = -I$(KOKKOS_ROOT)/include -I./programming_models/kokkos -I./
	LIBS = -L$(KOKKOS_ROOT)/lib -lkokkoscore
	SOURCES = ./programming_models/kokkos/Compute.cpp 

else ifeq ($(ARCH),openmptarget_icx)
	CXX = icpx
	CXXFLAGS = -fiopenmp -fopenmp-targets=spir64 -D__STRICT_ANSI__ 
	INCLUDES = -I./programming_models/openmp -I./
	LIBS = 
	SOURCES = ./programming_models/openmp/Compute.cpp 

else ifeq ($(ARCH),openmptarget_clang-ykt)
	CXX = clang++
	CXXFLAGS = -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda
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
	CXX=icpx
	CXXFLAGS = -L/opt/intel/oneapi/compiler/2021.2.0/linux/lib/ -I/opt/intel/oneapi/compiler/2021.2.0/linux/include  -lOpenCL
	INCLUDES = -I./programming_models/opencl -I./
	SOURCES = ./programming_models/opencl/Compute.cpp  

else ifeq ($(ARCH),sycl-4cuda)
	CXX=clang++
	CXXFLAGS = -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -DUSE_SYCL
	INCLUDES = -I./programming_models/sycl -I./
	SOURCES = ./programming_models/sycl/Compute.cpp  

else ifeq ($(ARCH),sycl-4cuda-usm)
	CXX=clang++
	CXXFLAGS = -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -DUSE_SYCL
	INCLUDES = -I./programming_models/sycl_usm -I./
	SOURCES = ./programming_models/sycl_usm/Compute.cpp  

else ifeq ($(ARCH),sycl-intel)
	CXX=mpiicpc
	CXXFLAGS = -qnextgen -fsycl -fdebug-info-for-profiling -gline-tables-only -DUSE_SYCL  -DUSE_MPI -DMAXTHREADS
	INCLUDES = -I./programming_models/sycl -I./
	SOURCES = ./programming_models/sycl/Compute.cpp  

else ifeq ($(ARCH),sycl-intel-usm)
	CXX=icpx
	CXXFLAGS = -fsycl -DUSM -DUSE_SYCL -g -DMAXTHREADS
	INCLUDES = -I./programming_models/sycl_usm -I./
	SOURCES = ./programming_models/sycl_usm/Compute.cpp  


else ifeq ($(ARCH),sycl-codeplay)
	CXX=compute++
	CXXFLAGS = -lComputeCpp -sycl-driver  -DUSE_SYCL
	INCLUDES = -I./programming_models/sycl -I./
	SOURCES = ./programming_models/sycl/Compute.cpp  

else ifeq ($(ARCH),sycl-hipsycl)
	CXX=syclcc-clang-wrapper -DUSE_SYCL
	CXXFLAGS = 
	INCLUDES = -I./programming_models/sycl -I./
	SOURCES = ./programming_models/sycl/Compute.cpp  

else ifeq ($(ARCH),hip)
	CXX=hipcc
	CXXFLAGS = 
	INCLUDES = -I./programming_models/hip -I./
	SOURCES = ./programming_models/hip/Compute.cpp  


else ifeq ($(ARCH),cpu)
	CXX=/usr/bin/clang++
	CXXFLAGS = -fopenmp
	INCLUDES = -I./programming_models/cpu -I./
	SOURCES = ./programming_models/cpu/Compute.cpp  
endif


COMPFLAGS = -std=c++17 -O2 
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


