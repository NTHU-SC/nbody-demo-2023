ifeq ($(ARCH),kokkos_cuda)
	CXX = nvcc_wrapper
	OPTFLAGS = -qopenmp --expt-extended-lambda -DKOKKOS_CUDA# compiler specific
	CXXFLAGS = $(OPTFLAGS)
	KOKKOS_DIR=/home/pvelesko/local/tarballs/kokkos/build_cuda_sm70/install
	INCLUDES = -I$(KOKKOS_DIR)/include
	LIBS = -L$(KOKKOS_DIR)/lib -lkokkos 
	SOURCES = GSimulation.cpp main.cpp
else
	CXX=icpc	
	OPTFLAGS = -xhost -qopenmp # compiler specific
	CXXFLAGS = $(OPTFLAGS)
	KOKKOS_DIR=/home/pvelesko/local/tarballs/kokkos/build_latest/install
	SOURCES = GSimulation.cpp main.cpp
endif


COMPFLAGS = -g -std=c++11 -O2 -DASALIGN 


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


