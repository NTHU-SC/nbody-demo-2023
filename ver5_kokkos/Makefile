CXX = nvcc
COMPFLAGS = -g -std=c++11 -O2 -DASALIGN  -L/home/pvelesko/local/tarballs/kokkos/build_cuda_sm60/install/lib -lkokkos -ccbin icpc
OPTFLAGS = -xhost -qopenmp

INCLUDES = -I/home/pvelesko/local/tarballs/kokkos/build_cuda_sm60/install/include

CXXFLAGS = -Xcompiler "$(OPTFLAGS)" $(COMPFLAGS) $(REPFLAGS) $(OMPFLAGS) 

SOURCES = GSimulation.cpp main.cpp

.SUFFIXES: .o .cpp

##########################################
OBJSC = $(SOURCES:.cpp=.o)
##########################################

EXEC=nbody.x

all: cpu

%.o: %.cpp
	$(info )
	$(info Compiling the object file for CPU: )
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@ 

cpu: $(OBJSC)
	$(info )
	$(info Linking the CPU executable:)
	$(CXX) -Xcompiler "$(OPTFLAGS)" $(INCLUDES) -o $(EXEC) $(OBJSC) $(COMPFLAGS)
	
run: 
	$(info )
	$(info Run the default test case on CPU: )
	./nbody.x 
	
clean: 
	rm -f $(OBJSC) nbody.x *.optrpt 


