module purge
source /soft/compilers/intel-2019/parallel_studio_xe_2019/psxevars.sh
module load opencl/neo
cd ver5_opencl; make clean; make; 
PERF_OCL=`./nbody.x 20000 150 gpu | tail -n 2 | head -n 1 | awk '{print $5}'`
cd ..

module purge
module load dpcpp
cd ver5_sycl; make clean; make -f makefile.dpcpp;
PERF_SYCL=`./nbody.x 20000 150 gpu | tail -n 2 | head -n 1 | awk '{print $5}'`
cd ..

module purge
module load omp
cd ver5_openmp; make clean; make -f makefile.icx; 
PERF_OMP=`./nbody.x 20000 150 | tail -n 2 | head -n 1 | awk '{print $5}'`
cd ..

echo "OpenCL GPU perf: $PERF_OCL"
echo "SYCL GPU perf: $PERF_SYCL"
echo "OpenMP GPU perf: $PERF_OMP"
