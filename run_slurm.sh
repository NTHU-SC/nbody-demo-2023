#!/bin/bash
#SBATCH -J nbody_vtune
#SBATCH -A ACD114003
#SBATCH -p hpcxai1
#SBATCH -o nbody_vtube_out_%j.log
#SBATCH -e nbody_vtube_err_%j.log
#SBATCH -n 1
#SBATCH -n 1
#SBATCH -c 112

module load intel/2022_3_1
source /pkg/compiler/intel/2024/vtune/2024.0/vtune-vars.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd ver7
make
rm -rf ../vtune_lab5_ver7
vtune -collect performance-snapshot -r ../vtune_lab5_ver7 ./nbody.x 5000 2000

cd ver8
make
rm -rf ../vtune_lab5_ver8
vtune -collect performance-snapshot -r ../vtune_lab5_ver8 ./nbody.x 5000 2000
