#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 200
#include "GSimulation.hpp"
#include <CL/cl2.hpp>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include "OCL.hpp"


void GSimulation :: update_accel()
{
   for (int i = 0; i < _npart; i++)// update acceleration
   {
#ifdef ASALIGN
     __assume_aligned(particles->pos_x, ALIGNMENT);
     __assume_aligned(particles->pos_y, ALIGNMENT);
     __assume_aligned(particles->pos_z, ALIGNMENT);
     __assume_aligned(particles->acc_x, ALIGNMENT);
     __assume_aligned(particles->acc_y, ALIGNMENT);
     __assume_aligned(particles->acc_z, ALIGNMENT);
     __assume_aligned(particles->mass, ALIGNMENT);
#endif
     real_type ax_i = particles->acc_x[i];
     real_type ay_i = particles->acc_y[i];
     real_type az_i = particles->acc_z[i];
#pragma omp simd reduction(+:ax_i,ay_i,az_i)
     for (int j = 0; j < _npart; j++)
     {
         real_type dx, dy, dz;
	 real_type distanceSqr = 0.0f;
	 real_type distanceInv = 0.0f;
		  
	 dx = particles->pos_x[j] - particles->pos_x[i];	//1flop
	 dy = particles->pos_y[j] - particles->pos_y[i];	//1flop	
	 dz = particles->pos_z[j] - particles->pos_z[i];	//1flop
	
 	 distanceSqr = dx*dx + dy*dy + dz*dz + softeningSquared;	//6flops
 	 distanceInv = 1.0f / sqrtf(distanceSqr);			//1div+1sqrt

	 ax_i+= dx * G * particles->mass[j] * distanceInv * distanceInv * distanceInv; //6flops
	 ay_i += dy * G * particles->mass[j] * distanceInv * distanceInv * distanceInv; //6flops
	 az_i += dz * G * particles->mass[j] * distanceInv * distanceInv * distanceInv; //6flops
     }
     particles->acc_x[i] = ax_i;
     particles->acc_y[i] = ay_i;
     particles->acc_z[i] = az_i;
   }

}



