/*
    This file is part of the example codes which have been used
    for the "Code Optmization Workshop".
    
    Copyright (C) 2016  Fabio Baruffa <fbaru-dev@gmail.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "GSimulation.hpp"
#include "cpu_time.hpp"
#include "compute.hpp"
#include <random>

#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <stdlib.h>

#include <omp.h>


GSimulation :: GSimulation()
{
  std::cout << "===============================" << std::endl;
  std::cout << " Initialize Gravity Simulation" << std::endl;
  set_npart(2000); 
  set_nsteps(500);
  set_tstep(0.1); 
  set_sfreq(50);
}

void GSimulation :: set_number_of_particles(int N)  
{
  set_npart(N);
}

void GSimulation :: set_number_of_steps(int N)  
{
  set_nsteps(N);
}

void GSimulation :: init_pos()
{
  std::random_device rd;        //random number generator
  std::mt19937 gen(42);
  std::uniform_real_distribution<real_type> unif_d(0,1.0);

  for(int i=0; i<get_npart(); ++i)
  {
    particles->pos_x[i] = unif_d(gen);
    particles->pos_y[i] = unif_d(gen);
    particles->pos_z[i] = unif_d(gen);
  }
}

void GSimulation :: init_vel()
{
  std::random_device rd;        //random number generator
  std::mt19937 gen(42);
  std::uniform_real_distribution<real_type> unif_d(-1.0,1.0);

  for(int i=0; i<get_npart(); ++i)
  {
    particles->vel_x[i] = unif_d(gen) * 1.0e-3f;
    particles->vel_y[i] = unif_d(gen) * 1.0e-3f;
    particles->vel_z[i] = unif_d(gen) * 1.0e-3f;
  }
}

void GSimulation :: init_acc()
{
  for(int i=0; i<get_npart(); ++i)
  {
    particles->acc_x[i] = 0.f;
    particles->acc_y[i] = 0.f;
    particles->acc_z[i] = 0.f;
  }
}

void GSimulation :: init_mass()
{
  real_type n   = static_cast<real_type> (get_npart());
  std::random_device rd;        //random number generator
  std::mt19937 gen(42);
  std::uniform_real_distribution<real_type> unif_d(0.0,1.0);

  for(int i=0; i<get_npart(); ++i)
  {
    particles->mass[i] = n * unif_d(gen);
  }
}


void GSimulation :: start() 
{
  real_type energy;
  real_type dt = get_tstep();
  int n = get_npart();
  int i,j;
 
  particles = (ParticleSoA*) _mm_malloc(sizeof(ParticleSoA),ALIGNMENT);

  particles->pos_x = (real_type*) _mm_malloc(n*sizeof(real_type),ALIGNMENT);
  particles->pos_y = (real_type*) _mm_malloc(n*sizeof(real_type),ALIGNMENT);
  particles->pos_z = (real_type*) _mm_malloc(n*sizeof(real_type),ALIGNMENT);
  particles->vel_x = (real_type*) _mm_malloc(n*sizeof(real_type),ALIGNMENT);
  particles->vel_y = (real_type*) _mm_malloc(n*sizeof(real_type),ALIGNMENT);
  particles->vel_z = (real_type*) _mm_malloc(n*sizeof(real_type),ALIGNMENT);
  particles->acc_x = (real_type*) _mm_malloc(n*sizeof(real_type),ALIGNMENT);
  particles->acc_y = (real_type*) _mm_malloc(n*sizeof(real_type),ALIGNMENT);
  particles->acc_z = (real_type*) _mm_malloc(n*sizeof(real_type),ALIGNMENT);
  particles->mass  = (real_type*) _mm_malloc(n*sizeof(real_type),ALIGNMENT);
 


  init_pos();	
  init_vel();
  init_acc();
  init_mass();
  
  print_header();
  
  _totTime = 0.; 
 
  
  CPUTime time;
  double ts0 = 0;
  double ts1 = 0;
  double nd = double(n);
  double gflops = 1e-9 * ( (11. + 18. ) * nd*nd  +  nd * 19. );
  double av=0.0, dev=0.0;
  int nf = 0;

  OCL::OCL OpenCL = OCL::OCL();  // initialize OpenCL environment w/default settings

  std::string src_str = R"CLC(
   #include <Constants.hpp>
   #include <Particle.hpp>
   __kernel void comp(__global struct ParticleSoA* particles) {
   int i = get_global_id(0);
/*#ifdef ASALIGN
     __assume_aligned(particles->pos_x, ALIGNMENT);
     __assume_aligned(particles->pos_y, ALIGNMENT);
     __assume_aligned(particles->pos_z, ALIGNMENT);
     __assume_aligned(particles->acc_x, ALIGNMENT);
     __assume_aligned(particles->acc_y, ALIGNMENT);
     __assume_aligned(particles->acc_z, ALIGNMENT);
     __assume_aligned(particles->mass, ALIGNMENT);
#endif
*/
     real_type ax_i = particles->acc_x[i];
     real_type ay_i = particles->acc_y[i];
     real_type az_i = particles->acc_z[i];
   int j = get_global_id(1);
     real_type dx, dy, dz;
	 real_type distanceSqr = 0.0f;
	 real_type distanceInv = 0.0f;
		  
	 dx = particles->pos_x[j] - particles->pos_x[i];	//1flop
	 dy = particles->pos_y[j] - particles->pos_y[i];	//1flop	
	 dz = particles->pos_z[j] - particles->pos_z[i];	//1flop
	
 	 distanceSqr = dx*dx + dy*dy + dz*dz + softeningSquared;	//6flops
 	 distanceInv = 1.0f / sqrt(distanceSqr);			//1div+1sqrt

	 ax_i+= dx * G * particles->mass[j] * distanceInv * distanceInv * distanceInv; //6flops
	 ay_i += dy * G * particles->mass[j] * distanceInv * distanceInv * distanceInv; //6flops
	 az_i += dz * G * particles->mass[j] * distanceInv * distanceInv * distanceInv; //6flops

     particles->acc_x[i] = ax_i;
     particles->acc_y[i] = ay_i;
     particles->acc_z[i] = az_i;
   }
    )CLC";

    cl::Program program(OpenCL.context, src_str);
    cl::Kernel kernel;
    cl::Buffer particles_d;
    try {
        program.build("-cl-std=CL2.0");
        auto buildInfo = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>();
        for (auto &pair : buildInfo)
            std::cerr << pair.second << std::endl << std::endl;
        kernel = cl::Kernel(program, "comp"); 
        
        // Make buffer
        cl::Buffer particles_d = cl::Buffer(OpenCL.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(ParticleSoA), particles);
        kernel.setArg(0, particles);



    } catch (cl::Error &e) {
        // Print build info for all devices
        cl_int buildErr;
        auto buildInfo = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&buildErr);
        for (auto &pair : buildInfo) {
            std::cerr << pair.second << std::endl << std::endl;
        }
        std::cout << OCL::getErrorString(e.err()) << std::endl;
        return;
    }


  
  const double t0 = time.start();
  for (int s=1; s<=get_nsteps(); ++s)
  {   
   ts0 += time.start(); 

   OpenCL.queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(_npart, _npart), cl::NullRange);
   OpenCL.queue.enqueueReadBuffer(particles_d, CL_TRUE, 0, sizeof(particles), particles);
   OpenCL.queue.finish();

  // update_accel(); 
   energy = 0;

   for (i = 0; i < n; ++i)// update position
   {
     particles->vel_x[i] += particles->acc_x[i] * dt; //2flops
     particles->vel_y[i] += particles->acc_y[i] * dt; //2flops
     particles->vel_z[i] += particles->acc_z[i] * dt; //2flops
	  
     particles->pos_x[i] += particles->vel_x[i] * dt; //2flops
     particles->pos_y[i] += particles->vel_y[i] * dt; //2flops
     particles->pos_z[i] += particles->vel_z[i] * dt; //2flops

     particles->acc_x[i] = 0.;
     particles->acc_y[i] = 0.;
     particles->acc_z[i] = 0.;
	
     energy += particles->mass[i] * (
	       particles->vel_x[i]*particles->vel_x[i] + 
               particles->vel_y[i]*particles->vel_y[i] +
               particles->vel_z[i]*particles->vel_z[i]); //7flops
   }
  
    _kenergy = 0.5 * energy; 
    
    ts1 += time.stop();
    if(!(s%get_sfreq()) ) 
    {
      nf += 1;      
      std::cout << " " 
		<<  std::left << std::setw(8)  << s
		<<  std::left << std::setprecision(5) << std::setw(8)  << s*get_tstep()
		<<  std::left << std::setprecision(5) << std::setw(12) << _kenergy
		<<  std::left << std::setprecision(5) << std::setw(12) << (ts1 - ts0)
		<<  std::left << std::setprecision(5) << std::setw(12) << gflops*get_sfreq()/(ts1 - ts0)
		<<  std::endl;
      if(nf > 2) 
      {
	av  += gflops*get_sfreq()/(ts1 - ts0);
	dev += gflops*get_sfreq()*gflops*get_sfreq()/((ts1-ts0)*(ts1-ts0));
      }
      
      ts0 = 0;
      ts1 = 0;
    }
  
  } //end of the time step loop
  
  const double t1 = time.stop();
  _totTime  = (t1-t0);
  _totFlops = gflops*get_nsteps();
  
  av/=(double)(nf-2);
  dev=sqrt(dev/(double)(nf-2)-av*av);
  
  int nthreads=1;

  std::cout << std::endl;
  std::cout << "# Number Threads     : " << nthreads << std::endl;	   
  std::cout << "# Total Time (s)     : " << _totTime << std::endl;
  std::cout << "# Average Perfomance : " << av << " +- " <<  dev << std::endl;
  std::cout << "===============================" << std::endl;

}



void GSimulation :: print_header()
{
	    
  std::cout << " nPart = " << get_npart()  << "; " 
	    << "nSteps = " << get_nsteps() << "; " 
	    << "dt = "     << get_tstep()  << std::endl;
	    
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << " " 
	    <<  std::left << std::setw(8)  << "s"
	    <<  std::left << std::setw(8)  << "dt"
	    <<  std::left << std::setw(12) << "kenergy"
	    <<  std::left << std::setw(12) << "time (s)"
	    <<  std::left << std::setw(12) << "GFlops"
	    <<  std::endl;
  std::cout << "------------------------------------------------" << std::endl;


}

GSimulation :: ~GSimulation()
{
  _mm_free(particles->pos_x);
  _mm_free(particles->pos_y);
  _mm_free(particles->pos_z);
  _mm_free(particles->vel_x);
  _mm_free(particles->vel_y);
  _mm_free(particles->vel_z);
  _mm_free(particles->acc_x);
  _mm_free(particles->acc_y);
  _mm_free(particles->acc_z);
  _mm_free(particles->mass);
  _mm_free(particles);

}
