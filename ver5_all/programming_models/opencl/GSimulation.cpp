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
#include <random>

#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <stdlib.h>

#include <omp.h>
#include "OCL.hpp"


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

  std::cout << "device option: " << get_devices() << std::endl;
  OCL::OCL OpenCL = OCL::OCL(get_devices());  // initialize OpenCL environment w/default settings

  // Kernel Source
  std::string src_str = R"CLC(
   #include <types.hpp>
   __kernel void comp(
       __global real_type* restrict particles_pos_x, 
       __global real_type* restrict particles_pos_y, 
       __global real_type* restrict particles_pos_z, 

       __global real_type* restrict particles_vel_x, 
       __global real_type* restrict particles_vel_y, 
       __global real_type* restrict particles_vel_z, 

       __global real_type* restrict particles_acc_x, 
       __global real_type* restrict particles_acc_y, 
       __global real_type* restrict particles_acc_z, 

       __global real_type* restrict particles_mass, 

       real_type dt,
       int n
       ) 
   {
   const float softeningSquared = 1.e-3f;
   const float G = 6.67259e-11f;
   int i = get_global_id(0);
   real_type ax_i = 0;
   real_type ay_i = 0;
   real_type az_i = 0;
   real_type dx, dy, dz;
	 real_type distanceSqr = 0.0f;
	 real_type distanceInv = 0.0f;
		  
	 for (int j = 0; j < n; j++)
   {
	   dx = particles_pos_x[j] - particles_pos_x[i];	//1flop
	   dy = particles_pos_y[j] - particles_pos_y[i];	//1flop	
	   dz = particles_pos_z[j] - particles_pos_z[i];	//1flop
	
 	   distanceSqr = dx*dx + dy*dy + dz*dz + softeningSquared;	//6flops
	   distanceInv = 1.0f / sqrt(distanceSqr);			//1div+1sqrt

	   ax_i+= dx * G * particles_mass[j] * distanceInv * distanceInv * distanceInv; //6flops
	   ay_i += dy * G * particles_mass[j] * distanceInv * distanceInv * distanceInv; //6flops
	   az_i += dz * G * particles_mass[j] * distanceInv * distanceInv * distanceInv; //6flops
 }

   //barrier(CLK_GLOBAL_MEM_FENCE);
   particles_acc_x[i] = ax_i;
   particles_acc_y[i] = ay_i;
   particles_acc_z[i] = az_i;
   }
    )CLC";

    int num_devices = OpenCL.clcu.size();

    /* Set up workgroup sizes
     * very naive implementaiton assuming device[0] is CPU
     * and device[1] is GPU. If using 1 device, say only GPU
     * which is normally the 2nd platform that's found it's
     * best to use the same wgsize arguments:
     * ./nbody.x 2000 5000 -1 256 256
     * -1 for no tuning of work-split
     *  256 local size for all devices. Currently max of 2
     */
    cl::NDRange local[2];
    if (_cpu_wgsize != 0 and _gpu_wgsize != 0) {
      local[0] = cl::NDRange(_cpu_wgsize);
      local[1] = cl::NDRange(_gpu_wgsize);
      printf("CPU WorkGroup Size:%d\n", _cpu_wgsize);
      printf("GPU WorkGroup Size:%d\n", _gpu_wgsize);
    } else {
      printf("Using automatic WorkGroup sizes\n");
    }

    /* Set work ratio between CPU/GPU
     * if no arg is passed it will test all
     * if -1 is passed it will test all
     * else, use the provided ratio
     */
    float cpu_ratio;
    bool tuning; if (num_devices > 1) {
      cpu_ratio = _cpu_ratio;
      tuning = (cpu_ratio < 0);
      if (tuning) cpu_ratio = 0;
    } else {
      cpu_ratio = 1.0f;
      tuning = false;
    }

    cl::Program program[num_devices];

    // data/array offsets for splitting work CPU/GPU
    size_t* shares = new size_t(num_devices);
    size_t* offsets = new size_t(num_devices);

    for (int i = 0; i < num_devices; i++) 
      program[i] = cl::Program(OpenCL.clcu[i].context, src_str);

    cl::Kernel kernel[num_devices];
    cl::Buffer particles_pos_x_d[num_devices];
    cl::Buffer particles_pos_y_d[num_devices];
    cl::Buffer particles_pos_z_d[num_devices];
                               
    cl::Buffer particles_vel_x_d[num_devices];
    cl::Buffer particles_vel_y_d[num_devices];
    cl::Buffer particles_vel_z_d[num_devices];
                              
    cl::Buffer particles_acc_x_d[num_devices];
    cl::Buffer particles_acc_y_d[num_devices];
    cl::Buffer particles_acc_z_d[num_devices];
                                
    cl::Buffer particles_mass_d[num_devices];
    for (int i = 0; i < num_devices; i++) {
    try {
        // since 2 platforms, have to build the same kernel twice
        program[i].build("");
#ifdef USE_OCL_2
        auto buildInfo = program[i].getBuildInfo<CL_PROGRAM_BUILD_LOG>();
        for (auto &pair : buildInfo)
            std::cerr << pair.second << std::endl << std::endl;
#endif
        kernel[i] = cl::Kernel(program[i], "comp"); 
        
        // Make buffer
        particles_pos_x_d[i] = cl::Buffer(OpenCL.clcu[i].context, CL_MEM_WRITE_ONLY, sizeof(real_type)*n, NULL);
        particles_pos_y_d[i] = cl::Buffer(OpenCL.clcu[i].context, CL_MEM_WRITE_ONLY, sizeof(real_type)*n, NULL);
        particles_pos_z_d[i] = cl::Buffer(OpenCL.clcu[i].context, CL_MEM_WRITE_ONLY, sizeof(real_type)*n, NULL);

        particles_vel_x_d[i] = cl::Buffer(OpenCL.clcu[i].context, CL_MEM_READ_WRITE, sizeof(real_type)*n, NULL);
        particles_vel_y_d[i] = cl::Buffer(OpenCL.clcu[i].context, CL_MEM_READ_WRITE, sizeof(real_type)*n, NULL);
        particles_vel_z_d[i] = cl::Buffer(OpenCL.clcu[i].context, CL_MEM_READ_WRITE, sizeof(real_type)*n, NULL);

        particles_acc_x_d[i] = cl::Buffer(OpenCL.clcu[i].context, CL_MEM_READ_ONLY, sizeof(real_type)*n, NULL);
        particles_acc_y_d[i] = cl::Buffer(OpenCL.clcu[i].context, CL_MEM_READ_ONLY, sizeof(real_type)*n, NULL);
        particles_acc_z_d[i] = cl::Buffer(OpenCL.clcu[i].context, CL_MEM_READ_ONLY, sizeof(real_type)*n, NULL);

        particles_mass_d[i] = cl::Buffer(OpenCL.clcu[i].context, CL_MEM_WRITE_ONLY , sizeof(real_type)*n, NULL);


        kernel[i].setArg(0, particles_pos_x_d[i]);
        kernel[i].setArg(1, particles_pos_y_d[i]);
        kernel[i].setArg(2, particles_pos_z_d[i]);

        kernel[i].setArg(3, particles_vel_x_d[i]);
        kernel[i].setArg(4, particles_vel_y_d[i]);
        kernel[i].setArg(5, particles_vel_z_d[i]);

        kernel[i].setArg(6, particles_acc_x_d[i]);
        kernel[i].setArg(7, particles_acc_y_d[i]);
        kernel[i].setArg(8, particles_acc_z_d[i]);

        kernel[i].setArg(9, particles_mass_d[i]);

        kernel[i].setArg(10, dt);
        kernel[i].setArg(11, n);


    } catch (cl::Error &e) {
        printf("failed to setup the ocl kernel\n");
        // Print build info for all devices
        cl_int buildErr;
        auto buildInfo = program[i].getBuildInfo<CL_PROGRAM_BUILD_LOG>(OpenCL.clcu[i].device);
//        for (auto &pair : buildInfo) {
//            std::cerr << pair.second << std::endl << std::endl;
//        }
        std::cout << buildInfo << std::endl;
        std::cout << OCL::getErrorString(e.err()) << std::endl;
        return;
    }
    }


  for (int i = 0; i < num_devices; i++)
    OpenCL.clcu[i].queue.enqueueWriteBuffer(particles_mass_d[i],  CL_TRUE, 0, n*sizeof(real_type), particles->mass);

  const double t0 = time.start();
  for (int s=1; s<=get_nsteps(); ++s)
  {   
    shares[0] = n * cpu_ratio;
    offsets[0] = 0;
    for (int i = 1; i < num_devices; i++) {
      shares[i] = n - n * cpu_ratio;
      offsets[i] = n * cpu_ratio;
    }

   ts0 += time.start(); 
   energy = 0;

   try {
     for (int i = 0; i < num_devices; i++) {
       int off = offsets[i];
       int off_m = offsets[i] * sizeof(real_type);
       int shr = shares[i];
       int shr_m = shares[i] * sizeof(real_type);

       //the nature of the kernel requires the full write
       OpenCL.clcu[i].queue.enqueueWriteBuffer(particles_pos_x_d[i], CL_FALSE, 0, n*sizeof(real_type), particles->pos_x);
       OpenCL.clcu[i].queue.enqueueWriteBuffer(particles_pos_y_d[i], CL_FALSE, 0, n*sizeof(real_type), particles->pos_y);
       OpenCL.clcu[i].queue.enqueueWriteBuffer(particles_pos_z_d[i], CL_FALSE, 0, n*sizeof(real_type), particles->pos_z);
       OpenCL.clcu[i].queue.finish(); // need this because using non-blocking writes

       OpenCL.clcu[i].queue.enqueueNDRangeKernel(kernel[i], cl::NDRange(off), cl::NDRange(shr), local[i]);

       OpenCL.clcu[i].queue.enqueueReadBuffer(particles_acc_x_d[i], CL_FALSE, off_m, shr_m, off+(particles->acc_x));
       OpenCL.clcu[i].queue.enqueueReadBuffer(particles_acc_y_d[i], CL_FALSE, off_m, shr_m, off+(particles->acc_y));
       OpenCL.clcu[i].queue.enqueueReadBuffer(particles_acc_z_d[i], CL_FALSE, off_m, shr_m, off+(particles->acc_z));

       OpenCL.clcu[i].queue.enqueueReadBuffer(particles_mass_d[i],  CL_FALSE, off_m, shr_m, off+(particles->mass));
     }

     // flush devices
     for (int i = 0; i < num_devices; i++)  
       OpenCL.clcu[i].queue.finish(); // need this because using non-blocking reads

   } catch (cl::Error &e) {
       // print build info for all devices

       std::cout << "failed to launch kernel" << std::endl;
       std::cout << OCL::getErrorString(e.err()) << std::endl;
       return;
   }

   energy = 0;
   // global mem accumulate
   for (int i = 0; i < n; ++i)// update position
   {
     particles->vel_x[i] += particles->acc_x[i] * dt; //2flops
     particles->vel_y[i] += particles->acc_y[i] * dt; //2flops
     particles->vel_z[i] += particles->acc_z[i] * dt; //2flops
	  
     particles->pos_x[i] += particles->vel_x[i] * dt; //2flops
     particles->pos_y[i] += particles->vel_y[i] * dt; //2flops
     particles->pos_z[i] += particles->vel_z[i] * dt; //2flops

//     no need since ocl overwrites
//     particles->acc_x[i] = 0.;
//     particles->acc_y[i] = 0.;
//     particles->acc_z[i] = 0.;
	
     energy += particles->mass[i] * (
	       particles->vel_x[i]*particles->vel_x[i] + 
               particles->vel_y[i]*particles->vel_y[i] +
               particles->vel_z[i]*particles->vel_z[i]); //7flops
   }
  
    _kenergy = 0.5 * energy; 

    ts1 += time.stop();
    if(!(s%get_sfreq()) ) 
    {
      if (tuning) {
        printf("cpu/gpu ratio = %f\n", cpu_ratio);
        cpu_ratio += 0.01;
        if (cpu_ratio > 1.0f) cpu_ratio = 1.0f;
      }
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
