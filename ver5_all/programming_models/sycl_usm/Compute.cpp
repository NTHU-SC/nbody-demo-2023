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
#include <stdlib.h>
#include <CL/sycl.hpp>

#include "GSimulation.hpp"
#include "cpu_time.hpp"

using namespace cl::sycl;
void GSimulation :: start() 
{
  q = queue(gpu_selector{});
  std::cout << "Using SYCL Device: ";
  std::cout << q.get_device().get_info<info::device::name>() << std::endl;
  real_type energy;
  real_type dt = get_tstep();
  int n = get_npart();
  int i,j;
 
  particles = (ParticleSoA*)      malloc_shared(sizeof(ParticleSoA), q);

  particles->pos_x = (real_type*) malloc_shared(n*sizeof(real_type), q);
  particles->pos_y = (real_type*) malloc_shared(n*sizeof(real_type), q);
  particles->pos_z = (real_type*) malloc_shared(n*sizeof(real_type), q);
  particles->vel_x = (real_type*) malloc_shared(n*sizeof(real_type), q);
  particles->vel_y = (real_type*) malloc_shared(n*sizeof(real_type), q);
  particles->vel_z = (real_type*) malloc_shared(n*sizeof(real_type), q);
  particles->acc_x = (real_type*) malloc_shared(n*sizeof(real_type), q);
  particles->acc_y = (real_type*) malloc_shared(n*sizeof(real_type), q);
  particles->acc_z = (real_type*) malloc_shared(n*sizeof(real_type), q);
  particles->mass  = (real_type*) malloc_shared(n*sizeof(real_type), q);
 
  init_pos();	
  init_vel();
  init_acc();
  init_mass();
  
  print_header();
  
  _totTime = 0.; 
 
  const float softeningSquared = 1.e-3f;
  const float G = 6.67259e-11f;
  
  ts0 = 0;
  ts1 = 0;
  nd = double(n);
  gflops = 1e-9 * ( (11. + 18. ) * nd*nd  +  nd * 19. );
  av=0.0, dev=0.0;
  nf = 0;

  const double t0 = time.start();
  auto* particles_acc_x = particles->acc_x;
  auto* particles_acc_y = particles->acc_y;
  auto* particles_acc_z = particles->acc_z;

  auto* particles_pos_x = particles->pos_x;
  auto* particles_pos_y = particles->pos_y;
  auto* particles_pos_z = particles->pos_z;

  auto* particles_mass = particles->mass;

  // Set up Max Total Threads
  auto num_groups = q.get_device().get_info<info::device::max_compute_units>();
  auto work_group_size =q.get_device().get_info<info::device::max_work_group_size>();
  auto total_threads = (int)(num_groups * work_group_size);

  for (s=1; s<=get_nsteps(); ++s) {
    ts0 += time.start(); 

    int start, end;
#ifdef USE_MPI
    mpi_bcast_all();
    start = world_rank * npp;
    end = start + npp_global[0];
#else
    start = 0;
    end = n;
#endif
    auto r = range<1>(end-start);
    q.submit([&] (handler& cgh) {

       cgh.parallel_for<class update_accel>(
#ifdef MAXTHREADS
       range<1>(total_threads), [=](item<1> item)
#else
       r, [=](item<1> item)
#endif

       { // lambda start
#ifdef MAXTHREADS
      for (int i = item.get_id()[0] + start; i < end; i+=total_threads)
#else
      auto i = item.get_id() + start;
#endif
      { 
      real_type ax_i = particles_acc_x[i];
      real_type ay_i = particles_acc_y[i];
      real_type az_i = particles_acc_z[i];
      for (int j = 0; j < n; j++) {
        real_type dx, dy, dz;
     	  real_type distanceSqr = 0.0f;
     	  real_type distanceInv = 0.0f;
     	     
     	  dx = particles_pos_x[j] - particles_pos_x[i];	//1flop
     	  dy = particles_pos_y[j] - particles_pos_y[i];	//1flop	
     	  dz = particles_pos_z[j] - particles_pos_z[i];	//1flop
     	
        distanceSqr = dx*dx + dy*dy + dz*dz + softeningSquared;	//6flops
        distanceInv = 1.0f / sqrt(distanceSqr);			//1div+1sqrt
     
     	  ax_i += dx * G * particles_mass[j] * distanceInv * distanceInv * distanceInv; //6flops
     	  ay_i += dy * G * particles_mass[j] * distanceInv * distanceInv * distanceInv; //6flops
     	  az_i += dz * G * particles_mass[j] * distanceInv * distanceInv * distanceInv; //6flops
      }
      particles_acc_x[i] = ax_i;
      particles_acc_y[i] = ay_i;
      particles_acc_z[i] = az_i;
      } // end of max_threads loop or just scope
     }); // parallel_for
   }); // queue scope
   q.wait();

#ifdef USE_MPI
    mpi_gather_acc(start);
#endif

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
    print_stats();
  } //end of the time step loop
  
  const double t1 = time.stop();
  _totTime  = (t1-t0);
  _totFlops = gflops*get_nsteps();
  
  av/=(double)(nf-2);
  dev=sqrt(dev/(double)(nf-2)-av*av);
  
  print_flops();
}

#ifdef  USM
GSimulation :: ~GSimulation() {
  free(particles->pos_x, q);
  free(particles->pos_y, q);
  free(particles->pos_z, q);
  free(particles->vel_x, q);
  free(particles->vel_y, q);
  free(particles->vel_z, q);
  free(particles->acc_x, q);
  free(particles->acc_y, q);
  free(particles->acc_z, q);
  free(particles->mass, q);
  free(particles, q);

#ifdef USE_MPI
  MPI_Finalize();
#endif
};
#endif
