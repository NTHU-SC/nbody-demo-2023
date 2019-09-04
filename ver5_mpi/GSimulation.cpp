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

#define dump(X, xx_len) for (int xx = 0; xx < xx_len; xx++) std::cout << X[xx] << " "; std::cout << std::endl;
#include "GSimulation.hpp"
#include "cpu_time.hpp"
#include "mpi.h"

GSimulation :: GSimulation()
{
  //std::cout << "===============================" << std::endl;
  //std::cout << " Initialize Gravity Simulation" << std::endl;
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

  MPI_Init(NULL, NULL);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  real_type energy;
  real_type dt = get_tstep();
  int n = get_npart();
  int world_n;
  // get share n
  if (world_rank == 0) {
    world_n = n / world_size + n % world_size;
  } else {
    world_n = n / world_size;
  }
  int max_world_n = n / world_size + n % world_size;
  std::cout << "Rank: " << world_rank << " Share: " << world_n << std::endl;
  int i,j;
 
  const int alignment = 32;
  particles = (ParticleSoA*) _mm_malloc(sizeof(ParticleSoA),alignment);

  particles->pos_x = (real_type*) _mm_malloc(n*sizeof(real_type),alignment);
  particles->pos_y = (real_type*) _mm_malloc(n*sizeof(real_type),alignment);
  particles->pos_z = (real_type*) _mm_malloc(n*sizeof(real_type),alignment);
  particles->vel_x = (real_type*) _mm_malloc(n*sizeof(real_type),alignment);
  particles->vel_y = (real_type*) _mm_malloc(n*sizeof(real_type),alignment);
  particles->vel_z = (real_type*) _mm_malloc(n*sizeof(real_type),alignment);
  particles->acc_x = (real_type*) _mm_malloc(n*sizeof(real_type),alignment);
  particles->acc_y = (real_type*) _mm_malloc(n*sizeof(real_type),alignment);
  particles->acc_z = (real_type*) _mm_malloc(n*sizeof(real_type),alignment);
  particles->mass  = (real_type*) _mm_malloc(n*sizeof(real_type),alignment);
 
  init_pos();	
  init_vel();
  init_acc();
  init_mass();
  MPI_Bcast(particles->mass, n, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  
  if (world_rank == 0) 
    print_header();
  
  _totTime = 0.; 
 
  const float softeningSquared = 1.e-3f;
  const float G = 6.67259e-11f;
  
  CPUTime time;
  double ts0 = 0;
  double ts1 = 0;
  double nd = double(n);
  double gflops = 1e-9 * ( (11. + 18. ) * nd*nd  +  nd * 19. );
  double av=0.0, dev=0.0;
  int nf = 0;
  
  const double t0 = time.start();
  for (int s=1; s<=get_nsteps(); ++s)
  {   
   // update all ranks with latest data from master
   MPI_Bcast(particles->vel_x, n, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(particles->vel_y, n, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(particles->vel_z, n, MPI_FLOAT, 0, MPI_COMM_WORLD);

   MPI_Bcast(particles->pos_x, n, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(particles->pos_y, n, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(particles->pos_z, n, MPI_FLOAT, 0, MPI_COMM_WORLD);

   MPI_Bcast(particles->acc_x, n, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(particles->acc_y, n, MPI_FLOAT, 0, MPI_COMM_WORLD);
   MPI_Bcast(particles->acc_z, n, MPI_FLOAT, 0, MPI_COMM_WORLD);

   MPI_Barrier(MPI_COMM_WORLD);
   ts0 += time.start(); 

   int start, end;
   start = world_rank * world_n;
   end = start + world_n;

   for (i = start; i < end; i++)// update acceleration
   {
#ifdef ASALIGN
     __assume_aligned(particles->pos_x, alignment);
     __assume_aligned(particles->pos_y, alignment);
     __assume_aligned(particles->pos_z, alignment);
     __assume_aligned(particles->acc_x, alignment);
     __assume_aligned(particles->acc_y, alignment);
     __assume_aligned(particles->acc_z, alignment);
     __assume_aligned(particles->mass, alignment);
#endif
     real_type ax_i = particles->acc_x[i];
     real_type ay_i = particles->acc_y[i];
     real_type az_i = particles->acc_z[i];
#pragma omp simd reduction(+:ax_i,ay_i,az_i)
     for (j = 0; j < n; j++)
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

   for (i = start; i < end; ++i)// update position
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
   }

   // send new to master
   if (world_rank != 0) {
     // make a payload
     float send_buf[max_world_n * 9 + 2]; // 3 * 3 + sender
       send_buf[0] = float(world_rank);
       send_buf[1] = float(world_n);

       int idx = int(send_buf[0] * send_buf[1]);
     for (int ii = 0; ii < world_n; ii++) {
       send_buf[2 + ii + 0*0*max_world_n] = particles->vel_x[ii + idx];
       send_buf[2 + ii + 0*1*max_world_n] = particles->vel_y[ii + idx];
       send_buf[2 + ii + 0*2*max_world_n] = particles->vel_z[ii + idx];

       send_buf[2 + ii + 1*0*max_world_n] = particles->pos_x[ii + idx];
       send_buf[2 + ii + 1*1*max_world_n] = particles->pos_y[ii + idx];
       send_buf[2 + ii + 1*2*max_world_n] = particles->pos_z[ii + idx];

       send_buf[2 + ii + 2*0*max_world_n] = particles->acc_x[ii + idx];
       send_buf[2 + ii + 2*1*max_world_n] = particles->acc_y[ii + idx];
       send_buf[2 + ii + 2*2*max_world_n] = particles->acc_z[ii + idx];
     }
     //dump(send_buf, 10)
     MPI_Send(send_buf, max_world_n * 9 + 2, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
 } else {
     MPI_Status status;
     float buf[max_world_n * 9 + 2]; // buffer for MPI recv
     int sender = 0;

     MPI_Recv(buf, max_world_n * 9 + 2, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
     //dump(buf, 10)
     int sender_rank = int(buf[0]);
     int sender_size = int(buf[1]);
     int idx = int(buf[1]) * int(buf[0]);
//     std::cout << "Received message from: " << sender_rank << std::endl;
//     std::cout << "Elements received: " << sender_size << std::endl;
     
     for (int ii = 0; ii < sender_size; ii++) {
       particles->vel_x[ii + idx] = buf[2 + ii + 0*0*max_world_n];
       particles->vel_y[ii + idx] = buf[2 + ii + 0*1*max_world_n];
       particles->vel_z[ii + idx] = buf[2 + ii + 0*2*max_world_n];

       particles->pos_x[ii + idx] = buf[2 + ii + 1*0*max_world_n];
       particles->pos_y[ii + idx] = buf[2 + ii + 1*1*max_world_n];
       particles->pos_z[ii + idx] = buf[2 + ii + 1*2*max_world_n];

       particles->acc_x[ii + idx] = buf[2 + ii + 2*0*max_world_n];
       particles->acc_y[ii + idx] = buf[2 + ii + 2*1*max_world_n];
       particles->acc_z[ii + idx] = buf[2 + ii + 2*2*max_world_n];
     }


   }

   // print energy
  if (world_rank == 0) {
   energy = 0;
   for (i = 0; i < n; ++i)// update position
   {
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
  }
  
  } //end of the time step loop
  
  const double t1 = time.stop();
  _totTime  = (t1-t0);
  _totFlops = gflops*get_nsteps();
  
  av/=(double)(nf-2);
  dev=sqrt(dev/(double)(nf-2)-av*av);
  
  int nthreads=1;

  if (world_rank == 0) {
    std::cout << std::endl;
    std::cout << "# Number Threads     : " << nthreads << std::endl;	   
    std::cout << "# Total Time (s)     : " << _totTime << std::endl;
    std::cout << "# Average Perfomance : " << av << " +- " <<  dev << std::endl;
    std::cout << "===============================" << std::endl;
  }

  MPI_Finalize();

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
