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

#include "Kokkos_Core.hpp"
#include "GSimulation.hpp"
#include "cpu_time.hpp"


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
  const int alignment = 32;

  Kokkos::initialize(); 
  {

  typedef Kokkos::Cuda espace;
  typedef Kokkos::CudaSpace mspace;
  typedef Kokkos::LayoutLeft layout;

  typedef Kokkos::View<float*, layout, mspace> floatView;


//  Kokkos::View<ParticleSoA*, layout, mspace> particles_view_d("particles", n);

  particles = (ParticleSoA*) _mm_malloc(sizeof(ParticleSoA),alignment);
  //auto particles_view_h = Kokkos::create_mirror_view(particles_view_d);

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

  // Create views for data
  floatView pos_x_d("pos_x", n);
  floatView pos_y_d("pos_y", n);
  floatView pos_z_d("pos_z", n);

  floatView vel_x_d("vel_x", n);
  floatView vel_y_d("vel_y", n);
  floatView vel_z_d("vel_z", n);

  floatView acc_x_d("acc_x", n);
  floatView acc_y_d("acc_y", n);
  floatView acc_z_d("acc_z", n);

  floatView mass_d("mass", n);

  // Create host mirrors
  auto pos_x_h = Kokkos::create_mirror_view(pos_x_d);
  auto pos_y_h = Kokkos::create_mirror_view(pos_y_d);
  auto pos_z_h = Kokkos::create_mirror_view(pos_z_d);

  auto vel_x_h = Kokkos::create_mirror_view(vel_x_d);
  auto vel_y_h = Kokkos::create_mirror_view(vel_y_d);
  auto vel_z_h = Kokkos::create_mirror_view(vel_z_d);

  auto acc_x_h = Kokkos::create_mirror_view(acc_x_d);
  auto acc_y_h = Kokkos::create_mirror_view(acc_y_d);
  auto acc_z_h = Kokkos::create_mirror_view(acc_z_d);
 
  auto mass_h = Kokkos::create_mirror_view(mass_d);
  init_pos();	
  init_vel();
  init_acc();
  init_mass();


//  // copy to device
//  Kokkos::deep_copy(pos_x_d, pos_x_h);
//  Kokkos::deep_copy(pos_y_d, pos_y_h);
//  Kokkos::deep_copy(pos_z_d, pos_z_h);
//
//  Kokkos::deep_copy(vel_x_d, vel_x_h);
//  Kokkos::deep_copy(vel_y_d, vel_y_h);
//  Kokkos::deep_copy(vel_z_d, vel_z_h);
//
//  Kokkos::deep_copy(acc_x_d, acc_x_h);
//  Kokkos::deep_copy(acc_y_d, acc_y_h);
//  Kokkos::deep_copy(acc_z_d, acc_z_h);
//
//  Kokkos::deep_copy(mass_d, mass_h);
  
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
   ts0 += time.start(); 

  // Copy data to views from host
  for (int i = 0; i < n; i++) {
    pos_x_h(i) = particles->pos_x[i];
    pos_y_h(i) = particles->pos_y[i];
    pos_z_h(i) = particles->pos_z[i];

    vel_x_h(i) = particles->vel_x[i];
    vel_y_h(i) = particles->vel_y[i];
    vel_z_h(i) = particles->vel_z[i];

    acc_x_h(i) = particles->acc_x[i];
    acc_y_h(i) = particles->acc_y[i];
    acc_z_h(i) = particles->acc_z[i];

    mass_h(i) = particles->mass[i];
  }
  
  // copy to device
  Kokkos::deep_copy(pos_x_d, pos_x_h);
  Kokkos::deep_copy(pos_y_d, pos_y_h);
  Kokkos::deep_copy(pos_z_d, pos_z_h);

  Kokkos::deep_copy(vel_x_d, vel_x_h);
  Kokkos::deep_copy(vel_y_d, vel_y_h);
  Kokkos::deep_copy(vel_z_d, vel_z_h);

  Kokkos::deep_copy(acc_x_d, acc_x_h);
  Kokkos::deep_copy(acc_y_d, acc_y_h);
  Kokkos::deep_copy(acc_z_d, acc_z_h);

  Kokkos::deep_copy(mass_d, mass_h);

   Kokkos::parallel_for(n, KOKKOS_LAMBDA (const int i)
   //for (i = 0; i < n; i++)// update acceleration
   {
//#ifdef ASALIGN
//     __assume_aligned(particles->pos_x, alignment);
//     __assume_aligned(particles->pos_y, alignment);
//     __assume_aligned(particles->pos_z, alignment);
//     __assume_aligned(particles->acc_x, alignment);
//     __assume_aligned(particles->acc_y, alignment);
//     __assume_aligned(particles->acc_z, alignment);
//     __assume_aligned(particles->mass, alignment);
//#endif
     real_type ax_i = acc_x_d[i];
     real_type ay_i = acc_y_d[i];
     real_type az_i = acc_z_d[i];
//#pragma omp simd reduction(+:ax_i,ay_i,az_i)
     for (int j = 0; j < n; j++)
     {
         real_type dx, dy, dz;
	 real_type distanceSqr = 0.0f;
	 real_type distanceInv = 0.0f;
		  
	 dx = pos_x_d[j] - pos_x_d[i];	//1flop
	 dy = pos_y_d[j] - pos_y_d[i];	//1flop	
	 dz = pos_z_d[j] - pos_z_d[i];	//1flop
	
 	 distanceSqr = dx*dx + dy*dy + dz*dz + softeningSquared;	//6flops
 	 distanceInv = 1.0f / sqrtf(distanceSqr);			//1div+1sqrt

	 ax_i += dx * G * mass_d[j] * distanceInv * distanceInv * distanceInv; //6flops
	 ay_i += dy * G * mass_d[j] * distanceInv * distanceInv * distanceInv; //6flops
	 az_i += dz * G * mass_d[j] * distanceInv * distanceInv * distanceInv; //6flops
     }
     acc_x_d[i] = ax_i;
     acc_y_d[i] = ay_i;
     acc_z_d[i] = az_i;
   } ); //end parallel for


  Kokkos::deep_copy(pos_x_h, pos_x_d);
  Kokkos::deep_copy(pos_y_h, pos_y_d);
  Kokkos::deep_copy(pos_z_h, pos_z_d);

  Kokkos::deep_copy(vel_x_h, vel_x_d);
  Kokkos::deep_copy(vel_y_h, vel_y_d);
  Kokkos::deep_copy(vel_z_h, vel_z_d);

  Kokkos::deep_copy(acc_x_h, acc_x_d);
  Kokkos::deep_copy(acc_y_h, acc_y_d);
  Kokkos::deep_copy(acc_z_h, acc_z_d);


  // Copy data to views from host
  for (int i = 0; i < n; i++) {
    particles->pos_x[i] = pos_x_h(i);
    particles->pos_y[i] = pos_y_h(i);
    particles->pos_z[i] = pos_z_h(i);

    particles->vel_x[i] = vel_x_h(i);
    particles->vel_y[i] = vel_y_h(i);
    particles->vel_z[i] = vel_z_h(i);

    particles->acc_x[i] = acc_x_h(i);
    particles->acc_y[i] = acc_y_h(i);
    particles->acc_z[i] = acc_z_h(i);
  }

   energy = 0;

   for (int i = 0; i < n; ++i)// update position
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

} // kokkos scope
  Kokkos::finalize();
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
