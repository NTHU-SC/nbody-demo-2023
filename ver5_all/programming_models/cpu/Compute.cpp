#include "GSimulation.hpp"
#include "cpu_time.hpp"
void GSimulation :: start() 
{
  real_type energy;
  real_type dt = get_tstep();
  int n = get_npart();
  int i,j;
 
  const int alignment = 32;
  particles = (ParticleSoA*) aligned_alloc(alignment, sizeof(ParticleSoA));

  particles->pos_x = (real_type*) aligned_alloc(alignment, n*sizeof(real_type));
  particles->pos_y = (real_type*) aligned_alloc(alignment, n*sizeof(real_type));
  particles->pos_z = (real_type*) aligned_alloc(alignment, n*sizeof(real_type));
  particles->vel_x = (real_type*) aligned_alloc(alignment, n*sizeof(real_type));
  particles->vel_y = (real_type*) aligned_alloc(alignment, n*sizeof(real_type));
  particles->vel_z = (real_type*) aligned_alloc(alignment, n*sizeof(real_type));
  particles->acc_x = (real_type*) aligned_alloc(alignment, n*sizeof(real_type));
  particles->acc_y = (real_type*) aligned_alloc(alignment, n*sizeof(real_type));
  particles->acc_z = (real_type*) aligned_alloc(alignment, n*sizeof(real_type));
  particles->mass  = (real_type*) aligned_alloc(alignment, n*sizeof(real_type));
 
  init_pos();	
  init_vel();
  init_acc();
  init_mass();
  
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
  for (int s=1; s<=get_nsteps(); ++s) {
    ts0 += time.start(); 
    for (i = 0; i < n; i++) { // update acceleration
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
      for (j = 0; j < n; j++) {
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
    if(!(s%get_sfreq())) {
      nf += 1;      
      std::cout << " " 
      <<  std::left << std::setw(8)  << s
      <<  std::left << std::setprecision(5) << std::setw(8)  << s*get_tstep()
      <<  std::left << std::setprecision(5) << std::setw(12) << _kenergy
      <<  std::left << std::setprecision(5) << std::setw(12) << (ts1 - ts0)
      <<  std::left << std::setprecision(5) << std::setw(12) << gflops*get_sfreq()/(ts1 - ts0)
      <<  std::endl;
      if(nf > 2) {
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
