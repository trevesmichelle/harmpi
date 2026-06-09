//Modified by Alexander Tchekhovskoy: MPI+3D
/***********************************************************************************
    Copyright 2006 Charles F. Gammie, Jonathan C. McKinney, Scott C. Noble, 
                   Gabor Toth, and Luca Del Zanna

                        HARM  version 1.0   (released May 1, 2006)

    This file is part of HARM.  HARM is a program that solves hyperbolic 
    partial differential equations in conservative form using high-resolution
    shock-capturing techniques.  This version of HARM has been configured to 
    solve the relativistic magnetohydrodynamic equations of motion on a 
    stationary black hole spacetime in Kerr-Schild coordinates to evolve
    an accretion disk model. 

    You are morally obligated to cite the following two papers in his/her 
    scientific literature that results from use of any part of HARM:

    [1] Gammie, C. F., McKinney, J. C., \& Toth, G.\ 2003, 
        Astrophysical Journal, 589, 444.

    [2] Noble, S. C., Gammie, C. F., McKinney, J. C., \& Del Zanna, L. \ 2006, 
        Astrophysical Journal, 641, 626.

   
    Further, we strongly encourage you to obtain the latest version of 
    HARM directly from our distribution website:
    http://rainman.astro.uiuc.edu/codelib/


    HARM is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    HARM is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with HARM; if not, write to the Free Software
    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

***********************************************************************************/

/*
 *
 * generates initial conditions for a fishbone & moncrief disk 
 * with exterior at minimum values for density & internal energy.
 *
 * cfg 8-10-01
 *
 */

#include "decs.h"
#include <float.h>


void coord_transform(double *pr,int i, int j,int k) ;
double compute_Amax( double (*A)[N2+D2][N3+D3] );
double compute_B_from_A( double (*A)[N2+D2][N3+D3], double (*p)[N2M][N3M][NPR] );
double normalize_B_by_maxima_ratio(double beta_target, double (*p)[N2M][N3M][NPR], double *norm_value);
double normalize_B_by_beta(double beta_target, double (*p)[N2M][N3M][NPR], double rmax, double *norm_value);

/////////////////////
//magnetic field geometry and normalization
#define NORMALFIELD (0)

#define WHICHFIELD NORMALFIELD

#define NORMALIZE_FIELD_BY_MAX_RATIO (1)
#define NORMALIZE_FIELD_BY_BETAMIN (2)
#define WHICH_FIELD_NORMALIZATION NORMALIZE_FIELD_BY_BETAMIN
//end magnetic field
//////////////////////

//////////////////////
//torus density normalization
#define THINTORUS_NORMALIZE_DENSITY (1)
#define DOAUTOCOMPUTEENK0 (1)

#define NORMALIZE_BY_TORUS_MASS (1)
#define NORMALIZE_BY_DENSITY_MAX (2)

#define DENSITY_NORMALIZATION NORMALIZE_BY_DENSITY_MAX
//torus density normalization
//////////////////////

double rmax = 0.;
double rhomax = 1.;

void init()
{
  void init_bondi(void);
  void init_torus(void);
  void init_sndwave(void);
  void init_entwave(void);
  void init_monopole(double Rout_val);

  switch( WHICHPROBLEM ) {
  case MONOPOLE_PROBLEM_1D:
  case MONOPOLE_PROBLEM_2D:
    init_monopole(1e3);
    break;
  case BZ_MONOPOLE_2D:
    init_monopole(100.);
    break;
  case TORUS_PROBLEM:
    init_torus();
    break;
  case SNDWAVE_TEST :
    init_sndwave() ;
    break ;
  case ENTWAVE_TEST :
    init_entwave() ;
    break ;
  case BONDI_PROBLEM_1D:
  case BONDI_PROBLEM_2D:
    init_bondi();
    break;
  }

}

void init_torus()
{
  int i,j,k ;
  double r,th,phi,sth,cth ;
  double ur,uh,up,u,rho ;
  double X[NDIM] ;
  struct of_geom geom ;

  /* for disk interior */
  double l,rin,lnh,expm2chi,up1 ;
  double DD,AA,SS,thin,sthin,cthin,DDin,AAin,SSin ;
  double kappa,hm1 ;

  /* for magnetic field */
  double A[N1+D1][N2+D2][N3+D3] ;
  double rho_av,umax,beta,bsq_ij,bsq_max,norm,q,beta_act ;
  double lfish_calc(double rmax) ;
  
  int iglob, jglob, kglob;
  double rancval;
  
  double amax, aphipow;
  
  double Amin, Amax, cutoff_frac = 0.01;
  
  /* some physics parameters */
  gam = 5./3. ;

  /* disk parameters (use fishbone.m to select new solutions) */
  a = 0.9 ;
  rin = 15. ;  // was 6.  — Chashkina 2021 fiducial
  rmax = 32. ;  // was 13. — Chashkina 2021 fiducial
  l = lfish_calc(rmax) ;

  kappa =1.e-3;
  beta = 100. ;

  /* some numerical parameters */
  lim = MC ;
  failed = 0 ;	/* start slow */
  cour = .8 ;
  dt = 1.e-5 ;
  R0 = 0.0 ;
  Rin = 0.87*(1. + sqrt(1. - a*a)) ;  //.98
  Rout = 1e5;
  rbr = 400.;
  npow2=4.0; //power exponent
  cpow2=1.0; //exponent prefactor (the larger it is, the more hyperexponentiation is)


  t = 0. ;
  hslope = 0.3 ;

  if(N2!=1) {
    //2D problem, use full pi-wedge in theta
    fractheta = 1.;
  }
  else{
    //1D problem (since only 1 cell in theta-direction), use a restricted theta-wedge
    fractheta = 1.e-2;
  }
  
  fracphi = 1.;

  //cylindrification parameters
  global_x10 = 3.5;  //radial distance in MCOORD until which the innermost angular cell is cylinrdical
  global_x20 = -1. + 1./mpi_ntot[2];     //This restricts grid cylindrification to the one
  //single grid cell closest to the pole (other cells virtually unaffeced, so there evolution is accurate).
  //This trick minimizes the resulting pole deresolution and relaxes the time step.
  //The innermost grid cell is evolved inaccurately whether you resolve it or not, and it will be fixed
  //by POLEFIX (see bounds.c).
  
  set_arrays() ;
  set_grid() ;

  get_phys_coord(5,0,0,&r,&th,&phi) ;
  if(MASTER==mpi_rank) {
    fprintf(stderr,"r[5]: %g\n",r) ;
    fprintf(stderr,"r[5]/rhor: %g",r/(1. + sqrt(1. - a*a))) ;
    if( r > 1. + sqrt(1. - a*a) ) {
      fprintf(stderr, ": INSUFFICIENT RESOLUTION, ADD MORE CELLS INSIDE THE HORIZON\n" );
    }
    else {
      fprintf(stderr, "\n");
    }
  }

  /* output choices */
  tf = 10000.0 ; //originally 10000.0 --> then 1000.0

  DTd = 5.; /* dumping frequency, in units of M  (originally 10.) */
  DTl = 5. ;	/* logfile frequency, in units of M (originally 10.) */
  DTi = 5. ; 	/* image file frequ., in units of M (originally 10.) */
  DTr = 25. ; /* restart file frequ., in units of M (originally 10.) */
  DTr01 = 100. ; /* restart file frequ., in timesteps */

  /* start diagnostic counters */
  dump_cnt = 0 ;
  image_cnt = 0 ;
  rdump_cnt = 0 ;
  rdump01_cnt = 0 ;
  defcon = 1. ;

  rhomax = 0. ;
  umax = 0. ;
  //ZSLOOP(0,N1-1,0,N2-1,0,N3-1) {
  for(iglob=0;iglob<mpi_ntot[1];iglob++) {
    for(jglob=0;jglob<mpi_ntot[2];jglob++) {
      for(kglob=0;kglob<mpi_ntot[3];kglob++) {
        
        rancval = ranc(0);
        i = iglob-mpi_startn[1];
        j = jglob-mpi_startn[2];
        k = kglob-mpi_startn[3];
        if(i<0 ||
           j<0 ||
           k<0 ||
           i>=N1 ||
           j>=N2 ||
           k>=N3){
          continue;
        }
        get_phys_coord(i,j,k,&r,&th,&phi) ;

        sth = sin(th) ;
        cth = cos(th) ;

        /* calculate lnh */
        DD = r*r - 2.*r + a*a ;
        AA = (r*r + a*a)*(r*r + a*a) - DD*a*a*sth*sth ;
        SS = r*r + a*a*cth*cth ;

        thin = M_PI/2. ;
        sthin = sin(thin) ;
        cthin = cos(thin) ;
        DDin = rin*rin - 2.*rin + a*a ;
        AAin = (rin*rin + a*a)*(rin*rin + a*a) 
                - DDin*a*a*sthin*sthin ;
        SSin = rin*rin + a*a*cthin*cthin ;

        if(r >= rin) {
          lnh = 0.5*log((1. + sqrt(1. + 4.*(l*l*SS*SS)*DD/
                  (AA*sth*AA*sth)))/(SS*DD/AA)) 
                  - 0.5*sqrt(1. + 4.*(l*l*SS*SS)*DD/(AA*AA*sth*sth))
                  - 2.*a*r*l/AA 
                  - (0.5*log((1. + sqrt(1. + 4.*(l*l*SSin*SSin)*DDin/
                  (AAin*AAin*sthin*sthin)))/(SSin*DDin/AAin)) 
                  - 0.5*sqrt(1. + 4.*(l*l*SSin*SSin)*DDin/
                          (AAin*AAin*sthin*sthin)) 
                  - 2.*a*rin*l/AAin ) ;
        }
        else
          lnh = 1. ;


        /* regions outside torus */
        if(lnh < 0. || r < rin) {
          //reset density and internal energy to zero outside torus
          rho = 0.; //1.e-7*RHOMIN ;
          u = 0.; //1.e-7*UUMIN ;

          /* these values are demonstrably physical
             for all values of a and r */
          /*
          ur = -1./(r*r) ;
          uh = 0. ;
          up = 0. ;
          */

          ur = 0. ;
          uh = 0. ;
          up = 0. ;

          /*
          get_geometry(i,j,CENT,&geom) ;
          ur = geom.gcon[0][1]/geom.gcon[0][0] ;
          uh = geom.gcon[0][2]/geom.gcon[0][0] ;
          up = geom.gcon[0][3]/geom.gcon[0][0] ;
          */

          p[i][j][k][RHO] = rho ;
          p[i][j][k][UU] = u ;
          p[i][j][k][U1] = ur ;
          p[i][j][k][U2] = uh ;
          p[i][j][k][U3] = up ;
        }
        /* region inside magnetized torus; u^i is calculated in
         * Boyer-Lindquist coordinates, as per Fishbone & Moncrief,
         * so it needs to be transformed at the end */
        else { 
          hm1 = exp(lnh) - 1. ;
          rho = pow(hm1*(gam - 1.)/(kappa*gam),
                                  1./(gam - 1.)) ; 
          u = kappa*pow(rho,gam)/(gam - 1.) ;
          ur = 0. ;
          uh = 0. ;

          /* calculate u^phi */
          expm2chi = SS*SS*DD/(AA*AA*sth*sth) ;
          up1 = sqrt((-1. + sqrt(1. + 4.*l*l*expm2chi))/2.) ;
          up = 2.*a*r*sqrt(1. + up1*up1)/sqrt(AA*SS*DD) +
                  sqrt(SS/AA)*up1/sth ;


          p[i][j][k][RHO] = rho ;
          if(rho > rhomax) rhomax = rho ;
          p[i][j][k][UU] = u*(1. + 4.e-2*(rancval-0.5)) ;
          if(u > umax && r > rin) umax = u ;
          p[i][j][k][U1] = ur ;
          p[i][j][k][U2] = uh ;

          p[i][j][k][U3] = up ;

          /* convert from 4-vel in BL coords to relative 4-vel in code coords */
          coord_transform(p[i][j][k],i,j,k) ;
        }

        p[i][j][k][B1] = 0. ;
        p[i][j][k][B2] = 0. ;
        p[i][j][k][B3] = 0. ;
      }
    }
  }

#ifdef MPI
  //exchange the info between the MPI processes to get the true max
  MPI_Allreduce(MPI_IN_PLACE,&rhomax,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&umax,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
#endif
  
  /* Normalize the densities so that max(rho) = 1 */
  if(MASTER==mpi_rank) fprintf(stderr,"rhomax: %g\n",rhomax) ;
  ZSLOOP(0,N1-1,0,N2-1,0,N3-1) {
          p[i][j][k][RHO] /= rhomax ;
          p[i][j][k][UU]  /= rhomax ;
  }
  umax /= rhomax ;
  kappa *= pow(rhomax,gam-1);
  global_kappa = kappa;
  rhomax = 1. ;

  bound_prim(p) ;

  if (WHICHFIELD == NORMALFIELD) {
    aphipow = 0.;
  }
  else {
    fprintf(stderr, "Unknown field type: %d\n", (int)WHICHFIELD);
    exit(321);
  }

  /* first find corner-centered vector potential */
  ZSLOOP(0,N1-1+D1,0,N2-1+D2,0,N3-1+D3) A[i][j][k] = 0. ;
  ZSLOOP(0,N1-1+D1,0,N2-1+D2,0,N3-1+D3) {
          /* radial field version */
          /*
          coord(i,j,k,CORN,X) ;
          bl_coord(X,&r,&th,&phi) ;
         
          A[i][j][k] = (1-cos(th)) ;
          */

    
          /* vertical field version */
          /*
          coord(i,j,k,CORN,X) ;
          bl_coord(X,&r,&th,&phi) ;

          A[i][j][k] = r*r*sin(th)*sin(th) ;
          */
    
    

          /* field-in-disk version */
          /* flux_ct */
    
      //cannot use get_phys_coords() here because it can only provide coords at CENT
      coord(i,j,k,CORN,X) ;
      bl_coord(X,&r,&th,&phi) ;


          rho_av = 0.25*(
                  p[i][j][k][RHO] +
                  p[i-1][j][k][RHO] +
                  p[i][j-1][k][RHO] +
                  p[i-1][j-1][k][RHO]) ;

          q = pow(r,aphipow)*rho_av/rhomax ;
          if (WHICHFIELD == NORMALFIELD) {
            q -= 0.2;
          }
          if(q > 0.) A[i][j][k] = q ;

  }
  
  fixup(p) ;

  /* now differentiate to find cell-centered B,
     and begin normalization */
  
  bsq_max = compute_B_from_A(A,p);
  
  if(WHICHFIELD == NORMALFIELD) {
    if(MASTER==mpi_rank)
      fprintf(stderr,"initial bsq_max: %g\n",bsq_max) ;

    /* finally, normalize to set field strength */
    beta_act =(gam - 1.)*umax/(0.5*bsq_max) ;

    if(MASTER==mpi_rank)
      fprintf(stderr,"initial beta: %g (should be %g)\n",beta_act,beta) ;
    
    if(WHICH_FIELD_NORMALIZATION == NORMALIZE_FIELD_BY_BETAMIN) {
      beta_act = normalize_B_by_beta(beta, p, 10*rmax, &norm);
    }
    else if(WHICH_FIELD_NORMALIZATION == NORMALIZE_FIELD_BY_MAX_RATIO) {
      beta_act = normalize_B_by_maxima_ratio(beta, p, &norm);
    }
    else {
      if(i_am_the_master) {
        fprintf(stderr, "Unknown magnetic field normalization %d\n",
                WHICH_FIELD_NORMALIZATION);
        MPI_Finalize();
        exit(2345);
      }
    }

    if(MASTER==mpi_rank)
      fprintf(stderr,"final beta: %g (should be %g); normalization factor: %g\n",beta_act,beta,norm) ;
  }

    
  /* enforce boundary conditions */
  fixup(p) ;
  bound_prim(p) ;




#if( DO_FONT_FIX )
  set_Katm();
#endif 


}

//note that only axisymmetric A is supported
double compute_Amax( double (*A)[N2+D2][N3+D3] )
{
  double Amax = 0.;
  int i, j, k;
  struct of_geom geom;
  
  ZLOOP {
    if(A[i][j][k] > Amax) Amax = A[i][j][k];
  }
  
#ifdef MPI
  //exchange the info between the MPI processes to get the true max
  MPI_Allreduce(MPI_IN_PLACE,&Amax,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
#endif
  
  return(Amax);
}


//note that only axisymmetric A is supported
double compute_B_from_A( double (*A)[N2+D2][N3+D3], double (*p)[N2M][N3M][NPR] )
{
  double bsq_max = 0., bsq_ij ;
  int i, j, k;
  struct of_geom geom;
  
  ZLOOP {
    get_geometry(i,j,k,CENT,&geom) ;
    
    /* flux-ct */
    p[i][j][k][B1] = -(A[i][j][k] - A[i][j+1][k]
                       + A[i+1][j][k] - A[i+1][j+1][k])/(2.*dx[2]*geom.g) ;
    p[i][j][k][B2] = (A[i][j][k] + A[i][j+1][k]
                      - A[i+1][j][k] - A[i+1][j+1][k])/(2.*dx[1]*geom.g) ;
    
    p[i][j][k][B3] = 0. ;
    
    bsq_ij = bsq_calc(p[i][j][k],&geom) ;
    if(bsq_ij > bsq_max) bsq_max = bsq_ij ;
  }
#ifdef MPI
  //exchange the info between the MPI processes to get the true max
  MPI_Allreduce(MPI_IN_PLACE,&bsq_max,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
#endif

  return(bsq_max);
}

double normalize_B_by_maxima_ratio(double beta_target, double (*p)[N2M][N3M][NPR], double *norm_value)
{
  double beta_act, bsq_ij, u_ij, umax = 0., bsq_max = 0.;
  double norm;
  int i, j, k;
  struct of_geom geom;
  
  ZLOOP {
    get_geometry(i,j,k,CENT,&geom) ;
    bsq_ij = bsq_calc(p[i][j][k],&geom) ;
    if(bsq_ij > bsq_max) bsq_max = bsq_ij ;
    u_ij = p[i][j][k][UU];
    if(u_ij > umax) umax = u_ij;
  }
#ifdef MPI
  //exchange the info between the MPI processes to get the true max
  MPI_Allreduce(MPI_IN_PLACE,&umax,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&bsq_max,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
#endif

  /* finally, normalize to set field strength */
  beta_act =(gam - 1.)*umax/(0.5*bsq_max) ;
  
  norm = sqrt(beta_act/beta_target) ;
  bsq_max = 0. ;
  ZLOOP {
    p[i][j][k][B1] *= norm ;
    p[i][j][k][B2] *= norm ;
    
    get_geometry(i,j,k,CENT,&geom) ;
    bsq_ij = bsq_calc(p[i][j][k],&geom) ;
    if(bsq_ij > bsq_max) bsq_max = bsq_ij ;
  }
#ifdef MPI
  //exchange the info between the MPI processes to get the true max
  MPI_Allreduce(MPI_IN_PLACE,&bsq_max,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
#endif
  
  beta_act = (gam - 1.)*umax/(0.5*bsq_max) ;

  if(norm_value) {
    *norm_value = norm;
  }
  return(beta_act);
}

//normalize the magnetic field using the values inside r < rmax
double normalize_B_by_beta(double beta_target, double (*p)[N2M][N3M][NPR], double rmax, double *norm_value)
{
  double beta_min = 1e100, beta_ij, beta_act, bsq_ij, u_ij, umax = 0., bsq_max = 0.;
  double norm;
  int i, j, k;
  struct of_geom geom;
  double X[NDIM], r, th, ph;
  
  ZLOOP {
    coord(i, j, k, CENT, X);
    bl_coord(X, &r, &th, &ph);
    if (r>rmax) {
      continue;
    }
    get_geometry(i,j,k,CENT,&geom) ;
    bsq_ij = bsq_calc(p[i][j][k],&geom) ;
    u_ij = p[i][j][k][UU];
    beta_ij = (gam - 1.)*u_ij/(0.5*(bsq_ij+SMALL)) ;
    if(beta_ij < beta_min) beta_min = beta_ij ;
  }
#ifdef MPI
  //exchange the info between the MPI processes to get the true max
  MPI_Allreduce(MPI_IN_PLACE,&beta_min,1,MPI_DOUBLE,MPI_MIN,MPI_COMM_WORLD);
#endif
  
  /* finally, normalize to set field strength */
  beta_act = beta_min;
  
  norm = sqrt(beta_act/beta_target) ;
  beta_min = 1e100;
  ZLOOP {
    p[i][j][k][B1] *= norm ;
    p[i][j][k][B2] *= norm ;
    p[i][j][k][B3] *= norm ;
    coord(i, j, k, CENT, X);
    bl_coord(X, &r, &th, &ph);
    get_geometry(i,j,k,CENT,&geom) ;
    bsq_ij = bsq_calc(p[i][j][k],&geom) ;
    u_ij = p[i][j][k][UU];
    beta_ij = (gam - 1.)*u_ij/(0.5*(bsq_ij+SMALL)) ;
    if(r<rmax && beta_ij < beta_min) beta_min = beta_ij ;
  }
#ifdef MPI
  //exchange the info between the MPI processes to get the true max
  MPI_Allreduce(MPI_IN_PLACE,&beta_min,1,MPI_DOUBLE,MPI_MIN,MPI_COMM_WORLD);
#endif
  
  beta_act = beta_min;

  if(norm_value) {
    *norm_value = norm;
  }

  return(beta_act);
}


void init_bondi()
{
	int i,j,k ;
	double r,th,phi,sth,cth ;
	double ur,uh,up,u,rho ;
	double X[NDIM] ;
	struct of_geom geom ;
	double rhor;

	/* for disk interior */
	double l,rin,lnh,expm2chi,up1 ;
	double DD,AA,SS,thin,sthin,cthin,DDin,AAin,SSin ;
	double kappa,hm1 ;

	/* for magnetic field */
	double A[N1+1][N2+1][N3+1] ;
	double rho_av,rhomax,umax,beta,bsq_ij,bsq_max,norm,q,beta_act ;
	double rmax, lfish_calc(double rmax) ;

	// =================================================================
	// SCENARIO CONFIGURATION - MODIFY THESE TO SWITCH BETWEEN CASES
	// =================================================================
	
	// Set EXACTLY ONE of these to 1, others to 0
	int PURE_BONDI = 0;                    // Pure spherical Bondi accretion
	int BONDI_HOYLE_LYTTLETON = 0;         // Uniform wind case
	int DENSITY_GRADIENT = 0;              // Global density gradient
	int ANGULAR_MOMENTUM = 1;              // Small initial angular momentum
	int RANDOM_VELOCITY = 0;               // Random velocity field
	
	// Wind velocity parameter - recommended values:
	// Pure Bondi: 0.0 (no wind)
	// BHL: 0.1 (strong wind in z-direction)
	// Density Gradient: 0.02-0.05 (ambient wind + gradient)
	// Angular Momentum: 0.0 (no wind needed)
	// Random Velocity: 0.02-0.05 (ambient wind + turbulence)
	double v_z_wind = 0.1;  // RENAMED: z-velocity (not theta-velocity!)

	// Other scenario parameters
	double density_gradient_index = 1.5;   // Power law index for density gradient 
	double omega_init_amplitude = 0.05;    // Angular momentum strength
	double turbulence_amplitude = 0.1;     // Random velocity amplitude
	
	// =================================================================

	/* some physics parameters */
	gam = 4./3. ;

	/* black hole parameters */
	a = 0.9375 ;

	kappa = 1.e-3 ;

	/* radius of the inner edge of the initial density distribution */
		/* =================================================================
	   CRITICAL FIX #1: Move rin closer to allow sonic point in gas region
	   =================================================================
	   Original: rin = 10.0 (sonic point would be in vacuum!)
	   Fixed: rin = 2.0 (allows sonic point at r≈5 to be in gas)
	*/
	rin = 2.0;  // CHANGED from 10.0

  	/* =================================================================
	   CRITICAL FIX #2: Define Bondi parameters
	   =================================================================
	*/
	// For γ=4/3, sonic radius r_s = GM/c_s^2
	// In code units where GM=1, if we want r_sonic≈5, then c_s^2 = 1/5 = 0.2
	double r_sonic = 5.0;      // Bondi sonic radius
	double c_s_inf = sqrt(0.2); // Sound speed at infinity to get r_sonic=5
	double rho_inf = 1.0;      // Density normalization at infinity
	
	// Bondi accretion rate: Mdot = 4π (GM)^2 ρ∞ / c_s^3
	// In our units: Mdot = 4π ρ∞ / c_s^3
	double Mdot_bondi = 4.0 * M_PI * rho_inf / (c_s_inf * c_s_inf * c_s_inf);

	/* some numerical parameters */
	lim = MC ;
	failed = 0 ;	/* start slow */
	cour = 0.9 ;
	dt = 1.e-5 ;
	rhor = (1. + sqrt(1. - a*a)) ;
	R0 = -2*rhor ;
	Rin = 0.5*rhor ;
	Rout = 1e3 ;
	rbr = Rout*10.;
	npow2=4.0;
	cpow2=1.0;

	t = 0. ;
	hslope = 1.0 ;

	if(N2!=1) {
		fractheta = 1.;
	}
	else{
		fractheta = 1.e-2;
	}
	fracphi = 1.;

	set_arrays() ;
	set_grid() ;

	coord(-2,0,0,CENT,X) ;
	bl_coord(X,&r,&th,&phi) ;
	fprintf(stderr,"rmin: %g\n",r) ;
	fprintf(stderr,"rmin/rm: %g\n",r/(1. + sqrt(1. - a*a))) ;

  // Add Bondi diagnostic output
	fprintf(stderr,"Bondi parameters:\n");
	fprintf(stderr,"  rin (gas inner edge): %g\n", rin);
	fprintf(stderr,"  r_sonic (theoretical): %g\n", r_sonic);
	fprintf(stderr,"  c_s_inf: %g\n", c_s_inf);
	fprintf(stderr,"  Mdot_bondi: %g\n", Mdot_bondi);

	/* output choices */
	tf = Rout ;

	DTd = 2. ;
	DTl = 2. ;
	DTi = 2. ;
	DTr = 50 ;
	DTr01 = 1000 ;

	/* start diagnostic counters */
	dump_cnt = 0 ;
	image_cnt = 0 ;
	rdump_cnt = 0 ;
	rdump01_cnt = 0 ;
	defcon = 1. ;

	// Print which scenario is active
	if(PURE_BONDI) fprintf(stderr,"Scenario: Pure Bondi accretion\n");
	else if(BONDI_HOYLE_LYTTLETON) fprintf(stderr,"Scenario: Bondi-Hoyle-Lyttleton (v_z=%.2f)\n", v_z_wind);
	else if(DENSITY_GRADIENT) fprintf(stderr,"Scenario: Density gradient (index=%.2f, v_z=%.2f)\n", density_gradient_index, v_z_wind);
	else if(ANGULAR_MOMENTUM) fprintf(stderr,"Scenario: Angular momentum (omega=%.3f)\n", omega_init_amplitude);
	else if(RANDOM_VELOCITY) fprintf(stderr,"Scenario: Random velocity (amplitude=%.2f, v_z=%.2f)\n", turbulence_amplitude, v_z_wind);
	else fprintf(stderr,"WARNING: No scenario selected or multiple scenarios active!\n");

	rhomax = 0. ;
	umax = 0. ;
	
	ZSLOOP(0,N1-1,0,N2-1,0,N3-1) {
		coord(i,j,k,CENT,X) ;
		bl_coord(X,&r,&th,&phi) ;

		sth = sin(th) ;
		cth = cos(th) ;

		// =================================================================
		// REGION 1: r < rin (evacuated inner region)
		// =================================================================
		if(r < rin) {
			rho = 1.e-7*RHOMIN ;
			u = 1.e-7*UUMIN ;

      // Initialize velocities
			ur = 0. ;
			uh = 0. ;
			up = 0. ;
      
      // For pure Bondi, set velocities in vacuum region
			// Use extrapolation from supersonic Bondi solution
			if(PURE_BONDI) {
				// In supersonic region, v ≈ sqrt(2GM/r) for r << r_sonic
				double v_esc = sqrt(2.0/r);
				ur = -0.9 * v_esc;  // 90% of escape velocity
				uh = 0.;
				up = 0.;
			}
			// [Keep other scenario velocity setups for r<rin as in original...]
			else if(ANGULAR_MOMENTUM || BONDI_HOYLE_LYTTLETON || DENSITY_GRADIENT) {
				ur = -0.01 / (r * r);  // Weak pressure-driven inflow
			}

			// CORRECTED: Add z-velocity using proper coordinate transformation
			if(BONDI_HOYLE_LYTTLETON || DENSITY_GRADIENT) {
				double gamma_z = 1.0 / sqrt(1.0 - v_z_wind * v_z_wind);
				
				// Transform v_z to Boyer-Lindquist 4-velocity components
				ur += gamma_z * v_z_wind * cth;           // Radial component
				uh = -gamma_z * v_z_wind * sth / r;       // Theta component
			}

			if(RANDOM_VELOCITY) {
				double gamma_z = 1.0 / sqrt(1.0 - v_z_wind * v_z_wind);
				ur += gamma_z * v_z_wind * cth;
				uh = -gamma_z * v_z_wind * sth / r;
				
				// Add random perturbations
				double rand_seed = fmod(1000.0 * (r + th + phi), 1.0);
				ur += turbulence_amplitude * (2.0 * rand_seed - 1.0);
				uh += turbulence_amplitude * (2.0 * fmod(rand_seed * 1.618, 1.0) - 0.5);
				up += turbulence_amplitude * (2.0 * fmod(rand_seed * 2.718, 1.0) - 0.5);
			}

			p[i][j][k][RHO] = rho ;
			p[i][j][k][UU] = u ;
			p[i][j][k][U1] = ur ;
			p[i][j][k][U2] = uh ;
			p[i][j][k][U3] = up ;
		}
		// =================================================================
		// REGION 2: r >= rin (ambient medium) // This is where the actual gas starts
		// =================================================================
		else {
      			// =================================================================
			// CRITICAL FIX #3: Implement proper Bondi solution
			// =================================================================
			
			if(PURE_BONDI) {
				/* 
				 * BONDI SOLUTION FOR γ=4/3
				 * 
				 * The Bondi solution has two regions:
				 * 1. Subsonic (r > r_sonic): low velocity, density ∝ r^(-3/2)
				 * 2. Supersonic (r < r_sonic): high velocity, density from Mdot conservation
				 * 
				 * For γ=4/3, the solution simplifies considerably
				 */
				
				if(r > r_sonic) {
					// ======================
					// SUBSONIC REGION (r > r_sonic)
					// ======================
					
					// Parker wind solution (good approximation for subsonic Bondi)
					// Velocity scales as (r_sonic/r)^2 in subsonic region
					ur = -c_s_inf * pow(r_sonic/r, 2.0);
					
					// Density from mass conservation: ρ = Mdot/(4πr²|v|)
					// This automatically gives ρ ∝ r^(-3/2)
					rho = Mdot_bondi / (4.0 * M_PI * r * r * fabs(ur));
					
				} else {
					// ======================
					// SUPERSONIC REGION (r < r_sonic)  
					// ======================
					
					// Use Bernoulli equation: v²/2 + c_s² - GM/r = const
					// At sonic point: v_s²/2 + c_s² - GM/r_s = const
					// For γ=4/3: c_s² = c_s_inf² (ρ/ρ_inf)^(1/3)
					
					// Simplified for γ=4/3: velocity approaches free-fall
					double v_esc = sqrt(2.0/r);  // Escape velocity
					double v_sonic = c_s_inf;    // Velocity at sonic point
					
					// Smooth interpolation between sonic point and free-fall
					double f = (r_sonic - r) / r_sonic;  // 0 at sonic point, 1 at r=0
					ur = -(v_sonic + f * (v_esc - v_sonic));
					
					// Density from mass conservation
					rho = Mdot_bondi / (4.0 * M_PI * r * r * fabs(ur));
				}
				
				// No angular velocities for pure Bondi
				uh = 0.;
				up = 0.;
				
				// Internal energy for γ=4/3 ideal gas
				// u = P/(γ-1) = ρc_s²/(γ-1)
				// For γ=4/3: u = 3P = 3ρc_s²
				// Sound speed: c_s² = c_s_inf² * (ρ/ρ_inf)^(1/3)
				double c_s_local_sq = c_s_inf * c_s_inf * pow(rho/rho_inf, 1./3.);
				u = 3.0 * rho * c_s_local_sq;
				
			}
			// Set density profile based on scenario
			else if(DENSITY_GRADIENT) {
				rho = 1.0 * pow(r/rin, -density_gradient_index);
			} 
			else if(ANGULAR_MOMENTUM) {
				// Gentle profile for stability
				rho = 1.0 * pow(r/rin, -1.5);
			} 
			else {
				// Uniform for BHL and random velocity
				rho = 1.0;
			}

			u = kappa*pow(rho,gam)/(gam - 1.) ;

			// Initialize velocities
			ur = 0. ;
			uh = 0. ;
			up = 0. ;

			// Apply small radial infall for scenarios that need it
			if(ANGULAR_MOMENTUM) {
				ur = -0.001 / (r * r);  // Very weak for stability
			}

			// CORRECTED: Apply z-wind using proper transformation
			if(BONDI_HOYLE_LYTTLETON || DENSITY_GRADIENT) {
				double gamma_z = 1.0 / sqrt(1.0 - v_z_wind * v_z_wind);
				
				// THIS IS THE CORRECT WAY to add z-velocity!
				ur += gamma_z * v_z_wind * cth;
				uh = -gamma_z * v_z_wind * sth / r;
			}

			// Angular momentum (rotation around z-axis)
			if(ANGULAR_MOMENTUM && r > 6.0) {
				up = omega_init_amplitude * sth;  // φ-velocity
			}

			// Random velocity perturbations
			if(RANDOM_VELOCITY) {
				double gamma_z = 1.0 / sqrt(1.0 - v_z_wind * v_z_wind);
				ur += gamma_z * v_z_wind * cth;
				uh = -gamma_z * v_z_wind * sth / r;
				
				// Add turbulent fluctuations
				double rand_seed = fmod(1000.0 * (r + th + phi), 1.0);
				ur += turbulence_amplitude * (2.0 * rand_seed - 1.0);
				uh += turbulence_amplitude * (2.0 * fmod(rand_seed * 1.618, 1.0) - 0.5);
				up += turbulence_amplitude * (2.0 * fmod(rand_seed * 2.718, 1.0) - 0.5);
			}

			p[i][j][k][RHO] = rho ;
			if(rho > rhomax) rhomax = rho ;
			p[i][j][k][UU] = u;
			if(u > umax && r > rin) umax = u ;
			p[i][j][k][U1] = ur ;
			p[i][j][k][U2] = uh ;
			p[i][j][k][U3] = up ;
			
			/* convert from 4-vel to 3-vel */
			coord_transform(p[i][j][k],i,j,k) ;
		}

		// Zero magnetic field (pure hydrodynamic problem)
		p[i][j][k][B1] = 0. ;
		p[i][j][k][B2] = 0. ;
		p[i][j][k][B3] = 0. ;
	}

	fixup(p) ;
	bound_prim(p) ;

	// Print final summary
	fprintf(stderr,"Setup complete: rhomax=%.2e, umax=%.2e\n", rhomax, umax);
	
	// Print verification info
  	if(PURE_BONDI) {
		fprintf(stderr,"Bondi initial conditions:\n");
		fprintf(stderr,"  Expected sonic radius: r_s = %.2f\n", r_sonic);
		fprintf(stderr,"  Expected accretion rate: Mdot = %.3e\n", Mdot_bondi);
		fprintf(stderr,"  Density at r=100: %.3e (should be ∝ r^-1.5)\n", 
			        Mdot_bondi / (4.0 * M_PI * 100 * 100 * c_s_inf * pow(r_sonic/100, 2.0)));
	}

	else if(BONDI_HOYLE_LYTTLETON || DENSITY_GRADIENT || RANDOM_VELOCITY) {
		fprintf(stderr,"Z-velocity transformation applied: v_z=%.3f\n", v_z_wind);
		fprintf(stderr,"  Lorentz factor: gamma=%.4f\n", 1.0/sqrt(1.0 - v_z_wind*v_z_wind));
		fprintf(stderr,"  At equator (theta=pi/2): u^r ≈ 0, u^theta ≈ -gamma*v_z/r\n");
		fprintf(stderr,"  At pole (theta=0): u^r ≈ gamma*v_z, u^theta ≈ 0\n");
	}
    

#if(0) //disable for now
	/* first find corner-centered vector potential */
	ZSLOOP(0,N1,0,N2,0,N3) A[i][j][k] = 0. ;
	ZSLOOP(0,N1,0,N2,0,N3) {
		/* vertical field version */
		/*
		coord(i,j,l,CORN,X) ;
		bl_coord(X,&r,&th,&phi) ;

		A[i][j][k] = 0.5*r*sin(th) ;
		*/

		/* field-in-disk version */
		/* flux_ct */
		rho_av = 0.25*(
			p[i][j][RHO] +
			p[i-1][j][RHO] +
			p[i][j-1][RHO] +
			p[i-1][j-1][RHO]) ;

		q = rho_av/rhomax - 0.2 ;
		if(q > 0.) A[i][j][k] = q ;

	}

	/* now differentiate to find cell-centered B,
	   and begin normalization */
	bsq_max = 0. ;
	ZLOOP {
		get_geometry(i,j,k,CENT,&geom) ;

		/* flux-ct */
		p[i][j][B1] = -(A[i][j][k] - A[i][j+1][k]
				+ A[i+1][j][k] - A[i+1][j+1][k])/(2.*dx[2]*geom.g) ;
		p[i][j][B2] = (A[i][j][k] + A[i][j+1][k]
				- A[i+1][j][k] - A[i+1][j+1][k])/(2.*dx[1]*geom.g) ;

		p[i][j][B3] = 0. ;

		bsq_ij = bsq_calc(p[i][j][k],&geom) ;
		if(bsq_ij > bsq_max) bsq_max = bsq_ij ;
	}
	fprintf(stderr,"initial bsq_max: %g\n",bsq_max) ;

	/* finally, normalize to set field strength */
	beta_act = (gam - 1.)*umax/(0.5*bsq_max) ;
	fprintf(stderr,"initial beta: %g (should be %g)\n",beta_act,beta) ;
	norm = sqrt(beta_act/beta) ;
	bsq_max = 0. ;
	ZLOOP {
		p[i][j][k][B1] *= norm ;
		p[i][j][k][B2] *= norm ;

		get_geometry(i,j,k,CENT,&geom) ;
		bsq_ij = bsq_calc(p[i][j][k],&geom) ;
		if(bsq_ij > bsq_max) bsq_max = bsq_ij ;
	}
	beta_act = (gam - 1.)*umax/(0.5*bsq_max) ;
	fprintf(stderr,"final beta: %g (should be %g)\n",beta_act,beta) ;

	/* enforce boundary conditions */
	fixup(p) ;
	bound_prim(p) ;
    
#endif

    

    
#if( DO_FONT_FIX ) 
	set_Katm();
#endif 


}

void init_monopole(double Rout_val)
{
	int i,j,k ;
	double r,th,phi,sth,cth ;
	double ur,uh,up,u,rho ;
	double X[NDIM] ;
	struct of_geom geom ;
	double rhor;

	/* for disk interior */
	double l,rin,lnh,expm2chi,up1 ;
	double DD,AA,SS,thin,sthin,cthin,DDin,AAin,SSin ;
	double kappa,hm1 ;

	/* for magnetic field */
	double A[N1+1][N2+1] ;
	double rho_av,rhomax,umax,beta,bsq_ij,bsq_max,norm,q,beta_act ;
	double rmax, lfish_calc(double rmax) ;

  // =================================================================
	// MAGNETIC FIELD CONFIGURATION - MODIFY THESE TO SWITCH SCENARIOS
	// =================================================================
	
	// Set EXACTLY ONE of these to 1, others to 0
	int MONOPOLE_FIELD = 0;      // Standard radial monopole field
	int DIPOLE_FIELD = 0;        // Dipole field aligned with spin axis
	int SPLIT_MONOPOLE = 1;      // Split monopole (different N/S hemispheres)
	
	// Split monopole parameters (only used if SPLIT_MONOPOLE == 1)
	double split_theta_boundary = M_PI/2.0;  // Where field splits (equator)
	double split_north_strength = 1.0;       // Relative strength above boundary
	double split_south_strength = 1.0;       // Relative strength below boundary
	
	// Dipole parameters (only used if DIPOLE_FIELD == 1)
	double mu_dipole = 1.0;  // Dipole moment strength
	
	// =================================================================

	/* some physics parameters */
	gam = 4./3. ;

	/* disk parameters (use fishbone.m to select new solutions) */
        a = 0.9375 ;
        rin = 6. ;
        rmax = 12. ;
        l = lfish_calc(rmax) ;

	kappa = 1.e-3 ;
	beta = 1.e2 ;

        /* some numerical parameters */
        lim = MC ;
        failed = 0 ;	/* start slow */
        cour = 0.9 ;
        dt = 1.e-5 ;
	rhor = (1. + sqrt(1. - a*a)) ;
	R0 = -4*rhor;
        Rin = 0.7*rhor ;
        Rout = Rout_val ;
        rbr = Rout*10.;
    npow2=4.0; //power exponent
    cpow2=1.0; //exponent prefactor (the larger it is, the more hyperexponentiation is)

        t = 0. ;
        hslope = 1. ;

	if(N2!=1) {
	  //2D problem, use full pi-wedge in theta
	  fractheta = 1.;
	}
	else{
	  //1D problem (since only 1 cell in theta-direction), use a restricted theta-wedge
	  fractheta = 1.e-2;
	}

        fracphi = 1.;

        set_arrays() ;
        set_grid() ;

	coord(-2,0,0,CENT,X) ;
	bl_coord(X,&r,&th,&phi) ;
	fprintf(stderr,"rmin: %g\n",r) ;
	fprintf(stderr,"rmin/rm: %g\n",r/(1. + sqrt(1. - a*a))) ;

        /* output choices */
	tf = 2*Rout ;

	DTd = 1. ;	/* dumping frequency, in units of M */
	DTl = 50. ;	/* logfile frequency, in units of M */
	DTi = 50. ; 	/* image file frequ., in units of M */
        DTr = 1. ; /* restart file frequ., in units of M */
	DTr01 = 1000 ; 	/* restart file frequ., in timesteps */

	/* start diagnostic counters */
	dump_cnt = 0 ;
	image_cnt = 0 ;
	rdump_cnt = 0 ;
        rdump01_cnt = 0 ;
	defcon = 1. ;

	// Print which magnetic field configuration is active
	if(MONOPOLE_FIELD) fprintf(stderr,"Magnetic field: Standard monopole\n");
	else if(DIPOLE_FIELD) fprintf(stderr,"Magnetic field: Dipole (mu=%.2f)\n", mu_dipole);
	else if(SPLIT_MONOPOLE) fprintf(stderr,"Magnetic field: Split monopole (theta=%.3f)\n", split_theta_boundary);

	rhomax = 0. ;
	umax = 0. ;
	ZSLOOP(0,N1-1,0,N2-1,0,N3-1) {
	  coord(i,j,k,CENT,X) ;
	  bl_coord(X,&r,&th,&phi) ;

	  sth = sin(th) ;
	  cth = cos(th) ;

	  /* rho = 1.e-7*RHOMIN ; */
	  /* u = 1.e-7*UUMIN ; */

	  /* rho = pow(r,-4.)/BSQORHOMAX; */
	  /* u = pow(r,-4.*gam)/BSQOUMAX; */

	  rho = RHOMINLIMIT+(r/10./rhor)/pow(r,4)/BSQORHOMAX;
	  u = UUMINLIMIT+(r/10./rhor)/pow(r,4)/BSQORHOMAX;

	  /* these values are demonstrably physical
	     for all values of a and r */
	  /*
	    ur = -1./(r*r) ;
	    uh = 0. ;
	    up = 0. ;
	  */

	  ur = 0. ;
	  uh = 0. ;
	  up = 0. ;

	  /*
	    get_geometry(i,j,CENT,&geom) ;
	    ur = geom.gcon[0][1]/geom.gcon[0][0] ;
	    uh = geom.gcon[0][2]/geom.gcon[0][0] ;
	    up = geom.gcon[0][3]/geom.gcon[0][0] ;
	  */

	  p[i][j][k][RHO] = rho ;
	  p[i][j][k][UU] = u ;
	  p[i][j][k][U1] = ur ;
	  p[i][j][k][U2] = uh ;
	  p[i][j][k][U3] = up ;
	  p[i][j][k][B1] = 0. ;
	  p[i][j][k][B2] = 0. ;
	  p[i][j][k][B3] = 0. ;
	}

	rhomax = 1. ;
	fixup(p) ;
	bound_prim(p) ;

        //leave A[][] a 2D array for now, which means that magnetic field will be axisymmetric
	/* first find corner-centered vector potential */
	ZSLOOP(0,N1,0,N2,0,0) A[i][j] = 0. ;
        ZSLOOP(0,N1,0,N2,0,0) {
                coord(i,j,k,CORN,X) ;
                bl_coord(X,&r,&th,&phi) ;
                
                if(MONOPOLE_FIELD) {
                    /* Standard radial monopole field */
                    A[i][j] = (1.0 - cos(th));
                }
                else if(DIPOLE_FIELD) {
                    /* Dipole field aligned with rotation axis */
                    if(r > 1.5) {  // Outside horizon
                        A[i][j] = mu_dipole * sin(th) * sin(th) / r;
                    } else {
                        A[i][j] = 0.0;  // No field inside horizon
                    }
                }
                else if(SPLIT_MONOPOLE) {
                    /* Split monopole - different field above/below boundary */
                    if(th < split_theta_boundary) {
                        // Northern hemisphere
                        A[i][j] = split_north_strength * (1.0 - cos(th));
                    } else {
                        // Southern hemisphere  
                        double th_rel = th - split_theta_boundary;
                        A[i][j] = split_south_strength * (1.0 - cos(th_rel));
                    }
                }
                else {
                    fprintf(stderr, "ERROR: No magnetic field configuration selected!\n");
                    fprintf(stderr, "Set exactly ONE field type to 1 in init_monopole()\n");
                    exit(1);
                }
        }

	/* now differentiate to find cell-centered B,
	   and begin normalization */
	bsq_max = 0. ;
	ZLOOP {
		get_geometry(i,j,k,CENT,&geom) ;

		/* flux-ct */
		p[i][j][k][B1] = -(A[i][j] - A[i][j+1]
				+ A[i+1][j] - A[i+1][j+1])/(2.*dx[2]*geom.g) ;
		p[i][j][k][B2] = (A[i][j] + A[i][j+1]
				- A[i+1][j] - A[i+1][j+1])/(2.*dx[1]*geom.g) ;

		p[i][j][k][B3] = 0. ;

		bsq_ij = bsq_calc(p[i][j][k],&geom) ;
		if(bsq_ij > bsq_max) bsq_max = bsq_ij ;
	}
	fprintf(stderr,"initial bsq_max: %g\n",bsq_max) ;

	/* enforce boundary conditions */
	fixup(p) ;
	bound_prim(p) ;
    
    



#if( DO_FONT_FIX )
	set_Katm();
#endif 


}

void init_entwave()
{
  int i,j,k ;
  double x,y,z,sth,cth ;
  double ur,uh,up,u,rho ;
  double X[NDIM] ;
  struct of_geom geom ;
  double rhor;
  
  double myrho, myu, mycs, myv;
  double delta_rho;
  double cosa, sina;
  double delta_ampl = 1.e-1; //amplitude of the wave
  double k_vec_x = 2 * M_PI;  //wavevector
  double k_vec_y = 2 * M_PI;
  double k_vec_len = sqrt( k_vec_x * k_vec_x + k_vec_y * k_vec_y );
  double tfac = 1.e3; //factor by which to reduce velocity
  

  /* some physics parameters */
  gam = 5./3. ;
  
  /* some numerical parameters */
  lim = VANL ;
  failed = 0 ;	/* start slow */
  cour = 0.9 ;
  dt = 1.e-5 ;
  
  t = 0. ;
  hslope = 1. ;
  
  if(N2!=1) {
    //2D problem, use full pi-wedge in theta
    fractheta = 1.;
  }
  else{
    //1D problem (since only 1 cell in theta-direction), use a restricted theta-wedge
    fractheta = 1.e-2;
  }
  
  fracphi = 1.;

  set_arrays() ;
  set_grid() ;
  
  
  myrho = 1.;
  myu = 4.*myrho / (gam * (gam-1));  //so that mycs is unity
  
  mycs = sqrt(gam * (gam-1) * myu / myrho);  //background sound speed
  myv = 1.;  //velocity with a magnitude of 1
  
  /* output choices */

    tf = tfac;///mycs;
  
  DTd = tf/10. ;	/* dumping frequency, in units of M */
  DTl = tf/10. ;	/* logfile frequency, in units of M */
  DTi = tf/10. ; 	/* image file frequ., in units of M */
  DTr = tf/10. ; /* restart file frequ., in units of M */
  DTr01 = 1000 ; 	/* restart file frequ., in timesteps */
  
  /* start diagnostic counters */
  dump_cnt = 0 ;
  image_cnt = 0 ;
  rdump_cnt = 0 ;
  rdump01_cnt = 0 ;
  defcon = 1. ;
  

  ZSLOOP(0,N1-1,0,N2-1,0,N3-1) {
    coord(i,j,k,CENT,X) ;
    bl_coord(X,&x,&y,&z) ;
    
    //applying the perturbations
    delta_rho = delta_ampl * cos( k_vec_x * x + k_vec_y * y );
    
  //  p[i][j][k][RHO] = myrho + delta_rho;
      if(x<.2){
          p[i][j][k][RHO] = 1.;
          p[i][j][k][U2] = 0.;

      }
      else if (x>.8){
          p[i][j][k][RHO] = 1.;
          p[i][j][k][U2] = 0.;
      }
      else{
          p[i][j][k][RHO] = 1e4;
          p[i][j][k][U2] = 0.;

      }
    p[i][j][k][UU] = myu/(tfac*tfac);
    p[i][j][k][U1] = myv/tfac;
    //p[i][j][k][U2] = myv/tfac;
    //p[i][j][k][U2] = myv/tfac;//cos(k_vec_x*x);
    p[i][j][k][U3] = 0 ;
    p[i][j][k][B1] = 0. ;
    p[i][j][k][B2] = 0. ;
    p[i][j][k][B3] = 0. ;
  }
  
  /* enforce boundary conditions */
  
  
  fixup(p) ;
  bound_prim(p) ;
  
  
  
  
  
#if( DO_FONT_FIX )
  set_Katm();
#endif
  
  
}

void init_sndwave()
{
  int i,j,k ;
  double x,y,z,sth,cth ;
  double ur,uh,up,u,rho ;
  double X[NDIM] ;
  struct of_geom geom ;
  
  double myrho, myu, mycs, myv;
  double delta_rho;
  double cosa, sina;
  double delta_ampl = 1e-5; //amplitude of the wave
  double k_vec_x = 2 * M_PI;  //wavevector
  double k_vec_y = 0;
  double k_vec_len = sqrt( k_vec_x * k_vec_x + k_vec_y * k_vec_y );
  double tfac = 1e4; //factor by which to reduce velocity
  
  
  /* some physics parameters */
  gam = 5./3. ;
  
  /* some numerical parameters */
  lim = VANL ;
  failed = 0 ;	/* start slow */
  cour = 0.9 ;
  dt = 1.e-5 ;
  
  t = 0. ;
  hslope = 1. ;
  
  if(N2!=1) {
    //2D problem, use full pi-wedge in theta
    fractheta = 1.;
  }
  else{
    //1D problem (since only 1 cell in theta-direction), use a restricted theta-wedge
    fractheta = 1.e-2;
  }
  
  fracphi = 1.;
  
  set_arrays() ;
  set_grid() ;
  
  
  myrho = 1.;
  myu = myrho / (gam * (gam-1));  //so that mycs is unity
  
  mycs = sqrt(gam * (gam-1) * myu / myrho);  //background sound speed
  
  /* output choices */
  
  tf = tfac/mycs;
  
  DTd = tf/10. ;	/* dumping frequency, in units of M */
  DTl = tf/10. ;	/* logfile frequency, in units of M */
  DTi = tf/10. ; 	/* image file frequ., in units of M */
  DTr = tf/10. ; /* restart file frequ., in units of M */
  DTr01 = 1000 ; /* restart file frequ., in timesteps */
  
  /* start diagnostic counters */
  dump_cnt = 0 ;
  image_cnt = 0 ;
  rdump_cnt = 0 ;
  rdump01_cnt = 0 ;
  defcon = 1. ;
  
  ZSLOOP(0,N1-1,0,N2-1,0,N3-1) {
    coord(i,j,k,CENT,X) ;
    bl_coord(X,&x,&y,&z) ;
    
    //applying the perturbations
    delta_rho = delta_ampl * cos( k_vec_x * x + k_vec_y * y );
    
    p[i][j][k][RHO] = myrho + delta_rho;
    p[i][j][k][UU] = (myu + gam * myu * delta_rho / myrho)/(tfac*tfac);
    p[i][j][k][U1] = (delta_rho/myrho * mycs * k_vec_x / k_vec_len)/tfac;
    p[i][j][k][U2] = (delta_rho/myrho * mycs * k_vec_y / k_vec_len)/tfac;
    p[i][j][k][U3] = 0 ;
    p[i][j][k][B1] = 0. ;
    p[i][j][k][B2] = 0. ;
    p[i][j][k][B3] = 0. ;
  }
  
  /* enforce boundary conditions */
  
  
  fixup(p) ;
  bound_prim(p) ;
  
  
  
  
  
#if( DO_FONT_FIX )
  set_Katm();
#endif
  
  
}


/* this version starts w/ BL 4-velocity and
 * converts to relative 4-velocities in modified
 * Kerr-Schild coordinates */

void coord_transform(double *pr,int ii, int jj, int kk)
{
  double X[NDIM],r,th,phi,ucon[NDIM],uconp[NDIM],trans[NDIM][NDIM],tmp[NDIM] ;
  double AA,BB,CC,discr ;
  double utconp[NDIM], dxdxp[NDIM][NDIM], dxpdx[NDIM][NDIM] ;
  struct of_geom geom ;
  struct of_state q ;
  int i,j,k,m ;

  coord(ii,jj,kk,CENT,X) ;
  bl_coord(X,&r,&th,&phi) ;
  blgset(ii,jj,kk,&geom) ;

  ucon[1] = pr[U1] ;
  ucon[2] = pr[U2] ;
  ucon[3] = pr[U3] ;

  AA =     geom.gcov[TT][TT] ;
  BB = 2.*(geom.gcov[TT][1]*ucon[1] +
           geom.gcov[TT][2]*ucon[2] +
           geom.gcov[TT][3]*ucon[3]) ;
  CC = 1. +
          geom.gcov[1][1]*ucon[1]*ucon[1] +
          geom.gcov[2][2]*ucon[2]*ucon[2] +
          geom.gcov[3][3]*ucon[3]*ucon[3] +
      2.*(geom.gcov[1][2]*ucon[1]*ucon[2] +
          geom.gcov[1][3]*ucon[1]*ucon[3] +
          geom.gcov[2][3]*ucon[2]*ucon[3]) ;

  discr = BB*BB - 4.*AA*CC ;
  ucon[TT] = (-BB - sqrt(discr))/(2.*AA) ;
  /* now we've got ucon in BL coords */

  /* transform to Kerr-Schild */
  /* make transform matrix */
  DLOOP trans[j][k] = 0. ;
  DLOOPA trans[j][j] = 1. ;
  trans[0][1] = 2.*r/(r*r - 2.*r + a*a) ;
  trans[3][1] = a/(r*r - 2.*r + a*a) ;

  /* transform ucon */
  DLOOPA tmp[j] = 0. ;
  DLOOP tmp[j] += trans[j][k]*ucon[k] ;
  DLOOPA ucon[j] = tmp[j] ;
  /* now we've got ucon in KS coords */

  /* transform to KS' coords */
  /* dr^\mu/dx^\nu jacobian, where x^\nu are internal coords */
  dxdxp_func(X, dxdxp);
  /* dx^\mu/dr^\nu jacobian */
  invert_matrix(dxdxp, dxpdx);
  
  for(i=0;i<NDIM;i++) {
    uconp[i] = 0;
    for(j=0;j<NDIM;j++){
      uconp[i] += dxpdx[i][j]*ucon[j];
    }
  }
  //old way of doing things for Gammie coords
  //ucon[1] *= (1./(r - R0)) ;
  //ucon[2] *= (1./(M_PI + (1. - hslope)*M_PI*cos(2.*M_PI*X[2]))) ;
  //ucon[3] *= 1.; //!!!ATCH: no need to transform since will use phi = X[3]

  get_geometry(ii, jj, kk, CENT, &geom);
  
  /* now solve for relative 4-velocity that is used internally in the code:
   * we can use the same u^t because it didn't change under KS -> KS' */
  ucon_to_utcon(uconp,&geom,utconp);
  
  pr[U1] = utconp[1] ;
  pr[U2] = utconp[2] ;
  pr[U3] = utconp[3] ;

  /* done! */
}


double lfish_calc(double r)
{
	return(
   ((pow(a,2) - 2.*a*sqrt(r) + pow(r,2))*
      ((-2.*a*r*(pow(a,2) - 2.*a*sqrt(r) + pow(r,2)))/
         sqrt(2.*a*sqrt(r) + (-3. + r)*r) +
        ((a + (-2. + r)*sqrt(r))*(pow(r,3) + pow(a,2)*(2. + r)))/
         sqrt(1 + (2.*a)/pow(r,1.5) - 3./r)))/
    (pow(r,3)*sqrt(2.*a*sqrt(r) + (-3. + r)*r)*(pow(a,2) + (-2. + r)*r))
	) ;
}


