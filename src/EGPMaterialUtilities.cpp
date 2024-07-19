
//-----------------------------------------------------------------------
// this file contains some of the functions needed for EGPMaterial update
//-----------------------------------------------------------------------


#include <jem/base/Array.h>
#include <jem/numeric/utilities.h>
#include <jem/base/String.h>
#include <jem/base/Error.h>
#include <jem/numeric/algebra/LUSolver.h>
#include <jem/numeric/algebra/matmul.h>
#include <jem/numeric/algebra/MatmulChain.h>
#include <jem/numeric/algebra/EigenUtils.h>
#include <jem/base/System.h>

#include "utilities.h"
#include "mcmlib.h"
#include "EGPMaterial.h"

using jem::String;
using jem::ALL;
using jem::Error;
using jem::numeric::inverse;
using jem::numeric::matmul;
using jem::numeric::MatmulChain;
using jem::slice;
using jem::System;
using jem::io::endl;

using namespace mcmlib;
using namespace jem;


typedef MatmulChain<double,3>   MChain3;

typedef Tuple<double,9,1> M91;



void EGPMaterial::deviatoricStress_ 
    
    (       M33&        mSs,
      const M33         DF, 
            M33&        mtB, 
            Array<M33>& mtBe,
            Array<M33>& mtBeB, 
            Vector&     lambda, 
            Vector&     h, 
            Vector&     teq,     
            Vector&     mteq, 
            Vector&     S, 
            Vector&     gamma, 
            double&     eqps, 
            double      JB, 
            double&     eqpsB )

{
  // define additional variables

  idx_t noe = nom_ + 1; // number of equations

  idx_t nsic; // number of subincrements

  double Jn, eigsum, dl1, dl2, dl3, dJn, p, eqpsBi; 

  Tuple<double,3> eigval, leig, lsq; 

  M33 mtFn, mtCn, evecs, mtUn;
  M33 mtUni, mRn, mbSs;
  
  Array<M33> mCpnB(nom_), mCpn(nom_), mbtBen(nom_);

  // initialize some variables

  eqpsBi = eqpsB; // eqpsBi updated in each i: iteration

  // mtBe = mtBeB.clone(); 


  for (idx_t i = 0; i < nom_; i++)
  {
    mCpnB [i] = inverse (mtBeB [i]);
  }

  // calculate incremental deformation variables

  Jn = jem::numeric::det( DF );  

//  System::out() << "Jn df " << Jn << "\n";

  mtFn = DF / pow( Jn, 1.0 / 3.0 );
  
  mtCn = matmul ( mtFn.transpose(), mtFn );

 // System::out() << "mtCn " << jem::numeric::det ( mtCn ) << "\n"; 

  // calculate eigenvalues and eigenvectors of mtCn
  // this will provide lambda^2 (not plasticity parameters)

  jem::numeric::EigenUtils::symSolve (eigval, evecs, mtCn);

  eigsum  = eigval[0] + eigval[1] + eigval[2];

  leig[0] = sqrt (eigval[0]); // vector of eigenvalues, lambda
  leig[1] = sqrt (eigval[1]); // do not confuse with plsticity parameters
  leig[2] = sqrt (eigval[2]);


  // in elements with large incremental deformations,
  // sub-incremental stress calculation is used
  // Global time increment is divided into smaller time increments
  // Deformation tensors are also divided such that all sub-incr.
  // deformation tensors yield the total incremental deformation

  // number of subincrements

  if ( eigsum - 3. > 1.e-4 )
  {
    int par = 10 + (eigsum-3.) / (1.e-3 - 1.e-4) * 40;

    nsic = jem::min(par,50);
  }
  else
  {
    nsic = 1;
  }  

  //System::out() << "dt_ " << dt_ << "\n"; //"nsci " << nsic << " DF " << DF << endl;               

  double dt = dt_ / nsic;

  // incremental eigenvalues

  dl1 = pow( leig[0],  1.0 / nsic );
  dl2 = pow( leig[1],  1.0 / nsic );
  dl3 = pow( leig[2],  1.0 / nsic );

  dJn = pow ( Jn, 1.0 / nsic);

  // subincremental stress integration

  for (idx_t sic = 1; sic <= nsic; sic++)
  {
    leig[0] = pow (dl1, sic);
    leig[1] = pow (dl2, sic);
    leig[2] = pow (dl3, sic);

    double J = pow(dJn, sic) * JB;

    p = -1.0 * k_ * (J - 1.0);

    // eigen lambda^2 incremental

    lsq[0] = leig[0] * leig[0];
    lsq[1] = leig[1] * leig[1];
    lsq[2] = leig[2] * leig[2];

    for (idx_t k = 0; k < 3; k++)
    {
      for (idx_t l = 0; l < 3; l++)
      {
        mtUn(k,l) = 0.0;
        mtCn(k,l) = 0.0;

        for (idx_t m = 0; m < 3; m++)
        {
          mtUn(k,l) += leig[m] * evecs(k,m) * evecs(l,m);
          mtCn(k,l) += lsq[m] * evecs(k,m) * evecs(l,m);
        }
      }
    }

    // calculate driving stress and updated state variables

    newtonrap_ (dt, eqps, eqpsBi, p,
                  
                    mCpnB, mtCn, mtUn, 
                    mtB, mCpn, mbSs, mbtBen,

                    lambda, h, teq, mteq,  
                    S, gamma, noe, Jn );

    eqpsBi = eqps;

    mCpnB  = mCpn;

  }

  // calculate rotation tensor

  mtUni = inverse (mtUn);

  mRn = matmul (mtFn, mtUni);

/*  System::out() << "mRn " << mRn << "\n";

  System::out() << "mtUn " << mtUn << "\n";
*/
  //rotate elastic finger and Cauchy tensor back to main coordinate system

  for (idx_t i = 0; i < nom_; i++)
  {
    mtBe[i] = matmul(matmul(mRn,mbtBen[i]), mRn.transpose());
  }

  // calculate increment of deviatoric stress

  //System::out() << "mbSs " << mbSs << "\n";

  mSs = matmul(matmul(mRn,mbSs), mRn.transpose()); 

  //System::out() << "mSs " << mSs << "\n";

}

//-----------------------------------------------------------------------
//   Newton-Raphson iteration procedure to solve the system of eqns
//-----------------------------------------------------------------------

void EGPMaterial::newtonrap_ 

    (const double dt, double& eqps, const double& eqpsBi, const double& p,
                  
    const Array<M33>& mCpnB, const M33& mtCn, const M33& mtUn,

    const M33& mtB, Array<M33>& mCpn, M33& mbSs, Array<M33>& mbtBen,

    Vector& lambda, Vector& h, Vector& teq, Vector& mteq,

    Vector& S, Vector& gamma, const idx_t noe, const double Jn )

{
  // nit: number of iteration
  // maxnit: max number of iterations
  // convpar: convergence parameter

  idx_t nit, maxnit, convpar;

  double epsLambda, epsEqps, deqpsdt, dSdea, dSdeb;

  double trmtB, secinvmtB, htB, nrmEqps, nrmLambda;

  double norm1, norm2;

  Vector Rg(mode_);

  Vector hB(nom_);

  // res: residue

  Vector res(noe), res1(noe), sol(noe); 

  M33 mI, mCpni, mbtBend, mbsdi; 
  M33 mbsda, mbsdb, mbsdsq;

  Matrix mat(noe, noe), mati (noe, noe);

  // iteration parameters

  nit         = 1;
  maxnit      = 16;
  epsLambda   = 1.0e-4;
  epsEqps     = 1.0e-4;
  convpar     = 1;

  inimI( mI );

  double t0a, t0b;

  if ( strHardening_ == "NEOHOOKEAN" )
  {
    t0a = t0a_;
    t0b = t0b_;
  }
  else
  {
    t0a = kb_ * T_ / Vacta_;
    t0b = kb_ * T_ / Vactb_;
  }

  norm2 = 0.; norm1 = 1.;

  sol = 0.0;

  hB = h.clone();

  if ( Jn != 1.0 )
  {
    updateEqStress_ ( mCpnB, mCpn, mtCn, mtUn, mbtBen, lambda, teq, mteq, mbsda, mbsdb);

    // calculate softening function for both processes

    calcSofteningFn_ ( eqps, Rg);

    // calculate state parameter for both processes

    calcAgingPar_ ( Rg, S );

    // viscous hardening: deformation dependence of viscosity
    
    trace (mtB, trmtB);

    secInv( mtB, secinvmtB);

    htB = sqrt(1.0 / 3.0 * trmtB * trmtB - secinvmtB);

    updateViscosity_ (h, Rg, htB, S, p, t0a, t0b, teq);

    if ( h[0] / hB[0] < 0.0001 ) { h = hB * 0.0001;} // limit viscosity drop

    gamma   = G_ / h;

    deqpsdt = mteq[0] / h[0];

    res[slice(0,nom_)] = 1.0 - lambda * (dt * gamma + 1.0);

    res[noe-1]         = eqpsBi + deqpsdt * dt - eqps;

    // derivative of state parameter wrt equivalent plastic strain

    calcDSDeqps_ ( dSdea, dSdeb, eqps);

    // calculate Jacobi matrix and its inverse 

   /* System::out() << "gamma " << gamma << " lambda " << lambda << "\n";
    System::out() << "mteq " << mteq << " teq " << teq << "\n";
    System::out() << "noe " << noe << " deqpsdt " << deqpsdt << "\n";
    System::out() << "dt " << dt << " dSdea " << dSdea << "\n";
    System::out() << "dSdeb " << dSdeb << " t0a " << t0a << " t0b " << t0a << "\n";
*/
    jacobimat_ (mat, gamma, lambda, mteq, teq, noe,

             deqpsdt, dt, dSdea, dSdeb, t0a, t0b);

 //   System::out() << "mat:\n" << mat << "\n";;

    mati = jem::numeric::inverse (mat);

    // calculate and adjust solution

    sol    = matmul(mati, res);

    norm1  = sqrt ( dot (res, res) );

    // update system variables

    lambda += sol[slice(0,nom_)];

    eqps   += sol[noe-1];

    // if ( eqps > .3 ) { eqps = .3; }

    for (idx_t i = 0; i < nom_; i++)
    {
      if (lambda[i] > 1.0)
      {
        lambda[i] = 1.0;  
      }
      if (lambda[i] < 0.0)
      {
        lambda[i] = 0.0;    
      } 
    }
  }

  
  while (convpar == 1 && nit <= maxnit)
  {
    if ( Jn != 1.0 )
    {
      updateEqStress_ ( mCpnB, mCpn, mtCn, mtUn, mbtBen, lambda, teq, mteq, mbsda, mbsdb);

      // calculate softening function for both processes

      calcSofteningFn_ ( eqps, Rg);

      // calculate state parameter for both processes

      calcAgingPar_ ( Rg, S );

      // viscous hardening: deformation dependence of viscosity
      
      trace (mtB, trmtB);

      secInv( mtB, secinvmtB);

      htB = sqrt(1.0 / 3.0 * trmtB * trmtB - secinvmtB);

      updateViscosity_ (h, Rg, htB, S, p, t0a, t0b, teq);

      if ( h[0] / hB[0] < 0.0001 ) { h = hB * 0.0001;}

      gamma   = G_ / h;

      deqpsdt = mteq[0] / h[0];

      res[slice(0,nom_)] = 1.0 - lambda * (dt * gamma + 1.0);

      res[noe-1]         = eqpsBi + deqpsdt * dt - eqps;

      // derivative of state parameter wrt equivalent plastic strain

      calcDSDeqps_ ( dSdea, dSdeb, eqps);

      // calculate Jacobi matrix and its inverse 

      jacobimat_ (mat, gamma, lambda, mteq, teq, noe,

               deqpsdt, dt, dSdea, dSdeb, t0a, t0b);

          
      mati = jem::numeric::inverse (mat);


      // calculate and adjust solution

      sol    = matmul(mati, res);

      // update system variables

      lambda += sol[slice(0,nom_)];

      eqps   += sol[noe-1];

      // if ( eqps > .3 ) { eqps = .3; }

      for (idx_t i = 0; i < nom_; i++)
      {
        if (lambda[i] > 1.0)
        {
          lambda[i] = 1.0;  
        }
        if (lambda[i] < 0.0)
        {
          lambda[i] = 0.0;    
        } 
      } 

      res[slice(0,nom_)] = 1.0 - lambda * (dt * gamma + 1.0);

      res[noe-1]         = eqpsBi + deqpsdt * dt - eqps;

      norm2 = sqrt ( dot ( res, res ) );

    } 

    // convergence criterium

    // L2 norm criteria

    convpar = 0;

    if ( norm2 > 1.e-4 * norm1 )
    {
      convpar = 1;
    }

    // absolute, term-by-term criteria

    // nrmEqps = fabs(sol[noe-1]);

    // if (nrmEqps > epsEqps)
    // {
    //   convpar = 1;  
    // }

    // if (convpar == 0)
    // {
    //   idx_t i = 1;
      
    //   while (i <= nom_ && convpar == 0)
    //   {
    //     nrmLambda = fabs(sol[i-1]);

    //     if (nrmLambda > epsLambda)
    //     {
    //       convpar = 1;
    //     }

    //     i += 1;
    //   } 
    // }
    
    nit += 1;

  }

  // if ( convpar == 1 )
  // {
  //   throw Error ( JEM_FUNC, "No convergence on the EGP constitutive level ..." );
  // }

  // update the state

  updateEqStress_ ( mCpnB, mCpn, mtCn, mtUn, mbtBen, lambda, teq, mteq, mbsda, mbsdb);

  // calculate softening function for both processes

  calcSofteningFn_ ( eqps, Rg);

  // calculate state parameter for both processes

  calcAgingPar_ ( Rg, S );

  // viscous hardening: deformation dependence of viscosity
  
  trace (mtB, trmtB);

  secInv( mtB, secinvmtB);

  htB = sqrt(1.0 / 3.0 * trmtB * trmtB - secinvmtB);

  updateViscosity_ (h, Rg, htB, S, p, t0a, t0b, teq);

  if ( h[0] / hB[0] < 0.0001 ) { h = hB * 0.0001;}

  // calculate total stress

  mbSs = mbsda + mbsdb;

}

//-----------------------------------------------------------------------
//   update equivalent stress: total and for each mode
//-----------------------------------------------------------------------

void       EGPMaterial::updateEqStress_

    (const Array<M33>& mCpnB, const Array<M33>& mCpn, const M33& mtCn, 

      const M33& mtUn, const Array<M33>& mbtBen, const Vector& lambda, 

      const Vector& teq, const Vector& mteq, M33& mbsda, M33& mbsdb)
    
{
  M33 mI, mCpni, mbtBend, mbsdi; 

  M33 mbsdsq;

  double trmbtBen, trmbsdsq;

  inimI( mI );

  mbsda = 0.0;
  mbsdb = 0.0;

  for (idx_t i = 0; i < nom_; i++)
  {
    // calculate kinematic variables

    mCpn[i] = (1.0 - lambda[i]) * mtCn +
                                lambda[i] * mCpnB[i];

    mCpni = inverse(mCpn[i]);

    mbtBen[i] = matmul(matmul(mtUn, mCpni), mtUn.transpose());

    trace (mbtBen[i], trmbtBen);

    mbtBend = mbtBen[i] - 1.0 / 3.0 * trmbtBen * mI;


    // calculate rotation neutralized stress 
    // and determine coressponding process (alpha or beta)

    mbsdi = G_[i] * mbtBend;

    if (i <= nam_ - 1)
    {
      mbsda += mbsdi;
    }
    else
    {
      mbsdb += mbsdi;
    }


    // calculate modal equivalent stress

    mbsdsq = matmul(mbsdi, mbsdi.transpose());

    trace (mbsdsq, trmbsdsq);

    mteq[i] = sqrt(1.0 / 2.0 * trmbsdsq);
      
  }

  // calculate total equivalent stress for both processes

  mbsdsq = matmul(mbsda, mbsda.transpose());

  trace (mbsdsq, trmbsdsq);

  teq[0] = sqrt(1.0 / 2.0 * trmbsdsq);

  if (mode_ == 2)
  {
    mbsdsq = matmul(mbsdb, mbsdb.transpose());
    
    trace (mbsdsq, trmbsdsq);

    teq[1] = sqrt(1.0 / 2.0 * trmbsdsq);  
  }
}

//-----------------------------------------------------------------------
//   calculate softening function
//-----------------------------------------------------------------------

void       EGPMaterial::calcSofteningFn_

                (const double& eqps, const Vector& Rg)

{
  // calculate softening function for both processes

  Rg[0] = pow(1. + pow(r0a_ * exp(eqps), r1a_), (r2a_ - 1.) / r1a_) / 

          pow(1. + pow(r0a_, r1a_), (r2a_ - 1.) / r1a_ );

  if (mode_ == 2)
  {
    Rg[1] = pow(1. + pow(r0b_ * exp(eqps), r1b_), (r2b_ - 1.) / r1b_) / 

          pow(1. + pow(r0b_, r1b_), (r2b_ - 1.) / r1b_);  
  }
}

//-----------------------------------------------------------------------
//   derivative of the state parameter wrt eqps
//-----------------------------------------------------------------------

void            EGPMaterial::calcDSDeqps_

         (double& dSdea, double& dSdeb, const double& eqps)

{
  dSdea = SSa_ * (r2a_ - 1.0) * pow(r0a_ * exp(eqps), r1a_) *

          pow(1.0 + pow(r0a_ * exp(eqps), r1a_), (r2a_ - 1.0 -r1a_) / r1a_);

  if (mode_ == 2)
  {
    dSdeb = SSb_ * (r2b_ - 1.0) * pow(r0b_ * exp(eqps), r1b_) * 

          pow(1.0 + pow(r0b_ * exp(eqps), r1b_), (r2b_ - 1.0 -r1b_) / r1b_);  
  }
}

//-----------------------------------------------------------------------
//   calculate aging (state) parameter
//-----------------------------------------------------------------------

void       EGPMaterial::calcAgingPar_

                  (const Vector& Rg, const Vector& S)
                  
{
  S[0] = SSa_ * Rg[0];

  if (mode_ == 2)
  {
    S[1] = SSb_ * Rg[1];  
  }
}

//-----------------------------------------------------------------------
//   update viscosity
//-----------------------------------------------------------------------

void            EGPMaterial::updateViscosity_

    (const Vector& h, const Vector& Rg, const double& htB, const Vector& S,

      const double& p, const double& t0a, const double& t0b, const Vector& teq)

{
  if ( strHardening_ == "NEOHOOKEAN" )
  {
    if (teq[0] > 1.0e-10)
    {
      for (idx_t i = 0; i < nam_; i++)
      {
        h[i] = h0_[i] * exp(S[0] + (ma_ * p) / t0a) *

              (teq[0] / t0a) / sinh(teq[0] / t0a);  

        if (h[i] < 1.0e-10) { h[i] = 1.0e-10; }
      } 
    }

    if (mode_ == 2)
    {
      if (teq[1] > 1.0e-10)
      {
        for (idx_t i = nam_; i < nom_; i++)
        {
          h[i] = h0_[i] * exp(S[1] + (mb_ * p) / t0b) *

              (teq[1] / t0b) / sinh(teq[1] / t0b);

          if (h[i] < 1.0e-10) { h[i] = 1.0e-10; }          
        }  
      } 
    }
  }
  else
  {
    if (teq[0] > 1.0e-10)
    {
      for (idx_t i = 0; i < nam_; i++)
      {
        h[i] = h0_[i] * exp((dH0a_ + SHa_ * Rg[0] + Cr2a_ * htB * htB) / R_ / T_) *

               exp(S[0]) * exp(ma_ * p / t0a) * 

               (teq[0] / t0a / sinh(teq[0] / t0a)) *

               exp(-Cr1a_ * htB * htB); 

        if (h[i] < 1.0e-10)
        {
          h[i] = 1.0e-10; 
        }
      } 
    }

    if (mode_ == 2)
    {
      if (teq[1] > 1.0e-10)
      {
        for (idx_t i = nam_; i < nom_; i++)
        {
          h[i] = h0_[i] * exp((dH0b_ + SHb_ * Rg[1] + Cr2b_ * htB) / R_ / T_) *

               exp(S[1]) * exp(mb_ * p / t0b) * 

               (teq[1] / t0b / sinh(teq[1] / t0b)) *

               exp(-Cr1b_ * htB); 

          if (h[i] < 1.0e-10)
          {
            h[i] = 1.0e-10; 
          }          
        }  
      } 
    }
  }
}

//-----------------------------------------------------------------------
//   convergence criteria
//-----------------------------------------------------------------------

void         EGPMaterial::checkConvergence_

             (idx_t& convpar, const double& Reta, const double& R0)

{
  convpar = 0;

  if ( fabs ( Reta / R0 ) > 0.5  ) { convpar = 1; }
}

//-----------------------------------------------------------------------
//   construct Jacobi matrix of the system of equations
//-----------------------------------------------------------------------

void EGPMaterial::jacobimat_ (Matrix& mat, 

          const Vector& gamma, const Vector& lambda, const Vector& mteq,

          const Vector& teq, const idx_t& noe, 

          const double& deqpsdt, const double& dt,
           
          const double& dSdea, const double& dSdeb,

          const double& t0a, const double& t0b)
{
  
  double vala, valb, tmp;

  vala = 1.0 / teq[0] - 1.0 / t0a;

  if (mode_ == 2)
    {
      valb = 1.0 / teq[1] - 1.0 / t0b;  
    }

  mat = 0.0;

  //   calculate df_i/dlambda_i, where i: mode number

  for (idx_t i = 0; i < nom_; i++)
  {
    if (i <= nam_ - 1)
    {
      if (teq[0] <= 1.0e-10)
      {
        mat(i,i) = 1.0 + dt * gamma[i];
      } 
      else
      {
        mat(i,i) = 1.0 + dt * gamma[i] * (1.0 - lambda[i] * vala * mteq[i]);
      }
    }
    else if (i > nam_ - 1)
    {
      if (teq[1] <= 1e-10)
      {
        mat(i,i) = 1.0 + dt * gamma[i];
      }
      else
      {
        mat(i,i) = 1.0 + dt * gamma[i] * (1.0 - lambda[i] * valb * mteq[i]);
      } 
    }
  }  
  
  //   calculate df_i/dlambda_j, where i,j: mode numbers

  for (idx_t i = 0; i < nom_; i++)
  {
    for (idx_t j = 0; j < nom_; j++)
    {
      if (i == j)
      {
        tmp = 1.0;
      } 
      else
      {
        if (i <= nam_-1 && j > nam_-1)
        {
          mat(i,j) = 0.0; 
        }
        else if (i > nam_-1 && j <= nam_-1)
        {
          mat(i,j) = 0.0; 
        }
        else if (i <= nam_-1)
        {
          if (teq[0] <= 1.0e-10)
          {
            mat(i,j) = 0.0;
          } 
          else
          {
            mat(i,j) = - lambda[i] * dt * gamma[i] * vala * mteq[j];
          }
        }
        else if (i > nam_-1)
        {
          if (teq[1] <= 1.0e-10)
          {
            mat(i,j) = 0.0;
          }
          else
          {
            mat(i,j) = -lambda[i] * dt * gamma[i] * valb * mteq[j];
          } 
        }
      }
    }
  }

 
  //   calculate df_i/deqps, where i: mode number

  for (idx_t i = 0; i < nom_; i++)
  {
    if (i <= nam_-1)
    {
      mat(i,noe-1) = -lambda[i] * dt * gamma[i] * dSdea;  
    }
    else
    {
      mat(i,noe-1) = -lambda[i] * dt * gamma[i] * dSdeb; 
    }
  }


  //   calculate dg/dlambda_i, where i: mode number

  for (idx_t i = 0; i < nom_; i++)
  {
    if (teq[0] <= 1.0e-10)
    {
      if (i == 0)
      {
        mat(noe-1,0) = -dt * deqpsdt;
      }
      else
      {
        mat(noe-1,i) = 0.0;
      } 
    }
    else
    {
      if (i == 0)
      {
        mat(noe-1,0) = -dt * deqpsdt * (1.0 - mteq[0] * vala);
      }
      else
      {
        if (i <= nam_-1)
        {
          mat(noe-1,i) = dt * deqpsdt * mteq[i] * vala; 
        }
        else if (i >= nam_-1)
        {
          mat(noe-1,i) = 0.0; 
        }
      } 
    }
  }

  //   calculate dg/deqps

  mat(noe-1,noe-1) = 1.0 + dt * deqpsdt * dSdea;

}

//-----------------------------------------------------------------------
//   calculate Edwards-Vilgis hardening stress
//-----------------------------------------------------------------------

void EGPMaterial::edvilStress_ 
    
    (M33& mSr, const M33& mI, double J, const M33& mtB)

{
  // define some variables

  double trmtB, trmQ6, trmtemp;

  M33 mQ, mQ1, mQ6, mZ, mtemp;

  mQ = mI + evh_ * mtB;

  mQ1 = inverse ( mQ );

  mQ6 = matmul ( mtB, mQ1 );

  trace ( mtB, trmtB ); // calculate trmtB
  trace ( mQ6, trmQ6 ); // calculate trmQ6

  // calculate Edwards-Vilgis tensor

  mZ = ( pow(eva_,2.) * (1.0 + evh_) * (1.0 - pow(eva_,2.)) ) /

            pow( 1.0 - pow(eva_,2.) * trmtB ,2) * trmQ6 * mI +

            ( (1.0 + evh_) * (1.0 - pow(eva_,2.)) ) /

            ( 1.0 - pow(eva_,2.) * trmtB ) * mQ1 -

            evh_ * ( (1.0 + evh_) * (1.0 - pow(eva_,2.)) ) /

            ( 1.0 - pow(eva_,2.) * trmtB ) *

            matmul( matmul( mQ1,mQ1 ), mtB ) + evh_ * mQ1 -

            pow(eva_,2.) / ( 1.0 - pow(eva_,2.) * trmtB ) * mI;

            

  // mtemp = evNskT_ / J * matmul( mtB,mZ );

  mtemp = matmul( mtB,mZ );

  trace ( mtemp, trmtemp );

  // calculate strain hardening stress

  mSr = evNskT_ / J * (mtemp - 1.0 / 3.0 * trmtemp * mI);

  // for (idx_t i = 0; i < 3; i++)
  // {
  //   for (idx_t j = 0; j < 3; j++)
  //   {
  //     if ( abs (mSr(i,j)) < 1e-25)
  //     {
  //       mSr(i,j) = 0.0;
  //     }
  //   }
  // }

}

//-----------------------------------------------------------------------
//   calculate Edwards-Vilgis stiffness contribution
//-----------------------------------------------------------------------

void       EGPMaterial::edvilStiff_ 

    (M99& mmStiffrc, M33& mtB, M33& mB, const M33 mF, double J)
{
  
  // define some variables

  double trmtB, trmQ6, trBZ;

  double prf1, prf3, prf4, prf5, prf6, prf7;

  
  Tuple <double,1,1> prf2;


  M33 mI, mQ, mQ1, mQ5, mQ6, mZ, mFi;

  M33 mBZ, mdsrdJ, mdJdF;


  M99 mmI, mmQ1, mmQ1c, mmQ1cr, mmQ2, mmQ3;

  M99 mmtB, mmtBc, mmtBcr, mmQ4, mmdZdtB;

  M99 mmZ, mmZc, mmZcr, mmZQ, mmZQr;

  M99 mmdsrdtB, mmF, mmFc, mmFcr;

  M99 mmdtBdFc, mmdsrdtBc, mmtemp;


  M91 ccI, ccB, cctB, ccQ1, ccQ1t, ccQ5;

  M91 ccFit, ccdsrdJ, ccdJdF;

  // define identity tensors

  inimI (mI);

  m2cc (ccI, mI, 9);

  inimmI4 (mmI);

  // define some other variables

  trace (mtB, trmtB);

  m2cc (ccB, mB, 9);

  mQ = mI + evh_ * mtB;

  mQ1 = inverse (mQ);

  m2mm (mmQ1, mQ1, 9);

  mm2mmc (mmQ1c, mmQ1, 9);

  mm2mmr(mmQ1cr, mmQ1c, 9);

  m2cc(cctB, mtB, 9);

  m2cc(ccQ1, mQ1, 9);

  cc2cct(ccQ1t, ccQ1, 9);

  // prf: product factors 

  prf1 = 2. * pow (eva_,4.) * (1. + evh_) * (1. - pow(eva_,2.)) / 

        pow (1. - pow(eva_,2.) * trmtB, 3.);

  prf2 = matmul (cctB.transpose(), ccQ1t);

  prf3 = 2. * pow (eva_,2.) * (1. + evh_) * (1. - pow(eva_,2.)) / 

        pow(1. - pow(eva_,2.) * trmtB,2.);

  prf4 = 2. * pow(eva_,2.) * evh_ * (1. + evh_) * (1. - pow(eva_,2.)) /

        pow(1. - pow(eva_,2.) * trmtB,2.);

  prf5 = 2. * evh_ * (1. + evh_) * (1. - pow(eva_,2.)) / 

        (1. - pow(eva_,2.) * trmtB);

  prf6 = 2. * pow(evh_,2.) * (1. + evh_) * (1. - pow(eva_,2.)) / 
  
        (1. - pow(eva_,2.) * trmtB);

  prf7 = pow(eva_,4.) / pow(1. - pow(eva_,2.) * trmtB,2.);


  mmQ2 = matmul (mmI, mmQ1cr);

  mmQ3 = matmul (mmQ1, mmQ2);

  m2mm (mmtB, mtB, 9);

  mm2mmc (mmtBc, mmtB, 9);

  mm2mmr (mmtBcr, mmtBc, 9);

  mmQ4 = matmul (matmul(mmQ1,mmQ3), mmtBcr);

  mQ5 = matmul (mtB, matmul(mQ1,mQ1));

  m2cc (ccQ5, mQ5, 9);

  mmdZdtB = prf1 * prf2(0,0) * matmul (ccI, ccI.transpose())

              + prf3 * matmul (ccI, ccQ1.transpose())

              - prf4 * matmul (ccI, ccQ5.transpose())

              - prf5 * mmQ3 + prf6 * mmQ4

              - pow(evh_,2.) * mmQ3

              - prf7 * matmul(ccI, ccI.transpose());

  mQ6 = matmul (mtB,mQ1);

  trace (mQ6, trmQ6);


  mZ = ( pow(eva_,2.) * (1.0 + evh_) * (1.0 - pow(eva_,2.)) ) /

            pow( 1.0 - pow(eva_,2.) * trmtB ,2.) * trmQ6 * mI +

            ( (1.0 + evh_) * (1.0 - pow(eva_,2.)) ) /

            ( 1.0 - pow(eva_,2.) * trmtB ) * mQ1 -

            evh_ * ( (1.0 + evh_) * (1.0 - pow(eva_,2.)) ) /

            ( 1.0 - pow(eva_,2.) * trmtB ) *

            matmul( matmul( mQ1,mQ1 ), mtB ) + evh_ * mQ1 -
            
            pow(eva_,2.) / ( 1.0 - pow(eva_,2.) * trmtB ) * mI;


  m2mm (mmZ, mZ, 9);
  
  mm2mmc (mmZc, mmZ, 9);
  
  mm2mmr (mmZcr, mmZc, 9);

  mmZQ = matmul (mmI, mmZcr) + matmul (mmtB, mmdZdtB);

  mm2mmr (mmZQr, mmZQ, 9);

  mmtemp = mmI - 1.0 / 3.0 * matmul(ccI, ccI.transpose());

  mmdsrdtB = evNskT_ / J * matmul( mmtemp , mmZQr );
  
  m2mm(mmF,mF,9);

  mm2mmc (mmFc, mmF, 9);
  
  mm2mmr (mmFcr, mmFc, 9);

  mFi = inverse (mF);
  
  m2cc (ccFit, mFi.transpose(), 9);

  // derivative of left Cauchy-Green deformation tensor wrt def.grad.

  mmdtBdFc = pow (J,-2.0 / 3.0) * ( ( mmFc + mmFcr ) -
                   2.0 / 3.0 * matmul(ccB, ccFit.transpose()) );

  mBZ = matmul (mtB, mZ);

  trace (mBZ, trBZ);

  // derivative of hardening stress wrt volume change ratio

  mdsrdJ = -1.0 * evNskT_ / pow(J,2.) * ( mBZ - 1.0 / 3.0 * trBZ * mI );

  // derivative of volume change ratio wrt deformation gradient

  mdJdF = J * mFi.transpose();

  mm2mmc (mmdsrdtBc, mmdsrdtB, 9);
  
  m2cc (ccdsrdJ, mdsrdJ, 9);
  
  m2cc(ccdJdF, mdJdF, 9);

  mmStiffrc = matmul(mmdsrdtBc, mmdtBdFc) +

          matmul(ccdsrdJ, ccdJdF.transpose());
}

//-----------------------------------------------------------------------
//   update the tangent stiffness matrix
//-----------------------------------------------------------------------

void EGPMaterial::stiffness_

    (const bool& nn, const Matrix& StiffFull, const Matrix& Stiff, 
     
     const M33& mF, const M33& mFB, M33& mtB, 

    const M33& mStress, const Array<M33>& mtBe, const Array<M33>& mtBeB,

    const Vector& h, const Vector& mteq, const Vector& lambda,

    const Vector& teq, const Vector& gamma, double J, double JB )

{
  // define some variables

  Tuple <double,9,1> ccI, ccStress, ccF, ccFi;

  Tuple <double,9,1> cctC, cctB, cctF, ccTemp;


  M33   mI, mtF, mtFB, mtC, mFi;

  M33   mtBeBi, mCpB, mCp, mCpi, mtBed, mM;

  M33   mB, mC, evecs, mUi, mU, mR;


  M99   mmI, mmtF, mmtFt, mmtFtr, mmFt;

  M99   mmStiffh, mmStiffd, mmH1c, mmStiff;

  M99   mmStiffc, mmStiffr, mmStress;

  M99   mmK, mmM, mmMc, mmMcr, mmFtr, mmIc;

  M99   mmF4, mmKtot, mmSdtmp, mmSdtmp2;

  M99   mmSdtmp3, mmSda, mmSdb, mmUi, mmRi;

  M99   mmRic, mmE, mmEi, mmStressc, mmStressrc;

  M99   mmtFc, mmtFcr, mmTemp; 


  Array <M91> ccCpB(nom_), cctBed(nom_);


  
  Array <M99>   mmA1(nom_), mmA2(nom_), mmC4c(nom_);

  Array <M99>   mmB1c(nom_), mmB2c(nom_);


  Tuple <double,1,9> dtrmtBdf, dsimtBdf;


  Tuple <double,9,2> ccStressd;


  Tuple <double,3,1> eigval;


  double trmtBe;

  MChain3     mc3;


  // initialize second and fourth order unit tensors
  // convert them to corressponding column vectors

  inimI (mI);
  m2cc (ccI, mI, 9); // different than Voigt representation

  inimmI4 (mmI); 
  mm2mmc (mmIc, mmI, 9); // switch columns of expanded mI


  // stress matrix to corresponding column array

  m2cc (ccStress, mStress, 9);


  // calculate kinetic variables

  // Timer t;

  // t.start();

  mtF = mF / pow (J, 1. / 3.);

  mtFB = mFB / pow (JB, 1. / 3.);

  mtC = matmul (mtF.transpose(), mtF);

  m2cc (cctC, mtC, 9);

  m2cc (cctF, mtF, 9);

  mFi = inverse (mF);

  m2cc (ccF, mF, 9);

  m2cc (ccFi, mFi, 9);

  m2mm (mmtF, mtF, 9);


  mm2mmc (mmtFc, mmtF, 9);

  mm2mmr (mmtFcr, mmtFc, 9);


  m2mm (mmtFt, mtF.transpose(), 9);

  mm2mmr (mmtFtr, mmtFt, 9);

  m2mm (mmFt, mF.transpose(), 9);

  mm2mmr (mmFtr, mmFt, 9);

  // t.stop();

  // System::out() << "\n";

  // System::out() << "kinetic Timme = " << t << '\n';  

  // System::out() << "\n";


  // calculate modal kinetic variables

  // Timer t1;

  // t1.start();

  for (idx_t i = 0; i < nom_; i++)
  {

    mtBeBi = inverse ( mtBeB[i] );

    mCpB = matmul (mtFB.transpose(), matmul ( mtBeBi, mtFB) );

    mCp = (1. - lambda[i]) * mtC + lambda[i] * mCpB;

    mCpi = inverse (mCp);

    ccTemp = 0.0;

    m2cc (ccTemp, mCpB, 9);

    ccCpB[i] = ccTemp; 

    trace (mtBe[i], trmtBe );

    mtBed = mtBe[i] - 1. / 3. * trmtBe * mI;

    ccTemp = 0.0;

    m2cc (ccTemp, mtBed, 9);

    cctBed[i] = ccTemp;

    mM = matmul (mtF, mCpi);

    m2mm (mmM, mM, 9);

    mm2mmc (mmMc, mmM, 9);

    mm2mmr (mmMcr, mmMc, 9);

    mmA1[i] = mmMcr + mmMc;

    mmA2[i] = -1. * matmul (mmM, mmMcr);

  } 

  // t1.stop();

  // System::out() << "\n";

  // System::out() << "for loop Timme = " << t1 << '\n';  

  // System::out() << "\n";

  mmF4 = -1. / 3. / pow (J, 1./3.) * matmul (ccF, ccFi.transpose()) + 

         1. / pow (J, 1./3.) * mmI;


  // variation of stress to deformation gradient

  //-----------------------------------------------------------------------
  //   hardening stiffness
  //-----------------------------------------------------------------------

  // Timer t2;

  // t2.stop();

  mB = matmul (mF, mF.transpose());

  if ( strHardening_ == "NEOHOOKEAN" )
  {
    mmTemp = mmtFcr + mmtFc - 2. / 3. * 

             matmul(matmul(ccI,ccI.transpose()),mmtFc);

    mmStiffr = Gr_ * matmul(mmTemp,mmF4);
  }
  else
  {
    edvilStiff_ (mmStiffr, mtB, mB, mF, J);
  }

  // t2.stop();

  // System::out() << "\n";

  // System::out() << "hardening Timme = " << t2 << '\n';  

  // System::out() << "\n";

  //-----------------------------------------------------------------------
  //   hydrostatic stiffness
  //-----------------------------------------------------------------------

  mmStiffh = k_ * J * matmul (ccI, ccFi.transpose());

  //-----------------------------------------------------------------------
  //   deviatoric stiffness
  //-----------------------------------------------------------------------

  // Timer t3;

  // t3.start();

  mmSdtmp = 0.0;

  for (idx_t i = 0; i < nom_; i++)
  {
    mmC4c[i] = (1.0 - lambda[i]) * (mmtFtr + mmtFt);

    mmB1c[i] = mmA1[i] + matmul (mmA2[i], mmC4c[i]);

    mmB2c[i] = matmul (mmB1c[i], mmF4);

    mmSdtmp  += mmB2c[i] * G_[i];
  }

  // for (idx_t i = 0; i < nom_; i++)
  // {
  //  mmSdtmp = mmSdtmp + mmB2c[i] * G_[i];
  // }


  mmH1c = mmIc - 1.0 / 3.0 * matmul (ccI, ccI.transpose());

  mmStiffd = matmul (mmH1c, mmSdtmp);

  // t3.stop();

  // System::out() << "\n";

  // System::out() << "deviatoric Timme = " << t3 << '\n';  

  // System::out() << "\n";

  // adding the 3 parts together

  mmStiff = mmStiffh + mmStiffd + mmStiffr;

  //System::out() << "mmStiff " << mmStiff << "\n";

  // shift columns in stiffness matrix

  mm2mmc (mmStiffc, mmStiff, 9);

  // mmStiffc += mmStiffr;


  // -------------------------------------------------------------------
  // calculate mat.stiffness matrix as derivative of stress to strain
  // -------------------------------------------------------------------

  // mC = matmul ( mF.transpose(), mF );

  // jem::numeric::EigenUtils::symSolve (eigval, evecs, mC);

  // for (idx_t k = 0; k < 3; k++)
  // {
  //   for (idx_t l = 0; l < 3; l++)
  //   {
  //     mU(k,l) = 0.0;

  //     for (idx_t m = 0; m < 3; m++)
  //     {
  //       mU(k,l) = mU(k,l) + sqrt(eigval[m]) * evecs(k,m) * evecs(l,m);
  //     }
  //   }
  // }

  // mUi = inverse (mU);

  // // calculate rotation tensor

  // mR = matmul (mF, mUi);

  // m2mm (mmUi, mUi, 9);

  // m2mm (mmRi, mR.transpose(), 9);

  // mm2mmc (mmRic, mmRi, 9);

  // mmE = matmul (mmUi, mmRic);

  // mmEi = inverse (mmE);

  // mmKtot = matmul (mmStiffc, mmEi); 

  mmKtot = matmul (mmStiffc, mmFtr);

  // --------------------------------------------------------------------
  // calculate stress stiffness matrix
  // --------------------------------------------------------------------

  // similar part is already included in Updated Lagrangian formulation 

  // m2mm (mmStress, mStress, 9);

  // mm2mmc (mmStressc, mmStress, 9);

  // mm2mmr (mmStressrc, mmStressc, 9);

  // // calculate total stiffness matrix

  // mmKtot = mmK - mmStressrc;

//  System::out() << "mmFtr " << mmFtr << "\n";
 //  System::out() << "mmStiffc " << mmStiffc << "\n";
 //  System::out() << "mmKtot " << mmKtot << "\n";

  if ( stateString_ == "PLANE_STRAIN" )
  {
    Stiff(0,0) = mmKtot(0,0);
    Stiff(1,0) = .5 * ( mmKtot(1,0) + mmKtot(0,1) );
    Stiff(2,0) = .5 * ( mmKtot(3,0) + mmKtot(0,3) );
    Stiff(0,1) = Stiff(1,0);
    Stiff(1,1) = mmKtot(1,1);
    Stiff(2,1) = .5* ( mmKtot(3,1) + mmKtot(1,3) );
    Stiff(0,2) = Stiff(2,0);
    Stiff(1,2) = Stiff(2,1);
    Stiff(2,2) = mmKtot(3,3);
  }

  else if ( stateString_ == "AXISYMMETRIC" )
  {
    Stiff(0,0) = mmKtot(0,0);
    Stiff(1,0) = .5 * ( mmKtot(1,0) + mmKtot(0,1) );
    Stiff(2,0) = .5 * ( mmKtot(3,0) + mmKtot(0,3) );
    Stiff(3,0) = .5 * ( mmKtot(2,0) + mmKtot(0,2) );
    Stiff(0,1) = Stiff(1,0);
    Stiff(1,1) = mmKtot(1,1);
    Stiff(2,1) = .5 * ( mmKtot(3,1) + mmKtot(1,3) );
    Stiff(3,1) = .5 * ( mmKtot(2,1) + mmKtot(1,2) );
    Stiff(0,2) = Stiff(2,0);
    Stiff(1,2) = Stiff(2,1);
    Stiff(2,2) = mmKtot(3,3);
    Stiff(3,2) = .5 * ( mmKtot(3,2) + mmKtot(2,3) );
    Stiff(0,3) = Stiff(3,0);
    Stiff(1,3) = Stiff(3,1);
    Stiff(2,3) = Stiff(3,2);
    Stiff(3,3) = mmKtot(2,2);
  }
  else
  {
    Stiff(0,0) = mmKtot(0,0);
    Stiff(0,1) = .5 * ( mmKtot(0,1) + mmKtot(1,0) );
    Stiff(0,2) = .5 * ( mmKtot(0,2) + mmKtot(2,0) );
    Stiff(0,3) = .5 * ( mmKtot(0,3) + mmKtot(0,4) );
    Stiff(0,4) = .5 * ( mmKtot(0,5) + mmKtot(0,6) );
    Stiff(0,5) = .5 * ( mmKtot(0,7) + mmKtot(0,8) );

    Stiff(1,0) = Stiff(0,1);
    Stiff(1,1) = mmKtot(1,1);
    Stiff(1,2) = .5 * ( mmKtot(1,2) + mmKtot(2,1) );
    Stiff(1,3) = .5 * ( mmKtot(1,3) + mmKtot(1,4) );
    Stiff(1,4) = .5 * ( mmKtot(1,5) + mmKtot(1,6) );
    Stiff(1,5) = .5 * ( mmKtot(1,7) + mmKtot(1,8) );

    Stiff(2,0) = Stiff(0,2);
    Stiff(2,1) = Stiff(1,2);
    Stiff(2,2) = mmKtot(2,2);
    Stiff(2,3) = .5 * ( mmKtot(2,3) + mmKtot(2,4) );
    Stiff(2,4) = .5 * ( mmKtot(2,5) + mmKtot(2,6) );
    Stiff(2,5) = .5 * ( mmKtot(2,7) + mmKtot(2,8) );

    Stiff(3,0) = Stiff(0,3);
    Stiff(3,1) = Stiff(1,3);
    Stiff(3,2) = Stiff(2,3);
    Stiff(3,3) = .5 * ( mmKtot(3,3) + mmKtot(3,4) );
    Stiff(3,4) = .5 * ( mmKtot(3,5) + mmKtot(3,6) );
    Stiff(3,5) = .5 * ( mmKtot(3,7) + mmKtot(3,8) );

    Stiff(4,0) = Stiff(0,4);
    Stiff(4,1) = Stiff(1,4);
    Stiff(4,2) = Stiff(2,4);
    Stiff(4,3) = Stiff(3,4);
    Stiff(4,4) = .5 * ( mmKtot(5,5) + mmKtot(5,6) );
    Stiff(4,5) = .5 * ( mmKtot(5,7) + mmKtot(5,8) );

    Stiff(5,0) = Stiff(0,5);
    Stiff(5,1) = Stiff(1,5);
    Stiff(5,2) = Stiff(2,5);
    Stiff(5,3) = Stiff(3,5);
    Stiff(5,4) = Stiff(4,5);
    Stiff(5,5) = .5 * ( mmKtot(8,7) + mmKtot(8,8) );
  }

  //System::out() << "Stiff " << Stiff << "\n";

  for ( idx_t i = 0; i < 9; i++ )
  {
    for ( idx_t j = 0; j < 9; j++ )
    {
    //  StiffFull(i,j) = mmStiffc(i,j); 
       StiffFull(i,j) = mmKtot(i,j);
    }
  }
}
