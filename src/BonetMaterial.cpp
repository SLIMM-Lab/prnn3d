/*
 *
 *  Copyright (C) 2019 TU Delft. All rights reserved.
 *
 *  This class implements an orthotropic, transversely isotropic material
 *  for updated Lagrangian analysis. Hyperelastic constitutive equations
 *  enable large strain computations
 *
 *  The class works for 3D analysis
 *
 *  Reference: Bonet and Burton, Comput Methods Appl Mech Engrg, 1997
 *
 *
 *  Date: November 2019
 *
 */

#include <jem/base/limits.h>
#include <jem/base/Error.h>
#include <jem/base/System.h>
#include <jem/base/Array.h>
#include <jem/base/array/tensor.h>
#include <jem/base/PrecheckException.h>
#include <jem/util/Properties.h>
#include <jem/numeric/algebra/matmul.h>
#include <jem/numeric/algebra/MatmulChain.h>
#include <jem/numeric/algebra/EigenUtils.h>
#include <jem/numeric/algebra/LUSolver.h>
#include <jem/numeric/utilities.h>

#include "utilities.h"
#include "LargeDispUtilities.h"
#include "BonetMaterial.h"
#include "voigtUtilities.h"

using namespace jem;
using jem::numeric::matmul;
using jem::numeric::MatmulChain;
using jem::numeric::invert;
using jem::numeric::LUSolver;
using jem::io::endl;

const double one_third = 0.33333333;
const double PI        = 3.14159265;

typedef MatmulChain<double,1>   MChain1;
typedef MatmulChain<double,3>   MChain3;

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  BonetMaterial::YOUNG_1_PROP    = "Ea";
const char*  BonetMaterial::YOUNG_2_PROP    = "E";
const char*  BonetMaterial::POISSON_1_PROP  = "nua";
const char*  BonetMaterial::POISSON_2_PROP  = "nu";
const char*  BonetMaterial::SHEAR_PROP      = "Ga";
const char*  BonetMaterial::FIBDIR_PROP     = "fibdir";

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------

BonetMaterial::BonetMaterial 

  ( const idx_t        rank,
    const Properties&  globdat )

  : Material ( rank, globdat )

{
  JEM_PRECHECK ( rank == 3 );

  Ea_          = 1.;
  E_           = 1.;
  Ga_          = 1.;
  nua_         = 0.;
  nu_          = 0.;
  fibdir_      = 0.;
  nn_          = false;
}


BonetMaterial::~BonetMaterial ()
{}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void BonetMaterial::configure 

  ( const Properties& props )

{
  using jem::maxOf;

  JEM_PRECHECK ( rank_ > 1 );

  props.get ( Ea_   , YOUNG_1_PROP   , 0.0, maxOf( Ea_) );
  props.get ( E_    , YOUNG_2_PROP   , 0.0, maxOf( E_ ) );
  props.get ( Ga_   , SHEAR_PROP     , 0.0, maxOf( Ga_) );
  props.get ( nua_  , POISSON_1_PROP , 0.0, 0.5 );
  props.get ( nu_   , POISSON_2_PROP , 0.0, 0.5 );
  props.get ( fibdir_,FIBDIR_PROP               );

  JEM_PRECHECK ( fibdir_.size() == 3 );

  props.find ( nn_, "nn" );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void BonetMaterial::getConfig

  ( const Properties& conf ) const 

{
  conf.set ( YOUNG_1_PROP   , Ea_      );
  conf.set ( YOUNG_2_PROP   , E_       );
  conf.set ( SHEAR_PROP     , Ga_      );
  conf.set ( POISSON_1_PROP , nua_     );
  conf.set ( POISSON_2_PROP , nu_      );
  conf.set ( FIBDIR_PROP    , fibdir_  );
}

//-----------------------------------------------------------------------
//   getHistory
//-----------------------------------------------------------------------

void BonetMaterial::getHistory

  ( const Vector&  hvals,
    const idx_t    mpoint ) const

{
  (*latestHist_)[mpoint].toVector ( hvals );
 // System::out() << "getHistory: " << hvals << "\n";
}

//-----------------------------------------------------------------------
//   getInitHistory
//-----------------------------------------------------------------------

void BonetMaterial::getInitHistory

  ( const Vector&  hvals,
    const idx_t    mpoint ) const

{
  initHist_[0].toVector ( hvals );
 // System::out() << "getInitHistory.\n"; // << preHist_[mpoint].F << "\n";
}

//-----------------------------------------------------------------------
//   getHistoryCount
//-----------------------------------------------------------------------

idx_t BonetMaterial::getHistoryCount

  ( ) const

{
  return 9; 
}

//-----------------------------------------------------------------------
//   setHistory
//-----------------------------------------------------------------------

void BonetMaterial::setHistory

  ( const Vector&  hvals,
    const idx_t    mpoint )

{
  preHist_[mpoint].F(0,0)     = hvals[0];     // Total deformation gradient
  preHist_[mpoint].F(0,1)     = hvals[1];
  preHist_[mpoint].F(0,2)     = hvals[2];
  preHist_[mpoint].F(1,0)     = hvals[3];
  preHist_[mpoint].F(1,1)     = hvals[4];
  preHist_[mpoint].F(1,2)     = hvals[5];
  preHist_[mpoint].F(2,0)     = hvals[6];
  preHist_[mpoint].F(2,1)     = hvals[7];
  preHist_[mpoint].F(2,2)     = hvals[8];

  latestHist_      = &preHist_;
  newHist_[mpoint] = preHist_[mpoint];
}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------

void BonetMaterial::update

  ( const Vector&         stress,
    const Matrix&         stiff,
    const Matrix&         df,
    const idx_t           ip )

{
  idx_t strCount = STRAIN_COUNTS[rank_];

  Vector stressNH(strCount), stressTI(strCount);

  // define some constants
  
  double lambda, mu, m, n;

  n = Ea_ / E_;

  m = 1. - nu_ - 2. * n * pow(nua_,2.);

  lambda = (nu_ + n * pow(nua_,2.)) * E_ / ( (1. + nu_) * m );

  mu = E_ / (2. * (1. + nu_ ));


  double alpha, beta, gamma;

  alpha = mu - Ga_;

  // System::out() << "alpha " << alpha << endl;

  beta = E_ * ( nua_ + nu_*nua_ - nu_ - n * pow(nua_,2.)) / (1 + nu_) / (4.* m);

  // System::out() << "beta " << beta << endl;

  // beta = min(0., beta);

  gamma = Ea_ * (1. - nu_) / (8.* m) - 

          (lambda + 2. * mu) / 8. + alpha / 2. - beta;


  // calculate deformation gradient for the current step

  Matrix F0 ( preHist_[ip].F );
  Matrix F  ( newHist_[ip].F );

  Matrix DF ( 3, 3 );

  if ( nn_ )
  {
    // When NN is enabled, the total def. grad is passed
    F = df;

 /*   System::out() << "F " << F << "\n";
    System::out() << "F prev " << F0 << "\n";
  */  
    Matrix Finv ( F0 );

    invert ( Finv );

    DF = matmul ( F, Finv );
  //  System::out() << "F " << F << "\ndf " << DF << "\n";
  }
  else
  {
    F = matmul ( df, F0 );
//    System::out() << "F " << F << "\ndf " << df << "\n";
  }

  double J;

  J = jem::numeric::det (F);

  // left Cauchy-Green deformation tensor

  Matrix B(3,3);

  B = matmul (F, F.transpose());

  // vector defining fiber direction in deformed configuration: a
  // vector defining fiber direction in undeformed configuration: A

  Vector a(3), A(3), Ba(3);

  A = fibdir_;

  a  = matmul (F, A);
  Ba = matmul (B, a);

  TensorIndex i, j, k, l;

  // initialize some tensors

  Matrix mI (3,3);

  mI = 0.0;

  mI(0,0) = 1.0; mI(1,1) = 1.0; mI(2,2) = 1.0;


  Array <double,4> II(3,3,3,3), ii(3,3,3,3), aaaa(3,3,3,3), aaB(3,3,3,3);

  Array <double,4> Baa(3,3,3,3), aBBa(3,3,3,3), aaI(3,3,3,3), Iaa(3,3,3,3);

  ii   (i,j,k,l) = mI(i,k) * mI(j,l);

  II   (i,j,k,l) = mI(i,j) * mI(k,l);

  aaaa (i,j,k,l) = a[i] * a[j] * a[k] * a[l];

  aaB  (i,j,k,l) = a[i] * a[j] * B(k,l);

  Baa  (i,j,k,l) = B(i,j) * a[k] * a[l];

  // aaI  (i,j,k,l) = a[i] * a[j] * mI(k,l);

  // Iaa  (i,j,k,l) = mI(i,j) * a[k] * a[l];

  aBBa (i,j,k,l) = a[i] * a[l] * B(j,k) + B(i,k) * a[j] * a[l];


  Matrix aa (3,3), aBa(3,3), baa(3,3);

  aa (i,j) = a[i] *  a[j];

  aBa(i,j) = a[i] * Ba[j];

  baa(i,j) = Ba[i] * a[j];


  // calculate the Cauchy stress for isotropic part of the model
  // neo-Hookean expression for the Cauchy stress

  Matrix sigmaNH(3,3);

  double lnJ = log(J);

  sigmaNH = mu / J * (B - mI) + lambda / J * lnJ * mI; 

  // sigmaNH = mu / J * (B - mI) + lambda * (J-1.) * mI;  

  // calculate the Cauchy stress for trans. isotropic part of the model

  double I1, I4;

  I1 = B(0,0) + B(1,1) + B(2,2); // trace of tensor B

  I4 = dot (a,a);


  Matrix sigmaTI(3,3);

  // sigmaTI = 2. * beta * (I4 - 1.) * mI

  //          + 2. * (alpha + 2.*beta*lnJ + 2.*gamma*(I4 - 1.)) * aa

  //          - alpha * ( baa + aBa);

  // the following option for transversely isotropic part gives better
  // convergence rate

  sigmaTI = 2. * beta * (I4 - 1.) * B

           + 2. * (alpha + beta * (I1 - 3.) + 2. * gamma * (I4 - 1.)) * aa

           - alpha * (baa + aBa);

  sigmaTI /= J;

  
  // calculate the total Cauchy stress

  if ( nn_ )
  { 
    Matrix stressGen ( 3, 3 );
    stressGen = sigmaNH + sigmaTI;
    for ( idx_t i = 0; i < 3; i++ )
    {
      for ( idx_t j = 0; j < 3; j++ )
      {
        stress[i*3+j] = 0.5 * ( stressGen(i,j) + stressGen(j,i) );
      //  stress[i*3+j] = stressGen(i,j);
      }
    }

//    System::out() << "stressGen " << stressGen << "\n";
  }
  else
  {
    voigtUtilities::tensor2VoigtStress (stressNH, sigmaNH);

    voigtUtilities::tensor2VoigtStress (stressTI, sigmaTI);

    stress = stressNH + stressTI;
  }

  // calculate isotropic part of the stiffness

 /* if ( nn_ )
  {
    stiff = 0.0;
  }
  else
  {*/
  Array <double,4> stiffNH(3,3,3,3), stiffTI(3,3,3,3), C(3,3,3,3);


  stiffNH = lambda / J * II

            + 2. / J * (mu - lambda * lnJ ) * ii;

  // stiffNH = lambda * (2.*J - 1.) * II

  //           + 2. / J * (mu - lambda*J * (2.*J - 1.) ) * ii;

  // calculate transversely isotropic part of the stiffness

  // stiffTI = 8.*gamma*aaaa

  //          + 4.*beta*(aaI + Iaa)

  //          - alpha*aBBa

  //          - 4.*beta*(I4 - 1.)*ii;

  // the following option for transversely isotropic part gives better
  // convergence rate

  stiffTI = 8. * gamma * aaaa

           + 4. * beta * (aaB + Baa)

           - 2. * alpha * aBBa;

  stiffTI /= J;


  // calculate the total stiffness
  
  C = stiffNH + stiffTI;

  stiff(0,0) = C(0,0,0,0);
  stiff(0,1) = C(0,0,1,1);
  stiff(0,2) = C(0,0,2,2);
  stiff(0,3) = 0.5 * ( C(0,0,0,1) + C(0,0,1,0) );
  stiff(0,4) = 0.5 * ( C(0,0,1,2) + C(0,0,2,1) );
  stiff(0,5) = 0.5 * ( C(0,0,0,2) + C(0,0,2,0) );   
  
  stiff(1,0) = stiff(0,1);
  stiff(1,1) = C(1,1,1,1);
  stiff(1,2) = C(1,1,2,2);
  stiff(1,3) = 0.5 * ( C(1,1,0,1) + C(1,1,1,0) );
  stiff(1,4) = 0.5 * ( C(1,1,1,2) + C(1,1,2,1) );
  stiff(1,5) = 0.5 * ( C(1,1,0,2) + C(1,1,2,0) );

  stiff(2,0) = stiff(0,2);
  stiff(2,1) = stiff(1,2);
  stiff(2,2) = C(2,2,2,2);
  stiff(2,3) = 0.5 * ( C(2,2,0,1) + C(2,2,1,0) );
  stiff(2,4) = 0.5 * ( C(2,2,1,2) + C(2,2,2,1) );
  stiff(2,5) = 0.5 * ( C(2,2,0,2) + C(2,2,2,0) );

  stiff(3,0) = stiff(0,3);
  stiff(3,1) = stiff(1,3);
  stiff(3,2) = stiff(2,3);
  stiff(3,3) = 0.5 * ( C(0,1,0,1) + C(0,1,1,0) );
  stiff(3,4) = 0.5 * ( C(0,1,1,2) + C(0,1,2,1) );
  stiff(3,5) = 0.5 * ( C(0,1,0,2) + C(0,1,2,0) );

  stiff(4,0) = stiff(0,4);
  stiff(4,1) = stiff(1,4);
  stiff(4,2) = stiff(2,4);
  stiff(4,3) = stiff(3,4);
  stiff(4,4) = 0.5 * ( C(1,2,1,2) + C(1,2,2,1) );
  stiff(4,5) = 0.5 * ( C(1,2,0,2) + C(1,2,2,0) );

  stiff(5,0) = stiff(0,5);
  stiff(5,1) = stiff(1,5);
  stiff(5,2) = stiff(2,5);
  stiff(5,3) = stiff(3,5);
  stiff(5,4) = stiff(4,5);
  stiff(5,5) = 0.5 * ( C(0,2,0,2) + C(0,2,2,0) );

 // }
//  System::out() << "newHist after update \n" << newHist_[0].F << endl;

  //System::out() << "stress " << stress << "\n";
//  System::out() << "stiff " << stiff << "\n";

  latestHist_ = &newHist_;
}

//-----------------------------------------------------------------------
//   updateWriteTable
//-----------------------------------------------------------------------

// this is needed when the material is used in the same model  
// with a rate dependent material;

void BonetMaterial::updateWriteTable

  ( const Vector&         stress,
    const Matrix&         stiff,
    const Matrix&         df,
    const idx_t           ip )

{
  idx_t strCount = STRAIN_COUNTS[rank_];

  Vector stressNH(strCount), stressTI(strCount);

  // define some constants
  
  double lambda, mu, m, n;

  n = Ea_ / E_;

  m = 1. - nu_ - 2. * n * pow(nua_,2.);

  lambda = (nu_ + n * pow(nua_,2.)) * E_ / ( (1. + nu_) * m );

  mu = E_ / (2. * (1. + nu_ ));


  double alpha, beta, gamma;

  alpha = mu - Ga_;

  // System::out() << "alpha " << alpha << endl;

  beta = E_ * ( nua_ + nu_*nua_ - nu_ - n * pow(nua_,2.)) / (1 + nu_) / (4.* m);

  //System::out() << "beta " << beta << endl;

  // beta = min(0., beta);

  gamma = Ea_ * (1. - nu_) / (8.* m) - 

          (lambda + 2. * mu) / 8. + alpha / 2. - beta;

  // calculate deformation gradient for the current step

  Matrix F0 ( preHist_[ip].F );
  Matrix F  ( newHist_[ip].F );

  F = matmul ( df, F0 );

  double J;

  J = jem::numeric::det (F);

  // left Cauchy-Green deformation tensor

  Matrix B(3,3);

  B = matmul (F, F.transpose());

  // vector defining fiber direction in deformed configuration: a
  // vector defining fiber direction in undeformed configuration: A

  Vector a(3), A(3), Ba(3);

  A = fibdir_;

  a  = matmul (F, A);
  Ba = matmul (B, a);

  TensorIndex i, j, k, l;

  // initialize some tensors

  Matrix mI (3,3);

  mI = 0.0;

  mI(0,0) = 1.0; mI(1,1) = 1.0; mI(2,2) = 1.0;


  Matrix aa (3,3), aBa(3,3), baa(3,3);

  aa (i,j)  = a[i] *  a[j];

  aBa(i,j)  = a[i] * Ba[j];

  baa(i,j)  = Ba[i] * a[j];


  // calculate the Cauchy stress for isotropic part of the model
  // neo-Hookean expression for the Cauchy stress

  Matrix sigmaNH(3,3);

  double lnJ = log(J);

  sigmaNH = mu / J * (B - mI) + lambda / J * lnJ * mI;  

  // calculate the Cauchy stress for trans. isotropic part of the model

  double I1, I4;

  I1 = B(0,0) + B(1,1) + B(2,2); // trace of tensor B

  I4 = dot (a,a);


  Matrix sigmaTI(3,3);

  // sigmaTI = 2. * beta * (I4 - 1.) * mI

  //          + 2. * (alpha + 2.*beta*lnJ + 2.*gamma*(I4 - 1.)) * aa

  //          - alpha * ( baa + aBa);

  // the following option for transversely isotropic part gives better
  // convergence rate

  sigmaTI = 2. * beta * (I4 - 1.) * B

           + 2. * (alpha + beta * (I1 - 3.) + 2. * gamma * (I4 - 1.)) * aa

           - alpha * ( baa + aBa);

  sigmaTI /= J;

  // calculate the total Cauchy stress

  voigtUtilities::tensor2VoigtStress (stressNH, sigmaNH);

  voigtUtilities::tensor2VoigtStress (stressTI, sigmaTI);

  //System::out() << "stress " << stress.size(0) << " stressnh " << stressNH.size(0) << " stressTI " << stressTI.size(0) << " \n";

  stress = stressNH + stressTI;

  stiff = 0.0; // stiffness not necessary for printing the output

  stiff ( 0,0 ) = J;

}

//-----------------------------------------------------------------------
//   fill3DStrain
//-----------------------------------------------------------------------

// Tuple<double,6> BonetMaterial::fill3DStrain

//   ( const Vector&    v3 ) const

// {
//   if ( v3.size() != 6 )
//   {
//     throw Error ( JEM_FUNC, String::format (
                  
//                   "State problem other than 3D not supported" ) );
//   }
//   else
//   {
//     return fillFrom3D_ ( v3 );
//   }
// }

//-----------------------------------------------------------------------
//   fill3DStress
//-----------------------------------------------------------------------

// Tuple<double,6> BonetMaterial::fill3DStress

//   ( const Vector&    v3 ) const

// {
//   if ( v3.size() != 6 )
//   {
//     throw Error ( JEM_FUNC, String::format (
                  
//                   "State problem other than 3D not supported" ) );
//   }
//   else
//   {
//     return fillFrom3D_ ( v3 );
//   }
// }

//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------

Ref<Material> BonetMaterial::clone () const

{
  // use default copy constructor

  return newInstance<BonetMaterial> ( *this );
}
  

// --------------------------------------------------------------------
//  commit
// --------------------------------------------------------------------

void  BonetMaterial::commit()

{
  newHist_.swap ( preHist_ );

  latestHist_ = &preHist_;
}

// --------------------------------------------------------------------
//  allocPoints
// --------------------------------------------------------------------

void     BonetMaterial::allocPoints

  ( const idx_t      count,
    const Matrix&    transfer,
    const IdxVector& oldPoints )

{
  allocPoints ( count );
}

// --------------------------------------------------------------------
//  allocPoints
// --------------------------------------------------------------------

void  BonetMaterial::allocPoints

    ( const idx_t   count )

{
  Hist_ hist ( rank_ );

  initHist_.pushBack ( hist, 1 );
  preHist_.pushBack ( hist, count );
  newHist_.pushBack ( hist, count );
}

// --------------------------------------------------------------------
//  Hist_ constructor
// --------------------------------------------------------------------

BonetMaterial::Hist_::Hist_ 

  ( const idx_t  rank )

  : F ( rank, rank )

{
  TensorIndex i, j;

  F(i,j) = where ( i==j, 1., 0. );
}

// --------------------------------------------------------------------
//  Hist_ copy constructor
// --------------------------------------------------------------------

BonetMaterial::Hist_::Hist_ 

  ( const Hist_&  h )

  : F ( h.F.clone() ) 
{
}
 
// -------------------------------------------------------------------
//  Hist_::toVector
// -------------------------------------------------------------------

inline void BonetMaterial::Hist_::toVector

 ( const Vector&  vec ) const

{
//  System::out() << "vec size " << vec.size(0) << "\n";

  for ( idx_t i = 0; i < 3; i++ )
  {
    for ( idx_t j = 0; j < 3; j++ )
    {
      vec[i*3+j]  = F(i, j);
    }
  }

//  System::out() << "vec (toVector): " << vec << "\n";
 
}

 
  

