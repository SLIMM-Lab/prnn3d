/*
 * 
 *  Copyright (C) 2018 TU Delft. All rights reserved.
 *  
 *  This class implements an isotropic NeoHookean material.
 *  Its update function computes PK2 stress from the Green-Lagrange
 *  strain tensor and evaluates the associated gradient
 *  Following Belytschko Sec. 5.4
 *  
 *  Author: F.P. van der Meer, f.p.vandermeer@tudelft.nl
 *  Date: January 2018
 *
 */


#include <jem/base/limits.h>
#include <jem/base/Error.h>
#include <jem/base/System.h>
#include <jem/base/array/operators.h>
#include <jem/base/array/tensor.h>
#include <jem/base/array/intrinsics.h>
#include <jem/io/Writer.h>
#include <jem/numeric/algebra/matmul.h>
#include <jem/numeric/algebra/LUSolver.h>

#include "utilities.h"
#include "voigtUtilities.h"
#include "NeoHookeMaterial.h"

using namespace jem;
using namespace voigtUtilities;

using jem::maxOf;
using jem::Error;
using jem::TensorIndex;
using jem::System;
using jem::io::endl;
using jem::io::Writer;
using jem::numeric::matmul;
using jem::numeric::LUSolver;

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  NeoHookeMaterial::LAMBDA_PROP  = "lambda";
const char*  NeoHookeMaterial::MU_PROP      = "mu";
const char*  NeoHookeMaterial::STATE_PROP   = "state";

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


NeoHookeMaterial::NeoHookeMaterial 

  ( idx_t rank, const Properties& globdat )
    : Material ( rank, globdat )
{
  JEM_PRECHECK ( rank >= 1 && rank <= 3 );

  lambda_ = 1.0;
  mu_     = 1.0;
}


NeoHookeMaterial::~NeoHookeMaterial ()
{}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void NeoHookeMaterial::configure ( const Properties& props )
{
  props.get ( lambda_, LAMBDA_PROP );
  props.get ( mu_, MU_PROP, 0.0, maxOf( mu_ ) );

  idx_t strCount = STRAIN_COUNTS[rank_];

  // read problem type

  props.get( stateString_, STATE_PROP );

  if      ( stateString_ == "PLANE_STRAIN" )
  {
    state_ = PlaneStrain;
  }
  else if ( stateString_ == "PLANE_STRESS" )
  {
    throw Error ( JEM_FUNC, "Plane stress not supported" );
  }
  else if ( stateString_ == "AXISYMMETRIC" )
  {
    state_ = AxiSymmetric;

    ++strCount;
  }


  // historyNames_.resize ( 1 );
  // historyNames_[0] = "eqps";
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void NeoHookeMaterial::getConfig ( const Properties& conf ) const
{
  conf.set ( LAMBDA_PROP, lambda_ );
  conf.set ( MU_PROP, mu_ );

  conf.set ( STATE_PROP, stateString_ );
}

//-----------------------------------------------------------------------
//   getHistory
//-----------------------------------------------------------------------

// void NeoHookeMaterial::getHistory

//   ( const Vector&  hvals,
//     const idx_t    mpoint ) const

// {
//   (*latestHist_)[mpoint].toVector ( hvals );
// }


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------

void NeoHookeMaterial::update

    ( const Vector&         stress,
      const Matrix&         stiff,
      const Matrix&         df,
      idx_t                 ipoint )

{
  TensorIndex i, j, k;

  // Writer&     dbgOut     ( System::debug ( "ul" ) );

  idx_t strCount = STRAIN_COUNTS[rank_];

  Matrix F0 ( preHist_[ipoint].F );
  Matrix F  ( newHist_[ipoint].F );

  F = matmul ( df, F0 );

  double jac = determinant(F);

  // dbgOut << "jac " << jac << endl;
  // dbgOut << "df\n" << df << endl;
  // dbgOut << "incremental F\n" << F << endl;

  double lnJ = log(jac);

  Matrix  tau ( 3, 3 );  // tau ( rank_, rank_ ); 


  // kirchhoff stress: tau = lambda * lnJ * I - mu * ( B - I )


  tau(i,j)  = ( lambda_ * lnJ - mu_ ) * where ( i==j, 1., 0. );
  tau(i,j) += mu_ * dot ( F(i,k), F(j,k), k );


  // cauchy stress in voigt notation

  if ( stateString_ == "PLANE_STRAIN" )
  {
    stress[0] = tau(0,0);
    stress[1] = tau(1,1);

    stress[2] = ( tau(0,1) + tau(1,0) ) / 2.;
  }

  else if ( stateString_ == "AXISYMMETRIC" )
  {
    stress[0] = tau(0,0);
    stress[1] = tau(1,1);

    stress[2] = ( tau(0,1) + tau(1,0) ) / 2.;

    stress[3] = tau(2,2);
  }

  else if ( stateString_ == "3D")
  {
    stress[0] = tau(0,0);
    stress[1] = tau(1,1);
    stress[2] = tau(2,2);

    stress[3] = ( tau(0,1) + tau(1,0) ) / 2.;
    stress[4] = ( tau(1,2) + tau(2,1) ) / 2.;
    stress[5] = ( tau(2,0) + tau(0,2) ) / 2.;
  }

  stress /= jac;

  // tangent matrix (dtau/dE?)

  double muTangent = mu_ - lambda_ * lnJ;

  stiff = 0.;

  for ( idx_t i = 0; i < rank_; ++i )
  {
    for ( idx_t j = 0; j < rank_; ++j )
    {
      stiff(i,j) = lambda_; 
    }
    stiff(i,i) += 2. * muTangent;
  }
  for ( idx_t i = rank_; i < strCount; ++i )
  {
   stiff(i,i) = muTangent;
  }

  // dbgOut << "tau\n" << tau << endl;

  latestHist_ = &newHist_;

  // System::out() << "\n";

  // System::out() << "tau " << tau << "\n";

  // System::out() << "\n";
}


void NeoHookeMaterial::updateWriteTable

    ( const Vector&         stress,
      const Matrix&         stiff,
      const Matrix&         df,
      idx_t                 ipoint )
  {
  TensorIndex i, j, k;

  idx_t strCount = STRAIN_COUNTS[rank_];

  Matrix F0 ( preHist_[ipoint].F );
  Matrix F  ( newHist_[ipoint].F );

  // F(i,j) = dot ( df(i,k), preHist_[ipoint].F(k,j), k );

  F = matmul ( df, F0 );

  double jac = determinant(F);

  double lnJ = log(jac);

  Matrix  tau ( 3, 3 );  // tau ( rank_, rank_ ); 


  // kirchhoff stress: tau = lambda * lnJ * I - mu * ( B - I )


  tau(i,j)  = ( lambda_ * lnJ - mu_ ) * where ( i==j, 1., 0. );
  tau(i,j) += mu_ * dot ( F(i,k), F(j,k), k );


  // cauchy stress in voigt notation
  
  if ( stateString_ == "PLANE_STRAIN" )
  {
    stress[0] = tau(0,0);
    stress[1] = tau(1,1);

    stress[2] = ( tau(0,1) + tau(1,0) ) / 2.;

    stress[3] = tau(2,2);
  }

  else if ( stateString_ == "AXISYMMETRIC" )
  {
    stress[0] = tau(0,0);
    stress[1] = tau(1,1);

    stress[2] = ( tau(0,1) + tau(1,0) ) / 2.;

    stress[3] = tau(2,2);
  }

  else if ( stateString_ == "3D")
  {
    stress[0] = tau(0,0);
    stress[1] = tau(1,1);
    stress[2] = tau(2,2);

    stress[3] = ( tau(0,1) + tau(1,0) ) / 2.;
    stress[4] = ( tau(1,2) + tau(2,1) ) / 2.;
    stress[5] = ( tau(2,0) + tau(0,2) ) / 2.;
  }

  stress /= jac;

  stiff = 0.;

  // dbgOut << "tau\n" << tau << endl;
}

//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------

Ref<Material> NeoHookeMaterial::clone () const

{
  // use default copy constructor

  return newInstance<NeoHookeMaterial> ( *this );
}

//-----------------------------------------------------------------------
//   fill3DStrain
//-----------------------------------------------------------------------

// Tuple<double,6> NeoHookeMaterial::fill3DStrain

//   ( const Vector&    v3 ) const

// {
//   if ( v3.size() == 3 )
//   {
//     double poisson = lambda_ / ( 2 * (lambda_ + mu_));

//     double eps_zz = state_ == PlaneStress
//                   ? -poisson / (1.-poisson) * (v3[0]+v3[1])
//                   : 0.;

//     return fillFrom2D_ ( v3, eps_zz );
//   }
//   else if ( v3.size() == 4 )
//   {
//     return fillFrom2D_ ( v3[slice(0,3)], v3[3] );
//   }
//   else
//   {
//     return fillFrom3D_ ( v3 );
//   }
// }

// //-----------------------------------------------------------------------
// //   fill3DStress
// //-----------------------------------------------------------------------

// Tuple<double,6> NeoHookeMaterial::fill3DStress

//   ( const Vector&    v3 ) const

// {
//   if ( v3.size() == 3 )
//   {
//     double poisson = lambda_ / ( 2 * (lambda_ + mu_));

//     double sig_zz = state_ == PlaneStress
//                   ? 0.
//                   : poisson * ( v3[0] + v3[1] );

//     return fillFrom2D_ ( v3, sig_zz );
//   }
//   else if ( v3.size() == 4 )
//   {
//     return fillFrom2D_ ( v3[slice(0,3)], v3[3] );
//   }
//   else
//   {
//     return fillFrom3D_ ( v3 );
//   }
// }

// --------------------------------------------------------------------
//  commit
// --------------------------------------------------------------------

void  NeoHookeMaterial::commit()

{
  newHist_.swap ( preHist_ );

  latestHist_ = &preHist_;
}

// --------------------------------------------------------------------
//  allocPoints
// --------------------------------------------------------------------

void  NeoHookeMaterial::allocPoints

    ( const idx_t   count )

{
  Hist_ hist ( 3 );

  preHist_.pushBack ( hist, count );
  newHist_.pushBack ( hist, count );
}

// --------------------------------------------------------------------
//  Hist_ constructor
// --------------------------------------------------------------------

NeoHookeMaterial::Hist_::Hist_ 

  ( const idx_t  rank )
  
  : F(rank, rank)

{
  TensorIndex i, j;

  F(i,j) = where ( i==j, 1., 0. );

  eqps = 0.0;
}

// --------------------------------------------------------------------
//  Hist_ copy constructor
// --------------------------------------------------------------------

NeoHookeMaterial::Hist_::Hist_ 

  ( const Hist_&  h )

  : F ( h.F.clone() ), eqps ( h.eqps )

{}

// inline void NeoHookeMaterial::Hist_::toVector

//  ( const Vector&  vec ) const

// {
//   vec[0] = eqps;
// }
  

