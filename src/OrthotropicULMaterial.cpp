/*
 *
 *  Copyright (C) 2019 TU Delft. All rights reserved.
 *
 *  This class implements an orthotropic material for updated
 *  Lagrangian analysis. A linear relation between Green-Lagrange
 *  strain and second Piola-Kirchhoff stress is assumed
 *
 *  The class works for 3D and plane strain analysis
 *  Output of stress in material frame not supported
 *  Thermal strain is implemented but lacks proper interface with Model
 *
 *  Author: F.P. van der Meer, f.p.vandermeer@tudelft.nl
 *  Date: October 2019
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
#include "OrthotropicULMaterial.h"

using namespace jem;
using jem::numeric::matmul;
using jem::numeric::MatmulChain;
using jem::numeric::invert;
using jem::numeric::LUSolver;
using jem::io::endl;

const double one_third = 0.33333333;

typedef MatmulChain<double,1>   MChain1;
typedef MatmulChain<double,3>   MChain3;

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  OrthotropicULMaterial::YOUNG_1_PROP    = "young1";
const char*  OrthotropicULMaterial::YOUNG_2_PROP    = "young2";
const char*  OrthotropicULMaterial::YOUNG_3_PROP    = "young3";
const char*  OrthotropicULMaterial::ALPHA_1_PROP    = "alpha1";
const char*  OrthotropicULMaterial::ALPHA_2_PROP    = "alpha2";
const char*  OrthotropicULMaterial::ALPHA_3_PROP    = "alpha3";
const char*  OrthotropicULMaterial::POISSON_12_PROP = "poisson12";
const char*  OrthotropicULMaterial::POISSON_23_PROP = "poisson23";
const char*  OrthotropicULMaterial::POISSON_31_PROP = "poisson31";
const char*  OrthotropicULMaterial::SHEAR_12_PROP   = "shear12";
const char*  OrthotropicULMaterial::SHEAR_23_PROP   = "shear23";
const char*  OrthotropicULMaterial::SHEAR_31_PROP   = "shear31";
const char*  OrthotropicULMaterial::STATE_PROP      = "state";
const char*  OrthotropicULMaterial::THETA_PROP      = "theta";

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------

OrthotropicULMaterial::OrthotropicULMaterial 

  ( const idx_t        rank,
    const Properties&  globdat )

  : Material ( rank, globdat )

{
  JEM_PRECHECK ( rank >= 2 && rank <= 3 );

  young1_    = 1.0;
  young2_    = 1.0;
  young3_    = 1.0;
  alpha1_    = 0.0;
  alpha2_    = 0.0;
  alpha3_    = 0.0;
  poisson12_ =  .0;
  poisson23_ =  .0;
  poisson31_ =  .0;
  shear12_   = 0.5;
  shear23_   = 0.5;
  shear31_   = 0.5;
  theta_     = 0.0;

  idx_t strCount = STRAIN_COUNTS[rank];

  transformMat_     . resize ( strCount, strCount );
  transformMat_     = 0.0;
  transformMatInv_  . resize ( strCount, strCount );
  transformMatInv_  = 0.0;
  stiffMat_         . resize ( strCount, strCount );
  stiffMat_         = 0.0;
  materialCompMat_  . resize ( strCount, strCount );
  materialCompMat_  = 0.0;
  materialStiffMat_ . resize ( strCount, strCount );
  materialStiffMat_ = 0.0;

  tt_               . ref ( transformMatInv_.transpose() );
  thermStrain_      . resize ( strCount );
  thermStrain_      = 0.0;
}


OrthotropicULMaterial::~OrthotropicULMaterial ()
{}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void OrthotropicULMaterial::configure 

  ( const Properties& props )

{
  using jem::maxOf;

  bool isotropic = false;

  JEM_PRECHECK ( rank_ > 1 );

  props.get ( young1_   , YOUNG_1_PROP   , 0.0, maxOf( young1_ ) );
  props.get ( poisson12_, POISSON_12_PROP, 0.0, 1.0 );

  if ( props.find ( young2_, YOUNG_2_PROP, 0.0, maxOf( young2_) ) )
  {
    props.get ( shear12_  , SHEAR_12_PROP, 0.0, maxOf( shear12_) );
    props.find( theta_, THETA_PROP, -90. , 90. );
  }
  else
  {
    isotropic = true;
    young2_ = young1_;
    shear12_ = young1_ / 2. / ( 1. + poisson12_ );
    theta_ = 0.;
  }

  props.find ( alpha1_   , ALPHA_1_PROP   , 0.0, maxOf( alpha1_ ) );
  props.find ( alpha2_   , ALPHA_2_PROP   , 0.0, maxOf( alpha2_ ) );

  if ( rank_ == 2  )
  {
    props.get ( state_, STATE_PROP);

    if ( state_ == "PLANE_STRESS" )
    {
      throw Error ( JEM_FUNC, "Plane stress not supported." );
    }
  }
  else
  {
    state_ = "NOT_PLANE";
  }

  if ( rank_ == 3 || state_ == "PLANE_STRAIN" )
  {
    if ( isotropic ) 
    {
      poisson23_ = poisson12_;
    }
    else
    {
      props.get ( poisson23_, POISSON_23_PROP, 0.0, 0.5 );
    }

    if ( props.find ( young3_, YOUNG_3_PROP, 0.0, maxOf( young3_ ) ) )
    {
      // completely orthotropic material

      props.get ( shear23_  , SHEAR_23_PROP  , 0.0, maxOf( shear23_) );
      props.get ( poisson31_, POISSON_31_PROP, 0.0, 0.5 );
      props.get ( shear31_  , SHEAR_31_PROP  , 0.0, maxOf( shear31_) );
      props.find( alpha3_   , ALPHA_3_PROP   , 0.0, maxOf( alpha3_ ) );
    }
    else
    {
      // transversely isotropic material

      alpha3_    = alpha2_;
      young3_    = young2_;
      poisson31_ = poisson12_;
      shear31_   = shear12_;
      shear23_   = young2_ / ( 2. + 2. * poisson23_ );

      props.find( alpha3_   , ALPHA_3_PROP   , 0.0, maxOf( alpha3_ ) );
    }
  }
  // compute the elastic stiffness matrix, only one time

  computeTransformMats_ ();
  computeStiffMat_ ();
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void OrthotropicULMaterial::getConfig

  ( const Properties& conf ) const 

{
  if ( rank_ == 2 )
  {
    conf.set ( STATE_PROP     , state_     );
  }

  conf.set ( YOUNG_1_PROP   , young1_    );
  conf.set ( YOUNG_2_PROP   , young2_    );
  conf.set ( POISSON_12_PROP, poisson12_ );
  conf.set ( SHEAR_12_PROP  , shear12_   );
  conf.set ( THETA_PROP     , theta_     );
  conf.set ( ALPHA_1_PROP   , alpha1_    );
  conf.set ( ALPHA_2_PROP   , alpha2_    );


  if ( rank_ == 3 || state_ == "PLANE_STRAIN" )
  {
    conf.set ( YOUNG_3_PROP   , young3_    );
    conf.set ( ALPHA_3_PROP   , alpha3_    );
    conf.set ( POISSON_23_PROP, poisson23_ );
    conf.set ( POISSON_31_PROP, poisson31_ );
    conf.set ( SHEAR_23_PROP  , shear23_   );
    conf.set ( SHEAR_31_PROP  , shear31_   );
  }
}

//-----------------------------------------------------------------------
//   transform
//-----------------------------------------------------------------------

void OrthotropicULMaterial::transform

( const Vector&         strain,
  Vector&               mStrain) const

{
  Vector mechStrain ( strain - thermStrain_ );
  matmul ( mStrain,         tt_, mechStrain );
}

//-----------------------------------------------------------------------
//   transformInv
//-----------------------------------------------------------------------

void OrthotropicULMaterial::transformInv

( const Vector&         mStress,
  Vector&               stress) const

{
  matmul (  stress, transformMatInv_, mStress );
}

//-----------------------------------------------------------------------
//   changeFrame
//-----------------------------------------------------------------------

void OrthotropicULMaterial::changeFrame

( const Matrix&         mStiff,
  Matrix&               stiff) const

{
  MChain3      mc3;
  stiff = mc3.matmul ( transformMatInv_, mStiff, tt_ );
}


//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------

void OrthotropicULMaterial::update

  ( const Vector&         stress,
    const Matrix&         stiff,
    const Matrix&         df,
    const idx_t           ip )

{
  idx_t strCount = STRAIN_COUNTS[rank_];

  Matrix F0 ( preHist_[ip].F );
  Matrix F  ( newHist_[ip].F );

  // F(i,j) = dot ( df(i,k), preHist_[ipoint].F(k,j), k );

  F = matmul ( df, F0 );

  Vector strain ( strCount );
  Vector pk2    ( strCount );

  getGreenLagrangeStrain ( strain, F );
  
  linearUpdate_ ( pk2, stiff, strain );

  getCauchyStress ( stress, pk2, F );
}

//-----------------------------------------------------------------------
//   updateWriteTable
//-----------------------------------------------------------------------

// this is necessary when the material model is combined with a rate dependent
// material model

void OrthotropicULMaterial::updateWriteTable

  ( const Vector&         stress,
    const Matrix&         stiff,
    const Matrix&         df,
    const idx_t           ip )

{
  idx_t strCount = STRAIN_COUNTS[rank_];

  Matrix F0 ( preHist_[ip].F );
  Matrix F  ( newHist_[ip].F );

  // F(i,j) = dot ( df(i,k), preHist_[ipoint].F(k,j), k );

  F = matmul ( df, F0 );

  Vector strain ( strCount );
  Vector pk2    ( strCount );

  getGreenLagrangeStrain ( strain, F );
  
  linearUpdate_ ( pk2, stiff, strain );

  getCauchyStress ( stress, pk2, F );
}

//-----------------------------------------------------------------------
//   linearUpdate_
//-----------------------------------------------------------------------

void OrthotropicULMaterial::linearUpdate_

  ( const Vector&         stress,
    const Matrix&         stiff,
    const Vector&         strain )

{
  stiff = stiffMat_;

  // MChain3      mc3;

  // stiff = mc3.matmul ( transformMatInv_, stiffMat_, tt_ );

  Vector mechStrain ( strain - thermStrain_ );

  matmul ( stress, stiff, strain );
}

//-----------------------------------------------------------------------
//   clone
//-----------------------------------------------------------------------

Ref<Material> OrthotropicULMaterial::clone () const

{
  // use default copy constructor

  return newInstance<OrthotropicULMaterial> ( *this );
}
  
//-----------------------------------------------------------------------
//   computeTransformMats_
//-----------------------------------------------------------------------


void   OrthotropicULMaterial::computeTransformMats_ () 
{
  const double  pi = 3.1415926535897931;
  const double  c  = cos( theta_ * pi / 180.0 );
  const double  s  = sin( theta_ * pi / 180.0 );
  const double  sc = s*c;
  const double  c2 = c*c;
  const double  s2 = s*s;
  
  if ( rank_ == 3 ) 
  {
    transformMat_(0,0) = c2;
    transformMat_(0,1) = s2;
    transformMat_(0,3) = 2.0 * sc;

    transformMat_(1,0) = s2;
    transformMat_(1,1) = c2;
    transformMat_(1,3) = - 2.0 * sc;

    transformMat_(2,2) = 1.0;

    transformMat_(3,0) = - sc;
    transformMat_(3,1) = sc;
    transformMat_(3,3) = c2 - s2;

    transformMat_(4,4) = c;
    transformMat_(4,5) = - s;

    transformMat_(5,4) = s;
    transformMat_(5,5) = c;

    transformMatInv_(0,0) = c2;
    transformMatInv_(0,1) = s2;
    transformMatInv_(0,3) = - 2.0 * sc;

    transformMatInv_(1,0) = s2;
    transformMatInv_(1,1) = c2;
    transformMatInv_(1,3) = 2.0 * sc;

    transformMatInv_(2,2) = 1.0;

    transformMatInv_(3,0) = sc;
    transformMatInv_(3,1) = - sc;
    transformMatInv_(3,3) = c2 - s2;

    transformMatInv_(4,4) = c;
    transformMatInv_(4,5) = s;

    transformMatInv_(5,4) = - s;
    transformMatInv_(5,5) = c;
  }
  else if ( rank_ == 2 )
  {
    transformMat_(0,0) = c2;
    transformMat_(0,1) = s2;
    transformMat_(0,2) = 2.0 * sc;

    transformMat_(1,0) = s2;
    transformMat_(1,1) = c2;
    transformMat_(1,2) = - 2.0 * sc;

    transformMat_(2,0) = - sc;
    transformMat_(2,1) = sc;
    transformMat_(2,2) = c2 - s2;

    transformMatInv_(0,0) = c2;
    transformMatInv_(0,1) = s2;
    transformMatInv_(0,2) = - 2.0 * sc;

    transformMatInv_(1,0) = s2;
    transformMatInv_(1,1) = c2;
    transformMatInv_(1,2) = 2.0 * sc;

    transformMatInv_(2,0) = sc;
    transformMatInv_(2,1) = - sc;
    transformMatInv_(2,2) = c2 - s2;
  }
  else
  {
    throw Error ( JEM_FUNC, "unexpected rank: " + String ( rank_ ) );
  }
}


//-----------------------------------------------------------------------
//   computeStiffMat_
//-----------------------------------------------------------------------


void   OrthotropicULMaterial::computeStiffMat_ () 

{
  const double  e1   = young1_;
  const double  e2   = young2_;
  const double  e3   = young3_;
  const double  nu12 = poisson12_;
  const double  nu23 = poisson23_;
  const double  nu31 = poisson31_;
  const double  g12  = shear12_;
  const double  g23  = shear23_;
  const double  g31  = shear31_;

  MChain3    mc3;

  if ( rank_ == 3 ) 
  {
    materialCompMat_(0,0) = 1.0 / e1;
    materialCompMat_(1,1) = 1.0 / e2;
    materialCompMat_(2,2) = 1.0 / e3;

    materialCompMat_(0,1) = materialCompMat_(1,0) = -nu12 / e1;
    materialCompMat_(0,2) = materialCompMat_(2,0) = -nu31 / e1;
    materialCompMat_(1,2) = materialCompMat_(2,1) = -nu23 / e2;

    materialCompMat_(3,3) = 1.0 / g12;
    materialCompMat_(4,4) = 1.0 / g23;
    materialCompMat_(5,5) = 1.0 / g31;

    materialStiffMat_ = materialCompMat_;

    invert( materialStiffMat_ );
  }
  else if ( state_ == "PLANE_STRAIN" )
  {
    // NB: the plane strain materialCompMat_ contains the 
    // slice(0,3)-part of the full 3D compliance matrix
    // This cannot be used to compute strain from stress but it can be
    // for update and inversion to compute damaged stiffness matrix

    materialCompMat_(0,0) = 1.0 / e1;
    materialCompMat_(1,1) = 1.0 / e2;
    materialCompMat_(2,2) = 1.0 / e3;

    materialCompMat_(0,1) = materialCompMat_(1,0) = -nu12 / e1;
    materialCompMat_(0,2) = materialCompMat_(2,0) = -nu31 / e1;
    materialCompMat_(1,2) = materialCompMat_(2,1) = -nu23 / e2;

    materialStiffMat_ = materialCompMat_;

    invert( materialStiffMat_ );

    materialStiffMat_(ALL,  2) = 0.;
    materialStiffMat_(  2,ALL) = 0.;
    materialStiffMat_(  2,  2)   = g12;
  }
  else if ( state_ == "PLANE_STRESS" )
  {
    materialCompMat_(0,0) = 1.0 / e1;
    materialCompMat_(1,1) = 1.0 / e2;
    materialCompMat_(2,2) = 1.0 / g12;
    materialCompMat_(0,1) = materialCompMat_(1,0) = -nu12 / e1;
    materialCompMat_(0,2) = materialCompMat_(2,0) = 0.0;
    materialCompMat_(1,2) = materialCompMat_(2,1) = 0.0;

    materialStiffMat_ = materialCompMat_;

    invert( materialStiffMat_ );
  }
  else
  {
    throw Error ( JEM_FUNC, "unexpected rank (or state): " + String ( rank_ ) );
  }
 
  // System::out() << "computed orthotropic stiffness matrix: " << endl <<
    // materialStiffMat_ << endl;
  
  stiffMat_ = mc3.matmul ( transformMatInv_, materialStiffMat_, tt_);

  // System::out() << "transformed orthotropic stiffness matrix: " << endl <<
    // stiffMat_ << endl;

}


//-----------------------------------------------------------------------
//   updateThermStrain_
//-----------------------------------------------------------------------


void   OrthotropicULMaterial::updateThermStrain

  ( const Properties&  params ) 

{
  double dT = 0.;

  params.find ( dT, "deltaT" );

  Vector mStrain ( thermStrain_.size() );

  mStrain = 0.0;
 
  if ( dT )
  {
    mStrain[0] = alpha1_ * dT;
    mStrain[1] = alpha2_ * dT;

    if ( rank_ == 2 )
    {
      if (  state_ == "PLANE_STRAIN" )
      {
        System::warn() << "thermal expansion not correct for plane" << 
          " strain!\n";
      }
    }
    else if ( rank_ == 3 )
    {
      mStrain[2] = alpha3_ * dT;
    }

    matmul ( thermStrain_, transformMat_.transpose() , mStrain );
  }
}

// --------------------------------------------------------------------
//  commit
// --------------------------------------------------------------------

void  OrthotropicULMaterial::commit()

{
  newHist_.swap ( preHist_ );

  latestHist_ = &preHist_;
}

// --------------------------------------------------------------------
//  allocPoints
// --------------------------------------------------------------------

void     OrthotropicULMaterial::allocPoints

  ( const idx_t      count,
    const Matrix&    transfer,
    const IdxVector& oldPoints )

{
  allocPoints ( count );
}

// --------------------------------------------------------------------
//  allocPoints
// --------------------------------------------------------------------

void  OrthotropicULMaterial::allocPoints

    ( const idx_t   count )

{
  Hist_ hist ( rank_ );

  preHist_.pushBack ( hist, count );
  newHist_.pushBack ( hist, count );
}

// --------------------------------------------------------------------
//  Hist_ constructor
// --------------------------------------------------------------------

OrthotropicULMaterial::Hist_::Hist_ 

  ( const idx_t  rank )

  : F ( rank, rank )

{
  TensorIndex i, j;

  F(i,j) = where ( i==j, 1., 0. );
}

// --------------------------------------------------------------------
//  Hist_ copy constructor
// --------------------------------------------------------------------

OrthotropicULMaterial::Hist_::Hist_ 

  ( const Hist_&  h )

  : F ( h.F.clone() )

{}
  

