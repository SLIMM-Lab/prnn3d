#ifndef LARGEDISPUTILS_H 
#define LARGEDISPUTILS_H

#include <jive/Array.h>

using jive::Cubix ;
using jive::Vector;
using jive::Matrix;

//-----------------------------------------------------------------------
//   public functions
//-----------------------------------------------------------------------

//-----------------------------------------------------------------------
// Compute deformation gradient f 
//   from nodal displacements u and shape function gradients g
//-----------------------------------------------------------------------

void   evalDeformationGradient

  ( const Matrix&    f,
    const Vector&    u,
    const Matrix&    g );

//-----------------------------------------------------------------------
// sri version
//-----------------------------------------------------------------------

void   evalDeformationGradient

  ( const Matrix&    f,
    const Vector&    u,
    const Matrix&    gii,
    const Matrix&    gij );

//-----------------------------------------------------------------------
// plane strain version
//-----------------------------------------------------------------------

void  planeStrainDeformationGradient

  ( const Matrix&  f,
    const Vector&  u,
    const Matrix&  g);

//-----------------------------------------------------------------------
// axisymmetric version
//-----------------------------------------------------------------------

void  axisymDeformationGradient

( const Matrix&  f,
  const Vector&  u,
  const Matrix&  g, 
  const Vector&  N,
  const double&  r);

//-----------------------------------------------------------------------
// Compute Green-LagrangeStrain vector from deformation gradient
//-----------------------------------------------------------------------

void   getGreenLagrangeStrain
  
  ( const Vector&    eps,
    const Matrix&    f );


//-----------------------------------------------------------------------
// Rotate normal vector n with deformation gradient F
//-----------------------------------------------------------------------

Vector rotateNormalVector

  ( const Vector&    n0,
    const Matrix&    f );

//-----------------------------------------------------------------------
// Update normal matrix 
//-----------------------------------------------------------------------

void   updateNormalMatrix

  ( const Matrix&    nMat,
    const Vector&    nVec );

//-----------------------------------------------------------------------
// Update rotation matrix 
//-----------------------------------------------------------------------

void   updateRotationMatrix

  ( const Matrix&    Q,
    const Vector&    nVec );

//-----------------------------------------------------------------------
// Rotate normal vector n0 with deformatin gradient f to compute
//   updated rotation matrix Q and derivative of angle phi w.r.t. f
//-----------------------------------------------------------------------

void  updateMatrices

  ( const Matrix&  Q,
    const Matrix&  dPhidF,
    const Vector&  n0,
    const Matrix&  f );

//-----------------------------------------------------------------------
// Compute B matrix for linear part of K
//   from deformation gradient f and shape function gradients g
//-----------------------------------------------------------------------

void  getBMatrixLin2D

  ( const Matrix&  b,
    const Matrix&  f,
    const Matrix&  g );

//-----------------------------------------------------------------------
// sri version
//-----------------------------------------------------------------------

void  getBMatrixLin2D

  ( const Matrix&  b,
    const Matrix&  f,
    const Matrix&  gii,
    const Matrix&  gij );

//-----------------------------------------------------------------------
// Assemble B matrix for nonlinear part of K
//   from shape function gradients g
// (more efficient to skip this and use addElemMatLargeD instead for 
//   direct assembly of additional term in K)
//-----------------------------------------------------------------------

void  getBMatrixNonlin2D

  ( const Matrix&  b,
    const Matrix&  g );

//-----------------------------------------------------------------------
// Add part nonlinear in the displacement field to stiffness matrix k
//   from second piola kirchhoff stress tau and shape func gradients g
//-----------------------------------------------------------------------

void  addElemMatLargeD_

  ( const Matrix&  k,
    const Vector&  tau,
    const Matrix&  g,
    const double   w );

//-----------------------------------------------------------------------
// sri version
//-----------------------------------------------------------------------

void  addElemMatLargeD_

  ( const Matrix&  k,
    const Vector&  tau,
    const Matrix&  gii,
    const Matrix&  gij,
    const double   w );

void addElemMatLargeD

  ( const Matrix& k,
    const Vector& tau,
    const Matrix& g,
    const double  w );

//-----------------------------------------------------------------------
// axisymmetric version
//-----------------------------------------------------------------------

void addAxsymElemMatLargeD

  ( const Matrix& k,
    const Vector& tau,
    const Matrix& g,
    const double  w,
    const double  r );

//-----------------------------------------------------------------------
// Compute nominal stress: stre=F*pk2
//-----------------------------------------------------------------------

void   getNominalStress

  ( const Matrix& stre,
    const Vector& pk2,
    const Matrix& f );

//----------------------------------------------------------------------
// Premultiply stiffness term DB with F, while going from Voigt notation
//   to tensor notation (because F*pk2 is asymmetric)
//----------------------------------------------------------------------

void   getNominalStiff

  ( const Cubix & FDB,
    const Matrix& DB,
    const Matrix& f );

//-----------------------------------------------------------------------
// add geometric stiffness to cohesive element
//-----------------------------------------------------------------------

void  addCohGeomStiff 

  ( const Matrix&  k,
    const Vector&  tracxy,
    const Matrix&  g,
    const Vector&  s,
    const double&   w );

void  addAxsymCohGeomStiff 

  ( const Matrix&  k,
    const Vector&  tracxy,
    const Matrix&  g,
    const Vector&  s,
    const double&  w,
    const double&  r );

//-----------------------------------------------------------------------
// Compute Cauchy stress from 2nd Piola Kirchhoff stress
//-----------------------------------------------------------------------

void   getCauchyStress
  
  ( const Vector&    sig,
    const Vector&    S,
    const Matrix&    f );

#endif
