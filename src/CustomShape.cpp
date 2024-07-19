/*
 *  Copyright (C) 2012 TU Delft. All rights reserved.
 *
 *  Object that works as an InternalShape, but optionally uses
 *  selective reduced integration.
 *  Takes "type=SriHex8", "type=SriQuad4" or normal IShape input
 *
 *  Author: F.P. van der Meer
 *  Date: March 2012
 *
 *  November 2014: also added large displacement optionality here
 *
 *  May 2020     : add Wedge12 shape  (Lu Ke)
 */

#include <jem/base/array/operators.h>
#include <jem/base/array/utilities.h>
#include <jem/base/IllegalInputException.h>
#include <jem/base/System.h>
#include <jem/numeric/algebra/matmul.h>

#include <jive/geom/IShapeFactory.h>
#include <jive/geom/ParametricShape.h>
#include <jive/geom/StdCube.h>
#include <jive/geom/StdLine.h>
#include <jive/geom/StdTriangle.h>
#include <jive/geom/StdWedge.h>
#include <jive/geom/StdSquare.h>
// #include "StdWedge12.h"
#include "Wedge12AIO.h"

#include "utilities.h"
#include "CustomShape.h"
#include "LargeDispUtilities.h"

using jem::io::endl;
using jem::numeric::matmul;

using jive::geom::IShapeFactory;
using jive::geom::ParametricShape;

const char* CustomShape::TYPE_PROP = "type";

//---------------------------------------------------------------------
//  constructors and destructor
//---------------------------------------------------------------------

CustomShape::CustomShape

( const String&      name,
  const Properties&  conf,
  const Properties&  props )

{
  String         type;
  Matrix         ischeme;

  Properties myProps = props.findProps ( name );
  Properties myConf  = conf. makeProps ( name );

  myProps.get ( type, TYPE_PROP );

  sri_ = ( type.size() >= 3 ) && ( type[slice ( 0, 3 )].equals ( "Sri" ) );

  if ( sri_ ) {
    String stype = type[slice ( 3, END )];

    sshape_ = getStdShape_ ( stype, ischeme );

    // child_ = newInstance<ParametricShape> ( "sri", ischeme, sshape_, sshape_ );
    props.set ( String ( name + "." + TYPE_PROP ), stype );
    child_ = IShapeFactory::newInstance ( name, conf, props );
    myConf .set ( TYPE_PROP, type );
  }
  else {
    sshape_ = getStdShape_ ( type, ischeme );

    child_ = IShapeFactory::newInstance ( name, conf, props );
  }

  initArrays_();

  largeDisp_ = false;
}

CustomShape::CustomShape

( const Ref<Self>     parent,
  const Matrix&       ischeme )

  : sri_ ( parent->sri_ ), sshape_ ( parent->sshape_ )

{
  child_ = newInstance<ParametricShape> ( "custom", ischeme, sshape_, sshape_ );

  initArrays_();
}

CustomShape::~CustomShape()
{}

//---------------------------------------------------------------------
//  initArrays_
//---------------------------------------------------------------------

void CustomShape::initArrays_ ()

{
  rank_      = globalRank();
  ipCount_   = ipointCount();
  nodeCount_ = nodeCount();

  grads_.  resize ( rank_, nodeCount_, ipCount_ );
  ischeme_.resize ( rank_ + 1,           ipCount_ );
  gradReg_.resize ( rank_, nodeCount_           );
  gradSri_.resize ( rank_, nodeCount_           );
  xiSri_.  resize ( rank_                       );
  defGrad_.resize ( rank_, rank_                );

  fillBMatrix_ = getBMatFunc_ ( rank_, sri_ );
}

//---------------------------------------------------------------------
//  setGradsForIntegration
//---------------------------------------------------------------------

void CustomShape::setGradsForIntegration

( const Vector&  ipWeights,
  const Matrix&  coords,
  const idx_t    ie )

{
  // get gradients in integration points

  child_->getShapeGradients ( grads_, ipWeights, coords );

  if ( sri_ ) {
    setGradSRI_ ( coords );
  }

  // store element number for check

  ieIntegr_ = ie;
}


//---------------------------------------------------------------------
//  setGradsForSampling
//---------------------------------------------------------------------

void CustomShape::setGradsForSampling

( const Matrix&  coords,
  const idx_t    ie )

{
  if ( sri_ ) {
    setGradSRI_ ( coords );
  }

  ieSample_ = ie;
}

//---------------------------------------------------------------------
//  setGradSRI_
//---------------------------------------------------------------------

void CustomShape::setGradSRI_

( const Matrix&  coords ) const

{
  ischeme_ = getIntegrationScheme();

  // compute mid point of integration scheme

  for ( idx_t i = 0; i < rank_; ++ i ) {
    xiSri_[i] = sum ( ischeme_ ( i + 1, ALL ) ) / ipCount_;
  }

  // evaluate gradients in that point

  child_->evalShapeGradients ( gradSri_, xiSri_, coords );
}

//---------------------------------------------------------------------
//  getBMatrix
//---------------------------------------------------------------------

void CustomShape::getBMatrix

( const Matrix&  b,
  const idx_t    ip,
  const idx_t    ie ) const

{
  // first check whether the right setGrads has been called for this element

  JEM_ASSERT ( ie == ieIntegr_ );

  // fill the B matrix (gradsri_ argument is ignored if !sri_)

  fillBMatrix_ ( b, grads_ ( ALL, ALL, ip ), gradSri_ );
}

//-----------------------------------------------------------------------
//  getStrain_
//-----------------------------------------------------------------------

void  CustomShape::getStrain

( const Vector&  strain,
  const Matrix&  b,
  const Vector&  disp,
  const idx_t    ip,
  const idx_t    ie ) const

{
  if ( largeDisp_ ) {
    Matrix grad ( grads_ ( ALL, ALL, ip ) );

    if ( sri_ ) {
      evalDeformationGradient ( defGrad_, disp, grad, gradSri_ );
      getGreenLagrangeStrain  ( strain, defGrad_ );
      getBMatrixLin2D         ( b, defGrad_, grad, gradSri_ );
    }
    else {
      evalDeformationGradient ( defGrad_, disp, grad );
      getGreenLagrangeStrain  ( strain, defGrad_ );
      getBMatrixLin2D         ( b, defGrad_, grad );
    }
  }
  else {
    getBMatrix ( b, ip, ie );
    matmul     ( strain, b, disp );
  }
}

//---------------------------------------------------------------------
// addElemMatLargeD
//---------------------------------------------------------------------

void CustomShape::addElemMatLargeD

( const Matrix&  k,
  const Vector&  tau,
  const double   w,
  const idx_t    ip,
  const idx_t    ie ) const

{
  // first check whether the right setGrads has been called for this element

  JEM_ASSERT ( ie == ieIntegr_ );

  if ( sri_ ) {
    addElemMatLargeD_ ( k, tau, grads_ ( ALL, ALL, ip ), gradSri_, w );
  }
  else {
    addElemMatLargeD_ ( k, tau, grads_ ( ALL, ALL, ip ), w );
  }
}

//---------------------------------------------------------------------
//  evalBMatrix
//---------------------------------------------------------------------

void CustomShape::evalBMatrix

( const Matrix&  b,
  const Vector&  u,
  const Matrix&  coords,
  const idx_t    ie ) const

{
  // first check whether the right setGrads has been called for this element

  JEM_ASSERT ( ie == ieSample_ );

  // evaluate shape function gradients at u

  child_->evalShapeGradients ( gradReg_, u, coords );

  // fill the B matrix (gradsri_ argument is ignored if !sri_)

  fillBMatrix_ ( b, gradReg_, gradSri_ );
}


//---------------------------------------------------------------------
// splitAtLocalCoord
//---------------------------------------------------------------------

void CustomShape::splitAtLocalCoord

( Ref<Self>&      shapeA,
  Ref<Self>&      shapeB,
  const Ref<Self> shape0,
  const double    xic,
  const idx_t     ixi )

{
  Matrix ischeme0  ( shape0->getIntegrationScheme() );

  Matrix ischemeA  ( shape0->rank_ + 1 , shape0->ipCount_ );
  Matrix ischemeB  ( shape0->rank_ + 1 , shape0->ipCount_ );

  double fA = ( xic + 1. ) / 2.;
  double fB = 1. - fA;

  ischemeA ( 0, ALL ) = fA;
  ischemeB ( 0, ALL ) = fB;

  ischemeA ( slice ( 1, END ), ALL ) = ischeme0 ( slice ( 1, END ), ALL );
  ischemeB ( slice ( 1, END ), ALL ) = ischeme0 ( slice ( 1, END ), ALL );

  idx_t ii = ixi + 1;

  ischemeA ( ii, ALL ) = ischeme0 ( ii, ALL ) * fA - 1. + fA;
  ischemeB ( ii, ALL ) = ischeme0 ( ii, ALL ) * fB + 1. - fB;

  shapeA = newInstance<Self> ( shape0, ischemeA );
  shapeB = newInstance<Self> ( shape0, ischemeB );
}

//=====================================================================
//  static functions
//=====================================================================

//---------------------------------------------------------------------
// getStdShape_
//---------------------------------------------------------------------

Ref<StdShape> CustomShape::getStdShape_

( const String& type,
  Matrix& ischeme )

{
  Ref<StdShape> ret;

  if ( type.equals ( "Line2" ) ) {
    ret = newInstance<jive::geom::StdLine2>();
    ischeme.ref ( jive::geom::StdLine2::getGaussScheme ( 2 ) );
  }
  else if ( type.equals ( "Triangle3" ) ) {
    ret = newInstance<jive::geom::StdTriangle3>();
    ischeme.ref ( jive::geom::StdTriangle3::getGaussScheme ( 1 ) );
  }
  else if ( type.equals ( "Quad4" ) ) {
    ret = newInstance<jive::geom::StdSquare4>();
    ischeme.ref ( jive::geom::StdSquare4::getGaussScheme ( 2, 2 ) );
  }
  else if ( type.equals ( "Triangle6" ) ) {
    ret = newInstance<jive::geom::StdTriangle6>();
    ischeme.ref ( jive::geom::StdTriangle6::getGaussScheme ( 3 ) );
  }
  else if ( type.equals ( "Wedge6" ) ) {
    ret = newInstance<jive::geom::StdWedge6>();
    ischeme.ref ( jive::geom::StdWedge6::getGaussScheme ( 1, 2 ) );
  }
  else if ( type.equals ( "Wedge12" ) ) {
    ret = newInstance<jive::geom::StdWedge12>();
    ischeme.ref ( jive::geom::StdWedge12::getGaussScheme ( 3, 1 ) );
  }
  else if ( type.equals ( "Hex8" ) ) {
    ret = newInstance<jive::geom::StdCube8>();
    ischeme.ref ( jive::geom::StdCube8::getGaussScheme ( 2, 2, 2 ) );
  }
  else {
    throw IllegalInputException ( JEM_FUNC, "Unknown CShape type " + type );
  }

  return ret;
}

//---------------------------------------------------------------------
// getBMatFunc_
//---------------------------------------------------------------------

CustomShape::bMatFunc_ CustomShape::getBMatFunc_

( const idx_t rank,
  const bool  sri )

{
  if ( sri ) {
    if ( rank == 1 ) { return & get1DBMatSRI_; }

    if ( rank == 2 ) { return & get2DBMatSRI_; }
    else { return & get3DBMatSRI_; }
  }
  else {
    if ( rank == 1 ) { return & get1DBMat_; }

    if ( rank == 2 ) { return & get2DBMat_; }
    else { return & get3DBMat_; }
  }
}

//---------------------------------------------------------------------
// get1DBMat_
//---------------------------------------------------------------------

void CustomShape::get1DBMat_

( const Matrix&       b,
  const Matrix&       g,
  const Matrix&       gsri )

{
  get1DShapeGrads ( b, g );
}

//---------------------------------------------------------------------
// get2DBMat_
//---------------------------------------------------------------------

void CustomShape::get2DBMat_

( const Matrix&       b,
  const Matrix&       g,
  const Matrix&       gsri )

{
  get2DShapeGrads ( b, g );
}

//---------------------------------------------------------------------
// get3DBMat_
//---------------------------------------------------------------------

void CustomShape::get3DBMat_

( const Matrix&       b,
  const Matrix&       g,
  const Matrix&       gsri )

{
  get3DShapeGrads ( b, g );
}

//---------------------------------------------------------------------
// get1DBMatSRI_
//---------------------------------------------------------------------

void CustomShape::get1DBMatSRI_

( const Matrix&       b,
  const Matrix&       g,
  const Matrix&       gsri )

{
  b = g;
}

//---------------------------------------------------------------------
// get2DBMatSRI_
//---------------------------------------------------------------------

void CustomShape::get2DBMatSRI_

( const Matrix&       b,
  const Matrix&       g,
  const Matrix&       gsri )

{
  const int nodeCount = 4;

  b = 0.0;

  for ( int in = 0; in < nodeCount; ++in ) {
    int i = 2 * in;

    b ( 0, i + 0 ) = g   ( 0, in );
    b ( 1, i + 1 ) = g   ( 1, in );
    b ( 2, i + 0 ) = gsri ( 1, in );
    b ( 2, i + 1 ) = gsri ( 0, in );
  }
}

//---------------------------------------------------------------------
// get3DBMatSRI_
//---------------------------------------------------------------------

void CustomShape::get3DBMatSRI_

( const Matrix&       b,
  const Matrix&       g,
  const Matrix&       gsri )

{
  const int nodeCount = 8;

  b = 0.0;

  for ( int in = 0; in < nodeCount; ++in ) {
    int i = 3 * in;

    b ( 0, i + 0 ) = g   ( 0, in );
    b ( 1, i + 1 ) = g   ( 1, in );
    b ( 2, i + 2 ) = g   ( 2, in );
    b ( 3, i + 0 ) = gsri ( 1, in );
    b ( 3, i + 1 ) = gsri ( 0, in );
    b ( 4, i + 1 ) = gsri ( 2, in );
    b ( 4, i + 2 ) = gsri ( 1, in );
    b ( 5, i + 2 ) = gsri ( 0, in );
    b ( 5, i + 0 ) = gsri ( 2, in );
  }
}


