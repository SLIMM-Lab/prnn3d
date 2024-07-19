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

#ifndef CUSTOM_SHAPE_H
#define CUSTOM_SHAPE_H

#include <jem/util/Properties.h>

#include <jive/Array.h>
#include <jive/geom/StdShape.h>
#include <jive/geom/InternalShape.h>

using namespace jem;

using jem::util::Properties;

using jive::Vector;
using jive::Matrix;
using jive::Cubix;
using jive::geom::StdShape;
using jive::geom::IShape;

//=====================================================================
//   class CustomShape
//=====================================================================

class CustomShape : public Object {
  public:

    typedef CustomShape  Self;

    typedef void         ( *bMatFunc_ )

    ( const Matrix&    b,
      const Matrix&    g,
      const Matrix&    gsri );

    static const char*   TYPE_PROP;

    CustomShape

    ( const String&       name,
      const Properties&   conf,
      const Properties&   props );

    CustomShape

    ( const Ref<Self>     parent,
      const Matrix&       ischeme );

  protected:

    ~CustomShape ();

  public:

    void                 setGradsForIntegration

    ( const Vector&       ipWeights,
      const Matrix&       coords,
      const idx_t         ie = 0 );

    void                 setGradsForSampling

    ( const Matrix&       coords,
      const idx_t         ie = 0 );

    void                 getBMatrix

    ( const Matrix&       b,
      const idx_t         ip,
      const idx_t         ie = 0 ) const;

    void                 getStrain

    ( const Vector&       strain,
      const Matrix&       b,
      const Vector&       disp,
      const idx_t         ip,
      const idx_t         ie = 0 ) const;

    void                 addElemMatLargeD

    ( const Matrix&      k,
      const Vector&      tau,
      const double       w,
      const idx_t        ip,
      const idx_t        ie = 0 )                   const;

    void                 evalBMatrix

    ( const Matrix&       b,
      const Vector&       u,
      const Matrix&       coords,
      const idx_t         ie = 0 ) const;

    static void          splitAtLocalCoord

    ( Ref<Self>&          shapeA,
      Ref<Self>&          shapeB,
      const Ref<Self>     shape0,
      const double        xic = 0.,
      const idx_t         ixi = 0 );

    inline void          setLargeDisp

    ( const bool         large );

  protected:

    void                 initArrays_ ();

    void                 setGradSRI_

    ( const Matrix&       coords ) const;

    static Ref<StdShape> getStdShape_

    ( const String&       type,
      Matrix&       ischeme );

    static bMatFunc_     getBMatFunc_

    ( const idx_t         rank,
      const bool          sri );

    static void          get1DBMat_

    ( const Matrix&       b,
      const Matrix&       g,
      const Matrix&       gsri );

    static void          get2DBMat_

    ( const Matrix&       b,
      const Matrix&       g,
      const Matrix&       gsri );

    static void          get3DBMat_

    ( const Matrix&       b,
      const Matrix&       g,
      const Matrix&       gsri );

    static void          get1DBMatSRI_

    ( const Matrix&       b,
      const Matrix&       g,
      const Matrix&       gsri );

    static void          get2DBMatSRI_

    ( const Matrix&       b,
      const Matrix&       g,
      const Matrix&       gsri );

    static void          get3DBMatSRI_

    ( const Matrix&       b,
      const Matrix&       g,
      const Matrix&       gsri );

  public:

  protected:

    Ref<IShape>          child_;
    bool                 sri_;
    Cubix                grads_;
    Matrix               gradReg_;
    Matrix               gradSri_;
    Matrix               ischeme_;
    Vector               xiSri_;
    Matrix               defGrad_;

    idx_t                ieSample_;
    idx_t                ieIntegr_;

    idx_t                rank_;
    idx_t                nodeCount_;
    idx_t                ipCount_;

    Ref<StdShape>        sshape_;
    bMatFunc_            fillBMatrix_;
    bool                 largeDisp_;

  public:

    // inline functions calling the normal InternalShape ones
    // (other InternalShape functions are available through the
    // public member child_)

    inline idx_t         nodeCount            () const;
    inline idx_t         vertexCount          () const;
    inline idx_t         ipointCount          () const;
    inline idx_t         localRank            () const;
    inline idx_t         globalRank           () const;
    inline String        getGeometry          () const;
    inline Matrix        getShapeFunctions    () const;
    inline Matrix        getIntegrationScheme () const;
    inline Matrix        getVertexCoords      () const;

    inline void          getGlobalIntegrationPoints

    ( const Matrix&      x,
      const Matrix&      c ) const;

    inline void          getIntegrationWeights

    ( const Vector&      w,
      const Matrix&      c ) const;

    inline void          evalShapeFunctions

    ( const Vector&      h,
      const Vector&      u )                   const;

    inline bool          getLocalPoint

    ( const Vector&      u,
      const Vector&      x,
      double             eps,
      const Matrix&      c )                   const;
};

//---------------------------------------------------------------------
// implementation of inline functions
//---------------------------------------------------------------------

idx_t CustomShape::nodeCount () const
{ return child_->nodeCount (); }

idx_t CustomShape::vertexCount () const
{ return child_->vertexCount (); }

idx_t CustomShape::ipointCount () const
{ return child_->ipointCount (); }

idx_t CustomShape::localRank () const
{ return child_->localRank (); }

idx_t CustomShape::globalRank () const
{ return child_->globalRank (); }

String CustomShape::getGeometry () const
{ return child_->getGeometry (); }

Matrix CustomShape::getShapeFunctions () const
{ return child_->getShapeFunctions (); }

Matrix CustomShape::getIntegrationScheme () const
{ return child_->getIntegrationScheme (); }

Matrix CustomShape::getVertexCoords () const
{ return child_->getVertexCoords (); }

void CustomShape::setLargeDisp
( const bool large )
{
  largeDisp_ = large;

  if ( sri_ && largeDisp_ ) {
    System::warn() << "SRI in combination with large displacenents"
                   << " is not working optimally! Something wrong with tangent."
                   << jem::io::endl;
  }
}

void CustomShape::evalShapeFunctions
( const Vector& h,
  const Vector& u ) const
{
  child_->evalShapeFunctions ( h, u );
}

void CustomShape::getGlobalIntegrationPoints
( const Matrix& x,
  const Matrix& c ) const
{
  child_->getGlobalIntegrationPoints ( x, c );
}

void CustomShape::getIntegrationWeights
( const Vector&      w,
  const Matrix&      c ) const
{
  child_->getIntegrationWeights ( w, c );
}

bool CustomShape::getLocalPoint
( const Vector& u,
  const Vector& x,
  double        eps,
  const Matrix& c ) const
{
  return child_->getLocalPoint ( u, x, eps, c );
}

#endif
