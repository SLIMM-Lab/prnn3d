/*
 *
 *  This file contains all necessary classes for Wedge12 shape.
 *
 *
 *  Lu Ke, 04-05-2020
 *
 *
 */


#include <jem/base/assert.h>
#include <jem/base/array/select.h>
#include <jem/base/IllegalInputException.h>
#include <jem/base/ClassTemplate.h>
#include <jem/util/Properties.h>
#include <jive/geom/IntegrationSchemes.h>
#include <jive/geom/StdShapeFactory.h>
#include <jive/geom/ShapeBoundary.h>
#include <jive/geom/Names.h>
#include <jive/geom/Geometries.h>
#include <jive/geom/StdLine.h>
#include <jive/geom/StdPrism.h>
#include <jive/geom/StdBoundary.h>
#include <jive/geom/StdTriangle.h>
#include <jive/geom/StdSquare.h>
#include <jive/geom/ParametricVolume.h>
#include <jive/geom/ParametricSurface.h>
#include <jive/geom/BoundaryTriangle.h>
#include <jive/geom/BoundaryQuad.h>
#include <jive/geom/IShapeFactory.h>
#include <jive/util/utilities.h>

#include "Wedge12AIO.h"


JEM_DEFINE_SERIABLE_CLASS ( jive::geom::StdWedge12 );


JIVE_BEGIN_PACKAGE ( geom )

using jem::newInstance;
using jive::util::joinNames;
using jive::util::Topology;
//=======================================================================
//   class StdWedge12
//=======================================================================


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


StdWedge12::StdWedge12 ()
{}


StdWedge12::~StdWedge12 ()
{}


//-----------------------------------------------------------------------
//   readFrom & writeTo
//-----------------------------------------------------------------------


void StdWedge12::readFrom ( ObjectInput& in )
{}


void StdWedge12::writeTo  ( ObjectOutput& out ) const
{}


//-----------------------------------------------------------------------
//   shapeFuncCount
//-----------------------------------------------------------------------


idx_t StdWedge12::shapeFuncCount () const
{
  return ( idx_t ) SHAPE_FUNC_COUNT;
}


//-----------------------------------------------------------------------
//   vertexCount
//-----------------------------------------------------------------------


idx_t StdWedge12::vertexCount () const
{
  return ( idx_t ) VERTEX_COUNT;
}


//-----------------------------------------------------------------------
//   vertexCoords
//-----------------------------------------------------------------------


Matrix StdWedge12::vertexCoords ()
{
  return StdPrism::getCoords (
           StdQuadraticTriangle::vertexCoords (),
           StdLinearLine       ::vertexCoords ()
         );
}


//-----------------------------------------------------------------------
//   getVertexCoords
//-----------------------------------------------------------------------


Matrix StdWedge12::getVertexCoords () const
{
  return vertexCoords ();
}


//-----------------------------------------------------------------------
//   evalShapeFunctions
//-----------------------------------------------------------------------


void StdWedge12::evalShapeFunctions

( const Vector&  f,
  const Vector&  u ) const

{
  JIVE_EVAL_WEDGE12_FUNCS ( f, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradients
//-----------------------------------------------------------------------


void StdWedge12::evalShapeGradients

( const Vector&  f,
  const Matrix&  g,
  const Vector&  u ) const

{
  JIVE_EVAL_WEDGE12_GRADS ( f, g, u );
}


//-----------------------------------------------------------------------
//   evalShapeGradGrads
//-----------------------------------------------------------------------


void StdWedge12::evalShapeGradGrads

( const Vector&  f,
  const Matrix&  g,
  const Matrix&  h,
  const Vector&  u ) const

{
  JIVE_EVAL_WEDGE12_GRADS2 ( f, g, h, u );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<StdShape> StdWedge12::makeNew

( const String&      geom,
  const Properties&  conf,
  const Properties&  props )

{
  return newInstance<Self> ();
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void StdWedge12::declare ()
{
  StdShapeFactory::declare ( "QuadraticLinear", GEOMETRY, & makeNew );
}


//=======================================================================
//   class Wedge12::myUtils_
//=======================================================================


class Wedge12::myUtils_ {
  public:

    typedef Ref<BShape>     ( *BShapeCtor )

    ( const String&           name,
      const Matrix&           ischeme,
      const Ref<SShape>&      sfuncs );

    typedef Ref<IShape>     ( *IShapeCtor )

    ( const String&           name,
      const MatrixVector&     ischeme,
      const Ref<SShape>&      sfuncs );


    static ShapeBoundary      getBoundary

    ( const String&           name,
      const MatrixVector&     ischeme,
      const Ref<SShape>&      xfuncs,
      const Ref<SShape>&      sfuncs );

    static ShapeBoundary      getBoundary

    ( const String&           name,
      BShapeCtor              tctor,
      BShapeCtor              qctor,
      const MatrixVector&     ischeme,
      const Topology&         topo,
      const Ref<SShape>&      sfuncs );

    static Ref<IShape>        getShape

    ( const String&           name,
      IShapeCtor              ctor,
      String&                 ischeme,
      String&                 tscheme,
      String&                 qscheme,
      const Ref<SShape>&      sfuncs );

    static Ref<IShape>        getShape

    ( const String&           name,
      IShapeCtor              ctor,
      const Properties&       conf,
      const Properties&       props,
      const StringVector&     ischeme,
      const String&           sfType );

};


//-----------------------------------------------------------------------
//   getBoundary
//-----------------------------------------------------------------------


ShapeBoundary Wedge12::myUtils_::getBoundary

( const String&        name,
  const MatrixVector&  ischeme,
  const Ref<SShape>&   xfuncs,
  const Ref<SShape>&   sfuncs )

{
  const idx_t    nodeCount = xfuncs->shapeFuncCount ();

  BShapeVector   bounds    ( BOUNDARY_COUNT );
  SBShapeVector  bxfuncs   ( BOUNDARY_COUNT );
  SBShapeVector  bsfuncs   ( BOUNDARY_COUNT );

  IdxVector      offsets   ( BOUNDARY_COUNT + 1 );
  IdxVector      inodes    =

    Wedge6::getBoundaryTopology().getColumnIndices ();

  Matrix         u         = StdWedge6::vertexCoords ();
  Matrix         v         = Matrix ( u[inodes] );

  String         baseName  = joinNames ( name, "boundary" );

  idx_t          ipos;
  int            j = 0;


  // The first triangular boundary.

  bxfuncs[0] = newInstance<StdBoundary> (
                 v[slice ( j, j + 3 )],
                 newInstance<StdTriangle3> (),
                 xfuncs
               );

  j += 3;

  // The three quadrilateral boundaries.

  bxfuncs[1] = newInstance<StdBoundary> (
                 v[slice ( j, j + 4 )],
                 newInstance<StdSquare4> (),
                 xfuncs
               );

  j += 4;

  for ( int i = 2; i < 4; i++, j += 4 ) {
    bxfuncs[i] = newInstance<StdBoundary> (
                   bxfuncs[1],
                   v[slice ( j, j + 4 )]
                 );
  }

  // The last triangular boundary.

  bxfuncs[4] = newInstance<StdBoundary> (
                 bxfuncs[0],
                 v[slice ( j, j + 3 )]
               );

  if ( sfuncs != nullptr && sfuncs != xfuncs ) {
    j = 0;

    bsfuncs[0] = newInstance<StdBoundary> (
                   v[slice ( j, j + 3 )],
                   newInstance<StdTriangle3> (),
                   sfuncs
                 );

    j += 3;

    bsfuncs[1] = newInstance<StdBoundary> (
                   v[slice ( j, j + 4 )],
                   newInstance<StdSquare4> (),
                   sfuncs
                 );

    j += 4;

    for ( int i = 2; i < 4; i++, j += 4 ) {
      bsfuncs[i] = newInstance<StdBoundary> (
                     bsfuncs[1],
                     v[slice ( j, j + 4 )]
                   );
    }

    bsfuncs[4] = newInstance<StdBoundary> (
                   bsfuncs[0],
                   v[slice ( j, j + 3 )]
                 );
  }

  for ( int i = 0; i < BOUNDARY_COUNT; i++ ) {
    bounds[i] = newInstance<ParametricSurface> (
                  baseName + String ( i ),
                  ischeme[i],
                  bxfuncs[i],
                  bsfuncs[i]
                );
  }

  inodes.resize ( BOUNDARY_COUNT * nodeCount );

  ipos = 0;

  for ( int i = 0; i < BOUNDARY_COUNT; i++ ) {
    offsets[i] = ipos;

    for ( idx_t j = 0; j < nodeCount; j++ ) {
      inodes[ipos++] = j;
    }
  }

  offsets[BOUNDARY_COUNT] = ipos;

  return ShapeBoundary (
           bounds,
           bxfuncs,
           Topology (
             jem::shape ( BOUNDARY_COUNT, nodeCount ),
             offsets,
             inodes
           )
         );
}


//-----------------------------------------------------------------------
//   getBoundary
//-----------------------------------------------------------------------


ShapeBoundary Wedge12::myUtils_::getBoundary

( const String&        name,
  BShapeCtor           tctor,
  BShapeCtor           qctor,
  const MatrixVector&  ischeme,
  const Topology&      topo,
  const Ref<SShape>&   sfuncs )

{
  BShapeVector   bounds   ( BOUNDARY_COUNT );
  SBShapeVector  bfuncs   ( BOUNDARY_COUNT );

  String         baseName = joinNames ( name, "boundary" );

  Matrix         u        = StdWedge6::vertexCoords ();
  Matrix         v;

  IdxVector      inodes;
  int            j = 0;


  if ( topo.size ( 1 ) == Wedge6::NODE_COUNT ) {
    inodes.ref ( topo.getColumnIndices() );
  }
  else {
    inodes.ref (
      Wedge6::getBoundaryTopology().getColumnIndices ()
    );
  }

  v.ref ( Matrix ( u[inodes] ) );

  // The first triangular boundary.

  bfuncs[0] = newInstance<StdBoundary> (
                v[slice ( j, j + 3 )],
                newInstance<StdTriangle3> (),
                sfuncs
              );

  j += 3;

  bounds[0] = tctor ( baseName  + '0',
                      ischeme[0], bfuncs[0] );

  // The three quadrilateral boundaries.

  bfuncs[1] = newInstance<StdBoundary> (
                v[slice ( j, j + 4 )],
                newInstance<StdSquare4> (),
                sfuncs
              );

  j += 4;

  for ( int i = 2; i < 4; i++, j += 4 ) {
    bfuncs[i] = newInstance<StdBoundary> (
                  bfuncs[1],
                  v[slice ( j, j + 4 )]
                );
  }

  for ( int i = 1; i < 4; i++ ) {
    bounds[i] = qctor ( baseName  + String ( i ),
                        ischeme[i], bfuncs[i] );
  }

  // The last triangular boundary.

  bfuncs[4] = newInstance<StdBoundary> (
                bfuncs[0],
                v[slice ( j, j + 3 )]
              );

  bounds[4] = tctor ( baseName  + '4',
                      ischeme[4], bfuncs[4] );

  return ShapeBoundary ( bounds, bfuncs, topo );
}


//-----------------------------------------------------------------------
//   getShape (given integration schemes)
//-----------------------------------------------------------------------


Ref<IShape> Wedge12::myUtils_::getShape

( const String&        name,
  IShapeCtor           ctor,
  String&              ischeme,
  String&              tscheme,
  String&              qscheme,
  const Ref<SShape>&   sfuncs )

{
  MatrixVector  s ( BOUNDARY_COUNT + 1 );

  s[0].ref ( StdWedge   ::getIntegrationScheme ( ischeme ) );
  s[1].ref ( StdTriangle::getIntegrationScheme ( tscheme ) );
  s[5].ref ( s[1] );
  s[2].ref ( StdSquare  ::getIntegrationScheme ( qscheme ) );
  s[3].ref ( s[2] );
  s[4].ref ( s[2] );

  return ctor ( name, s, sfuncs );
}


//-----------------------------------------------------------------------
//   getShape (given properties)
//-----------------------------------------------------------------------


Ref<IShape> Wedge12::myUtils_::getShape

( const String&        name,
  IShapeCtor           ctor,
  const Properties&    conf,
  const Properties&    props,
  const StringVector&  ischeme,
  const String&        sfType )

{
  JEM_ASSERT2 ( ischeme.size() == BOUNDARY_COUNT + 1,
                "invalid integration scheme" );

  MatrixVector  s ( BOUNDARY_COUNT + 1 );
  StringVector  t ( ischeme );

  Ref<SShape>   sfuncs;
  Ref<IShape>   shape;


  sfuncs = StdShapeFactory::findShapeFuncs (
             GEOMETRY,
             conf, props, sfType,
             StdWedge::RANK
           );

  if ( props.find ( t, PropNames::ISCHEME ) ) {
    if      ( t.size() == 1 ) {
      ischeme[0] = t[0];
    }
    else if ( t.size() == 3 ) {
      ischeme[0] = t[0];
      ischeme[1] = t[1];
      ischeme[2] = t[2];
      ischeme[3] = t[2];
      ischeme[4] = t[2];
      ischeme[5] = t[1];
    }
    else if ( t.size() == s.size() ) {
      ischeme = t;
    }
    else {
      props.propertyError (
        PropNames::ISCHEME,
        "array must have length 1, 3 or 6"
      );
    }
  }

  try {
    s[0].ref ( StdWedge   ::getIntegrationScheme ( ischeme[0] ) );
    s[1].ref ( StdTriangle::getIntegrationScheme ( ischeme[1] ) );
    s[5].ref ( StdTriangle::getIntegrationScheme ( ischeme[5] ) );

    for ( int i = 2; i < 5; i++ ) {
      s[i].ref ( StdSquare::getIntegrationScheme ( ischeme[i] ) );
    }
  }
  catch ( jem::IllegalInputException& ex ) {
    ex.setContext ( props.getContext ( PropNames::ISCHEME ) );
    throw;
  }

  conf.set ( PropNames::ISCHEME, ischeme );

  return ctor ( name, s, sfuncs );
}

//=======================================================================
//   class Wedge12
//=======================================================================


//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  Wedge12::TYPE_NAME = "Wedge12";
const char*  Wedge12::ISCHEME   = "3*1";
const char*  Wedge12::TSCHEME   = "3";
const char*  Wedge12::QSCHEME   = "1*1";


//-----------------------------------------------------------------------
//   getShape
//-----------------------------------------------------------------------


Ref<IShape> Wedge12::getShape

( const String&       name,
  const Ref<SShape>&  sfuncs )

{
  JEM_PRECHECK2 ( sfuncs == nullptr ||
                  sfuncs->getGeometry() == GEOMETRY,
                  "invalid shape functions" );

  String  ischeme = ISCHEME;
  String  tscheme = TSCHEME;
  String  qscheme = QSCHEME;

  return myUtils_::getShape ( name,    getShape,
                              ischeme, tscheme, qscheme, sfuncs );
}


Ref<IShape> Wedge12::getShape

( const String&       name,
  String&             ischeme,
  String&             tscheme,
  String&             qscheme,
  const Ref<SShape>&  sfuncs )

{
  JEM_PRECHECK2 ( sfuncs == nullptr ||
                  sfuncs->getGeometry() == GEOMETRY,
                  "invalid shape functions" );

  return myUtils_::getShape ( name,    getShape,
                              ischeme, tscheme, qscheme, sfuncs );
}


Ref<IShape> Wedge12::getShape

( const String&        name,
  const MatrixVector&  ischeme,
  const Ref<SShape>&   sfuncs )

{
  JEM_PRECHECK2 ( ischeme.size() == BOUNDARY_COUNT + 1,
                  "invalid integration scheme" );
  JEM_PRECHECK2 ( sfuncs == nullptr ||
                  sfuncs->getGeometry() == GEOMETRY,
                  "invalid shape functions" );


  Ref<SShape>  xshape = newInstance<StdWedge12> ();
  Ref<SShape>  sshape = ( sfuncs != nullptr ) ? sfuncs : xshape;

  return newInstance<ParametricVolume> (
           name,
           ischeme[0],
           myUtils_::getBoundary (
             name,
             & BoundaryTriangle6::getShape,
             & BoundaryQuad6    ::getShape,
             ischeme[slice ( 1, END )],
             getBoundaryTopology (),
             sshape
           ),
           xshape,
           sshape
         );
}


Ref<IShape> Wedge12::getShape

( const String&      name,
  const Properties&  conf,
  const Properties&  props )

{
  StringVector  ischeme ( BOUNDARY_COUNT + 1 );

  ischeme[0] = ISCHEME;
  ischeme[1] = TSCHEME;
  ischeme[2] = QSCHEME;
  ischeme[3] = QSCHEME;
  ischeme[4] = QSCHEME;
  ischeme[5] = TSCHEME;

  return myUtils_::getShape ( name,
                              & getShape,
                              conf .makeProps ( name ),
                              props.findProps ( name ),
                              ischeme,
                              "QuadraticLinear" );
}


//-----------------------------------------------------------------------
//   getBoundaryTopology
//-----------------------------------------------------------------------


Topology Wedge12::getBoundaryTopology ()
{
  IdxVector  offsets ( BOUNDARY_COUNT + 1 );
  IdxVector  indices ( 3 * 6 + 2 * 6 );
  idx_t      i;


  i = 0;

  offsets[i++] = 0;
  offsets[i++] = 6;
  offsets[i++] = 12;
  offsets[i++] = 18;
  offsets[i++] = 24;
  offsets[i++] = 30;

  i = 0;

  // First boundary

  indices[i++] = 0;
  indices[i++] = 5;
  indices[i++] = 4;
  indices[i++] = 3;
  indices[i++] = 2;
  indices[i++] = 1;

  // Second boundary

  indices[i++] = 0;
  indices[i++] = 1;
  indices[i++] = 2;
  indices[i++] = 8;
  indices[i++] = 7;
  indices[i++] = 6;

  // Third boundary

  indices[i++] = 2;
  indices[i++] = 3;
  indices[i++] = 4;
  indices[i++] = 10;
  indices[i++] = 9;
  indices[i++] = 8;

  // Fourth boundary

  indices[i++] = 4;
  indices[i++] = 5;
  indices[i++] = 0;
  indices[i++] = 6;
  indices[i++] = 11;
  indices[i++] = 10;

  // Fifth boundary

  indices[i++] = 6;
  indices[i++] = 7;
  indices[i++] = 8;
  indices[i++] = 9;
  indices[i++] = 10;
  indices[i++] = 11;

  /*
           w
           ^
           |
           6
         ,/|`\
       ,7  |  `11
     ,/    |    `\
    8------9------10
    |      |      |
    |    ,/|`\    |
    |  ,/  |  `\  |
    |,/    |    `\|
   ,|      |      |\
  ,/|      0      | `\
  u |    ,/ `\    |   v
    |  ,1     `5  |
    |,/         `\|
    2------3------4

  */
  return Topology ( jem::shape ( BOUNDARY_COUNT, NODE_COUNT ),
                    offsets,
                    indices );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------

void Wedge12::declare ()
{
  IShapeFactory::declare ( TYPE_NAME,             & getShape );
  IShapeFactory::declare ( "jive::geom::Wedge12", & getShape );
}

JIVE_END_PACKAGE ( geom )
