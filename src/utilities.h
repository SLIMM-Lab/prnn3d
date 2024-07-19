#ifndef UTILITIES_H
#define UTILITIES_H

#include <jem/base/Array.h>
#include <jive/Array.h>
#include <jem/base/Tuple.h>

#include "power.h"

using jive::Vector;
using jive::Matrix;
using jive::IdxVector;
using jem::Tuple;
using jem::idx_t;

//-----------------------------------------------------------------------
//   typedefs
//-----------------------------------------------------------------------

// A pointer to a function that computes the spatial derivatives of
// the interpolation matrix. This is the so-called B-matrix.

typedef void        (*ShapeGradsFunc)

  ( const Matrix&       b,
    const Matrix&       g );

// Similar stuff, from the shape functions N_i, compute the shape matrix N

typedef void        (*ShapeFuncsFunc)

  ( const Matrix&       sfuncs,
    const Vector&       n );

// Similar stuff, from computing grad(U)

typedef void        (*GradUFunc)

  ( const Matrix&       gradU,
    const Matrix&       gradN,
    const Vector&       elemDisp );

//-----------------------------------------------------------------------
//   constants
//-----------------------------------------------------------------------

// An integer array that maps the number of spatial dimensions (1, 2,
// or 3) to the number of strain/stress components.

extern const idx_t    STRAIN_COUNTS[4];


//-----------------------------------------------------------------------
//   public functions
//-----------------------------------------------------------------------

// These functions compute the B-matrix given an interpolation matrix.

void                  get1DShapeGrads

  ( const Matrix&       b,
    const Matrix&       g );

void                  get2DShapeGrads

  ( const Matrix&       b,
    const Matrix&       g );

void                  get3DShapeGrads

  ( const Matrix&       b,
    const Matrix&       g );

// A function that returns a pointer to a function that computes the
// B-matrix given the number of spatial dimensions.

ShapeGradsFunc        getShapeGradsFunc

  ( idx_t               rank );

// -----------------------------------------------------------------------
//   matrix of shape functions  N
// -----------------------------------------------------------------------

void                  get1DShapeFuncs

  ( const Matrix&       sfuncs,
    const Vector&       n );

void                  get2DShapeFuncs

  ( const Matrix&       sfuncs,
    const Vector&       n );

void                  get3DShapeFuncs

  ( const Matrix&       sfuncs,
    const Vector&       n );

// A function that returns a pointer to a function that computes the
// B-matrix given the number of spatial dimensions.

ShapeFuncsFunc        getShapeFuncsFunc

  ( idx_t               rank );

// -----------------------------------------------------------------------
//   computing grad(u) from grad(N) and elemDisp
// -----------------------------------------------------------------------

// gradU = du_i/dx_j,  size: ( rank,rank )
// gradN = dN_i/dx_j,  size: ( rank,nodeCount )
// elemDisp = u        numbering: (0x,0y,1x,1y...nx,ny)

void                  get1DGradU

  ( const Matrix&       gradU,
    const Matrix&       gradN,
    const Vector&       elemDisp );

void                  get2DGradU

  ( const Matrix&       gradU,
    const Matrix&       gradN,
    const Vector&       elemDisp );

void                  get3DGradU

  ( const Matrix&       gradU,
    const Matrix&       gradN,
    const Vector&       elemDisp );

// A function that returns a pointer to a function that computes the
// B-matrix given the number of spatial dimensions.

GradUFunc             getGradUFunc

  ( idx_t               rank );

// -----------------------------------------------------------------------
//   other utility functions
// -----------------------------------------------------------------------

// Solving linear equation of one variable
// a*x + b = 0

void                  solveLinearEqua

  ( double&      ans,
    const double a,
    const double b );

// Solving quadratic equation
// a*x^2 + b*x + c = 0

void                  solveQuadEqua

  ( Vector&      ans,
    const double a,
    const double b,
    const double c );

// Solving cubic equation
// a*x^3 + b*x^2 + c*x + d = 0
// only for functions with real roots
// returns the number of roots

idx_t                 solveCubicEqua

  ( Tuple<double,3>& ans,
    const double     a,
    const double     b,
    const double     c,
    const double     d);

// Invert 2x2 matrix (no size check performed!)

void                  invert2x2

  ( const Matrix& mat );

// compute determinant for 1x1, 2x2 or 3x3 matrix 

double                determinant

  ( const Matrix& mat );

// Compute the ramp function <x> = 1/2(x+x)

double                  evalMcAuley

  ( const double );

double                  evalHeaviside

  ( const double );

// compute the trace of a matrix

void trace (const Tuple<double,3,3>& m, double& tr );

// compute the second invariant of a matrix

void secInv (const Tuple<double,3,3>& m, double& inv );

double                  powerMethod

  ( const Matrix&           mat       );

// Computes the QR decomposition of 'mat' by using a 
// series of Householder reflections 

void                    QR

  (       Matrix&           Q,
          Matrix&           R,
    const Matrix&           mat       ); 

void                    polarDecomposition

  (       Matrix&           R,
          Matrix&           U,
    const Matrix&           F       ); 


void                    kronProd

  (       Matrix&           prod,
    const Matrix&           A,
    const Matrix&           B       ); 

#endif

