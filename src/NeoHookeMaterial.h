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

#ifndef NEOHOOKEMATERIAL_H
#define NEOHOOKEMATERIAL_H

#include <jem/base/String.h>
#include <jem/util/Flex.h>

#include "Material.h"


using jem::String;
using jem::Tuple;
using jem::util::Flex;


// =======================================================
//  class NeoHookeMaterial
// =======================================================

// This class implements an isotropic elastic material

class NeoHookeMaterial : public Material
{
 public:

  static const char*      LAMBDA_PROP;
  static const char*      MU_PROP;
  static const char*      RHO_PROP;
  static const char*      STATE_PROP;

  enum ProblemType {
    PlaneStrain,
    PlaneStress,
    AxiSymmetric
  };

  explicit                NeoHookeMaterial

    ( idx_t                 rank,
      const Properties&     globdat );

  virtual void            configure

    ( const Properties&     props );

  virtual void            getConfig

    ( const Properties&     conf )         const;

  virtual void            update

    ( const Vector&         stress,
      const Matrix&         stiff,
      const Matrix&         df,
      idx_t                 ipoint );

  Ref<Material>           clone           () const;

  virtual void            updateWriteTable

    ( const Vector&         stress,
      const Matrix&         stiff,
      const Matrix&         df,
      idx_t                 ipoint );

  virtual void            commit ();

  virtual void            allocPoints

    ( const idx_t           count );

//   virtual void            getHistory

// ( const Vector&         hvals,
//   const idx_t           mpoint ) const;

// Tuple<double,6>         fill3DStrain

//     ( const Vector&         v3 )           const;

// Tuple<double,6>         fill3DStress

//   ( const Vector&         v3 )             const;

 protected:

  virtual                ~NeoHookeMaterial   ();

 protected:

  double                  mu_;
  double                  lambda_;

  String                  stateString_;
  ProblemType             state_;

  // history variables

  class                   Hist_
  {
    public:

                            Hist_

      ( const idx_t           rank );

                            Hist_

      ( const Hist_&          h );

      // void   toVector ( const Vector& vec ) const;

      Matrix  F;

      double  eqps;
  };

  Flex<Hist_>             preHist_;    // history of previous load step
  Flex<Hist_>             newHist_;    // history of current iteration
  Flex<Hist_>*            latestHist_; // points to latest history
};



#endif 
