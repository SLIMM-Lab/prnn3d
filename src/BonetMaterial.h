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
 *  Reference: A simple orthotropic, transversely isotropic hyperelastic
 *             constitutive equation for large strain computations
 *             Bonet and Burton, Comput Methods Appl Mech Engrg, 1997 
 *
 *
 *  Date: November 2019
 *
 */

#ifndef BONET_MATERIAL_H
#define BONET_MATERIAL_H

#include <jem/base/String.h>
#include <jem/base/System.h>
#include <jem/util/Flex.h>

#include "Material.h"

using jem::String;
using jem::System;
using jem::util::Flex;
using jem::io::endl;

// =======================================================
//  class BonetMaterial
// =======================================================


class BonetMaterial : public Material
{
 public:

  typedef BonetMaterial  Self;
  typedef Material       Super;

  static const char*      YOUNG_1_PROP;
  static const char*      YOUNG_2_PROP;
  static const char*      POISSON_1_PROP;
  static const char*      POISSON_2_PROP;
  static const char*      SHEAR_PROP;
  static const char*      FIBDIR_PROP;

  explicit                BonetMaterial

    ( const idx_t           rank,
      const Properties&     globdat );

                         ~BonetMaterial ();

  virtual void            configure

    ( const Properties&     props );

  virtual void            getConfig

    ( const Properties&     conf ) const;

  virtual void            update

    ( const Vector&         stress,
      const Matrix&         stiff,
      const Matrix&         df,
      const idx_t           ip );

  virtual void            updateWriteTable

  ( const Vector&         stress,
    const Matrix&         stiff,
    const Matrix&         df,
    const idx_t           ip );

  // Tuple<double,6>         fill3DStrain

  //   ( const Vector&         v3 )           const;

  // Tuple<double,6>         fill3DStress

  //   ( const Vector&         v3 )           const;
  
  virtual void            getHistory

  ( const Vector&         hvals,
    const idx_t           mpoint ) const;

  virtual void            getInitHistory

  ( const Vector&         hvals,
    const idx_t           mpoint ) const;

  virtual idx_t                getHistoryCount ()    const override;

  virtual void                setHistory

    ( const Vector&             hvals,
      const idx_t               mpoint );  

  Ref<Material>           clone           () const;

  virtual void            commit ();

  virtual void            allocPoints     
    
    ( const idx_t           count );

  virtual void            allocPoints

    ( const idx_t           count,
      const Matrix&         transfer,
      const IdxVector&      oldPoints );

  inline virtual idx_t    pointCount      () const;


 private:
  
 protected:


  double                  Ea_;
  double                  E_;  
  double                  Ga_;
  double                  nua_;
  double                  nu_;
  bool                    nn_;

  Vector                  fibdir_;

  // history variables

  class                   Hist_
  {
    public:

                            Hist_

      ( const idx_t           rank );

                            Hist_

      ( const Hist_&          h );

    void  toVector ( const Vector& vec ) const;
      
    Matrix                  F;
  };
  
  Flex<Hist_>             initHist_; 
  Flex<Hist_>             preHist_;    // history of previous load step
  Flex<Hist_>             newHist_;    // history of current iteration
  Flex<Hist_>*            latestHist_; // points to latest history
};

//-----------------------------------------------------------------------
//   pointCount
//-----------------------------------------------------------------------

inline idx_t  BonetMaterial::pointCount  () const

{
  return preHist_.size();
}


#endif 
