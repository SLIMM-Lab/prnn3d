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

#ifndef ORTHOTROPIC_UL_MATERIAL_H
#define ORTHOTROPIC_UL_MATERIAL_H

#include <jem/base/String.h>
#include <jem/base/System.h>
#include <jem/util/Flex.h>

#include "Material.h"

using jem::String;
using jem::System;
using jem::util::Flex;
using jem::io::endl;

// =======================================================
//  class OrthotropicULMaterial
// =======================================================

// for transversely isotropic material, input:
//
//   - young1
//   - young2
//   - poisson12
//   - poisson23
//   - shear12
//   - theta
//
// for 2D input
//
//   - young1
//   - young2
//   - poisson12
//   - shear12
//   - theta

class OrthotropicULMaterial : public Material
{
 public:

  typedef OrthotropicULMaterial  Self;
  typedef Material               Super;
  typedef Tuple<double,6>        Vec6;
  typedef Tuple<double,6,6>      Mat66;

  static const char*      YOUNG_1_PROP;
  static const char*      YOUNG_2_PROP;
  static const char*      YOUNG_3_PROP;
  static const char*      ALPHA_1_PROP;
  static const char*      ALPHA_2_PROP;
  static const char*      ALPHA_3_PROP;
  static const char*      POISSON_12_PROP;
  static const char*      POISSON_23_PROP;
  static const char*      POISSON_31_PROP;
  static const char*      SHEAR_12_PROP;
  static const char*      SHEAR_23_PROP;
  static const char*      SHEAR_31_PROP;
  static const char*      STATE_PROP;
  static const char*      THETA_PROP;

  explicit                OrthotropicULMaterial

    ( const idx_t           rank,
      const Properties&     globdat );

                         ~OrthotropicULMaterial ();

  virtual void            configure

    ( const Properties&     props );

  virtual void            getConfig

    ( const Properties&     conf ) const;

  void                    transform

    ( const Vector&         strain,
      Vector&               mStrain) const;

  void                    transformInv

    ( const Vector&         mStress,
      Vector&               stress) const;

  void                    changeFrame

    ( const Matrix&         mStiff,
      Matrix&               stiff) const;

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

  Ref<Material>           clone           () const;

  virtual void            commit ();

  virtual void            allocPoints     
    
    ( const idx_t           count );

  virtual void            allocPoints

    ( const idx_t           count,
      const Matrix&         transfer,
      const IdxVector&      oldPoints );

  inline virtual idx_t    pointCount      () const;

  virtual void            updateThermStrain

   ( const Properties&   params );

 private:

  virtual void            linearUpdate_

    ( const Vector&         stress,
      const Matrix&         stiff,
      const Vector&         strain ); 

  void                    computeStiffMat_();
  void                    computeTransformMats_();
  
 protected:


  double                  young1_;
  double                  young2_;
  double                  young3_;
  double                  alpha1_;
  double                  alpha2_;
  double                  alpha3_;
  double                  poisson12_;
  double                  poisson23_;
  double                  poisson31_;
  double                  shear12_;
  double                  shear23_;
  double                  shear31_;
  double                  theta_;
  Matrix                  stiffMat_;
  Matrix                  materialStiffMat_;
  Matrix                  materialCompMat_;
  Matrix                  transformMat_;
  Matrix                  transformMatInv_;
  Matrix                  tt_;
	Vector                  thermStrain_;
  String                  state_;

  // history variables

  class                   Hist_
  {
    public:

                            Hist_

      ( const idx_t           rank );

                            Hist_

      ( const Hist_&          h );

    Matrix                  F;
  };

  Flex<Hist_>             preHist_;    // history of previous load step
  Flex<Hist_>             newHist_;    // history of current iteration
  Flex<Hist_>*            latestHist_; // points to latest history
};

//-----------------------------------------------------------------------
//   pointCount
//-----------------------------------------------------------------------

inline idx_t  OrthotropicULMaterial::pointCount  () const

{
  return preHist_.size();
}


#endif 
