#ifndef VOIGT_UTILITIES_H
#define VOIGT_UTILITIES_H

#include <jem/base/Tuple.h>
#include <jive/Array.h>


using jem::idx_t;
using jem::Tuple;
using jive::Vector;
using jive::Matrix;

namespace voigtUtilities
{
  double                  getPrincipalDirection2D

    ( const Vector& stress );

  void                    tensor2VoigtStress3D

    ( Tuple<double,6> vec,
      const Matrix&   tens );

  void                    tensor2VoigtStress

    ( const Vector&   vec,
      const Matrix&   tens );


  void                    tensor2VoigtStrain

    ( const Vector& vec,
      const Matrix& tens );

  void                    voigt2TensorStress

    ( const Matrix& tens,
      const Vector& vec  );

  void                    voigt2TensorStrain

    ( const Matrix& tens,
      const Vector& vec  );

  void                    multiplyVoigtNormal

    ( const Vector& sign,
      const Vector& sig,
      const Vector& n );

  void                    multiplyVoigtNormal

    ( const Matrix& dbn,
      const Matrix& db,
      const Vector& n );

  void                   getVectorRotationMat

    ( const Matrix& q,
      const Vector& normal );

  void                   getStressRotationMat

    ( const Matrix& q,
      const Vector& normal );

  void                   getStrainRotationMat

    ( const Matrix& q,
      const Vector& normal );

  void                    transformStress2D

    ( const Vector&         sign,
      const double          co,
      const double          si,
      const Vector&         sigx );

  void                    transformStrain2D

    ( const Vector&         epsn,
      const double          co,
      const double          si,
      const Vector&         epsx );

  Tuple<idx_t,2>          tensorIndex

    ( const idx_t   iv,
      const idx_t   rank );

  void                    tensorIndex

    (       idx_t&  i,
            idx_t&  j,
      const idx_t   iv,
      const idx_t   rank );

  idx_t                   voigtIndex

    ( const idx_t   i,
      const idx_t   j,
      const idx_t   rank );


  //-----------------------------------------------------------------------
  //   engng2tensorial
  //-----------------------------------------------------------------------

  void                    engng2tensorial

    ( const Vector&         tens,
      const Vector&         eng );

  void                    engng2tensorial

    ( const Vector&         eng2tens );

};

#endif
