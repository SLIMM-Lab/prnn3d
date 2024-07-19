/*
 *  TU Delft 
 *
 *  Iuri Barcelos, July 2019
 *
 *  Variance-based normalizer:
 *
 *  Forward:
 *
 *  x' = (x - mean)/sdev
 *
 *  Backward:
 *
 *  x = mean + x' sdev
 *
 */

#ifndef VARIANCENORMALIZER_H
#define VARIANCENORMALIZER_H

#include "Normalizer.h"

//-----------------------------------------------------------------------
//   class VarianceNormalizer
//-----------------------------------------------------------------------

class VarianceNormalizer : public Normalizer
{
 public:
  
                  VarianceNormalizer ();

		  VarianceNormalizer
 
    ( const String& fname              );
 
  virtual Vector  normalize 
 
    ( const Vector& vec                );
 
  virtual Vector  denormalize
 
    ( const Vector& vec                );

  virtual Vector  getJacobianFactor
 
    ( const Vector& vec                );

  virtual void    update

    ( const Cubix&  inp                );

  virtual void    write

    ( const String& fname              );

 private:

  Vector          means_;
  Vector          sdevs_;

};

#endif
