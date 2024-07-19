/*
 *  TU Delft 
 *
 *  Iuri Barcelos, July 2019
 *
 *  Normalizer for sigmoid units:
 *
 *  Forward:
 *
 *  x'  = x - mean
 *  x'' = ( x' - min(x') ) / ( max(x') - min(x') )
 *
 *  Backward:
 *
 *  x' = x'' ( max(x') - min(x') ) + min(x') 
 *  x  = x' + mean
 *
 *  Reference:
 *  
 *  Gonzalez, F. J. and Balajewicz, M. (2018). Deep convolutional 
 *  autoencoders for learning low-dimensional feature dynamics of
 *  fluid systems. arXiv:1808.01346v2
 *
 */

#ifndef SIGMOIDNORMALIZER_H
#define SIGMOIDNORMALIZER_H

#include "Normalizer.h"

//-----------------------------------------------------------------------
//   class SigmoidNormalizer
//-----------------------------------------------------------------------

class SigmoidNormalizer : public Normalizer
{
 public:
  
                  SigmoidNormalizer ();

		  SigmoidNormalizer

    ( const String& fname            );
 
  virtual Vector  normalize 
 
    ( const Vector& vec              );
 
  virtual Vector  denormalize
 
    ( const Vector& vec              );

  virtual Vector  getJacobianFactor
 
    ( const Vector& vec              );

  virtual void    update

    ( const Cubix&  inp              );

  virtual void    write

    ( const String& fname            );
 
 private:

  Vector means_;
  Vector upper_;
  Vector lower_;

};

#endif
