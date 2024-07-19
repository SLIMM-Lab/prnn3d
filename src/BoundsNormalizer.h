/*
 *  TU Delft 
 *
 *  Iuri Barcelos, December 2019
 *
 */

#ifndef BOUNDSNORMALIZER_H
#define BOUNDSNORMALIZER_H

#include "Normalizer.h"

//-----------------------------------------------------------------------
//   class BoundsNormalizer
//-----------------------------------------------------------------------

class BoundsNormalizer : public Normalizer
{
 public:
  
                  BoundsNormalizer ();

		  BoundsNormalizer

    ( const String& fname            );
 
  virtual Vector  normalize 
 
    ( const Vector& vec              );
 
  virtual Vector  denormalize
 
    ( const Vector& vec              );

  virtual Vector  getJacobianFactor
 
    ( const Vector& vec              );

  virtual Vector  getDenormFactor
 
    ( const Vector& vec              );

  virtual Vector  getNormFactor
 
    ( const Vector& vec              );

  virtual void    update

    ( const Cubix&  inp              );

  virtual void    write

    ( const String& fname            );

  virtual void    getBounds
 
    ( Vector& upper, Vector& lower );
 
 private:

  Vector upper_;
  Vector lower_;

};

#endif
