/*
 *  TU Delft 
 *
 *  Iuri Barcelos, July 2019
 *
 *  Base class for ANN input/output normalization
 *
 */

#ifndef NORMALIZER_H
#define NORMALIZER_H

#include <jem/base/Object.h>
#include <jem/base/Ref.h>
#include <jem/base/Array.h>
#include <jem/base/array/tensor.h>
#include <jem/numeric/algebra/utilities.h>
#include <jem/util/Properties.h>

#include <jive/Array.h>

using jem::Object;
using jem::Ref;
using jem::Array;
using jem::String;
using jem::util::Properties;

using jive::Matrix;
using jive::Cubix;
using jive::Vector;

//-----------------------------------------------------------------------
//   class Normalizer
//-----------------------------------------------------------------------

class Normalizer : public Object
{
 public:
  
                  Normalizer ();

		  Normalizer

    ( const String& fname     );
 
  virtual Vector  normalize 
 
    ( const Vector& vec       );
 
  virtual Vector  denormalize
 
    ( const Vector& vec       );

  virtual Vector  getJacobianFactor  

    ( const Vector& vec       );

  virtual Vector  getDenormFactor  

    ( const Vector& vec       );

  virtual Vector  getNormFactor  

    ( const Vector& vec       );

  virtual void    update

    ( const Cubix&  inp       );

  virtual void    write

    ( const String& fname     );

  virtual void   getBounds

    ( Vector& upper,
      Vector& lower           );

};

//-----------------------------------------------------------------------
//   newNormalizer
//-----------------------------------------------------------------------

Ref<Normalizer> newNormalizer

  ( const String& type );

#endif
