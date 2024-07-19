/*
 *  TU Delft 
 *
 *  Iuri Barcelos, July 2019
 *
 *  Base class for ANN input/output normalization
 *
 */

#include <jem/base/Error.h>
#include <jem/base/System.h>
#include <jem/io/FileReader.h>
#include <jem/util/StringUtils.h>

#include "Normalizer.h"
#include "SigmoidNormalizer.h"
#include "VarianceNormalizer.h"
#include "BoundsNormalizer.h"

using jem::idx_t;
using jem::System;
using jem::newInstance;
using jem::Error;
using jem::io::FileReader;
using jem::util::StringUtils;

using jive::StringVector;

//-----------------------------------------------------------------------
//   newNormalizer
//-----------------------------------------------------------------------

Ref<Normalizer> newNormalizer

  ( const String& name )

{
  Ref<Normalizer> nml;

  System::out() << "Normalizer " << name << "\n";

  StringVector strings = StringUtils::split ( name, '.' );

  if ( strings.size() == 1 )
  {
    if      ( name.equalsIgnoreCase ( "none" ) )
    {
      nml = newInstance<Normalizer> ();
    }
    else if ( name.equalsIgnoreCase ( "variance" ) )
    {
      nml = newInstance<VarianceNormalizer> ();  
    }
    else if ( name.equalsIgnoreCase ( "sigmoid" ) )
    {
      nml = newInstance<SigmoidNormalizer> ();
    }
    else if ( name.equalsIgnoreCase ( "bounds" ) )
    {
      nml = newInstance<BoundsNormalizer> ();
    }
    else
    {
      throw Error ( JEM_FUNC, "Unknown normalizer type" );
    }
  }
  else if ( strings.size() >= 2 && strings[strings.size()-1] == "nml" )
  {
    Ref<FileReader> in    = newInstance<FileReader> ( name );

    StringVector fst = StringUtils::split ( in->readLine() );

    String type ( fst[0] );

    System::out() << "Normalizer: " << type << "\n";

    if      ( type.equalsIgnoreCase ( "none" ) )
    {
      nml = newInstance<Normalizer> ( name );
    }
    else if ( type.equalsIgnoreCase ( "variance" ) )
    {
      nml = newInstance<VarianceNormalizer> ( name );
    }
    else if ( type.equalsIgnoreCase ( "sigmoid"  ) )
    {
      nml = newInstance<SigmoidNormalizer>  ( name );
    }
    else if ( type.equalsIgnoreCase ( "bounds"  ) )
    {
      System::out() << "File name " << name << "\n";
      nml = newInstance<BoundsNormalizer>  ( name );
    }
    else
    {
      throw Error ( JEM_FUNC, "Unknown normalizer type" );
    }
  }
  else
  {
    throw Error ( JEM_FUNC, "Unknown normalizer data file format" );
  }

  return nml;
}

//-----------------------------------------------------------------------
//   class Normalizer
//-----------------------------------------------------------------------

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------

Normalizer::Normalizer ()
{}

Normalizer::Normalizer

  ( const String& fname )
{}

//-----------------------------------------------------------------------
//   normalize
//-----------------------------------------------------------------------

Vector Normalizer::normalize

  ( const Vector& vec )

{
  return vec;
}

//-----------------------------------------------------------------------
//   denormalize
//-----------------------------------------------------------------------

Vector Normalizer::denormalize

  ( const Vector& vec )

{
  return vec;
}

//-----------------------------------------------------------------------
//   getJacobianFactor
//-----------------------------------------------------------------------

Vector Normalizer::getJacobianFactor

  ( const Vector& vec )

{
  Vector ret ( vec.shape() );
  ret = 1.0;

  return ret;
}

//-----------------------------------------------------------------------
//   getDenormFactor
//-----------------------------------------------------------------------

Vector Normalizer::getDenormFactor

  ( const Vector& vec )

{
  Vector ret ( vec.shape() );
  ret = 1.0;

  return ret; 
}

//-----------------------------------------------------------------------
//   getNormFactor
//-----------------------------------------------------------------------

Vector Normalizer::getNormFactor

  ( const Vector& vec )

{
  Vector ret ( vec.shape() );
  ret = 1.0;

  return ret;
}

//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------

void Normalizer::update

  ( const Cubix& vec )

{
}

//-----------------------------------------------------------------------
//   write
//-----------------------------------------------------------------------

void Normalizer::write

  ( const String& fname )

{
}

void Normalizer::getBounds
 
  ( Vector& upper,
    Vector& lower )
{
}
