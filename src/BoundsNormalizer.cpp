/*
 *  TU Delft 
 *
 *  Iuri Barcelos, December 2019
 *
 */

#include <jem/base/IllegalInputException.h>
#include <jem/base/Error.h>
#include <jem/base/System.h>
#include <jem/io/FileReader.h>
#include <jem/io/FileInputStream.h>
#include <jem/io/InputStreamReader.h>
#include <jem/io/PrintWriter.h>
#include <jem/io/FileWriter.h>
#include <jem/util/Tokenizer.h>

#include "BoundsNormalizer.h"

using namespace jem;

//-----------------------------------------------------------------------
//   class BoundsNormalizer
//-----------------------------------------------------------------------

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------

BoundsNormalizer::BoundsNormalizer ()
{
  upper_.resize ( 0 );
  lower_.resize ( 0 );
}

BoundsNormalizer::BoundsNormalizer

  ( const String& fname )

{
  using jem::IllegalInputException;
  using jem::util::Tokenizer;
  using jem::io::FileInputStream;
  using jem::io::InputStreamReader;

  Ref<Tokenizer> tizer = 
    newInstance<Tokenizer> ( 
      newInstance<InputStreamReader> ( 
        newInstance<FileInputStream> ( fname ) ) );

  tizer->nextToken();
  tizer->nextToken();

  idx_t count = tizer->getInteger();

  upper_.resize ( count );
  lower_.resize ( count );
  upper_ = 0.0; lower_ = 0.0;

  System::out() << "Normalizer file: " << fname << "\n";

  for ( idx_t i = 0; i < count; ++i )
  {
    tizer->nextToken();

    upper_[i] = tizer->getFloat();

    tizer->nextToken();

    lower_[i] = tizer->getFloat();
  }

  System::out() << "lower " << lower_ << "\n";
  System::out() << "upper " << upper_ << "\n";
}

//-----------------------------------------------------------------------
//   normalize
//-----------------------------------------------------------------------

Vector BoundsNormalizer::normalize

  ( const Vector& vec )

{
//  System::out() << "vec " << vec << "\n";
  Vector ret ( 2. * ( vec - lower_ ) / ( upper_ - lower_ ) - 1. );
//  System::out() << "ret " << ret << "\n";
 
  return ret;
}

//-----------------------------------------------------------------------
//   denormalize
//-----------------------------------------------------------------------

Vector BoundsNormalizer::denormalize

  ( const Vector& vec )

{
//  System::out() << "vecde " << vec << "\n";
  Vector ret ( ( vec + 1. ) * ( upper_ - lower_ ) / 2. + lower_ );
//  System::out() << "retde " << ret << "\n";

  return ret;
}

//-----------------------------------------------------------------------
//   getJacobianFactor
//-----------------------------------------------------------------------

Vector BoundsNormalizer::getJacobianFactor

  ( const Vector& vec )

{
  return Vector ( 1. / ( upper_ - lower_ ) );
}

//-----------------------------------------------------------------------
//   getDenormFactor
//-----------------------------------------------------------------------

Vector BoundsNormalizer::getDenormFactor

  ( const Vector& vec )

{
  return Vector ( ( upper_ - lower_ ) / 2. );
}

//-----------------------------------------------------------------------
//   getNormFactor
//-----------------------------------------------------------------------

Vector BoundsNormalizer::getNormFactor

  ( const Vector& vec )

{
  return Vector ( 2. / ( upper_ - lower_ ) );
}

//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------

void BoundsNormalizer::update

  ( const Cubix& inp )

{
  System::out() << "update normalizer bounds\n";
 
  idx_t inpsize = inp.size(1);
  idx_t smpsize = inp.size(2);
  idx_t seqsize = inp.size(0);

  upper_.resize ( inpsize );
  lower_.resize ( inpsize );
  upper_ = 0.0; lower_ = 0.0;

  Vector bounds ( 2 );
  bounds = 0.0;

  for ( idx_t j = 0; j < inpsize; ++j )
  {
    bounds[0] = max ( inp(ALL,j,ALL) );
    bounds[1] = abs ( min ( inp(ALL,j,ALL) ) );

    upper_[j] = max ( bounds );
    lower_[j] = -upper_[j]; //min ( bounds );

    if ( jem::numeric::abs ( upper_[j] - lower_[j] ) < 1.e-20 )
    {
      lower_[j] = -1.0;

      if ( jem::numeric::abs ( upper_[j] ) < 1.e-20 ) 
      {
	upper_[j] = 1.0;
      }
    }
  }

  System::out() << "Upper " << upper_ << "\nLower " << lower_ << "\n";
}

//-----------------------------------------------------------------------
//   write
//-----------------------------------------------------------------------

void BoundsNormalizer::write

  ( const String& fname )

{
  using jem::io::PrintWriter;
  using jem::io::FileWriter;

  idx_t count = upper_.size();

  Ref<PrintWriter> fout = newInstance<PrintWriter> (
    newInstance<FileWriter> ( fname + ".nml" ) );

  fout->nformat.setFractionDigits ( 10 );

  System::out() << "Writing nml file. Upper " << upper_ << "\nLower " << lower_ << "\n";

  *fout << "bounds " << count << "\n";

  for ( idx_t i = 0; i < count; ++i )
  {
    *fout << upper_[i] << " " << lower_[i] << "\n";
  }
}

void BoundsNormalizer::getBounds

   ( Vector& upper,
     Vector& lower   )
{
  upper.resize ( upper_.size(0) );
  lower.resize ( lower_.size(0) );

  upper = 0.0;
  lower = 0.0;

  upper = upper_.clone();
  lower = lower_.clone();

}
