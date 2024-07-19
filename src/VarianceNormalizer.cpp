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

#include <jem/base/Error.h>
#include <jem/base/System.h>
#include <jem/base/IllegalInputException.h>
#include <jem/io/FileReader.h>
#include <jem/io/FileInputStream.h>
#include <jem/io/InputStreamReader.h>
#include <jem/io/PrintWriter.h>
#include <jem/io/FileWriter.h>
#include <jem/io/FileFlags.h>
#include <jem/util/Tokenizer.h>

#include "VarianceNormalizer.h"

using namespace jem;

//-----------------------------------------------------------------------
//   class VarianceNormalizer
//-----------------------------------------------------------------------

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------

VarianceNormalizer::VarianceNormalizer ()
{
  means_.resize ( 0 );
  sdevs_.resize ( 0 );
}

VarianceNormalizer::VarianceNormalizer

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

  idx_t count = tizer->getInteger();

  means_.resize ( count );
  sdevs_.resize ( count );
  means_ = 0.0; sdevs_ = 0.0;

  for ( idx_t i = 0; i < count; ++i )
  {
    tizer->nextToken();

    means_[i] = tizer->getFloat();
    tizer->nextToken();

    sdevs_[i] = tizer->getFloat();
  }
}

//-----------------------------------------------------------------------
//   normalize
//-----------------------------------------------------------------------

Vector VarianceNormalizer::normalize

  ( const Vector& vec )

{
  Vector ret ( ( vec - means_ ) / sdevs_ );

  return ret;
}

//-----------------------------------------------------------------------
//   denormalize
//-----------------------------------------------------------------------

Vector VarianceNormalizer::denormalize

  ( const Vector& vec )

{
  Vector ret ( means_ + vec * sdevs_ );

  return ret;
}

//-----------------------------------------------------------------------
//   getJacobianFactor
//-----------------------------------------------------------------------

Vector VarianceNormalizer::getJacobianFactor

  ( const Vector& vec )

{
  return Vector ( 1. / sdevs_ );
}

//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------

void VarianceNormalizer::update

  ( const Cubix& inp )

{
  idx_t inpsize = inp.size(1);
  idx_t smpsize = inp.size(2);
  idx_t seqsize = inp.size(0);

  means_.resize ( inpsize );
  sdevs_.resize ( inpsize );
  means_ = 0.0; sdevs_ = 0.0;

  for ( idx_t j = 0; j < inpsize; ++j )
  {
    means_[j] = sum ( inp(ALL,j,ALL) ) / smpsize / seqsize;

    double den = (double) ( smpsize * seqsize ) - 1.0;

    for ( idx_t i = 0; i < seqsize; ++i )
    {
      for ( idx_t k = 0; k < smpsize; ++k )
      {
        sdevs_[j] += ( inp(i,j,k) - means_[j] ) * 
	             ( inp(i,j,k) - means_[j] ) / den;
      }
    }
    sdevs_[j] = sqrt ( sdevs_[j] );

    if ( sdevs_[j] < 1.e-20 )
    {
      sdevs_[j] = 1.0;
    }
  }
}

//-----------------------------------------------------------------------
//   write
//-----------------------------------------------------------------------

void VarianceNormalizer::write

  ( const String& fname )

{
  using jem::io::PrintWriter;
  using jem::io::FileWriter;
  using jem::io::FileFlags;

  idx_t count = means_.size();

  Ref<PrintWriter> fout = newInstance<PrintWriter> (
    newInstance<FileWriter> ( fname + ".nml", FileFlags::WRITE ) );

  fout->nformat.setFractionDigits ( 5 );

  *fout << "variance " << count << "\n";

  for ( idx_t i = 0; i < count; ++i )
  {
    *fout << means_[i] << " " << sdevs_[i] << "\n";
  }
}
