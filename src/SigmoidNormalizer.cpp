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

#include <jem/base/IllegalInputException.h>
#include <jem/base/Error.h>
#include <jem/base/System.h>
#include <jem/io/FileReader.h>
#include <jem/io/FileInputStream.h>
#include <jem/io/InputStreamReader.h>
#include <jem/io/PrintWriter.h>
#include <jem/io/FileWriter.h>
#include <jem/io/FileFlags.h>
#include <jem/util/Tokenizer.h>

#include "SigmoidNormalizer.h"

using namespace jem;

//-----------------------------------------------------------------------
//   class SigmoidNormalizer
//-----------------------------------------------------------------------

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------

SigmoidNormalizer::SigmoidNormalizer ()
{
  means_.resize ( 0 );
  upper_.resize ( 0 );
  lower_.resize ( 0 );

  System::out() << "Bounds normalizer sigmoid. No files found. \n";
}

SigmoidNormalizer::SigmoidNormalizer

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

  means_.resize ( count );
  upper_.resize ( count );
  lower_.resize ( count );
  means_ = 0.0; upper_ = 0.0; lower_ = 0.0;

  for ( idx_t i = 0; i < count; ++i )
  {
    tizer->nextToken();

    means_[i] = tizer->getFloat();

    tizer->nextToken();

    upper_[i] = tizer->getFloat();

    tizer->nextToken();

    lower_[i] = tizer->getFloat();
  }

  System::out() << "Bounds normalizer: " << means_ << " upper " << upper_ << " lower " << lower_ << "\n";
}

//-----------------------------------------------------------------------
//   normalize
//-----------------------------------------------------------------------

Vector SigmoidNormalizer::normalize

  ( const Vector& vec )

{
  Vector ret ( ( vec - means_ - lower_ ) / ( upper_ - lower_ ) );

  return ret;
}

//-----------------------------------------------------------------------
//   denormalize
//-----------------------------------------------------------------------

Vector SigmoidNormalizer::denormalize

  ( const Vector& vec )

{
  Vector ret ( means_ + vec * ( upper_ - lower_ ) + lower_ );
  
  return ret;
}

//-----------------------------------------------------------------------
//   getJacobianFactor
//-----------------------------------------------------------------------

Vector SigmoidNormalizer::getJacobianFactor

  ( const Vector& vec )

{
  return Vector ( 1. / ( upper_ - lower_ ) );
}

//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------

void SigmoidNormalizer::update

  ( const Cubix& inp )

{
  idx_t inpsize = inp.size(1);
  idx_t smpsize = inp.size(2);
  idx_t seqsize = inp.size(0);

  means_.resize ( inpsize );
  upper_.resize ( inpsize );
  lower_.resize ( inpsize );
  means_ = 0.0; upper_ = 0.0; lower_ = 0.0;

  for ( idx_t j = 0; j < inpsize; ++j )
  {
    means_[j] = sum ( inp(ALL,j,ALL) ) / smpsize / seqsize;
    upper_[j] = max ( inp(ALL,j,ALL) - means_[j] );
    lower_[j] = min ( inp(ALL,j,ALL) - means_[j] );

    if ( jem::numeric::abs ( upper_[j] - lower_[j] ) < 1.e-20 )
    {
      upper_[j] = 1.0;
      lower_[j] = 0.0;
    }
  }
}

//-----------------------------------------------------------------------
//   write
//-----------------------------------------------------------------------

void SigmoidNormalizer::write

  ( const String& fname )

{
  using jem::io::PrintWriter;
  using jem::io::FileWriter;
  using jem::io::FileFlags;

  idx_t count = means_.size();

  Ref<PrintWriter> fout = newInstance<PrintWriter> (
    newInstance<FileWriter> ( fname + ".nml", FileFlags::WRITE ) );

  fout->nformat.setFractionDigits ( 10 );

  *fout << "sigmoid " << count << "\n";

  for ( idx_t i = 0; i < count; ++i )
  {
    *fout << means_[i] << " " << upper_[i] << " " << lower_[i] << "\n";
  }
}
