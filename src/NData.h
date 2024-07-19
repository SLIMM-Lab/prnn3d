/*
 *  TU Delft 
 *
 *  Iuri Barcelos, June 2019
 *
 *  Object used to pass data to a neural network
 *
 */

#ifndef NDATA_H
#define NDATA_H

#include <jem/base/Object.h>
#include <jem/base/Ref.h>
#include <jem/base/Array.h>
#include <jem/util/Properties.h>

#include <jive/Array.h>

using jem::Object;
using jem::Ref;
using jem::idx_t;
using jem::Array;
using jem::String;
using jem::util::Properties;

using jive::IdxVector;
using jive::Matrix;
using jive::Cubix;
using jive::Vector;

//-----------------------------------------------------------------------
//   class NData
//-----------------------------------------------------------------------

class NData : public Object
{
 public:

  inline               NData 
  
    ( const idx_t        bsize,
      const idx_t        isize,
      const idx_t        osize    );

  inline void          init 

    ( const idx_t        nsize    );

  inline idx_t         batchSize ();

  inline idx_t         inpSize   ();

  inline idx_t         outSize   ();

 public:

  Matrix               inputs;
  Matrix               outputs;
  Matrix               targets;
  Matrix               history;
  Matrix               dts;

  Matrix               deltas;
  Matrix               values;
  Matrix               activations;

  Matrix               jacobian;
  Matrix               covariance;

 private:
  
  idx_t                bsize_;
  idx_t                isize_;
  idx_t                osize_;
};

//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------

inline NData::NData 

  ( const idx_t bsize,
    const idx_t isize,
    const idx_t osize ) : 

    bsize_ ( bsize ),
    isize_ ( isize ),
    osize_ ( osize )
{
  inputs.resize  ( isize, bsize );
  outputs.resize ( osize, bsize );
  targets.resize ( osize, bsize );
  history.resize ( 2, bsize );
  dts.resize ( 1, bsize );

  inputs  = 0.0;
  outputs = 0.0;
  targets = 0.0;
  history = 0.0;
  dts = 0.0;

  deltas.resize(0);
  values.resize(0);
  activations.resize(0);
}

//-----------------------------------------------------------------------
//   init 
//-----------------------------------------------------------------------

inline void NData:: init

  ( const idx_t nsize )

{
  deltas.resize ( nsize, bsize_ );
  values.resize ( nsize, bsize_ );
  activations.resize ( nsize, bsize_ );

  deltas = 0.0;
  values = 0.0;
  activations = 0.0;
}

//-----------------------------------------------------------------------
//   batchSize
//-----------------------------------------------------------------------

inline idx_t NData::batchSize()
{
  return bsize_; 
}

//-----------------------------------------------------------------------
//   inpSize
//-----------------------------------------------------------------------

inline idx_t NData::inpSize()
{
  return isize_; 
}

//-----------------------------------------------------------------------
//   outSize
//-----------------------------------------------------------------------

inline idx_t NData::outSize()
{
  return osize_; 
}

#endif

