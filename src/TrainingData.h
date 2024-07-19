/*
 * Copyright (C) 2019 TU Delft. All rights reserved.
 *
 * Class that reads input/output layer for ANN training. It can
 * also be used by an external agent (forked chain, module parallel
 * to the solver) to gradually add training data to be used by
 * the ANN solver.
 *
 * Author: Iuri Barcelos, i.barceloscarneiromrocha@tudelft.nl
 * Date:   May 2019
 * 
 */

#ifndef TRAININGDATA_H
#define TRAININGDATA_H

#include <jem/base/Object.h>
#include <jem/base/Ref.h>
#include <jem/base/Array.h>
#include <jem/util/Properties.h>
#include <jem/util/Event.h>

#include <jive/Array.h>

#include "NData.h"
#include "Normalizer.h"

using jem::Object;
using jem::Ref;
using jem::idx_t;
using jem::Array;
using jem::String;
using jem::util::Event;
using jem::util::Properties;

using jive::IdxVector;
using jive::Matrix;
using jive::Cubix;
using jive::Vector;

typedef Array< Ref<NData>, 1 > Batch;

//-----------------------------------------------------------------------
//   class TrainingData
//-----------------------------------------------------------------------

class TrainingData : public Object
{
 public:

  typedef TrainingData Self;

  Event < Self& >        newDataEvent;
  
  static const char*     GLOBNAME;
  static const char*     INPUTNORMALIZER;
  static const char*     OUTPUTNORMALIZER;
  static const char*     JACOBIANS;

                         TrainingData
    
    ( const Properties&    globdat         );
  
                         TrainingData

    ( const String&        fname,
      const Properties&    globdat         );

                         TrainingData

    ( const String&        fname,
      const IdxVector&     inputs,
      const IdxVector&     outputs,
      const Properties&    globdat         );


                         TrainingData

    ( const String&        fname,
      const IdxVector&     inputs,
      const IdxVector&     outputs,
      const IdxVector&     dts, 
      const Properties&    globdat         );
  
  virtual void           configure

    ( const Properties&    props,
      const Properties&    globdat         );

  inline idx_t           sampleSize       () const;

  inline idx_t           inputSize        () const;

  inline idx_t           outputSize       () const;

  inline idx_t           sequenceSize     () const;

  inline Ref<Normalizer> getInpNormalizer () const;

  inline Ref<Normalizer> getOutNormalizer () const;

  Batch                  getData

    ( const IdxVector&    ids              );

  Ref<NData>             stretchData

    ( const Ref<NData>     data            );

  Matrix                 getJacobian

    ( const IdxVector&    ids              );

  Matrix                 getHistory

    ( const IdxVector&    ids, 
      const idx_t timestep                 );

  idx_t                  addData

    ( const Vector&        inputs,
      const Vector&        outputs         );

  idx_t                  addData

    ( const Matrix&        inputs,
      const Matrix&        outputs         );

  void                   refreshData

    ( const idx_t          id,
      const Vector&        inputs,
      const Vector&        outputs         );

  static Ref<Self>       get

    ( const Properties&    globdat,        
      const String&        context         );

 private:
   
  void                   readJacs_

    ( const String&        fname,
      const Properties&    globdat         );

  void                   readHistory_

    ( const String&        fname,
      const Properties&    globdat         );

 private:
  
  idx_t                  size_;
  idx_t                  skipFirst_;
  idx_t                  subsetSize_;
  idx_t                  rseed_;

  Cubix                  inputs_;
  Cubix                  outputs_;
  Cubix                  jacobians_;
  Cubix                  history_;
  Cubix                  dts_;

  Ref<Normalizer>        inl_;
  Ref<Normalizer>        onl_;

  Cubix                  normalInps_;
};

//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   sampleSize
//-----------------------------------------------------------------------

inline idx_t TrainingData::sampleSize () const 
{
  return inputs_.size ( 2 );
}

//-----------------------------------------------------------------------
//   inputSize
//-----------------------------------------------------------------------

inline idx_t TrainingData::inputSize () const 
{
  return inputs_.size ( 1 );
}

//-----------------------------------------------------------------------
//   outputSize
//-----------------------------------------------------------------------

inline idx_t TrainingData::outputSize () const 
{
  return outputs_.size ( 1 );
}

//-----------------------------------------------------------------------
//   sequenceSize
//-----------------------------------------------------------------------

inline idx_t TrainingData::sequenceSize () const 
{
  return inputs_.size ( 0 );
}

//-----------------------------------------------------------------------
//   getInpNormalizer
//-----------------------------------------------------------------------

inline Ref<Normalizer> TrainingData::getInpNormalizer () const
{
  return inl_;
}

//-----------------------------------------------------------------------
//   getOutNormalizer
//-----------------------------------------------------------------------

inline Ref<Normalizer> TrainingData::getOutNormalizer () const
{
  return onl_;
}

#endif

