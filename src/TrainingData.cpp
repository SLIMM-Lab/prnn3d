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

#include <jem/base/System.h>
#include <jem/base/Error.h>
#include <jem/base/Slice.h>
#include <jem/util/Tokenizer.h>
#include <jem/util/ArrayBuffer.h>
#include <jem/io/FileReader.h>
#include <jem/io/FileInputStream.h>
#include <jem/io/InputStreamReader.h>
#include <jem/base/IllegalInputException.h>
#include <jive/util/Random.h>


#include "TrainingData.h"
#include "XNeuronSet.h"

using namespace jem;
using namespace jem::literals;
using jive::util::Random;

//-----------------------------------------------------------------------
//   class TrainingData
//-----------------------------------------------------------------------

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------

const char* TrainingData::GLOBNAME         = "trainingData";
const char* TrainingData::INPUTNORMALIZER  = "inpNormalizer"; 
const char* TrainingData::OUTPUTNORMALIZER = "outNormalizer";
const char* TrainingData::JACOBIANS        = "jacobians";

//-----------------------------------------------------------------------
//   constructors
//-----------------------------------------------------------------------

TrainingData::TrainingData

  ( const Properties& globdat )

{
  //JEM_ASSERT ( *this != jem::NIL );

  inputs_. resize ( 0, 0, 0 );
  outputs_.resize ( 0, 0, 0 );
  dts_.resize ( 0, 0, 0 );

  globdat.set ( GLOBNAME, const_cast<TrainingData*> ( this ) );
}

TrainingData::TrainingData

  ( const String&     fname,
    const Properties& globdat ) : Self ( globdat )

{
  using jem::util::Tokenizer;
  using jem::util::ArrayBuffer;
  using jem::io::FileInputStream;
  using jem::io::InputStreamReader;

  idx_t sampsize = 0;
  idx_t linesize = 0;
  idx_t seqsize  = 0;

  idx_t pos = 0;

  ArrayBuffer<double> linebuf;
  Matrix imat;

  Ref<Tokenizer> tok = 
    newInstance<Tokenizer> ( 
      newInstance<InputStreamReader> ( 
        newInstance<FileInputStream> ( fname ) ) );

  idx_t token = tok->nextToken(); 
  long int ln = tok->getLineNumber();

  while ( true )
  {
    bool eof = token == Tokenizer::EOF_TOKEN;
    bool eol = tok->getLineNumber() > ln;
    bool eos = tok->getLineNumber() - ln > 1;

    if ( eof || eol )
    {
      Vector line ( linebuf.toArray() );

      if ( !linesize )
      {
        linesize = line.size();
      }
      else if ( line.size() != linesize )
      {
        throw Error ( JEM_FUNC,
	  "TrainingData: Inconsistent number of rows" );
      }

      if ( !seqsize )
      {
	imat.reshape ( pos + 1, linesize );
      }

      imat(pos,ALL) = line;

      pos++;

      if ( eof || eos )
      {
	if ( !seqsize )
	{
          seqsize = pos;
	  inputs_.resize  ( seqsize, linesize, 0 );

	  inputs_  = 0.0;
	}
	else if ( pos != seqsize )
	{
          throw Error ( JEM_FUNC,
	    "TrainingData: Inconsistent sequence length" );
	}

	inputs_. reshape ( sampsize + 1 );

	inputs_ (ALL,ALL,sampsize) = imat;

	imat.resize ( seqsize, linesize );

	imat = 0.0; 

	pos = 0;
	sampsize++;
      }

      linebuf.clear();

      if ( eof )
      {
        break;
      }

      ln = tok->getLineNumber();
    }

    if ( token == Tokenizer::FLOAT_TOKEN ||
         token == Tokenizer::INTEGER_TOKEN  )
    {
      linebuf.pushBack ( tok->getFloat() );
    }
    else
    {
      throw Error ( JEM_FUNC,
        "TrainingData: Invalid input character" );
    }

    token = tok->nextToken();
  }

  outputs_.ref ( inputs_ );

  //updNorms_();

}

TrainingData::TrainingData

  ( const String&     fname,
    const IdxVector&  inputs,
    const IdxVector&  outputs,
    const Properties& globdat ) : Self ( globdat )

{

  System::out() << "TrainingData::TrainingData\n";

  using jem::util::Tokenizer;
  using jem::util::ArrayBuffer;
  using jem::io::FileInputStream;
  using jem::io::InputStreamReader;

  idx_t sampsize = 0;
  idx_t linesize = 0;
  idx_t seqsize  = 0;

  idx_t pos = 0;

  idx_t isize = inputs. size();
  idx_t osize = outputs.size();

  ArrayBuffer<double> linebuf;
  Matrix imat ( 0_idx, isize );
  Matrix omat ( 0_idx, osize );

  for ( idx_t i = 0; i < inputs.size(); ++i )
  {
    if ( testany ( outputs == inputs[i] ) )
    {
      throw Error ( JEM_FUNC,
	"TrainingData: Ambiguous input/output choice" );
    }
  }

  Ref<Tokenizer> tok = 
    newInstance<Tokenizer> ( 
      newInstance<InputStreamReader> ( 
        newInstance<FileInputStream> ( fname ) ) );

  idx_t token = tok->nextToken(); 
  long int ln = tok->getLineNumber();

  while ( true )
  {
    bool eof = token == Tokenizer::EOF_TOKEN;
    bool eol = tok->getLineNumber() > ln;
    bool eos = tok->getLineNumber() - ln > 1;

    if ( eof || eol )
    {
      Vector line ( linebuf.toArray() );

      if ( !linesize )
      {
        linesize = line.size();

	if ( linesize <= max(max(inputs),max(outputs)) )
	{
          throw Error ( JEM_FUNC,
	    "TrainingData: Incompatible input/output data" );
	}
      }
      else if ( line.size() != linesize )
      {
        throw Error ( JEM_FUNC,
	  "TrainingData: Inconsistent number of rows" );
      }

      if ( !seqsize )
      {
	imat.reshape ( pos + 1, isize );
	omat.reshape ( pos + 1, osize );
      }

      imat(pos,ALL) = line[inputs];
      omat(pos,ALL) = line[outputs];

      pos++;

      if ( eof || eos )
      {
	if ( !seqsize )
	{
          seqsize = pos;
	  inputs_.resize  ( seqsize, isize, 0 );
	  outputs_.resize ( seqsize, osize, 0 );

	  inputs_  = 0.0;
	  outputs_ = 0.0;
	}
	else if ( pos != seqsize )
	{
          throw Error ( JEM_FUNC,
	    "TrainingData: Inconsistent sequence length" );
	}

	inputs_. reshape ( sampsize + 1 );
	outputs_.reshape ( sampsize + 1 );

	inputs_ (ALL,ALL,sampsize) = imat;
	outputs_(ALL,ALL,sampsize) = omat;

	imat.resize ( seqsize, isize );
	omat.resize ( seqsize, osize );

	imat = 0.0; omat = 0.0;

	pos = 0;
	sampsize++;
      }

      linebuf.clear();

      if ( eof )
      {
        break;
      }

      ln = tok->getLineNumber();
    }

    if ( token == Tokenizer::FLOAT_TOKEN ||
         token == Tokenizer::INTEGER_TOKEN  )
    {
      linebuf.pushBack ( tok->getFloat() );
    }
    else
    {
      throw Error ( JEM_FUNC,
        "TrainingData: Invalid input character" );
    }

    token = tok->nextToken();
  }

  //updNorms_();
}

TrainingData::TrainingData

  ( const String&     fname,
    const IdxVector&  inputs,
    const IdxVector&  outputs,
    const IdxVector&  dts, 
    const Properties& globdat ) : Self ( globdat )

{

   System::out() << "TrainingData::TrainingData\n";

  using jem::util::Tokenizer;
  using jem::util::ArrayBuffer;
  using jem::io::FileInputStream;
  using jem::io::InputStreamReader;

  idx_t sampsize = 0;
  idx_t linesize = 0;
  idx_t seqsize  = 0;

  idx_t pos = 0;

  idx_t isize = inputs. size();
  idx_t osize = outputs.size();
  idx_t tsize = dts.size();

  ArrayBuffer<double> linebuf;
  Matrix imat ( 0_idx, isize );
  Matrix omat ( 0_idx, osize );
  Matrix tmat ( 0_idx, tsize );

  for ( idx_t i = 0; i < inputs.size(); ++i )
  {
    if ( testany ( outputs == inputs[i] ) )
    {
      throw Error ( JEM_FUNC,
	"TrainingData: Ambiguous input/output choice" );
    }
  }

  Ref<Tokenizer> tok = 
    newInstance<Tokenizer> ( 
      newInstance<InputStreamReader> ( 
        newInstance<FileInputStream> ( fname ) ) );

  idx_t token = tok->nextToken(); 
  long int ln = tok->getLineNumber();

  while ( true )
  {
    bool eof = token == Tokenizer::EOF_TOKEN;
    bool eol = tok->getLineNumber() > ln;
    bool eos = tok->getLineNumber() - ln > 1;

    if ( eof || eol )
    {
      Vector line ( linebuf.toArray() );

      if ( !linesize )
      {
        linesize = line.size();

	if ( linesize <= max(max(inputs),max(outputs)) )
	{
          throw Error ( JEM_FUNC,
	    "TrainingData: Incompatible input/output data" );
	}
      }
      else if ( line.size() != linesize )
      {
        throw Error ( JEM_FUNC,
	  "TrainingData: Inconsistent number of rows" );
      }

      if ( !seqsize )
      {
	imat.reshape ( pos + 1, isize );
	omat.reshape ( pos + 1, osize );
	tmat.reshape ( pos + 1, tsize );
      }

      imat(pos,ALL) = line[inputs];
      omat(pos,ALL) = line[outputs];
      tmat(pos,ALL) = line[dts];

      pos++;

      if ( eof || eos )
      {
	if ( !seqsize )
	{
          seqsize = pos;
	  inputs_.resize  ( seqsize, isize, 0 );
	  outputs_.resize ( seqsize, osize, 0 );
	  dts_.resize ( seqsize, tsize, 0 );

	  inputs_  = 0.0;
	  outputs_ = 0.0;
	  dts_ = 0.0;
	}
	else if ( pos != seqsize )
	{
          throw Error ( JEM_FUNC,
	    "TrainingData: Inconsistent sequence length" );
	}

	inputs_. reshape ( sampsize + 1 );
	outputs_.reshape ( sampsize + 1 );
	dts_.reshape ( sampsize + 1 );

	inputs_ (ALL,ALL,sampsize) = imat;
	outputs_(ALL,ALL,sampsize) = omat;
	dts_(ALL,ALL,sampsize) = tmat;

	imat.resize ( seqsize, isize );
	omat.resize ( seqsize, osize );
	tmat.resize ( seqsize, tsize );

	imat = 0.0; omat = 0.0; tmat = 0.0;

	pos = 0;
	sampsize++;
      }

      linebuf.clear();

      if ( eof )
      {
        break;
      }

      ln = tok->getLineNumber();
    }

    if ( token == Tokenizer::FLOAT_TOKEN ||
         token == Tokenizer::INTEGER_TOKEN  )
    {
      linebuf.pushBack ( tok->getFloat() );
    }
    else
    {
      throw Error ( JEM_FUNC,
        "TrainingData: Invalid input character" );
    }

    token = tok->nextToken();
  }

  //updNorms_();
}

//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------

void TrainingData::configure

  ( const Properties& props,
    const Properties& globdat )

{
  
  System::out( ) << "TrainingData::configure\n";  

  String type = "none";

  props.find ( type, INPUTNORMALIZER );

  inl_ = newNormalizer ( type );

  type = "none";

  props.find ( type, OUTPUTNORMALIZER );

  System::out() << "Output normalizer: " << type << "\n";
  onl_ = newNormalizer ( type );
  
  // Select only some curves out of pool of loading curves

  subsetSize_ = 0;
  skipFirst_ = 0;
  globdat.find ( subsetSize_, "subset");
  globdat.find ( skipFirst_, "skipFirst");
  globdat.find ( rseed_, "rseed");

  Ref<Random> generator = Random::get ( globdat );
  generator->restart( rseed_ );

  System::out() << "Here\n"; 
  

  if ( subsetSize_ > 0)
  {
    IdxVector randsamp(skipFirst_ + subsetSize_);

    for ( idx_t r = 0; r < subsetSize_ +skipFirst_; r++ )
    {
      if ( r < skipFirst_ )
      {
        randsamp[r] = r;
      }
      else
      {
        randsamp[r] = generator->next(outputs_.size(2)-skipFirst_) + skipFirst_;
      }
    }

    System::out() << "Subset " << randsamp << "\n";
    //System::out() << "Selected for training" << outputs_[randsamp[slice(skipFirst_, subsetSize_+skipFirst_)]]  << "\n";

    Cubix tempoutputs (outputs_.size(0), outputs_.size(1), skipFirst_ + subsetSize_ );
    tempoutputs = outputs_[randsamp];
    Cubix tempinputs (inputs_.size(0), inputs_.size(1), skipFirst_ + subsetSize_ );
    tempinputs = inputs_[randsamp];
    Cubix tempdts (dts_.size(0), dts_.size(1), skipFirst_ + subsetSize_ );
    tempdts = dts_[randsamp];

    outputs_.resize ( outputs_.size(0), outputs_.size(1), subsetSize_ + skipFirst_ );
    inputs_.resize ( inputs_.size(0), inputs_.size(1), subsetSize_ + skipFirst_ );
    dts_.resize ( dts_.size(0), dts_.size(1), subsetSize_ + skipFirst_ );
    
    outputs_ = 0.0; inputs_ = 0.0; dts_ = 0.0;

    outputs_ = tempoutputs;
    inputs_ = tempinputs;
    dts_ = tempdts;
  }
  else
  {
    System::out() << "outs " << outputs_.size(0) << " " << outputs_.size(1) << " " << outputs_.size(2) << "\n";
    subsetSize_ = outputs_.size(2) - skipFirst_; 
  }

  System::out() << "Subset size: " << subsetSize_ << "\n"; 
  
  inl_->update ( inputs_[slice(skipFirst_, skipFirst_ + subsetSize_)] );
  onl_->update ( outputs_[slice(skipFirst_, skipFirst_ + subsetSize_)] );

  System::out() << "Finished\n";

  String fname;

  if ( props.find ( fname, JACOBIANS ) )
  {
    readJacs_ ( fname, globdat );
  }

  if ( props.find ( fname, "history" ) )
  {
    readHistory_ ( fname, globdat );
  }

/* System::out() << "TrainingData. inputs_ " << inputs_ << " outputs_ " << 
	  outputs_ << "\n";
 System::out() << "TrainingData. dts_ " << dts_ << "\n"; */ 
}

//-----------------------------------------------------------------------
//   addData
//-----------------------------------------------------------------------

idx_t TrainingData::addData

  ( const Vector& inputs,
    const Vector& outputs )

{
  
 // System::out() << "TrainingData::addData. Vector.\n";

  idx_t n = sampleSize();
  idx_t i = inputs. size();
  idx_t o = outputs.size();

  if ( !n )
  {
    inputs_. resize ( 1_idx, i, 1_idx );
    outputs_.resize ( 1_idx, o, 1_idx );
    inputs_ = 0.0; outputs_ = 0.0;
  }
  else
  {
    JEM_ASSERT ( i == inputSize()  );
    JEM_ASSERT ( o == outputSize() );

    inputs_. reshape ( n + 1_idx ); 
    outputs_.reshape ( n + 1_idx );
  }

  inputs_ (0_idx,ALL,n) = inputs;
  outputs_(0_idx,ALL,n) = outputs;

//  System::out() << "Updating GP with " << inputs_ << " outputs " << outputs_ << "\n";

  inl_->update ( inputs_  );
  onl_->update ( outputs_ );

  newDataEvent.emit ( *this );

  return n;
}

//-----------------------------------------------------------------------
//   addData
//-----------------------------------------------------------------------

idx_t TrainingData::addData

  ( const Matrix& inputs,
    const Matrix& outputs )

{

  System::out() << "TrainingData::addData. Matrix.";

  idx_t n = sampleSize();
  idx_t i = inputs. size(0);
  idx_t o = outputs.size(0);

  JEM_PRECHECK ( inputs.size(1) == outputs.size(1) );

  if ( !n )
  {
    inputs_. resize ( 1_idx, i, inputs.size(1) ); 
    outputs_.resize ( 1_idx, o, inputs.size(1) ); 
    inputs_ = 0.0; outputs_ = 0.0;
  }
  else
  {
    JEM_PRECHECK ( i == inputSize () );
    JEM_PRECHECK ( o == outputSize() );

    inputs_. reshape ( n + inputs.size(1) );
    outputs_.reshape ( n + inputs.size(1) );
  }

  inputs_ (0_idx,ALL,slice(n,END)) = inputs;
  outputs_(0_idx,ALL,slice(n,END)) = outputs;

  inl_->update ( inputs_  );
  onl_->update ( outputs_ );

  newDataEvent.emit ( *this );

  return n;
}

//-----------------------------------------------------------------------
//   refreshData
//-----------------------------------------------------------------------

void TrainingData::refreshData

  ( const idx_t   id,
    const Vector& inputs,
    const Vector& outputs )

{
  System::out() << "TrainingData::RefreshData\n";
	
  inputs_ (0,ALL,id) = inputs;
  outputs_(0,ALL,id) = outputs;

  inl_->update ( inputs_  );
  onl_->update ( outputs_ );

  newDataEvent.emit ( *this );
}


//-----------------------------------------------------------------------
//   getData
//-----------------------------------------------------------------------

Batch TrainingData::getData

  ( const   IdxVector& ids )

{
//  System::out() << "TrainingData::getData. ids " << ids.size() << "\n";
//  System::out() << "inpSize " << inputSize() << " outputSize " << outputs_.size(0) << " " << outputs_.size(1) << " " << outputs_.size(2) << "\n"; 

  Batch data ( sequenceSize() );

  for ( idx_t t = 0; t < sequenceSize(); ++t )
  {
    data[t] = newInstance<NData> 
      ( ids.size(), inputSize(), outputSize() );

    for ( idx_t s = 0; s < ids.size(); ++s )
    {
   //   System::out() << "inp " << inputs_(0, ALL, 0) << "\n";
      data[t]->inputs(ALL,s)  = inl_->normalize ( inputs_ ( t, ALL, ids[s] ) );
      data[t]->targets(ALL,s) = onl_->normalize ( outputs_( t, ALL, ids[s] ) );
      if ( outputs_.size(0) == 1 && outputs_.size(1) == 1 )
      {
        data[t]->dts(0,s) = 1.; 
      }
      else
      {
        data[t]->dts(0,s) = dts_ ( t, 0, ids[s] );
      }
   //   if (t == sequenceSize() - 1) System::out() << "input: " << inputs_(t, ALL, ids[s] )
  //    System::out()   << " input (normalized): " << data[t]->inputs(ALL, s) << "\n"; 
  //    System::out() << " output (normalized): " << data[t]->outputs(ALL, s) << "\n"; 
    }
  }

  return data;
}

//-----------------------------------------------------------------------
//   stretchData
//-----------------------------------------------------------------------

Ref<NData> TrainingData::stretchData

  ( const Ref<NData> data )

{
  idx_t ns = data->batchSize();

  Ref<NData> ret = newInstance<NData> 
    ( ns * inputSize(), inputSize(), outputSize() );

  // NB: ugly, change later
  ret->init ( data->values.size(0) );

  for ( idx_t i = 0; i < inputSize(); ++i )
  {
    ret->values     (ALL,slice(i*ns,(i+1)*ns)) = data->values;
    ret->activations(ALL,slice(i*ns,(i+1)*ns)) = data->activations;
  }

  return ret;
}

//-----------------------------------------------------------------------
//   getJacobian
//-----------------------------------------------------------------------

Matrix TrainingData::getJacobian

  ( const IdxVector& ids ) 

{
  idx_t ns = ids.size();

  Matrix ret ( inputSize(), ns * inputSize() );
  ret = 0.0;

  for ( idx_t i = 0; i < inputSize(); ++i )
  {
    for ( idx_t s = 0; s < ns; ++s )
    {
      ret(ALL,i*ns+s) = jacobians_(i,ALL,s);
    }
  }

  return ret;
}

//-----------------------------------------------------------------------
//   getHistory
//-----------------------------------------------------------------------

Matrix TrainingData::getHistory

  ( const IdxVector& ids, const idx_t t) 

{
  idx_t ns = ids.size();

  Matrix ret ( 1, ns );
  ret = 0.0;

 // System::out() << "inputSize " << inputSize() << " ns " <<
 // ns << " ids " << ids << "\n";

  for ( idx_t i = 0; i < ns; ++i )
  {
      ret(ALL,i) = history_(t,6,ids[i]);
  }

 // System::out() << "Training data. getHistory: " << ret << "\n";

  return ret;
}

//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------

Ref<TrainingData> TrainingData::get

  ( const Properties&  globdat,
    const String&      context )

{
  Ref<TrainingData> data;

  globdat.find ( data, GLOBNAME );

  if ( data == nullptr )
  {
    throw jem::IllegalInputException (
      context,
      "No TrainingData object exists."
    );
  }

  return data;
}

//-----------------------------------------------------------------------
//   readHistory_
//-----------------------------------------------------------------------

void TrainingData::readHistory_

  ( const String&     fname,
    const Properties& globdat )

{
  using jem::io::FileReader;

  idx_t ni = sequenceSize(); //inputSize();
  idx_t no = 1; //outputSize();
  idx_t ns = sampleSize();

  history_.resize ( ni, no, ns );
  history_ = 0.0;

  System::out() << "Reading DT \n";

  Ref<FileReader> in    = newInstance<FileReader> ( fname );

  System::out() << "ni " << ni << " no " << no << " ns " << ns << "\n";

  for ( idx_t s = 0; s < ns; ++s )
  {
    for ( idx_t i = 0; i < ni; ++i )
    {
      for ( idx_t o = 0; o < no; ++o )
      {             
        history_( i, o, s ) = in->parseDouble();
      }
    }
  }
}


//-----------------------------------------------------------------------
//   readJacs_
//-----------------------------------------------------------------------

void TrainingData::readJacs_

  ( const String&     fname,
    const Properties& globdat )

{
  using jem::io::FileReader;

  idx_t ni = inputSize();
  idx_t no = outputSize();
  idx_t ns = sampleSize();

  if ( sequenceSize() > 1 )
  {
    throw Error ( JEM_FUNC, "J-prop augmentation not implemented for recurrent layers" );
  }

  jacobians_.resize ( ni, no, ns );
  jacobians_ = 0.0;

  Ref<FileReader> in    = newInstance<FileReader> ( fname );

  for ( idx_t s = 0; s < ns; ++s )
  {
    for ( idx_t i = 0; i < ni; ++i )
    {
      for ( idx_t o = 0; o < no; ++o )
      {
        jacobians_( i, o, s ) = in->parseDouble();
      }

      Vector fac = inl_->getJacobianFactor ( inputs_ ( 0, ALL, s ) );
      jacobians_ ( i, ALL, s ) /= fac;
    }
  }
}
