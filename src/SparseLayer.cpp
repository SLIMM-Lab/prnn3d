/*
 * Copyright (C) 2023 TU Delft. All rights reserved.
 *
 * Class that implements a sparsely-connected neural layer. Each
 * of its neurons will connect themselves with the neuron of
 * the previous layer (towards the input layer) in a component-wise
 * manner. If symmetric option is enabled, weights within each
 * material point are symmetric.
 *
 * Author: Marina Maia, m.alvesmaia@tudelft.nl
 * Date:   Jul 2023
 * 
 */

#include <jem/base/Error.h>
#include <jem/base/System.h>
#include <jem/base/Float.h>
#include <jem/numeric/algebra/matmul.h>
#include <jem/util/Timer.h>
#include <jive/Array.h>
#include <jive/model/Actions.h>
#include <jive/model/ModelFactory.h>
#include <jive/model/StateVector.h>
#include <jive/util/error.h>

#include "MatRecLayer.h"
#include "SparseLayer.h"
#include "LearningNames.h"

using jem::Error;
using jem::numeric::matmul;
using jem::util::Timer;

using jive::IdxVector;
using jive::util::XDofSpace;
using jive::util::sizeError;
using jive::model::StateVector;

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------

const char* SparseLayer::SIZE           = "size";
const char* SparseLayer::ACTIVATION     = "activation";
const char* SparseLayer::INITIALIZATION = "init";
const char* SparseLayer::USEBIAS        = "useBias";
const char* SparseLayer::DEBUG          = "debug";
const char* SparseLayer::PRUNING        = "pruning";

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------

SparseLayer::SparseLayer

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat ) : Model ( name )

{
  String func ( "identity" );

  size_ = 0;
  onl_ = nullptr;

  inputLayer_  = false;
  outputLayer_ = false;
  mirrored_    = false;
  useBias_     = false;
  debug_       = false; 
  pruning_      = true;
  actWeights_ = false;
  symmetric_ = false;
  String actFuncWeights = "relu";
  nComp_ = 9;
  nWtsBlock_ = 9;

  Properties myProps = props.getProps ( myName_ );
  Properties myConf  = conf.makeProps ( myName_ );

  myProps.get  ( size_, SIZE );
  myProps.find ( func, ACTIVATION );
  myProps.find ( useBias_, USEBIAS );
  myProps.find ( debug_, DEBUG );
  myProps.find ( pruning_, PRUNING );
  myProps.find ( symmetric_, "symmetric");
  myProps.find ( actWeights_, "activateWeights" );
  myProps.find ( actFuncWeights, "activationWeights" );

  myConf. set ( SIZE, size_ );
  myConf. set ( ACTIVATION, func );
  myConf. set ( USEBIAS, useBias_ );
  myConf. set ( "pruning", pruning_ );
  myConf. set ( "symmetric", symmetric_ );
  myConf. set ( "activateWeights", actWeights_ );
  myConf. set ( "activationWeights", actFuncWeights );

  func_ = NeuralUtils::getActivationFunc ( func );
  grad_ = NeuralUtils::getActivationGrad ( func );
  hess_ = NeuralUtils::getActivationHess ( func );
  actfuncweights_ = NeuralUtils::getActivationFunc ( actFuncWeights );
  gradfuncweights_ = NeuralUtils::getActivationGrad ( actFuncWeights );

  func = "glorot";

  myProps.find ( func, INITIALIZATION );
  myConf.set   ( INITIALIZATION, func );

  init_ = NeuralUtils::getInitFunc ( func );

  globdat.get ( inpNeurons_, LearningParams::PREDECESSOR );
  preSize_ = inpNeurons_.size();

  nBlocks_ = int ( size_/ nComp_ );
  nBlocksPrev_ = int ( preSize_/ nComp_ );

  symIndex_.resize ( 9 );
  if ( symmetric_ )
  {
    nWtsBlock_ = 6;

    symIndex_[0] = 0;
    symIndex_[1] = 1;
    symIndex_[2] = 2;
    symIndex_[3] = 1;
    symIndex_[4] = 3;
    symIndex_[5] = 4;
    symIndex_[6] = 2;
    symIndex_[7] = 4;
    symIndex_[8] = 5;

    selComp_.resize(6);
    selComp_[0] = 0; selComp_[1] = 1; selComp_[2] = 2;
    selComp_[3] = 4; selComp_[4] = 5; selComp_[5] = 8;
    remComp_.resize ( 3);
    remComp_[0] = 3; remComp_[1] = 6; remComp_[2]= 7;

  }
  else
  {
    symIndex_[0] = 0;
    symIndex_[1] = 1;
    symIndex_[2] = 2;
    symIndex_[3] = 3;
    symIndex_[4] = 4;
    symIndex_[5] = 5;
    symIndex_[6] = 6;
    symIndex_[7] = 7;
    symIndex_[8] = 8;
    selComp_.resize(9);
    selComp_ = symIndex_.clone();
  }

  iNeurons_.resize ( size_ );
  iNeurons_ = 0;

  Ref<XAxonSet>   aset = XAxonSet  ::get ( globdat, getContext()    );
  Ref<XNeuronSet> nset = XNeuronSet::get ( globdat, getContext()    );
  Ref<XDofSpace>  dofs = XDofSpace:: get ( aset->getData(), globdat );

  for ( idx_t in = 0; in < size_ ; ++in )
  {
    iNeurons_[in] = nset->addNeuron();
  }

  weightType_ = dofs->findType ( LearningNames::WEIGHTDOF );
  biasType_   = dofs->findType ( LearningNames::BIASDOF   );

  Ref<Model> layer;

  if ( myProps.find ( layer, LearningParams::IMAGE ) )
  {
    Ref<SparseLayer> image = dynamicCast<SparseLayer> ( layer );

    System::out() << "Sparse layer. Symmetric network.\n";

    if ( image == nullptr )
    {
      throw Error ( JEM_FUNC, "Broken symmetry" );
    }

    globdat.get ( inpNeurons_, LearningParams::PREDECESSOR );

    idx_t presize = inpNeurons_.size();

    iInpWts_.resize ( size_, presize );
    iInpWts_ = -1;

    IdxMatrix imagewts;
    image->getInpDofs ( imagewts );

    iInpWts_ = imagewts.transpose();


    weights_.resize ( size_, inpNeurons_.size() );
    weights_ = 0.0;

    Vector state;

    if ( useBias_ )
    {
      iBiases_.resize ( size_ );
      biases_. resize ( size_ );
      iBiases_ = -1;
      biases_  = 0.0;

      IdxVector newbaxons = aset->addAxons ( size_ );
      dofs->addDofs ( newbaxons, biasType_   );

      System::out() << "Mirrored bias: Creating " << size_ << " axons\n";

      StateVector::get ( state, dofs, globdat );

      dofs->getDofIndices ( iBiases_, newbaxons, biasType_ );
      state[iBiases_] = 0.0;
    }

    StateVector::get ( state, dofs, globdat );

    System::out() << "size_ mirrored " << size_ << " dim iInpWts_ " << iInpWts_.size(0) << " x " << iInpWts_.size(1) << "\n";
    for ( idx_t in = 0; in < size_; ++in )
    {
      weights_(in,ALL) = state[iInpWts_(in,ALL)];
      System::out() << "iInpWts_ mirrored " << iInpWts_(in,ALL) << "\n";
    }
    System::out() << "ok\n";
    mirrored_ = true;
  }
  else if ( globdat.find ( inpNeurons_, LearningParams::PREDECESSOR ) )
  {
    IdxVector idofs ( nBlocksPrev_ );      // Number of dofs per component 
    
    iInpWts_.resize ( nWtsBlock_*nBlocks_, nBlocksPrev_ );  // Dofs of weights
    iInpWts_ = -1;

    weights_.resize ( size_, preSize_ );  // Full weight matrix like a regular dense layer
    weights_ = 0.0;

    init_ ( weights_, globdat );

    Vector state;

    IdxVector newaxons  = aset->addAxons ( nWtsBlock_*nBlocksPrev_*nBlocks_ );
    System::out() << "Sparse layer. inpWeights: Creating " << preSize_ * nBlocks_ << " axons\n";
    System::out() << "Presize: " << preSize_ << " size: " << size_ << "\n";
    System::out() << "inpNeurons:" << inpNeurons_ << "\n";
    System::out() << "nWtsBlock: " << nWtsBlock_ << " nBlocks: " << nBlocks_ << " nBlocksPrev_ " << nBlocksPrev_ << "\n";

    dofs->addDofs ( newaxons,  weightType_ );

    StateVector::get ( state, dofs, globdat );

  //  System::out() << "newaxons: " << newaxons << "\n";
   
    for ( idx_t nb = 0; nb < nBlocks_; nb++ )
    {

      IdxVector selComp ( nBlocksPrev_ );
      IdxVector remComp ( preSize_ - nBlocksPrev_ );

      for ( idx_t in = 0; in < nComp_; in++ )
      {
        IdxVector myaxons ( newaxons[slice (nb*(nWtsBlock_*nBlocksPrev_)+symIndex_[in]*nBlocksPrev_,nb*(nWtsBlock_*nBlocksPrev_)+(symIndex_[in]+1)*nBlocksPrev_ )] );
        dofs->getDofIndices ( idofs,  myaxons,   weightType_ );
	iInpWts_(nb*nWtsBlock_+symIndex_[in],ALL) = idofs;

        System::out() << "i: " << in << "\n";
        System::out() << "myaxons: " << myaxons << "\n";

        for ( idx_t ii = 0; ii < nBlocksPrev_; ii++) 
	{
	  idx_t computedSelComp = 0;
	  selComp[ii] = in+ii*nComp_;

	  for ( idx_t jj = 0; jj < 9; jj++ )
	  {
	    if ( jj != in ) 
	    { 
	      remComp[ii*8+jj+computedSelComp] = ii*nComp_+jj;
	    }
	    else
	    {
	      computedSelComp = -1;
	    }
	  }
	}

  //    System::out() << "idofs: " << idofs << "\n";
  //    System::out() << "state[idofs]: " << state[idofs] << "\n";

	if ( pruning_ ) weights_(nb*nComp_+in,remComp) = 0.0;
        
        if ( symmetric_ )
	{
	  if ( in !=3 && in != 6 && in != 7 ) 
	  {
	    state[idofs] = weights_(nb*nComp_+in,selComp);
	  }
	  else
	  {
            weights_(nb*nComp_+in, selComp) = state[idofs];
	  }
	}
	else
	{
	  state[idofs] = weights_(nb*nComp_+in,selComp);
	}

	//if ( pruning_ ) weights_(nb*nComp_+in,remComp) = 0.0;
        
	System::out() << "weights_(in, ALL): " << weights_(in, selComp) << "\n";
      }
    }
  
    System::out() << "INIT OK\n";

    if ( useBias_ )
    {
      iBiases_.resize ( size_ );
      biases_. resize ( size_ );
      iBiases_ = -1;
      biases_  = 0.0;

      IdxVector newbaxons = aset->addAxons ( size_ );
      dofs->addDofs ( newbaxons, biasType_   );
      System::out() << "Normal biases: Creating " << size_ << " axons\n";

      StateVector::get ( state, dofs, globdat );
      System::out() << "newbaxons: " << newbaxons << "\n";
      System::out() << "state: " << state << "\n";

      dofs->getDofIndices ( iBiases_, newbaxons, biasType_ );
      state[iBiases_] = 0.0;

      System::out() << "iBiases_: " << iBiases_ << "\n";
    }
    
    
     System::out() << "Final weights" << weights_ << "\n";
  }
  //else
  //{
  //  inputLayer_ = true;
  //}

  myProps.find ( inputLayer_, LearningNames::FIRSTLAYER );
  myProps.find ( outputLayer_, LearningNames::LASTLAYER );

  System::out() << "Input? " << inputLayer_ << "\n";
  System::out() << "Output? " << outputLayer_ << "\n";

  axons_   = aset;
  neurons_ = nset;
  dofs_    = dofs;
  
    // Getting output normalizer for stresses

  String dummy;
  if ( props.find ( dummy, "normalizer" ) )
  {
    System::out() << "Found normalizer.\n";
    props.find ( onl_, "outNormalizer" );
  }
  else
  {
    Ref<TrainingData> data = TrainingData::get ( globdat, getContext() );
    onl_ = data->getOutNormalizer();
  }
   
  System::out() << "Predecessor: " << iNeurons_ << "\n";
  globdat.set ( LearningParams::PREDECESSOR, iNeurons_.clone() );

}

SparseLayer::~SparseLayer ()
{}

//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------

void SparseLayer::configure

  ( const Properties& props,
    const Properties& globdat )

{
  //outputLayer_ = inputLayer_ ? false :
  //               mirrored_   ? min ( iInpWts_ ) == 0 :
  //               max ( max ( iBiases_ ), max ( iInpWts_ ) ) == dofs_->dofCount() - 1;

}

//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------

void SparseLayer::getConfig

  ( const Properties& conf,
    const Properties& globdat )

{
}

//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------

bool SparseLayer::takeAction

  ( const String&     action,
    const Properties& params,
    const Properties& globdat )

{
  using jive::model::Actions;

  //if ( action == Actions::SHUTDOWN )
  //{
  //  System::out() << "Layer " << myName_ << " statistics:\n";
  //  System::out() << "Time spent propagating " << ptot_ << "\n";
  //  System::out() << "... matmul " << pmmul_ << "\n";
  //  System::out() << "... func   " << pfunc_ << "\n";
  //  System::out() << "... bias   " << pbias_ << "\n";
  //  System::out() << "... allocs " << palloc_ << "\n";
  //  System::out() << "Time spent backpropagating " << bptot_ << "\n";
  //  System::out() << "... matmul " << bpmmul_ << "\n";
  //  System::out() << "... func   " << bpfunc_ << "\n";
  //  System::out() << "... bias  "  << bpbias_ << "\n";
  //  System::out() << "... alloc "  << bpalloc_ << "\n";
  //  System::out() << '\n';

  //  return true;
  //}

  if ( action == LearningActions::UPDATE )
  {
    if ( !inputLayer_ )
    {
      update_ ( globdat );
    }

    return true;
  }

  if ( action == LearningActions::GETDETERMINANT )
 {
    Ref<NData> data;
    params.get ( data, LearningParams::DATA );

    if ( inputLayer_ )
    {
      if ( data->inpSize() != size_ )
      {
        sizeError ( JEM_FUNC, "Neural input", size_, data->inpSize() );
      }

      data->init ( neurons_->size() );

      select ( data->values,      iNeurons_, ALL ) = data->inputs;
      select ( data->activations, iNeurons_, ALL ) = data->inputs;
      
      IdxVector selNeurons ( 3 );
      selNeurons[0] = 0;
      selNeurons[1] = 4;
      selNeurons[2] = 8;

      select ( data->activations, selNeurons, ALL ) -= 1.0;
    
      System::out() << "Input: select(values): " << select ( data->values, iNeurons_, ALL) << "\n";
      //System::out() << "Input: select(activations): " << select ( data->activations, iNeurons_, ALL) << "\n";

      return true;
    }

    getDeterminant_ ( data, globdat );
    System::out() << "\n";

    return true;
  }

  if ( action == LearningActions::PROPAGATE ||
       action == LearningActions::RECALL       )
  {
    Ref<NData> data;
    params.get ( data, LearningParams::DATA );

    if ( inputLayer_ )
    {
      if ( data->inpSize() != size_ )
      {
        sizeError ( JEM_FUNC, "Neural input", size_, data->inpSize() );
      }

      data->init ( neurons_->size() );

      select ( data->values,      iNeurons_, ALL ) = data->inputs;
      select ( data->activations, iNeurons_, ALL ) = data->inputs;
      
      IdxVector selNeurons ( 3 );
      selNeurons[0] = 0;
      selNeurons[1] = 4;
      selNeurons[2] = 8;

      select ( data->activations, selNeurons, ALL ) -= 1.0;
    
     // System::out() << "Input: select(values): " << select ( data->values, iNeurons_, ALL) << "\n";
      //System::out() << "Input: select(activations): " << select ( data->activations, iNeurons_, ALL) << "\n";

      return true;
    }

    propagate_ ( data, globdat );
    System::out() << "\n";

    if ( outputLayer_ )
    {
      if ( data->outSize() != size_ )
      {
        sizeError ( JEM_FUNC, "Neural output", size_, data->outSize() );
      }

      data->outputs = select ( data->activations, iNeurons_, ALL );

    //  System::out() << "Output: select(values): " << select ( data->activations, iNeurons_, ALL ) << "\n";
    }

    return true;
  }

  if ( action == LearningActions::FORWARDJAC )
  {

    Ref<NData> data, rdata;
    params.get ( data,  LearningParams::DATA  );
    params.get ( rdata, LearningParams::RDATA );

    if ( inputLayer_ )
    {
      rdata->init ( neurons_->size() );

      rdata->inputs += select ( data->deltas, iNeurons_, ALL );

      select ( rdata->values, iNeurons_, ALL )      = rdata->inputs;
      select ( rdata->activations, iNeurons_, ALL ) = rdata->inputs;

      return true;
    }

    forwardJacobian_ ( data, rdata, globdat );

    return true;
  }

  if ( action == LearningActions::BACKPROPAGATE )
  {
    Ref<NData> data;
    params.get ( data, LearningParams::DATA );

    Vector grads;
    params.get ( grads, LearningParams::GRADS );

    if ( outputLayer_ )
    {
      select ( data->deltas, iNeurons_[selComp_], ALL ) = select ( data->outputs, selComp_, ALL );
      select ( data->deltas, iNeurons_[remComp_], ALL ) = 0.0;
// select ( data->deltas, iNeurons_, ALL ) = data->outputs;
    }

    if ( !inputLayer_ )
    {
      backPropagate_ ( data, grads, globdat );
    }

    return true;
  }

  if ( action == LearningActions::BACKWARDJAC )
  {
  
    Ref<NData> data, rdata;
    params.get ( data,  LearningParams::DATA  );
    params.get ( rdata, LearningParams::RDATA );

    Vector grads;
    params.get ( grads, LearningParams::GRADS );

    if ( outputLayer_ )
    {
      select ( rdata->deltas, iNeurons_, ALL ) = 0.0;
    }

    if ( !inputLayer_ )
    {
      backJacobian_ ( data, rdata, grads, globdat );
    }

    return true;
  }

  if ( action == LearningActions::GETJACOBIAN )
  {
    Ref<NData> data;
    params.get ( data, LearningParams::DATA );

    if ( !inputLayer_ )
    {
      getJacobian_ ( data, globdat );
    }

    return true;
  }

  return false;
}

//-----------------------------------------------------------------------
//   getInpDofs
//-----------------------------------------------------------------------

void SparseLayer::getInpDofs

  (       IdxMatrix& dofs )

{
  dofs.ref ( iInpWts_ );
}

//-----------------------------------------------------------------------
//   update_
//-----------------------------------------------------------------------

void SparseLayer::update_

  ( const Properties& globdat )

{
  Vector state;
  StateVector::get ( state, dofs_, globdat );

  if ( useBias_ )
  {
    biases_ = state[iBiases_];
  }

  if ( pruning_ )
  {
    for ( idx_t nb = 0; nb < nBlocks_; nb++ )
    {
      IdxVector selComp ( nBlocksPrev_ );
      IdxVector remComp ( preSize_ - nBlocksPrev_ );

      for ( idx_t in = 0; in < nComp_; in++ )
      {
        for ( idx_t ii = 0; ii < nBlocksPrev_; ii++)
        {
          idx_t computedSelComp = 0;
          selComp[ii] = in+ii*nComp_;

          for ( idx_t jj = 0; jj < 9; jj++ )
          {
            if ( jj != in )
            { 
              remComp[ii*8+jj+computedSelComp] = ii*nComp_+jj;
            }
            else
            { 
              computedSelComp = -1;
            }
          }
        }

        if ( debug_ )
	{
	  System::out() << "remComp " << remComp << "\n";
          System::out() << "selComp " << selComp << "\n";
        } 
	weights_(nb*nComp_+in, selComp) = state[iInpWts_(nb*nWtsBlock_+symIndex_[in],ALL)];
        weights_(nb*nComp_+in, remComp) = 0.0;
      }
    }

  }
  else
  {
    for ( idx_t in = 0; in < size_; ++in )
    {
      weights_(in,ALL) = state[iInpWts_(in,ALL)];
     //System::out() << "Updated weights (Sparse Layer): " << weights_(in, ALL) << "\n";
    }
  }
}

//-----------------------------------------------------------------------
//   propagate_
//-----------------------------------------------------------------------

void SparseLayer::propagate_

  ( const Ref<NData>  data,
    const Properties& globdat  )

{
  ptot_.start();

  palloc_.start();
  Matrix netValues ( size_, data->batchSize() );
  netValues = 0.0;

  Matrix acts        ( select ( data->activations, inpNeurons_, ALL ) );
  palloc_.stop();

  if ( debug_ ) System::out() << "acts: " << acts << "\n";

  if ( actWeights_ )
  {
    Matrix actWeights = weights_.clone();
    actfuncweights_ ( actWeights );
    
     if ( pruning_ )
     {
     /* if ( debug_ )
      {
        System::out() << "Propagate. Weights before fixing activated weights: " << actWeights << "\n";
      }
    */ 
      idx_t nip = (int) preSize_/9;

      if ( debug_ ) System::out() << "Starting to prune (propagate_). Presize: " << preSize_ << "\n";

     idx_t nCompsErased;

     for ( idx_t i = 0; i < 9; i++ )  // Total number of components
     {
       idx_t init = i + 1;
       idx_t ws = 0;
       idx_t remMax = i;

       for ( idx_t rem = 0; rem < remMax; rem++ )
       {
         actWeights(i, rem) = 0.0;
       }

       for ( idx_t p = 0; p < nip; p++ ) // Number of int. points - 1
       {
          if ( p == nip - 1 )
	  {
	    nCompsErased = 9 - remMax - 1;
	  }
	  else
	  {
	    nCompsErased = 8;
	  }

          for ( idx_t j = 0; j < nCompsErased; j++ ) // Number of components erased
          {
            actWeights(i, init+j+ws) = 0.0;
          }	       
          ws += 9;
        }
        
     //   System::out() << "i " << i << " weights after pruning " << actWeights(i, ALL) << "\n";
     }

     }
    
    pmmul_.start();
    //System::out() << "actweights " << actWeights << "\n";
    netValues = matmul ( actWeights, acts );
    pmmul_.stop();
  
 /*   if ( debug_ )
    {
      System::out() << "Propagate. Weights fixed after being activated: " << actWeights << "\n";
    }*/
  }
  else
  {
    pmmul_.start();    
    netValues = matmul ( weights_, acts );
    pmmul_.stop();
    if ( debug_ ) System::out() << "weights: " << weights_ << "\n";
  }

  pbias_.start();
  if ( useBias_ )
  {
    for ( idx_t is = 0; is < data->batchSize(); ++is )
    {
      netValues(ALL,is) += biases_;
      if ( debug_ ) System::out() << "bias: " << biases_ << "\n";
    }
  }
  pbias_.stop();

  select ( data->values, iNeurons_, ALL ) = netValues;

  if ( debug_ ) System::out() << "netValues: " << netValues << "\n";

  pfunc_.start();
  func_ ( netValues );
  pfunc_.stop();

  if ( debug_ ) System::out() << "new activations: " << netValues << "\n";
  select ( data->activations, iNeurons_, ALL ) = netValues;

  ptot_.stop();
}

//-----------------------------------------------------------------------
//   getDeterminant_
//-----------------------------------------------------------------------

void SparseLayer::getDeterminant_

  ( const Ref<NData>  data,
    const Properties& globdat  )

{
  ptot_.start();

  palloc_.start();
  Matrix netValues ( size_, data->batchSize() );
  netValues = 0.0;

  Matrix acts        ( select ( data->activations, inpNeurons_, ALL ) );
  palloc_.stop();

  if ( debug_ ) System::out() << "acts: " << acts << "\n";

  if ( actWeights_ )
  {
    Matrix actWeights = weights_.clone();
    actfuncweights_ ( actWeights );
    
     if ( pruning_ )
     {
      if ( debug_ )
      {
        System::out() << "Propagate. Weights before fixing activated weights: " << actWeights << "\n";
      }
     
      idx_t nip = (int) preSize_/3;

      if ( debug_ ) System::out() << "Begin pruning. Presize: " << preSize_ << "\n";
     
      idx_t ws = 0;
      idx_t init = 1;
      for ( idx_t p = 0; p < nip; ++p ) // Number of int. points - 1
      {
        for ( idx_t j = 0; j < 2; ++j ) // Number of components erased
        {
          actWeights(0, init+j+ws) = 0.0;
        }	       
        ws += 3;
      }

      ws = 0;
      init = 0;
      for ( idx_t p = 0; p < nip; ++p ) // Number of int. points - 1
      {
        actWeights(1, init+ws) = 0.0;
        actWeights(1, init+2+ws) = 0.0;        	       
        ws += 3;
      }    
      
      ws = 0;
      for ( idx_t p = 0; p < nip; ++p ) // Number of int. points - 1
      {
         for ( idx_t j = 0; j < 2; ++j ) // Number of components erased
         {
           actWeights(2, init+j+ws) = 0.0;
         }	       
         ws += 3;
       }    
     }
    
    pmmul_.start();
    netValues = matmul ( actWeights, acts );
    pmmul_.stop();
  
    if ( debug_ )
    {
      System::out() << "Propagate. Weights fixed after being activated: " << actWeights << "\n";
    }
  }
  else
  {
    pmmul_.start();    
    netValues = matmul ( weights_, acts );
    pmmul_.stop();
  }

  if ( debug_ ) System::out() << "weights: " << weights_ << "\n";

  pbias_.start();
  if ( useBias_ )
  {
    for ( idx_t is = 0; is < data->batchSize(); ++is )
    {
      netValues(ALL,is) += biases_;
      if ( debug_ ) System::out() << "bias: " << biases_ << "\n";
    }
  }
  pbias_.stop();

  select ( data->values, iNeurons_, ALL ) = netValues;

  if ( debug_ ) System::out() << "netValues: " << netValues << "\n";

  pfunc_.start();
  func_ ( netValues );
  pfunc_.stop();

  if ( debug_ ) System::out() << "new activations: " << netValues << "\n";
  select ( data->activations, iNeurons_, ALL ) = netValues;

  ptot_.stop();
}

//-----------------------------------------------------------------------
//   backPropagate_
//-----------------------------------------------------------------------

void SparseLayer::backPropagate_

  ( const Ref<NData>   data,
          Vector&      grads,
    const Properties&  globdat  )

{
  bptot_.start();
  bpalloc_.start();
  Matrix derivs ( select ( data->values, iNeurons_, ALL ) );
  bpalloc_.stop();

  if ( debug_ ) System::out() << "backPropagate. SparseLayer. derivs (values): " << derivs << "\n";

  bpfunc_.start();
  grad_ ( derivs );

  if ( debug_ ) System::out() << "backPropagate. SparseLayer. derivs (grad_): " << derivs << "\n";

  //select ( data->deltas, iNeurons_, ALL ) *= derivs;
  bpfunc_.stop();

  bpalloc_.start();
  Matrix acts        ( select ( data->activations, inpNeurons_, ALL ) );

  for ( idx_t nb = 0; nb < nBlocksPrev_; nb++ )
  {
    for ( idx_t in = 0; in < 9; in++ )
    {
      if ( in == 3 || in == 6 || in == 7 )
      {
        acts(nb*9+in, ALL) = 0.0;
      }
    }
  }

  if ( debug_ ) System::out() << "backPropagate. SparseLayer. acts: " << acts << "\n";

  //Matrix deltas ( select ( data->deltas, iNeurons_, ALL ) );
  Matrix deltas ( select ( data->deltas, iNeurons_, ALL ) * derivs );

  if ( debug_ ) 
  {
    System::out() << "backPropagate. SparseLayer. iNeuros: " << iNeurons_ << ".\n";
    System::out() << "backPropagate. SparseLayer. delta: " << select ( data->deltas, iNeurons_, ALL ) << ".\n";
    System::out() << "backPropagate. SparseLayer. delta*derivs: " << deltas << "\n";
  }

  Matrix gradmat ( size_, inpNeurons_.size() );
  gradmat = 0.0;
  bpalloc_.stop();

  bpmmul_.start();
  gradmat = matmul ( deltas, acts.transpose() );
  bpmmul_.stop();

  Matrix actweightscopy_ = weights_.clone();    
  Matrix weightscopy_ = weights_.clone();    
  
  if ( actWeights_ )
  {

    actfuncweights_ ( actweightscopy_ );
    gradfuncweights_ ( weightscopy_ );
    
  if ( pruning_ )
     {
      if ( debug_ )
      {
        System::out() << "backPropagate. Weights before fixing activated weights: " << weightscopy_ << "\n";
      }
     
      idx_t nip = (int) preSize_/nComp_;

      if ( debug_ ) System::out() << "Starting to prune (backPropagate_). Presize: " << preSize_ << "\n";

     idx_t nCompsErased;

     for ( idx_t i = 0; i < 9; i++ )  // Total number of components
     {
       idx_t init = i + 1;
       idx_t ws = 0;
       idx_t remMax = i;

       for ( idx_t rem = 0; rem < remMax; rem++ )
       {
         weightscopy_(i, rem) = 0.0;
         actweightscopy_(i, rem) = 0.0;
       }

       for ( idx_t p = 0; p < nip; p++ ) // Number of int. points - 1
       {
          if ( p == nip - 1 )
	  {
	    nCompsErased = 9 - remMax - 1;
	  }
	  else
	  {
	    nCompsErased = 8;
	  }

          for ( idx_t j = 0; j < nCompsErased; j++ ) // Number of components erased
          {
            weightscopy_(i, init+j+ws) = 0.0;
            actweightscopy_(i, init+j+ws) = 0.0;
          }	       
          ws += 9;
        }
        
        if ( debug_ ) System::out() << "i " << i << " weights after pruning " << weightscopy_(i, ALL) << "\n";
     }

     }
     
    if ( debug_ )
    {
      System::out() << "Backpropagate. Weights after fixing activated weights: " << weightscopy_ << "\n";
    }

    TensorIndex i, j;

    gradmat (i,j) = gradmat (i,j) * weightscopy_ ( i,j );
  }

  if ( debug_ )
  {
    System::out() << "backPropagate. SparseLayer. deltas*acts: " << gradmat << "\n";
    System::out() << "backPropagate. SparseLayer. size_: " << size_ << "\n";
    System::out() << "backPropagate. SparseLayer. iInptWts: " << iInpWts_ << "\n";
  }
  
  // Adding gradients contribution

  if ( pruning_ )
  {
    // Only selects the gradients from the selected components (if pruning is enabled)
    // else all weights contribute
    for ( idx_t nb = 0; nb < nBlocks_; nb++ )
    {
      IdxVector selComp ( nBlocksPrev_ );
      idx_t contGrad = 0;

      for ( idx_t in = 0; in < nComp_; in++ )
      {

        for ( idx_t ii = 0; ii < nBlocksPrev_; ii++)
        {
          selComp[ii] = in+ii*nComp_;        
        }
        if ( debug_ )  System::out() << "selComp " << selComp << "\n";
  
        if ( symmetric_ )
        {
          if ( in != 3 && in != 6 && in != 7 )
	  {
            grads[iInpWts_(nb*nWtsBlock_+contGrad,ALL)] += gradmat(nb*nComp_+in,selComp);
	    contGrad += 1;
	  }
        }
        else
        {
          grads[iInpWts_(nb*nWtsBlock_+in,ALL)] += gradmat(nb*nComp_+in,selComp);
//   System::out() << "backPropagate. grads Sparse layer " << in << ": " << grads[iInpWts_(nb*nComp_+in, selComp)] << "\n";
        }
      }
    }
  }
  else
  {
    for ( idx_t in = 0; in < size_; ++in )
    {
      grads[iInpWts_(in,ALL)] += gradmat     (in,ALL); 

      bpbias_.start();
      if ( useBias_ )
      {
        grads[iBiases_[in]]     += sum ( deltas(in,ALL) );
        if ( debug_ ) System::out() << "backPropagate. SparseLayer. biases " << iBiases_[in] << ": " << sum ( deltas(in,ALL) ) << "\n";
      }
      bpbias_.stop();
      
 //   System::out() << "backPropagate. grads Sparse layer " << in << ": " << grads[iInpWts_(in, ALL)] << "\n";
    }
  } 

  
  bpmmul_.start();
  // NB: 08-10-19 - was +=, removed the +
  if ( actWeights_ )
  {
    select ( data->deltas, inpNeurons_, ALL ) = matmul ( actweightscopy_.transpose(), deltas );
  }
  else
  {
     select ( data->deltas, inpNeurons_, ALL ) = matmul ( weights_.transpose(), deltas );
  }

  if ( debug_ ) 
  {
    System::out() << "backPropagate. SparseLayer. inpNeurons: " << inpNeurons_ << "\n";
    System::out() << "backPropagate. SparseLayer. deltas (inp): " << select (data->deltas, inpNeurons_, ALL) << "\n";
    System::out() << "deltas: " << deltas << "\n";
    System::out() << "weights: " << weights_ << "\n";
  }

  bpmmul_.stop();
  bptot_.stop();
}

//-----------------------------------------------------------------------
//   forwardJacobian_
//-----------------------------------------------------------------------

void SparseLayer::forwardJacobian_

  ( const Ref<NData>  data,
    const Ref<NData>  rdata,
    const Properties& globdat  )

{
  Matrix netValues ( size_, data->batchSize() );
  netValues = 0.0;

  Matrix acts ( select ( rdata->activations, inpNeurons_, ALL ) );

  netValues = matmul ( weights_, acts );

  select ( rdata->values, iNeurons_, ALL ) = netValues;

  Matrix vals ( select ( data->values, iNeurons_, ALL ) );

  grad_ ( vals );

  select ( rdata->activations, iNeurons_, ALL ) = vals * netValues;

  //System::out() << "J-forward vals" << netValues << "\n";
  //System::out() << "J-forward activation " << vals*netValues << "\n";
}

//-----------------------------------------------------------------------
//   backJacobian_
//-----------------------------------------------------------------------

void SparseLayer::backJacobian_

  ( const Ref<NData>   data,
    const Ref<NData>   rdata,
          Vector&      grads,
    const Properties&  globdat  )

{
  Matrix derivs  ( select ( data->values, iNeurons_, ALL ) );
  Matrix derivs2 ( select ( data->values, iNeurons_, ALL ) );

  grad_ ( derivs  );
  hess_ ( derivs2 );

  //System::out() << "fp " << derivs << " fpp " << derivs2 << "\n";

  Matrix predeltas ( select ( data->deltas, iNeurons_, ALL ) );
  Matrix deltas    ( select ( data->deltas, iNeurons_, ALL ) * derivs );
  Matrix rdeltas   ( select ( rdata->deltas, iNeurons_, ALL ) );
  Matrix rvalues   ( select ( rdata->values, iNeurons_, ALL ) );

  rdeltas = rdeltas*derivs + predeltas*derivs2*rvalues;

  //System::out() << "rdeltas " << rdeltas << "\n";

  Matrix acts    ( select ( data->activations,  inpNeurons_, ALL ) );
  Matrix racts   ( select ( rdata->activations, inpNeurons_, ALL ) );

  Matrix gradmat ( size_, inpNeurons_.size() );
  gradmat = 0.0;

  gradmat = matmul ( rdeltas, acts.transpose() ) + matmul ( deltas, racts.transpose() );

  for ( idx_t in = 0; in < size_; ++in )
  {
    grads[iInpWts_(in,ALL)] += gradmat (in,ALL);
    
    if ( useBias_ )
    {
      grads[iBiases_[in]] += sum ( rdeltas(in,ALL) );
    }
  }
  
  select ( rdata->deltas, inpNeurons_, ALL ) = matmul ( weights_.transpose(), rdeltas );  
}

//-----------------------------------------------------------------------
//   getJacobian_
//-----------------------------------------------------------------------

void SparseLayer::getJacobian_

  ( const Ref<NData>  data,
    const Properties& globdat )

{
  idx_t bsize = data->batchSize();
  idx_t jsize = data->outSize();

  if ( debug_ ) 
  {
     System::out() << "SparseLayer. getJacobian initial:\n" << jacobian_ << "\n";
     System::out() << "jsize: " << jsize << " bsize: " << bsize << 
	  " inpNeurons size: " << inpNeurons_.size() << "\n";
  }

  if ( jacobian_.size(0) != jsize * bsize )
  {
    jacobian_.resize ( jsize*bsize, inpNeurons_.size() );
    jacobian_ = 0.0;
  }

  Matrix derivs ( select ( data->values, iNeurons_, ALL ) );

  grad_ ( derivs );

  if ( onl_ != nullptr ) 
  {
    Vector normalized = onl_->getDenormFactor ( derivs [0] );
    for ( idx_t i = 0; i < 9; i++ ) derivs ( i, 0 ) = derivs( i, 0 ) * normalized [ i ];
  }

/*  if ( debug_ )
  {
    System::out() << "derivs:\n" << derivs << "\n";
    System::out() << "weights_:\n" << weights_ << "\n";
  }
*/

  if ( outputLayer_ )
  {

   Matrix weightscopy = weights_.clone();
  if ( actWeights_ )
  {
    actfuncweights_ ( weightscopy );
    
    if ( pruning_ )
     {
     /* if ( debug_ )
      {
        System::out() << "backPropagate. Weights before fixing activated weights: " << weightscopy << "\n";
      }*/
     
      idx_t presize = inpNeurons_.size();
      idx_t nip = (int) presize/9;

      if ( debug_ ) System::out() << "Starting to prune (getJacobian). Presize: " << presize << "\n";

     idx_t nCompsErased;

     for ( idx_t i = 0; i < 9; i++ )  // Total number of components
     {
       idx_t init = i + 1;
       idx_t ws = 0;
       idx_t remMax = i;

       for ( idx_t rem = 0; rem < remMax; rem++ )
       {
         weightscopy(i, rem) = 0.0;
       }

       for ( idx_t p = 0; p < nip; p++ ) // Number of int. points - 1
       {
          if ( p == nip - 1 )
	  {
	    nCompsErased = 9 - remMax - 1;
	  }
	  else
	  {
	    nCompsErased = 8;
	  }

          for ( idx_t j = 0; j < nCompsErased; j++ ) // Number of components erased
          {
            weightscopy(i, init+j+ws) = 0.0;
          }	       
          ws += 9;
        }
        
        //System::out() << "i " << i << " weights after pruning " << weightscopy(i, ALL) << "\n";
     }

     }
    
  }

    for ( idx_t b = 0; b < bsize; ++b )
    {
      for ( idx_t i = 0; i < size_; ++i )
      {
        if ( actWeights_ )
        { 
         jacobian_(i+b*jsize,ALL) = weightscopy (i,ALL) * derivs(i,b);
	}
        else
        {
         jacobian_(i+b*jsize,ALL) = weights_(i,ALL) * derivs(i,b);
        }
      }
    }
  }
  else
  {


   Matrix weightscopy_ = weights_.clone();
  if ( actWeights_ )
  {
    actfuncweights_ ( weightscopy_ );
  }


    for ( idx_t b = 0; b < bsize; ++b )
    {
      for ( idx_t i = 0; i < size_; ++i )
      {
        data->jacobian ( slice(b*jsize,(b+1)*jsize), i ) *= derivs(i,b);
      }
    }
   
    if ( actWeights_ )
    {
      jacobian_ = matmul ( data->jacobian, weightscopy_ );
    }
    else
    {
      jacobian_ = matmul ( data->jacobian, weights_ );
    }
  }

  if ( debug_ ) System::out() << "SparseLayer. getJacobian updated:\n" << jacobian_ << "\n";

  data->jacobian.ref ( jacobian_ );
}

//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   newSparseLayer
//-----------------------------------------------------------------------


Ref<Model>            newSparseLayer

  ( const String&       name,
    const Properties&   conf,
    const Properties&   props,
    const Properties&   globdat )

{
  // Return an instance of the class.

  return newInstance<SparseLayer> ( name, conf, props, globdat );
}

//-----------------------------------------------------------------------
//   declareSparseLayer
//-----------------------------------------------------------------------

void declareSparseLayer ()
{
  using jive::model::ModelFactory;

  // Register the StressModel with the ModelFactory.

  ModelFactory::declare ( "Sparse", newSparseLayer );
}

