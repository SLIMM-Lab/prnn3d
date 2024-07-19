/*
 * Copyright (C) 2022 TU Delft. All rights reserved.
 *
 * Class that implements a neural layer connected by blocks. Each
 * of its blocks will connect themselves with the previous blocks 
 * of neurons of the previous layer (towards the input layer). 
 * Note that this does not guarantee fully connectivity to the 
 * next layer.
 *
 * Author: Marina Maia, m.alvesmaia@tudelft.nl
 * Date:   Oct 2022
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
#include "BlockDecLayer.h"
#include "LearningNames.h"

#include "utilities.h"

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

const char* BlockDecLayer::SIZE           = "size";
const char* BlockDecLayer::ACTIVATION     = "activation";
const char* BlockDecLayer::INITIALIZATION = "init";
const char* BlockDecLayer::USEBIAS        = "useBias";
const char* BlockDecLayer::DEBUG          = "debug";
const char* BlockDecLayer::PRUNING        = "pruning";
const char* BlockDecLayer::POSWEIGHTS     = "posWeights";

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------

BlockDecLayer::BlockDecLayer

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat ) : Model ( name )

{
  String func ( "identity" );

  size_ = 0;
  blockDim_ = 3;
  blockSize_ = 9;
  nBlocks_ = 1;
  nBlocksPrev_ = 1;

  inputLayer_  = false;
  outputLayer_ = false;
  mirrored_    = false;
  useBias_     = false;
  debug_       = false; 
  pruning_      = false;
  posWeights_   = false;
  actWeights_ = false;
  postMult_ = false;
  transpMat_ = false;
  lower_     = false; 

  String actFuncWeights = "relu";
  prev = 0;

  Properties myProps = props.getProps ( myName_ );
  Properties myConf  = conf.makeProps ( myName_ );

  System::out() << "myProps " << myProps << "\n";
  myProps.get  ( size_, SIZE );
  myProps.find ( func, ACTIVATION );
  myProps.find ( useBias_, USEBIAS );
  myProps.find ( debug_, DEBUG );
  myProps.find ( pruning_, PRUNING );
  myProps.find ( posWeights_, POSWEIGHTS );
  myProps.find ( prev, "prev" );
  myProps.find ( actWeights_, "activateWeights" );
  myProps.find ( actFuncWeights, "activationWeights" );
  myProps.find ( postMult_, "postMult" );
  myProps.find ( transpMat_, "transpMat" );
  myProps.find ( lower_, "lower" );
  sizeTrue_ = size_;

  System::out() << "Activation function used for the weights: " << actFuncWeights << "\n";

  nBlocks_ = int (size_/blockSize_);

  myConf. set ( SIZE, size_ );
  myConf. set ( ACTIVATION, func );
  myConf. set ( USEBIAS, useBias_ );
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

  iNeurons_.resize ( size_ );
  iNeurons_ = 0;

  Ref<XAxonSet>   aset = XAxonSet  ::get ( globdat, getContext()    );
  Ref<XNeuronSet> nset = XNeuronSet::get ( globdat, getContext()    );
  Ref<XDofSpace>  dofs = XDofSpace:: get ( aset->getData(), globdat );

  for ( idx_t in = 0; in < size_; ++in )
  {
    iNeurons_[in] = nset->addNeuron();
  }

  weightType_ = dofs->findType ( LearningNames::WEIGHTDOF );
  biasType_   = dofs->findType ( LearningNames::BIASDOF   );

  Ref<Model> layer;

  if ( myProps.find ( layer, LearningParams::IMAGE ) )
  {
    Ref<BlockDecLayer> image = dynamicCast<BlockDecLayer> ( layer );

    System::out() << "TESTE Block.\n";

    if ( image == nullptr )
    {
      throw Error ( JEM_FUNC, "Broken symmetry" );
    }

    globdat.get ( inpNeurons_, LearningParams::PREDECESSOR );

    System::out() << "nBlocks_ " << nBlocks_ << " inpNeurons_ " << inpNeurons_ << "\n";    

    idx_t presize = inpNeurons_.size();
    nBlocksPrev_ = int ( presize / blockSize_ );

    System::out() << "nBlocksprev_ " << nBlocksPrev_ << "\n";

    nBlocksTrue_ = nBlocksPrev_;
    nBlocks_ = 1;
    size_ = 9;
    sizeTrue_ = presize;

    iInpWts_.resize ( nBlocksTrue_, 6 );
    iInpWts_ = -1;

    weightsDof_.resize ( nBlocksTrue_, 6);
    weightsDofcopy_.resize ( nBlocksTrue_, 6);
 
    image->getInpDofs ( iInpWts_ );

    System::out() << "dofs copied " << iInpWts_ << "\n";

    weights_.resize ( nBlocksTrue_*blockDim_, blockDim_ );
    weights_ = 0.0;

    weightsExp_.resize ( nBlocksTrue_*blockSize_, blockSize_ );
    weightsExp_ = 0.0;

    Vector state;

    StateVector::get ( state, dofs, globdat );

    System::out() << "nBlocks_ " << nBlocks_ << "\n";
    System::out() << "nBlocksTrue_ " << nBlocksTrue_ << "\n";

    for ( idx_t in = 0; in < nBlocksTrue_; ++in )
    {
      weightsDof_(in,ALL) = state[iInpWts_(in,ALL)];
      System::out() << "iInpWts_ mirrored (blockLayer)" << iInpWts_(in,ALL) << "\n";
    }

    System::out() << "weightsdof " << weightsDof_ << "\n";
    weightsDofcopy_ = weightsDof_.clone();
    
    Matrix wAux ( blockDim_, blockDim_ );
    Matrix wAuxExp ( blockSize_ , blockSize_ );

    for ( idx_t nb = 0; nb < nBlocksTrue_; nb++ )
    {
       wAux = 0.0;
       wAux ( 0, 0 ) = weightsDof_ ( nb, 0 );
       wAux ( 0, 1 ) = weightsDof_ ( nb, 1 );
       wAux ( 0, 2 ) = weightsDof_ ( nb, 2 );
       wAux ( 1, 1 ) = weightsDof_ ( nb, 3 );
       wAux ( 1, 2 ) = weightsDof_ ( nb, 4 );
       wAux ( 2, 2 ) = weightsDof_ ( nb, 5 );

       wAuxExp = 0.0;
       wAuxExp (0,0) = wAuxExp (1,1) = wAuxExp(2,2) = weightsDof_ (nb, 0 );
       wAuxExp (3,3) = wAuxExp (4,4) = wAuxExp(5,5) = weightsDof_ (nb, 3 );
       wAuxExp (6,6) = wAuxExp (7,7) = wAuxExp(8,8) = weightsDof_ (nb, 5 );
       wAuxExp (0,3) = wAuxExp (1,4) = wAuxExp(2,5) = weightsDof_ (nb, 1 ); 
       wAuxExp (0,6) = wAuxExp (1,7) = wAuxExp(2,8) = weightsDof_ (nb, 2 ); 
       wAuxExp (3,6) = wAuxExp (4,7) = wAuxExp(5,8) = weightsDof_ (nb, 4 ); 

       if ( postMult_ )
       {
         if ( transpMat_ )
	 {
 	   weights_ ( slice (nb*blockDim_, (nb+1)*blockDim_ ), ALL ) = wAux.clone(); 
         }
	 else
	 {
          weights_ ( slice (nb*blockDim_, (nb+1)*blockDim_ ), ALL ) = wAux.transpose().clone(); 
         }
       }
       else
       {
          weightsExp_ ( slice (nb*blockSize_, (nb+1)*blockSize_), ALL) = wAuxExp.transpose().clone();
          weights_ ( slice (nb*blockDim_, (nb+1)*blockDim_ ), ALL ) = wAux.transpose().clone(); 
       }
    }

    System::out() << "weights " << weights_ << "\nweightsdofs " << weightsDof_ << "\n";

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

    mirrored_ = true;
  }
  else if ( globdat.find ( inpNeurons_, LearningParams::PREDECESSOR ) )
  {
    System::out() << "nBlocks " << nBlocks_ << " size_ " << size_ << "\n";

    idx_t presize = inpNeurons_.size();

    nBlocksPrev_ = int ( presize / blockSize_ );

    System::out() << "nBlocks prev " << nBlocksPrev_ << "\n";

    IdxVector idofs ( 6 ); // number of dofs per block

    if ( nBlocks_ == 1 )
    {
      nBlocksTrue_ = nBlocksPrev_;
    }
    else
    {
      nBlocksTrue_ = nBlocks_;
    }
   
    iInpWts_.resize ( nBlocksTrue_, 6 );
    iInpWts_ = -1;

    weights_.resize ( nBlocksTrue_*blockDim_, blockDim_ );
    weights_ = 0.0;
    
    weightsExp_.resize( nBlocksTrue_*blockSize_, blockSize_ );
    weightsExp_ = 0.0;

    Matrix wAuxExp ( blockSize_, blockSize_ );
    Matrix wAux ( blockDim_, blockDim_ );

    weightsDof_.resize ( nBlocksTrue_, 6);
    weightsDofcopy_.resize ( nBlocksTrue_, 6);
    
    weightsDof_= 0.0; weightsDofcopy_ = 0.0;

    init_ ( weightsDof_, globdat );
    
    weightsDofcopy_ = weightsDof_.clone();

    System::out() << "weights " << weights_ << " weightsDof " << weightsDof_ << "\n";

    for ( idx_t nb = 0; nb < nBlocksTrue_; nb++ )
    {
       // Initialize upper triangular matrix
     
       wAux = 0.0;

       wAux ( 0, 0 ) = weightsDof_ ( nb, 0 );
       wAux ( 0, 1 ) = weightsDof_ ( nb, 1 ); 
       wAux ( 0, 2 ) = weightsDof_ ( nb, 2 );
       wAux ( 1, 1 ) = weightsDof_ ( nb, 3 );
       wAux ( 1, 2 ) = weightsDof_ ( nb, 4 );
       wAux ( 2, 2 ) = weightsDof_ ( nb, 5 );
      
       // Expanded matrix

       wAuxExp = 0.0;
       wAuxExp (0,0) = wAuxExp (1,1) = wAuxExp(2,2) = weightsDof_ (nb, 0 );
       wAuxExp (3,3) = wAuxExp (4,4) = wAuxExp(5,5) = weightsDof_ (nb, 3 );
       wAuxExp (6,6) = wAuxExp (7,7) = wAuxExp(8,8) = weightsDof_ (nb, 5 );
       wAuxExp (0,3) = wAuxExp (1,4) = wAuxExp(2,5) = weightsDof_ (nb, 1 ); 
       wAuxExp (0,6) = wAuxExp (1,7) = wAuxExp(2,8) = weightsDof_ (nb, 2 ); 
       wAuxExp (3,6) = wAuxExp (4,7) = wAuxExp(5,8) = weightsDof_ (nb, 4 ); 
     
       if ( lower_ )
       {
         weights_ ( slice ( nb*blockDim_, (nb+1)*blockDim_ ), ALL ) = wAux.transpose().clone();
         weightsExp_ ( slice (nb*blockSize_, (nb+1)*blockSize_), ALL) = wAuxExp.transpose().clone();
       }
       else
       {
         weights_ ( slice ( nb*blockDim_, (nb+1)*blockDim_ ), ALL ) = wAux.clone();
         weightsExp_ ( slice (nb*blockSize_, (nb+1)*blockSize_), ALL) = wAuxExp.clone();
       }
    }

    Vector state;

    IdxVector newaxons  = aset->addAxons ( 6 * nBlocksTrue_ );
    System::out() << "Block layer. inpWeights: Creating " << 6 * nBlocksTrue_ << " axons\n";
    System::out() << "Presize: " << presize << " size: " << size_ << "\n";
    System::out() << "inpNeurons:" << inpNeurons_ << "\n";

    dofs->addDofs ( newaxons,  weightType_ );

    StateVector::get ( state, dofs, globdat );

    for ( idx_t in = 0; in < nBlocksTrue_; ++in )
    {
      IdxVector myaxons ( newaxons[slice (in*6,(in+1)*6 )] );

      System::out() << "block i: " << in << "\n";
      System::out() << "myaxons: " << myaxons << "\n";

      dofs->getDofIndices ( idofs,  myaxons,   weightType_ );
      System::out() << "idofs: " << idofs << "\n";
      iInpWts_( in, ALL )= idofs;
      state[idofs] = weightsDof_(in, ALL);
      System::out() << "state[idofs] " << weightsDof_(in, ALL) << "\n";
    }

    System::out() << "Weights " << weights_ << "\n";

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
    
      // Weight pruning
   
    // System::out() << "state: " << state << "\n";
      idx_t ws;
      idx_t nip = (int) presize/3;
    
     if ( pruning_ )
     {
      idx_t init = nip*3*3 + nip*3 + 1;
//     idx_t init = prev*3 + prev + nip*3*prev + nip*3 + 1;

     System::out() << "Begin pruning\n";
     for ( idx_t k = 0; k < 3; ++k) // Number of components
     { 
       ws = 0;
       for ( idx_t p = 0; p < nip-1; ++p ) // Number of int. points - 1
       {
         for ( idx_t j = 0; j < 2; ++j ) // Number of components erased
         {
           state[init+j+ws] = 0.0;
         }	       
         ws += 3;
       }
      
       if ( k != 2 )
       { 
       for ( idx_t i = 0; i < 3; ++i )
       {
 	 state[init+i+ws] = 0.0;
       }
       }

       init += nip*3+1;
      }
    }
    
     /* 
     for (idx_t in = 0; in < size_; ++in )
     {
	weights_(in, ALL) = state[iInpWts_(in, ALL)];
     }*/
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

  System::out() << "Predecessor: " << iNeurons_ << "\n";
  globdat.set ( LearningParams::PREDECESSOR, iNeurons_.clone() );

}

BlockDecLayer::~BlockDecLayer ()
{}

//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------

void BlockDecLayer::configure

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

void BlockDecLayer::getConfig

  ( const Properties& conf,
    const Properties& globdat )

{
}

//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------

bool BlockDecLayer::takeAction

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

//  System::out() << "takeAction " << action << "\n";

  if ( action == LearningActions::UPDATE )
  {
    if ( !inputLayer_ )
    {
      update_ ( globdat );
    }

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

     // System::out() << "Input: select(values): " << select ( data->values, iNeurons_, ALL) << "\n";
     // System::out() << "Input: select(activations): " << select ( data->activations, iNeurons_, ALL) << "\n";

      return true;
    }

    propagate_ ( data, globdat );
    System::out() << "\n";

    if ( outputLayer_ )
    {

//      System::out() << "Output: select(values): " << iNeurons_[slice(0, size_)] <<"\n";

  /*    if ( data->outSize() != size_ )
      {
        sizeError ( JEM_FUNC, "Neural output", size_, data->outSize() );
      }
*/
      data->outputs = select ( data->activations, iNeurons_[slice(0, size_)], ALL );
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

     // System::out() << "Input: select(values): " << select ( data->values, iNeurons_, ALL) << "\n";
     // System::out() << "Input: select(activations): " << select ( data->activations, iNeurons_, ALL) << "\n";

      return true;
    }

    getDeterminant_ ( data, globdat, params);
    System::out() << "\n";

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
      if ( nBlocks_ == 1 )
      {
        for ( idx_t nb = 0; nb < nBlocksTrue_; nb++ ) select ( data->deltas, iNeurons_[slice(nb*blockSize_, (nb+1)*blockSize_)], ALL ) = data->outputs;
      }
      else
      {
        select ( data->deltas, iNeurons_[slice(0, size_)], ALL ) = data->outputs;
      }
      
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

void BlockDecLayer::getInpDofs

  (       IdxMatrix& dofs )

{
  dofs.ref ( iInpWts_ );
}

//-----------------------------------------------------------------------
//   update_
//-----------------------------------------------------------------------

void BlockDecLayer::update_

  ( const Properties& globdat )

{
  Vector state;
  StateVector::get ( state, dofs_, globdat );

  if ( useBias_ )
  {
    biases_ = state[iBiases_];
  }

  for ( idx_t in = 0; in < nBlocks_; ++in )
  {
    weightsDof_(in,ALL) = state[iInpWts_(in,ALL)];
  }
  
  weightsDofcopy_ = weightsDof_.clone();
 
  Matrix wAux ( blockDim_, blockDim_ );
  Matrix wAuxExp ( blockSize_, blockSize_ );  

  for ( idx_t nb = 0; nb < nBlocks_; nb++ )
  {
       wAux = 0.0;
       wAux ( 0, 0 ) = weightsDof_ ( nb, 0 );
       wAux ( 0, 1 ) = weightsDof_ ( nb, 1 );
       wAux ( 0, 2 ) = weightsDof_ ( nb, 2 );
       wAux ( 1, 1 ) = weightsDof_ ( nb, 3 );
       wAux ( 1, 2 ) = weightsDof_ ( nb, 4 );
       wAux ( 2, 2 ) = weightsDof_ ( nb, 5 );

       wAuxExp = 0.0;
       wAuxExp (0,0) = wAuxExp (1,1) = wAuxExp(2,2) = weightsDof_ (nb, 0 );
       wAuxExp (3,3) = wAuxExp (4,4) = wAuxExp(5,5) = weightsDof_ (nb, 3 );
       wAuxExp (6,6) = wAuxExp (7,7) = wAuxExp(8,8) = weightsDof_ (nb, 5 );
       wAuxExp (0,3) = wAuxExp (1,4) = wAuxExp(2,5) = weightsDof_ (nb, 1 ); 
       wAuxExp (0,6) = wAuxExp (1,7) = wAuxExp(2,8) = weightsDof_ (nb, 2 ); 
       wAuxExp (3,6) = wAuxExp (4,7) = wAuxExp(5,8) = weightsDof_ (nb, 4 ); 

       if ( mirrored_ || lower_ )
       {
         if ( postMult_ )
	 { 
	   if ( transpMat_ )
	   {
             weights_ ( slice (nb*blockDim_, (nb+1)*blockDim_ ), ALL ) = wAux.clone();
	   }
	   else
	   {
             weights_ ( slice (nb*blockDim_, (nb+1)*blockDim_ ), ALL ) = wAux.transpose().clone();
	   }
	 }
	 else
	 {
           weights_ ( slice (nb*blockDim_, (nb+1)*blockDim_ ), ALL ) = wAux.transpose().clone();
	   weightsExp_ ( slice (nb*blockSize_, (nb+1)*blockSize_ ), ALL ) = wAuxExp.transpose().clone();
	 }
       }
       else
       {
         weights_ ( slice (nb*blockDim_, (nb+1)*blockDim_ ), ALL ) = wAux.clone();
	 weightsExp_ ( slice (nb*blockSize_, (nb+1)*blockSize_ ), ALL ) = wAuxExp.clone();
       }
  }

  if ( debug_ ) System::out() << "BlockDecLayer. update. weights_ " << weightsDof_ << "\n";
  //if ( debug_ ) System::out() << "BlockDecLayer. update. weightsExp_ " << weightsExp_ << "\n";

}

//-----------------------------------------------------------------------
//   propagate_
//-----------------------------------------------------------------------

void BlockDecLayer::propagate_

  ( const Ref<NData>  data,
    const Properties& globdat  )

{
  ptot_.start();
  palloc_.start();
  if ( debug_ ) System::out() << "Started propagating blockLayer\n";
  Matrix netValues ( sizeTrue_, data->batchSize() );
  netValues = 0.0;
  
  Matrix actValues ( size_, data->batchSize() );
  actValues = 0.0;

  Matrix defgrad ( blockDim_, blockDim_ );
  Matrix actWeights ( blockDim_, blockDim_ );
  
  Matrix defgradblock ( blockDim_, blockDim_ );
  defgradblock = 0.0;
  
  Matrix acts        ( select ( data->activations, inpNeurons_, ALL ) );
  palloc_.stop();

  if ( debug_ ) System::out() << "acts: " << acts << "\n";

  pmmul_.start();  
  idx_t cont = 0; 

  for ( idx_t p = 0; p < data->batchSize(); p++ )
  {
    for ( idx_t nb = 0; nb < nBlocksTrue_; nb++ )
    { 
      for ( idx_t i = 0; i < blockDim_; i++ )
      {
        for ( idx_t j = 0; j < blockDim_; j++ )
        {
          defgrad(i, j) = acts(cont, p);
          cont += 1;
        }
      }

      if ( nBlocksPrev_ == 1 ) cont = 0;

      if ( debug_ ) 
      {
        System::out() << "blockLayer. defgrad " << defgrad << "\n";
      }

      if ( actWeights_ )
      {
        actWeights =  weights_(slice(nb*blockDim_, (nb+1)*blockDim_), ALL).clone();
        actfuncweights_ ( actWeights );

        if ( postMult_ )
        {
	  System::out() << "Inverting order of multiplication.\n";
          matmul ( defgradblock, defgrad, actWeights );
        }
        else
        {
          matmul ( defgradblock, actWeights, defgrad );
        }
      
        if ( debug_ ) System::out() << "weights block " << nb << " " << actWeights << "\n";  

      }
      else
      {
         if ( debug_ ) System::out() << "weights block " << nb << " " << weights_(slice(nb*blockDim_,(nb+1)*blockDim_), ALL) << "\n";
        
         if ( postMult_ )
         {
           matmul ( defgradblock, defgrad, weights_(slice(nb*blockDim_, (nb+1)*blockDim_), ALL) );
         }
         else
         {
           matmul ( defgradblock, weights_(slice(nb*blockDim_, (nb+1)*blockDim_), ALL), defgrad );
         }
      }
      
      if ( debug_ )
      {
        System::out() << "defgrad x weights = " << defgradblock << "\n";
      }

      if ( mirrored_ || lower_ )
      {
        // Saving it to netvalues
        
        for ( idx_t i = 0; i < blockDim_; i++ )
        {
          for ( idx_t j = 0; j < blockDim_; j++ )
          {
	    if ( nBlocks_ == 1 ) 
	    {
              netValues ( nb*blockSize_ + i*blockDim_+j, p ) = defgradblock (i,j);
              actValues ( i*blockDim_+j, p ) += defgradblock (i,j);
	    }
	    else
	    {
              netValues ( nb*blockSize_ + i*blockDim_+j, p ) = defgradblock (i,j);
              actValues ( nb*blockSize_ + i*blockDim_+j, p ) = defgradblock (i,j);
	    }
          }
        }

//	System::out() << "Checkpoint " << nb << " detWact " << determinant ( actWeights ) << " w " << actWeights << " detF " << detF << "\n";
      }
      else
      {
        for ( idx_t i = 0; i < blockDim_; i++ )
        {
          for ( idx_t j = 0; j < blockDim_; j++ )
          {
	    if ( nBlocks_ == 1 )
	    {
              netValues (nb*blockSize_ + i*blockDim_+j, p) = defgradblock (i,j);
              actValues (i*blockDim_+j, p) += defgradblock (i,j);
	    }
	    else
	    {
              netValues (nb*blockSize_ + i*blockDim_+j, p) = defgradblock (i,j);
              actValues (nb*blockSize_ + i*blockDim_+j, p) = defgradblock (i,j);
	    }
          }
        }
      }
    }
    cont = 0; 
    }
    pmmul_.stop();

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
  select ( data->activations, iNeurons_[slice(0, size_)], ALL ) = actValues;
  
  if ( debug_ )
  {
    System::out() << "netValues: " << netValues << "\n";
    System::out() << "actValues: " << actValues << "\n";
    System::out() << "iNeurons: " << iNeurons_ << "\n";
  }
  
  ptot_.stop();
}

//-----------------------------------------------------------------------
//   backPropagate_
//-----------------------------------------------------------------------

void BlockDecLayer::backPropagate_

  ( const Ref<NData>   data,
          Vector&      grads,
    const Properties&  globdat  )

{

  bptot_.start();
  bpalloc_.start();

  Matrix derivs ( select ( data->values, iNeurons_, ALL ) );
  bpalloc_.stop();

  if ( debug_ ) System::out() << "backPropagate. BlockDecLayer. derivs (values): " << derivs << "\n";

  bpfunc_.start();
  
  for ( idx_t p = 0; p < data->batchSize(); p++ )
  {
     for ( idx_t nb = 0; nb < nBlocksTrue_; nb++ )
     {
         Matrix defgradblock ( blockDim_, blockDim_ );
        
         for ( idx_t i = 0; i < blockDim_; i++ )
         {
            for ( idx_t j = 0; j < blockDim_; j++ )
            {
              defgradblock(i, j) = derivs(nb*blockSize_ + i*blockDim_ +j, p);
            }
         }
      
         if ( mirrored_ || lower_ )
         {

           // Apply softplus on the diagonals of F
         
          // Matrix defgradblocknosp = defgradblock.clone();

         //  actfuncweights_ ( defgradblock );

           // Define gradients

             grad_ ( defgradblock ); 
	//     gradfuncweights_ ( defgradblocknosp ); 
	     defgradblock = defgradblock;
         }
         else
         {
           grad_ ( defgradblock );
         }
    
         for ( idx_t i = 0; i < blockDim_; i++ )
         {
            for ( idx_t j = 0; j < blockDim_; j++ )
            {
              derivs(nb*blockSize_ + i*blockDim_+j, p) = defgradblock (i, j);
            }
         }
    } 
  }
  
  if ( debug_ ) System::out() << "backPropagate. BlockDecLayer. derivs (grad_): " << derivs << "\n";

  //select ( data->deltas, iNeurons_, ALL ) *= derivs;
  bpfunc_.stop();

  bpalloc_.start();
  Matrix acts        ( select ( data->activations, inpNeurons_, ALL ) );

  if ( debug_ ) System::out() << "backPropagate. BlockDecLayer. acts: " << acts << "\n";

  //Matrix deltas ( select ( data->deltas, iNeurons_, ALL ) );
  Matrix deltas ( select ( data->deltas, iNeurons_, ALL ) * derivs );

  if ( debug_ ) 
  {
    System::out() << "backPropagate. BlockDecLayer. iNeuros: " << iNeurons_ << ".\n";
    System::out() << "backPropagate. BlockDecLayer. delta: " << select ( data->deltas, iNeurons_, ALL ) << ".\n";
    System::out() << "backPropagate. BlockDecLayer. delta*derivs: " << deltas << "\n";
    System::out() << "backPropagate. BlockDecLayer. derivs: " << derivs << "\n";
    System::out() << "backPropagate. BlockDecLayer. acts " << acts.size(0) << " x " << acts.size(1) << "\n";
  }

  Matrix gradmat ( nBlocksTrue_, 6 );
  gradmat = 0.0;
  bpalloc_.stop();
  bpmmul_.start();
  
  idx_t cont = 0; idx_t cont2 = 0; idx_t cont3 = 0; idx_t cont4 = 0;
  Matrix defgradblock ( blockDim_, blockDim_ );
  Matrix defgrad ( blockDim_, blockDim_ );
  Matrix deltasperblock ( blockDim_, blockDim_ );
  deltasperblock = 0.0;
  defgradblock = 0.0; defgrad = 0.0;

  Matrix deltasupcondensed ( inpNeurons_.size(), data->batchSize());
  deltasupcondensed = 0.0;
     
//  System::out() << "acts all " << acts << "\n";

   for ( idx_t p = 0; p < data->batchSize(); p++ )
   {
    for ( idx_t nb = 0; nb < nBlocksTrue_; nb++ )
    { 
      for ( idx_t i = 0; i < blockDim_; i++ )
      {
        for ( idx_t j = 0; j < blockDim_; j++ )
        {
          defgrad(i,j) = acts(cont, p);
          deltasperblock(i,j) = deltas(cont2, p);
          cont += 1;
	  cont2 += 1;
        }
      }

      if ( nBlocksPrev_ == 1 ) cont = 0;

      if ( debug_ )
      {
        System::out() << "acts per block " << defgrad << "\n";
        System::out() << "deltas per block " << deltasperblock <<"\n";
      }
   
      if ( postMult_ )
      { 
        matmul ( defgradblock, deltasperblock.transpose(), defgrad );
      }
      else
      { 
        matmul ( defgradblock, deltasperblock, defgrad.transpose() );
      }

      if ( debug_) System::out() << "ok matmul: " << defgradblock << "\n";

      for ( idx_t i = 0; i < blockDim_; i++ )
      {
        for ( idx_t j = 0; j < blockDim_; j++ )
        {
          if ( mirrored_ || lower_ )
	  {
	    if ( postMult_ )
	    {
	      if ( transpMat_ )
	      {
                if ( j >= i )
                {
                  gradmat (nb, cont3) += defgradblock (i, j);
	          cont3 += 1;             
                }	      
	      }
	      else
	      {
	        if ( i >= j )
                {
                 gradmat (nb, cont3) += defgradblock (i, j);
	         cont3 += 1;             
                }
	      }
	    }
	    else
	    {
	       if ( i >= j )
               {
	         if ( cont3 == 2 )
		 {
		   gradmat ( nb, 3 ) += defgradblock ( i, j);
		 }
		 else if ( cont3 == 3 )
		 {
		   gradmat ( nb, 2 ) += defgradblock (i, j);
		 }
		 else
		 {
                   gradmat (nb, cont3) += defgradblock (i, j);
		 }
	         cont3 += 1;             
               }	    
	    }
	  }
	  else
	  {
            if ( j >= i )
            {
              gradmat (nb, cont3) += defgradblock (i, j);
	       cont3 += 1;             
            }	    
	  }
	}
      }
      
      cont3 = 0;

      if ( debug_ ) System::out() << "ok gradmat\n";

      Matrix deltasupperblock ( blockDim_, blockDim_);
      if ( actWeights_ )
      {
         Matrix weightscopy_ = weights_(slice(nb*blockDim_, (nb+1)*blockDim_), ALL).clone();
                              
         actfuncweights_ ( weightscopy_ );

	 if ( postMult_ )
	 {
	   deltasupperblock = matmul ( weightscopy_, deltasperblock.transpose() ); 
	 }
	 else
	 {      
           deltasupperblock = matmul ( weightscopy_.transpose(), deltasperblock );       
	 }
      }
      else
      {
         deltasupperblock = matmul ( weights_(slice(nb*blockDim_, (nb+1)*blockDim_), ALL).transpose(), deltasperblock );
      }

      for ( idx_t i = 0; i < blockDim_; i++)
      {
        for ( idx_t j = 0; j < blockDim_; j++ )
	{
            if ( nBlocksPrev_ == 1 )
            {
              deltasupcondensed(cont4,p) += deltasupperblock(i,j);
  	      cont4 += 1;
            }
            else
            {
              deltasupcondensed(cont4,p) = deltasupperblock(i,j);
              cont4 +=1;
            }
	}
      }
     
      if ( nBlocksPrev_ == 1 ) cont4 = 0;

    }
    
    cont = 0; cont2 =0; cont4 = 0;
  }

  bpmmul_.stop();

  if ( actWeights_ )
  {
    Matrix wA = weightsDof_.clone();

    gradfuncweights_ ( wA );
         
    if ( debug_ )
    {
      System::out() << "Backpropagate. Weights after fixing activated weights: " << wA << "\n";
      System::out() << "Backpropagate. gradmat " << gradmat << "\n";
    }

    TensorIndex i, j;

    gradmat (i,j) = gradmat (i,j) * wA ( i,j );
  }

  if ( debug_ )
  {
    System::out() << "backPropagate. BlockDecLayer. deltas*acts: " << gradmat << "\n";
 //   System::out() << "backPropagate. BlockDecLayer. size_: " << size_ << "\n";
 //   System::out() << "backPropagate. BlockDecLayer. iInptWts: " << iInpWts_ << "\n";
 //   System::out() << "backPropagate. BlockDecLayer. gradmat dim " << gradmat.size(0) << " x " << gradmat.size(1) << "\n";
  }

  for ( idx_t in = 0; in < nBlocks_; ++in )
  {
    grads[iInpWts_(in,ALL)] += gradmat(in,ALL);

    bpbias_.start();
    if ( useBias_ )
    {
      grads[iBiases_[in]]     += sum ( deltas(in,ALL) );
      if ( debug_ ) System::out() << "backPropagate. BlockDecLayer. biases " << iBiases_[in] << ": " << sum ( deltas(in,ALL) ) << "\n";
    }
    bpbias_.stop();
  }
  
  if ( debug_ ) System::out() << "grads ok \n";

  bpmmul_.start();
  
  select ( data->deltas, inpNeurons_, ALL ) = deltasupcondensed.clone(); 
 
 /* if ( debug_ ) 
  {
    System::out() << "backPropagate. BlockDecLayer. inpNeurons: " << inpNeurons_ << "\n";
    System::out() << "backPropagate. BlockDecLayer. deltas (inp): " << select (data->deltas, inpNeurons_, ALL) << "\n";
    System::out() << "deltas: " << deltas << "\n";
    System::out() << "weights: " << weights_ << "\n";
  }*/

  bpmmul_.stop();
  bptot_.stop();
}

//-----------------------------------------------------------------------
//   getDeterminant
//-----------------------------------------------------------------------

void BlockDecLayer::getDeterminant_

  ( const Ref<NData>  data,
    const Properties& globdat,
    const Properties& params )

{
  ptot_.start();
  palloc_.start();
  if ( debug_ ) System::out() << "Started calculating the determinant\n";
  
  Matrix detallpoints ( nBlocks_, data->batchSize() );
  detallpoints = 0.0;
  
  Matrix netValues ( size_, data->batchSize() );
  netValues = 0.0;

  Matrix defgrad ( 3, 3 );

  Matrix acts        ( select ( data->activations, inpNeurons_, ALL ) );
  palloc_.stop();

  pmmul_.start();  
  idx_t cont = 0;
  idx_t cont2 = 0; 
  Matrix defgradblock ( blockDim_, blockDim_ );
  defgradblock = 0.0;

  for ( idx_t p = 0; p < data->batchSize(); p++ )
  {
    for ( idx_t nb = 0; nb < nBlocks_; nb++ )
    { 
      for ( idx_t i = 0; i < blockDim_; i++ )
      {
        for ( idx_t j = 0; j < blockDim_; j++ )
        {
          defgrad(i, j) = acts(cont, p);
          cont += 1;
        }
      }

    if ( nBlocksPrev_ == 1 ) cont = 0;

     
    if ( actWeights_ )
    {
      Matrix actWeights =  weights_(slice(nb*blockDim_, (nb+1)*blockDim_), ALL).clone();
      actfuncweights_ ( actWeights );
      matmul ( defgradblock, actWeights, defgrad );
    
      if ( mirrored_ || lower_ ) 
      {
      
         defgradblock(0,0) += 1.0;
         defgradblock(1,1) += 1.0;
         defgradblock(2,2) += 1.0;
         double detF = determinant ( defgradblock );

         if ( debug_ ) System::out() << "Block " << nb << " detF " << detF << "\n";

         detallpoints(nb, p) = detF;                     
      }
      
    }
    else
    { 
       matmul ( defgradblock, weights_(slice(nb*blockDim_, (nb+1)*blockDim_), ALL), defgrad );
    }
    
    if ( debug_ ) System::out() << "defgrad " << defgradblock << "\n";
        
    for ( idx_t i = 0; i < blockDim_; i++ )
    {
      for ( idx_t j = 0; j < blockDim_; j++ )
      {
        netValues(cont2, p) = defgradblock (i, j);
	if ( nBlocksPrev_ > 1 && i == j ) netValues ( cont2, p ); // += 1.0;
	cont2 += 1;
      }
    }    
   }
   
    cont2 = 0; cont = 0;
     
    }
    pmmul_.stop();
    
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

  if ( debug_ ) System::out() << "iNeurons: " << iNeurons_ << "\n";
  select ( data->activations, iNeurons_, ALL ) = netValues;

  Vector lowestdet ( nBlocks_ );

  if ( mirrored_ || lower_ )
  {
    for ( idx_t nb = 0; nb < nBlocks_; nb++ ) lowestdet[nb] = min ( detallpoints( nb, ALL ) );
//    System::out() << "Determinants " << detallpoints << "\n";      
    params.set ( LearningParams::DETERMINANT, lowestdet );
  }

  ptot_.stop();
}


//-----------------------------------------------------------------------
//   forwardJacobian_
//-----------------------------------------------------------------------

void BlockDecLayer::forwardJacobian_

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

void BlockDecLayer::backJacobian_

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

void BlockDecLayer::getJacobian_

  ( const Ref<NData>  data,
    const Properties& globdat )

{
  idx_t bsize = data->batchSize();
  idx_t jsize = data->outSize();
  
  Matrix jacpt ( blockSize_, blockSize_ );
  jacpt = 0.0;

  if ( debug_ ) 
  {
     System::out() << "BlockDecLayer. getJacobian initial:\n" << jacobian_ << "\n";
     System::out() << "jsize: " << jsize << " bsize: " << bsize << 
	  " inpNeurons size: " << inpNeurons_.size() << "\n";
  }

  if ( jacobian_.size(0) != jsize * bsize )
  {
    jacobian_.resize ( jsize*bsize, inpNeurons_.size() );
    jacobian_ = 0.0;
  }
  else
  {
    jacobian_ = 0.0;
  }

  Matrix derivs ( select ( data->values, iNeurons_, ALL ) );

  if ( debug_ ) 
  {
    System::out() << "BlockDecLayer. derivs.\n";
    System::out() << derivs << "\n";
  }

  grad_ ( derivs );

  if ( debug_ )
  {
    System::out() << "BlockDecLayer. grad_ ( derivs ):\n" << derivs << "\n";
    System::out() << "weights_:\n" << weights_ << "\n";
  }


  Matrix weightscopy = weightsExp_.clone();

  if ( outputLayer_ )
  {
    if ( actWeights_ )
    {
      actfuncweights_ ( weightscopy );
    }

    for ( idx_t b = 0; b < bsize; ++b )
    {
      for ( idx_t nb = 0; nb < nBlocks_; nb++ )
      {
        for ( idx_t i = 0; i < size_; ++i )
        {
          if ( actWeights_ )
          { 
           jacobian_(i+b*jsize,slice(nb*blockSize_, (nb+1)*blockSize_ )) = weightscopy (nb*blockSize_+i,ALL) * derivs(i,b);
   	  }
          else
          {
           jacobian_(i+b*jsize,slice(nb*blockSize_, (nb+1)*blockSize_ )) = weightscopy (nb*blockSize_+i,ALL) * derivs(i,b);
          }
        }
      }
    }

    
     if ( debug_ )
     {
       System::out() << "Block (output). Weights " << weightscopy << "\nJacobian " << jacobian_ << "\n";
     } 
  }
  else
  {
   for ( idx_t p = 0; p < data->batchSize(); ++p )
   { 
    for ( idx_t nb = 0; nb < nBlocks_; nb++ )
    {
     if ( debug_ ) System::out() << "Block size " << data->jacobian.size(0) << " x " << data->jacobian.size(1) << "\n";

     for ( idx_t i = 0; i < blockSize_; i++ )
     {
       data->jacobian ( slice(p*jsize,(p+1)*jsize), i ) *= derivs(i,p);
     }

     jacpt = data->jacobian ( slice(p*jsize,(p+1)*jsize), slice( nb*blockSize_, (nb+1)*blockSize_ ) );

      // Multiply it by the weight matrix of the block

     if ( actWeights_ )
     {
       Matrix weightscopy_ = weightsExp_(slice(nb*blockSize_, (nb+1)*blockSize_), ALL).clone();
       actfuncweights_ ( weightscopy_ );
       jacpt = matmul ( jacpt, weightscopy_ );

       if ( debug_ ) System::out() << "Block " << jacpt << "\n";
       if ( debug_ ) System::out() << "Weights " << weightscopy_ << "\n";
     }
     else
     {
       jacpt = matmul ( jacpt, weightscopy ( slice (nb*blockSize_, (nb+1)*blockSize_), ALL ) );   

       if ( debug_ ) System::out() << "Block " << jacpt << "\n";
       if ( debug_ ) System::out() << "Weights " << weightscopy ( slice (nb*blockSize_, (nb+1)*blockSize_), ALL ) << "\n";   
     }

    // Save it for the final jacobian

     for ( idx_t i = 0; i < blockSize_; ++i )
     {
        if ( nBlocksPrev_ == 1 )
	{
         jacobian_( slice (p*jsize, (p+1)*jsize), i ) += jacpt (ALL, i);
	}
	else
	{ 	
         jacobian_( slice (p*jsize, (p+1)*jsize), nb*blockSize_ + i ) = jacpt (ALL, i);
	}
     }
    }  
    }    
  }

  if ( debug_ ) System::out() << "BlockDecLayer. getJacobian updated:\n" << jacobian_ << "\n";

  data->jacobian.ref ( jacobian_ );
}

//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   newBlockDecLayer
//-----------------------------------------------------------------------


Ref<Model>            newBlockDecLayer

  ( const String&       name,
    const Properties&   conf,
    const Properties&   props,
    const Properties&   globdat )

{
  // Return an instance of the class.

  return newInstance<BlockDecLayer> ( name, conf, props, globdat );
}

//-----------------------------------------------------------------------
//   declareBlockDecLayer
//-----------------------------------------------------------------------

void declareBlockDecLayer ()
{
  using jive::model::ModelFactory;

  // Register the StressModel with the ModelFactory.

  ModelFactory::declare ( "BlockDec", newBlockDecLayer );
}

