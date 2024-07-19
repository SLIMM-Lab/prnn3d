/*
 * Copyright (C) 2019 TU Delft. All rights reserved.
 *
 * Class that implements a fully-connected neural layer. Each
 * of its neurons will connect themselves with every neuron of
 * the previous layer (towards the input layer). Note that this
 * does not guarantee fully connectivity to the next layer.
 *
 * Author: Iuri Barcelos, i.barceloscarneiromrocha@tudelft.nl
 * Date:   May 2019
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
#include "DenseLayer.h"
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

const char* DenseLayer::SIZE           = "size";
const char* DenseLayer::ACTIVATION     = "activation";
const char* DenseLayer::INITIALIZATION = "init";
const char* DenseLayer::USEBIAS        = "useBias";
const char* DenseLayer::DEBUG          = "debug";
const char* DenseLayer::PRUNING        = "pruning";
const char* DenseLayer::POSWEIGHTS     = "posWeights";

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------

DenseLayer::DenseLayer

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
  pruning_      = false;
  posWeights_   = false;
  actWeights_ = false;
  symmetric_ = false;
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
    Ref<DenseLayer> image = dynamicCast<DenseLayer> ( layer );

    System::out() << "Dense layer. Symmetric network.\n";

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
    idx_t presize = inpNeurons_.size();

    IdxVector idofs ( presize );
    
    iInpWts_.resize ( size_, presize );
    iInpWts_ = -1;

    weights_.resize ( size_, inpNeurons_.size() );
    weights_ = 0.0;

    init_ ( weights_, globdat );

    Vector state;

    IdxVector newaxons  = aset->addAxons ( presize * size_ );
    System::out() << "Dense layer. inpWeights: Creating " << presize * size_ << " axons\n";
    System::out() << "Presize: " << presize << " size: " << size_ << "\n";
    System::out() << "inpNeurons:" << inpNeurons_ << "\n";
    System::out() << "Weight type: " << weightType_ << "\n";

    dofs->addDofs ( newaxons,  weightType_ );

    StateVector::get ( state, dofs, globdat );

//    System::out() << "newaxons: " << newaxons << "\n";
    
    for ( idx_t in = 0; in < size_; ++in )
    {
      IdxVector myaxons ( newaxons[slice (in*presize,(in+1)*presize )] );

      System::out() << "i: " << in << "\n";
      System::out() << "myaxons: " << myaxons << "\n";

      dofs->getDofIndices ( idofs,  myaxons,   weightType_ );
  //    System::out() << "idofs: " << idofs << "\n";
  //    System::out() << "state[idofs]: " << state[idofs] << "\n";
      iInpWts_(in,ALL) = idofs;
      state[idofs] = weights_(in,ALL);
   //   System::out() << "weights_(in, ALL): " << weights_(in, ALL) << "\n";
    }
    

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
      idx_t nip = (int) presize/9;
    
     if ( pruning_ )
     {
      idx_t init = nip*9*9 + nip*6 + 1;

     System::out() << "Begin pruning dense layer. Number of subgroups: " << nip << " init " << init << "\n";
     for ( idx_t k = 0; k < 9; ++k) // Number of components
     { 
       ws = 0;
       for ( idx_t p = 0; p < nip-1; ++p ) // Number of int. points - 1
       {
         for ( idx_t j = 0; j < 8; ++j ) // Number of components erased
         {
           state[init+j+ws] = 0.0;
         }	       
         ws += 9;
       }
      
       if ( k != 8 )
       { 
       for ( idx_t i = 0; i < 9; ++i )
       {
 	 state[init+i+ws] = 0.0;
       }
       }

       init += nip*9+1;
      }

      System::out() << "Pruned weights: " << state << "\n";
    }
    
   if ( posWeights_ ) 
   { 
    idx_t initp =  nip*3*3 + nip*3;
 //    idx_t initp = prev*3 + prev + nip*3*prev + nip*3 + 1;     
     for (idx_t i = initp; i < state.size(); ++i )
     {
       if ( state[i] < 0.0)
       {
	 state[i] = abs(state[i]);
       }
     }
   }
      
     for (idx_t in = 0; in < size_; ++in )
     {
	weights_(in, ALL) = state[iInpWts_(in, ALL)];
     }
//     System::out() << "state. denselayer: " << state << "\n";
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

DenseLayer::~DenseLayer ()
{}

//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------

void DenseLayer::configure

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

void DenseLayer::getConfig

  ( const Properties& conf,
    const Properties& globdat )

{
}

//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------

bool DenseLayer::takeAction

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
      select ( data->deltas, iNeurons_, ALL ) = data->outputs;
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

void DenseLayer::getInpDofs

  (       IdxMatrix& dofs )

{
  dofs.ref ( iInpWts_ );
}

//-----------------------------------------------------------------------
//   update_
//-----------------------------------------------------------------------

void DenseLayer::update_

  ( const Properties& globdat )

{
  Vector state;
  StateVector::get ( state, dofs_, globdat );

  if ( useBias_ )
  {
    biases_ = state[iBiases_];
  }

  for ( idx_t in = 0; in < size_; ++in )
  {
    weights_(in,ALL) = state[iInpWts_(in,ALL)];
    //System::out() << "Updated weights (Dense Layer): " << weights_(in, ALL) << "\n";
  }
}

//-----------------------------------------------------------------------
//   propagate_
//-----------------------------------------------------------------------

void DenseLayer::propagate_

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
      idx_t presize = inpNeurons_.size();
      idx_t nip = (int) presize/9;

     // if ( debug_ ) System::out() << "Starting to prune (propagate_). Presize: " << presize << "\n";

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
        
        //System::out() << "i " << i << " weights after pruning " << actWeights(i, ALL) << "\n";
     }

     }
    
    pmmul_.start();
   // System::out() << "actweights " << actWeights << "\n";
    netValues = matmul ( actWeights, acts );
    pmmul_.stop();
  
   /* if ( debug_ )
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

void DenseLayer::getDeterminant_

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
     
      idx_t presize = inpNeurons_.size();
      idx_t nip = (int) presize/3;

      if ( debug_ ) System::out() << "Begin pruning. Presize: " << presize << "\n";
     
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

void DenseLayer::backPropagate_

  ( const Ref<NData>   data,
          Vector&      grads,
    const Properties&  globdat  )

{

  bptot_.start();
  bpalloc_.start();
  Matrix derivs ( select ( data->values, iNeurons_, ALL ) );
  bpalloc_.stop();

  if ( debug_ ) System::out() << "backPropagate. DenseLayer. derivs (values): " << derivs << "\n";

  bpfunc_.start();
  grad_ ( derivs );

  if ( debug_ ) System::out() << "backPropagate. DenseLayer. derivs (grad_): " << derivs << "\n";

  //select ( data->deltas, iNeurons_, ALL ) *= derivs;
  bpfunc_.stop();

  bpalloc_.start();
  Matrix acts        ( select ( data->activations, inpNeurons_, ALL ) );

  if ( debug_ ) System::out() << "backPropagate. DenseLayer. acts: " << acts << "\n";

  //Matrix deltas ( select ( data->deltas, iNeurons_, ALL ) );
  Matrix deltas ( select ( data->deltas, iNeurons_, ALL ) * derivs );

  if ( debug_ ) 
  {
    System::out() << "backPropagate. DenseLayer. iNeuros: " << iNeurons_ << ".\n";
    System::out() << "backPropagate. DenseLayer. delta: " << select ( data->deltas, iNeurons_, ALL ) << ".\n";
    System::out() << "backPropagate. DenseLayer. delta*derivs: " << deltas << "\n";
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
     
      idx_t presize = inpNeurons_.size();
      idx_t nip = (int) presize/9;

      if ( debug_ ) System::out() << "Starting to prune (backPropagate_). Presize: " << presize << "\n";

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
        
        //System::out() << "i " << i << " weights after pruning " << weightscopy_(i, ALL) << "\n";
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
    System::out() << "backPropagate. DenseLayer. deltas*acts: " << gradmat << "\n";
    System::out() << "backPropagate. DenseLayer. size_: " << size_ << "\n";
    System::out() << "backPropagate. DenseLayer. iInptWts: " << iInpWts_ << "\n";
  }
  
  // Setting gradient contribution from pruned connections to zero (if enabled)
  
  if ( pruning_ )
  {
     if ( debug_ )
     {
       System::out() << "backPropagate. Gradients before fixing activated weights: " << gradmat << "\n";
     }
     
     idx_t presize = inpNeurons_.size();
     idx_t nip = (int) presize/9;

     idx_t nCompsErased;

     for ( idx_t i = 0; i < 9; i++ )  // Total number of components
     {
       idx_t init = i + 1;
       idx_t ws = 0;
       idx_t remMax = i;

       for ( idx_t rem = 0; rem < remMax; rem++ )
       {
         gradmat(i, rem) = 0.0;
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
            gradmat(i, init+j+ws) = 0.0;
          }	       
          ws += 9;
        }        
     }

  }
  
  // Adding gradients contribution

  for ( idx_t in = 0; in < size_; ++in )
  {
    grads[iInpWts_(in,ALL)] += gradmat     (in,ALL); 
    
 //   System::out() << "backPropagate. grads dense layer " << in << ": " << grads[iInpWts_(in, ALL)] << "\n";

    bpbias_.start();
    if ( useBias_ )
    {
      grads[iBiases_[in]]     += sum ( deltas(in,ALL) );
      if ( debug_ ) System::out() << "backPropagate. DenseLayer. biases " << iBiases_[in] << ": " << sum ( deltas(in,ALL) ) << "\n";
    }
    bpbias_.stop();
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
    System::out() << "backPropagate. DenseLayer. inpNeurons: " << inpNeurons_ << "\n";
    System::out() << "backPropagate. DenseLayer. deltas (inp): " << select (data->deltas, inpNeurons_, ALL) << "\n";
    System::out() << "deltas: " << deltas << "\n";
    System::out() << "weights: " << weights_ << "\n";
  }

  bpmmul_.stop();
  bptot_.stop();
}

//-----------------------------------------------------------------------
//   forwardJacobian_
//-----------------------------------------------------------------------

void DenseLayer::forwardJacobian_

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

void DenseLayer::backJacobian_

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

void DenseLayer::getJacobian_

  ( const Ref<NData>  data,
    const Properties& globdat )

{
  idx_t bsize = data->batchSize();
  idx_t jsize = data->outSize();

  if ( debug_ ) 
  {
     System::out() << "DenseLayer. getJacobian initial:\n" << jacobian_ << "\n";
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

  if ( debug_ ) System::out() << "DenseLayer. getJacobian updated:\n" << jacobian_ << "\n";

  data->jacobian.ref ( jacobian_ );
}

//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   newDenseLayer
//-----------------------------------------------------------------------


Ref<Model>            newDenseLayer

  ( const String&       name,
    const Properties&   conf,
    const Properties&   props,
    const Properties&   globdat )

{
  // Return an instance of the class.

  return newInstance<DenseLayer> ( name, conf, props, globdat );
}

//-----------------------------------------------------------------------
//   declareDenseLayer
//-----------------------------------------------------------------------

void declareDenseLayer ()
{
  using jive::model::ModelFactory;

  // Register the StressModel with the ModelFactory.

  ModelFactory::declare ( "Dense", newDenseLayer );
}

