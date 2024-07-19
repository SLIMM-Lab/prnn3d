/*
 * Copyright (C) 2021 TU Delft. All rights reserved.
 *
 * Class that implements a fully-connected recurrent neural layer.
 * Neurons are connected with the previous and next layers as well
 * as with its own state at the time when forward propagation
 * occurs. This layer can be stacked with other MatRecLayers
 * as well as with DenseLayers. 
 * 
 * This is not LSTM or GRU, be mindful of the representational 
 * limitations of a conventional RNN layer.
 *
 * Author: Marina Maia, m.alvesmaia@tudelft.nl
 * Date:   May 2021
 * 
 */

#include <jem/base/System.h>
#include <jem/base/Float.h>
#include <jem/base/Error.h>
#include <jem/numeric/algebra/matmul.h>
#include <jive/Array.h>
#include <jive/model/ModelFactory.h>
#include <jive/model/StateVector.h>
#include <jive/util/error.h>
#include "utilities.h"

#include "MatRecLayer.h"
#include "DenseLayer.h"
#include "LearningNames.h"

#include "Material.h" 

using jem::numeric::matmul;

using jive::IdxVector;
using jive::util::XDofSpace;
using jive::util::sizeError;
using jive::model::StateVector;

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------

const char* MatRecLayer::SIZE          = "size";
const char* MatRecLayer::ACTIVATION    = "activation";
const char* MatRecLayer::INPINIT       = "inpInit";
const char* MatRecLayer::RECINIT       = "recInit";
const char* MatRecLayer::MATERIAL_LIST = "matList";
const char* MatRecLayer::RATIOMODEL    = "ratioModel";
const char* MatRecLayer::USEBIAS       = "useBias";
const char* MatRecLayer::NMODELS       = "nModels";

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------

MatRecLayer::MatRecLayer

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat ) : Model ( name )

{
  String func ( "identity" );

  // Setting default values

  size_ = 0;            
  onl_ = nullptr;
  ratioModel_ = 1.0;      // Proportion of history-dependent materials
  debug_ = false;
  mirrored_ = false;      
  pruning_ = false;
  useBias_ = false;
  learnProp_ = 0;      // Number of learnable material parameters in each material point
  frommodel_ = false;   // Get increment of time from the solver instead of from the data
  dt_ = 1.0;
  useBlocks_ = true;    // If encoder is based on SPD matrices, skip weights creation here

  // Reading layer properties

  myProps = props.getProps ( myName_ );
  Properties myConf  = conf.makeProps ( myName_ );

  Ref<Model> layer;
  Ref<MatRecLayer> image = dynamicCast<MatRecLayer> ( layer );

  jive::StringVector matList;
  String             matName;
  globdat_         = globdat;

  myProps.get  ( size_, SIZE );
  myProps.find ( func, ACTIVATION );
  myProps.find ( ratioModel_, RATIOMODEL );
  myProps.find ( debug_, "debug");
  myProps.find ( pruning_, "pruning");
  myProps.find ( learnProp_, "learnProp" );
  myProps.find ( useBias_, USEBIAS ); 
  myProps.find ( nModels_, NMODELS );
  myProps.find ( useBlocks_, "useBlocks" );

  myConf. set ( SIZE, size_ );
  myConf. set ( ACTIVATION, func );
  myConf. set ( "learnProp", learnProp_ );
  myConf. set ( USEBIAS, useBias_ );

  func_ = NeuralUtils::getActivationFunc ( func );
  grad_ = NeuralUtils::getActivationGrad ( func );

  func = "glorot";

  myProps.find ( func, INPINIT );
  myConf. set  ( INPINIT, func );

  inpInit_ = NeuralUtils::getInitFunc ( func );

  func = "orthogonal";

  myProps.find ( func, RECINIT );
  myConf. set  ( RECINIT, func );

  recInit_ = NeuralUtils::getInitFunc ( func );

  // Create and configure material models
  
  myProps.get ( matList, MATERIAL_LIST );
  myConf .set ( MATERIAL_LIST, matList );

  if ( debug_ ) System::out() << "Create and configure material models.\n";
  
  // Create one material model with one integration point
  // for every subgroups of size number of strains and
  // learnable material parameters
  
  Properties matProps = myProps.getProps ( matList[0] ); // For obtaining rank only
  idx_t rank;

  idx_t STRAINS_COUNT[4] = {0, 1, 3, 9};
  matProps.get ( rank, "dim" );
  sSize_ = STRAINS_COUNT[rank];
  nIntPts_ = size_/(sSize_+learnProp_);

  nIntPts1_ = 0; nIntPts2_ = 0;
//  nIntPts1_ = ratioModel_*nIntPts_;     // Old stuff, now it is automatic 
//  nIntPts2_ = nIntPts_ - nIntPts1_;  

  modelType_.resize ( nIntPts_ );
  modelType_ = 0;

  if ( myProps.find ( nModels_, NMODELS ) )
  {
    idx_t totalNModels = sum(nModels_);

    if ( totalNModels != nIntPts_ )
    {
      throw Error ( JEM_FUNC, "Total # of specified material points is not equal to the layer size.");
    }
    else
    {
      idx_t sumpts = 0;
      for ( idx_t type = 0; type < nModels_.size(0); type++ )
      {
        for ( idx_t pt = 0; pt < nModels_[type]; pt++ )
	{
	  modelType_[sumpts] = type;
	  sumpts += 1;
	}
      }

      System::out() << "Types of material models used to the subgroups " << modelType_ << "\n"; 
    }
  }
  else
  {
    // N.B.: if not specified which mat model each subgroup should use, 
    // all material points will use the first material model in matList

    nModels_.resize(1);
    nModels_[0] = nIntPts_;

    System::out() << "Types of material models used to the subgroups " << modelType_ << "\n"; 
  }

  // Aux variable to distinguish weighted connections
  // from biases used to learn material properties
  
  wSize_ = size_ - nIntPts_*learnProp_;
  mpSize_ = nIntPts_*learnProp_;

  child_.resize ( nIntPts_ );

  // Create one material model for each group of three neurons

  hSizes_.resize ( nIntPts_ );
  hSizesIdx_.resize ( nIntPts_ + 1 );
  hSizes_ = 0; hSizesIdx_ = 0;

  for ( idx_t t = 0; t < nIntPts_; ++t )
  {
    matName = matList[modelType_[t]];
    
    child_[t] = newMaterial ( matName, Properties(), myProps, globdat_ );
    matProps = myProps.getProps ( matName );

    child_[t]->configure ( matProps ); 
    child_[t]->allocPoints ( 1 );

    hSizes_[t] = child_[t]->getHistoryCount();

    hSizesIdx_[t+1] = sum( hSizes_ ); 

    if ( hSizes_[t] > 0 ) nIntPts1_ += 1;
  }

  // Getting size of internal variables vector for path dependent material (always
  // the first in the matList)

  hNeurons_.resize ( sum(hSizes_) );
  hNeurons_ = 0;

  hSize_ = hSizes_[0]; // temporary

  if ( debug_ )
  {
    System::out() << nIntPts1_ << " points with path-dependency and remaining " << nIntPts2_ << " points are history-independent.\n";
    System::out() << "Indices of internal variables " << hSizesIdx_ << "\n";
    System::out() << "Finished configuring material. \n"
                      "Total # of integration points: " << nIntPts_ << "\n";
  }

  System::out() << "Number of internal variables in the path-dependent material: " << hSizes_ << "\n";
  System::out() << "Size of layer " << size_ << 
         " number of properties being learned " << learnProp_ << " wSize_ " << wSize_ << "\n";

  // Init neurons and connections

  iNeurons_.resize ( wSize_ );
  iNeurons_ = 0;

  mpNeurons_. resize ( mpSize_ );
  mpNeurons_ = 0;

  Ref<XAxonSet>   aset = XAxonSet  ::get ( globdat, getContext()    );
  Ref<XNeuronSet> nset = XNeuronSet::get ( globdat, getContext()    );
  Ref<XDofSpace>  dofs = XDofSpace:: get ( aset->getData(), globdat );

  for ( idx_t in = 0; in < wSize_; ++in )
  {
    iNeurons_[in] = nset->addNeuron();
  }

  for ( idx_t in = 0; in < mpSize_; ++in )
  {
    mpNeurons_[in] = nset->addNeuron();
  }

  for ( idx_t in = 0; in < hNeurons_.size(); ++in )
  {
    hNeurons_[in] = nset->addNeuron();
  }
  
  weightType_ = dofs->findType ( LearningNames::WEIGHTDOF );
  biasType_   = dofs->findType ( LearningNames::BIASDOF   );

//  Ref<Model> layer;

  if ( myProps.contains ( LearningParams::IMAGE ) )
  {
 //   throw Error ( JEM_FUNC,
 //     "RNN layers do not currently support symmetry" );
    Ref<DenseLayer> image = dynamicCast<DenseLayer> ( layer );

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
    
    inpWeights_.resize ( size_, inpNeurons_.size() );
    inpWeights_ = 0.0;
    
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
      
   //   dofs->getDofIndices ( iBiases_, newbaxons, biasType_ );
   //   state[iBiases_] = 0.0;
    }

    for ( idx_t in = 0; in < size_; ++in )
    {
      inpWeights_ (in,ALL) = state[iInpWts_(in,ALL)];
    }

    mirrored_ = true;
  }
  else if ( globdat.find ( inpNeurons_, LearningParams::PREDECESSOR ) )
  {
    idx_t presize = inpNeurons_.size();

    IdxVector idofs ( presize );
 
    if ( useBias_ == true )
    {
      iBiases_.   resize ( wSize_         );
      biases_.    resize ( wSize_ );
      iBiases_ = -1;
      biases_ = 0.0;
      idx_t id = 0;
      for (idx_t npt = 0; npt < nIntPts_; npt++ )
      {
        biases_[id] = biases_[id+4] = biases_[id+8] = 1.0;
	id = npt*9;
      }
    }
 
    if ( learnProp_ > 0 )
    {
      iBiasesProp_.resize ( mpSize_         );
      biasesProp_.resize ( mpSize_ );
      iBiasesProp_ = -1;
      biasesProp_ = 0.0;
    }

    iInits_.    resize ( size_          );
    iInpWts_.   resize ( wSize_, presize );
    inpWeights_.resize ( wSize_, presize );
//    Matrix phWeights ( wSize_, nIntPts_*hSize_ );

    iInits_     = -1; iBiasesProp_ = -1;
    iInpWts_    = -1; 
    biasesProp_ = 0.; inpWeights_ = 0.; 

    inpInit_ ( inpWeights_, globdat );
//    inpInit_ ( phWeights, globdat );

    Vector state;
    IdxVector newinpaxons;

    if ( useBlocks_ == false ) 
    {
      newinpaxons = aset->addAxons ( presize * wSize_ );
      dofs->addDofs ( newinpaxons, weightType_ );
      System::out() << "Creating " << presize * wSize_ << " axons\n";
    }
    
    System::out() << "Size of previous layer: " << presize << " wSize: " << wSize_ << "\n";
    System::out() << "Number of biases refering to material properties: " << mpSize_ << "\n";
 
    StateVector::get ( state, dofs, globdat );

    // List of connections set to zero
    delconn_.resize ( nIntPts_* 9 );
    delconn_ = 0;
    idx_t id = 2;
    for ( idx_t i = 0; i < nIntPts_; i++ )
    {
      delconn_[id] = delconn_[id+1] = delconn_[id+3] = delconn_[id+4] = delconn_[id+5] = 1.0; 
      id += 9;
    }

    delconn_ = 0.0;

    if ( useBlocks_ == false )
    {
      for ( idx_t in = 0; in < wSize_; ++in )
      {
        IdxVector myinpaxons ( newinpaxons[slice(in*presize,(in+1)*presize)] );
//      System::out() << "i: " << in << "\n";
//      System::out() << "myaxons: " << myinpaxons << "\n";
     
        dofs->getDofIndices ( idofs, myinpaxons, weightType_ );
 //     System::out() << "idofs " << idofs << " \n";
        iInpWts_(in,ALL) = idofs;
        if ( delconn_[in] == 1.0 )
        {
          inpWeights_(in, ALL) = 0.0;
        }
        else
        {
          inpWeights_(in, ALL) *= 1.0;
        }
	
        state[idofs] = inpWeights_(in,ALL);

//      System::out() << "weights: ";
    //  System::out() << state[idofs] << "\n";
        idx_t cont = 0;
      }
    }
   
    // Init regular biases

    if ( useBias_ == true && useBlocks_ == false ) 
    {
      IdxVector newbiaaxons = aset->addAxons ( wSize_           );
   //   dofs->addDofs ( newbiaaxons, biasType_   );

      StateVector::get ( state, dofs, globdat );
   //   dofs->getDofIndices ( iBiases_, newbiaaxons, biasType_ );

      System::out() << "Regular biases. Creating " << wSize_ << "\n";
      System::out() << "Regular biases. dofs: " << iBiases_ << "\n";
      System::out() << "newbiaaxons: " << newbiaaxons << "\n";
            
      state[iBiases_] = biases_;
    }

    // Init material properties biases

    if ( learnProp_ > 0 && useBlocks_ == false )
    {
      IdxVector newbiaaxonsprop = aset->addAxons ( mpSize_           );
 //     dofs->addDofs ( newbiaaxonsprop, biasType_   );
    //  dofs->getDofIndices ( iBiasesProp_, newbiaaxonsprop, biasType_ );

      System::out() << "Material biases. Creating " << mpSize_ << "\n";
      System::out() << "Material biases. dofs: " << iBiasesProp_ << "\n";
      System::out() << "newbiaaxonsprop: " << newbiaaxonsprop << "\n";

      StateVector::get ( state, dofs, globdat );
      state[iBiasesProp_] = 50.0;     // TODO: init properly material properties
    }

    // Summary

    System::out() << "inpNeurons: " << inpNeurons_ << "\n";
    System::out() << "iNeurons: " << iNeurons_ << "\n";
    System::out() << "hNeurons: " << hNeurons_ << "\n";
    System::out() << "mpNeurons: " << mpNeurons_ << "\n";  
  }

  bool inputLayer  = false;
  bool outputLayer = false;

  myProps.find ( inputLayer,  LearningNames::FIRSTLAYER );
  myProps.find ( outputLayer, LearningNames::LASTLAYER  );

  if ( inputLayer || outputLayer )
  {
    throw Error ( JEM_FUNC, "Material Recurrent layers must be hidden" );
  }

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
 
  System::out() << "Getting upper and lower bounds: \n";
  onl_->getBounds ( upper_, lower_ );
  System::out() << "Upper " << upper_ << " lower " << lower_ << "\n";

  axons_   = aset;
  neurons_ = nset;
  dofs_    = dofs;

  globdat.set ( LearningParams::PREDECESSOR, iNeurons_.clone() );
}

MatRecLayer::~MatRecLayer ()
{}

//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------

void MatRecLayer::configure

  ( const Properties& props,
    const Properties& globdat )

{
  if ( iInits_[size_-1] == dofs_->dofCount() - 1 )
  {
    throw Error ( JEM_FUNC, "Recurrent layers must be hidden" );
  }
}

//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------

void MatRecLayer::getConfig

  ( const Properties& conf,
    const Properties& globdat )

{
}

//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------

bool MatRecLayer::takeAction

  ( const String&     action,
    const Properties& params,
    const Properties& globdat )

{
//  System::out() << "@MatRecLayer::takeAction. action: " << action << "\n";

  bool offline = false;

  params.find ( offline, "offline" );

  if ( offline ) 
  {
    if ( params.find ( dt_, "dtglobal" ) )
    {
  //    System::out() << "dt_ " << dt_ << "\n";
      frommodel_ = true;
    }
  }

  if ( action == LearningActions::UPDATE )
  {
    update_ ( globdat );

    return true;
  }

  if ( action == LearningActions::PROPAGATE ||
       action == LearningActions::RECALL       )
  {

    Ref<NData> data;
    Ref<NData> state = nullptr;

    params.get  ( data,  LearningParams::DATA  );
    params.find ( state, LearningParams::STATE ); 

    propagate_ ( data, state, globdat );

    return true;
  }

  if ( action == LearningActions::BACKPROPAGATE )
  {
    Vector grads;

    Ref<NData> data;
    Ref<NData> state = nullptr;

    params.get  ( data,  LearningParams::DATA  );
    params.get  ( grads, LearningParams::GRADS );
    params.find ( state, LearningParams::STATE );

    backPropagate_ ( data, state, grads, globdat );

    return true;
  }
  
  if ( action == LearningActions::GETHISTORY )
  {
    System::out() << "MatRecLayer. getHistory\n";
  
    Ref<NData> data;
    Ref<NData> state = nullptr;

    params.get  ( data,  LearningParams::DATA  );
    params.find ( state, LearningParams::STATE ); 
    
    getHistory_ ( data, state, globdat, params );
    
    return true;
  }

  if ( action == LearningActions::GETJACOBIAN )
  {
    Ref<NData> data;
    params.get ( data, LearningParams::DATA );

    if ( debug_ ) System::out() << "getJacobian" << "\n";
    getJacobian_ ( data, globdat );

    return true;
  }

  return false;
}

//-----------------------------------------------------------------------
//   getInpDofs
//-----------------------------------------------------------------------

void MatRecLayer::getInpDofs

  (       IdxMatrix& dofs )

{
  dofs.ref ( iInpWts_ );
}

//-----------------------------------------------------------------------
//   update_
//-----------------------------------------------------------------------

void MatRecLayer::update_

  ( const Properties& globdat )

{
  if ( useBlocks_ == false )
  {
    Vector state;
    StateVector::get ( state, dofs_, globdat );

    if ( debug_ ) System::out() << "Update weights.\n"; 

    if ( useBias_ ) biases_ = state[iBiases_];

    if ( learnProp_ > 0 ) biasesProp_ = state[iBiasesProp_];
 
    for ( idx_t in = 0; in < wSize_; ++in )
    {
      inpWeights_(in,ALL) = state[iInpWts_(in,ALL)];
    }

    if ( debug_ ) 
    { 
 //   System::out() << "Updated weights: " << inpWeights_ << "\n";
      if ( useBias_ ) System::out() << "Updated biases "  << biases_ << "\n";
      if ( learnProp_ > 0 ) System::out() << "Updated mat props biases " << biasesProp_ << "\n";
    }
  }
}
//-----------------------------------------------------------------------
//   propagate_
//-----------------------------------------------------------------------

void MatRecLayer::propagate_

  ( const Ref<NData>  data,
    const Ref<NData>  state,
    const Properties& globdat  )

{
  Matrix netValues ( wSize_, data->batchSize() );
  netValues = 0.0;

  Matrix acts        ( select ( data->activations, inpNeurons_, ALL ) );

  Vector hall ( sum(hSizes_ ) );
  hall = 0.0;
  
  Matrix epeqallpoints ( nIntPts_, data->batchSize() );
  epeqallpoints = 0.0;


  if ( state == nullptr )
  {
    if ( debug_ ) System::out() << "\nPROPAGATE STATE NULL\n";
    for ( idx_t npt = 0; npt < nIntPts_; ++npt )
    {
       if ( npt < nIntPts1_ )
       {
          Vector hvals ( hSizes_[npt] );
   	  if ( learnProp_ > 0 )      // TODO: Generalize to other properties to include
	  {                          // all types of materials
	    idx_t init = npt*learnProp_;
            child_[npt]->updateYieldFunc ( biasesProp_[slice(init, init+learnProp_)] );
	  }
          child_[npt]->getInitHistory ( hvals, 0 );
          hall[slice(hSizesIdx_[npt], hSizesIdx_[npt+1])] = hvals;
       //   System::out() << "hvals " << hvals << "\n";
        }
	else
	{
          if ( debug_ ) System::out() << "No setHistory is implemented for path-independent material.\n";
	}
    }  
    for ( idx_t is = 0; is < data->batchSize(); ++is )  data->values ( hNeurons_, is ) = hall;
  }
  else
  {
    if ( debug_ ) System::out() << "STATE NONNULL.\n";
    for ( idx_t is = 0; is < data->batchSize(); ++is )
    {
      data->values ( hNeurons_, is ) = state->activations ( hNeurons_, is );
    }
  }

  if ( debug_ )
  {
    System::out() << "acts: " << acts << "\n";
//    System::out() << "weights: " << inpWeights_ << "\n";
  }

//  System::out() << "Checkpoint propagate_\n";

  // Calculate strains of fictitious points

  //netValues  = matmul ( inpWeights_, acts   );

  netValues = acts.clone();

  if ( useBias_ )
  {
    for ( idx_t is = 0; is < data->batchSize(); ++is )
    {
      netValues(ALL,is) += biases_;
      if ( debug_ )
      { 
        System::out() << "bias: " << biases_ << "\n";
      } 
    }
  }
  
  select ( data->values, iNeurons_, ALL ) = netValues;

  if ( debug_ ) System::out() << "netValues (acts*weights+bias):" << netValues << "\n";

 // Apply material model for every subgroup 

  Matrix stiff ( sSize_, sSize_ );
  Vector stress ( sSize_ );
  Matrix strain ( 3, 3);
  double dt = 0.0;

  for ( idx_t p = 0; p < data->batchSize(); ++p )
  { 
    if ( frommodel_ )
    {
      if ( debug_ ) System::out() << "From model " << dt_ << "\n";
      dt = dt_;
    }
    else
    {
      if ( debug_ ) System::out() << "From data " << dt << "\n";
      dt = data->dts(0, p);
    }


    if ( debug_ ) System::out() << "Batch " << p << " dt " << dt << "\n";

    for ( idx_t npt = 0; npt < nIntPts_; ++npt )
    { 
      Vector hvals ( hSizes_[npt] );
      hvals = 0.0;

      if ( npt < nIntPts1_ )
      {
        for ( idx_t nh = 0; nh < hSizes_[npt]; ++nh )
        {
          hvals[nh] = data->values ( hNeurons_, p )[hSizesIdx_[npt]+nh];
        }
	if ( debug_ ) System::out() << "\nSetting history for point " << npt << ": " << hvals << " and dt " << dt << "\n";
        
	child_[npt]->setHistory ( hvals, 0 );
        child_[npt]->setDT ( dt );
      }
      else
      {
        if ( debug_ ) System::out() << "No setHistory is implemented for path-independent material.\n";
      }
   
//      System::out() << "Determinant of weight matrix of point " << npt << ": " << determinant (inpWeights_(slice(npt*sSize_, npt*sSize_ + sSize_), ALL)) << "\n";

      stiff = 0.0; strain = 0.0; stress = 0.0;

      for ( idx_t i = 0; i < 3; i++ )
      { 
        strain(i, ALL) =  netValues(slice(npt*9+i*3, npt*9+(i+1)*3), p);
      }

      double detF = determinant ( strain );
     // System::out() << "Checkpoint " << npt << " det " << detF << "\n";
     // System::out() << "F " << strain << "\n";
      
      if ( debug_ ) System::out() << "propagate_ Call material.\n";
      child_[npt]->update ( stress, stiff, strain, 0 );
      //System::out() << "Without normalize " << stress << "\n";
      stress = onl_->normalize ( stress );
      //System::out() << "With normalize " << stress << "\n";
      if ( debug_ ) System::out() << "propagate_ End material call.\n";

      // Updating with stresses

      netValues( slice ( npt*sSize_, (npt+1)*sSize_), p ) = stress;
      
      if ( npt < nIntPts1_ )
      {
        child_[npt]->getHistory(hvals, 0);                    // Getting history of each int pt
      }
      else
      {
	 hvals = 0.0;     
      }

      hall[slice(hSizesIdx_[npt], hSizesIdx_[npt+1])] = hvals;

      epeqallpoints(npt, p) = hvals[6];                      // Experimental. Skip!
      
      if ( debug_ ) 
      {
   //     System::out() << "propagate_ Strain point " << npt << " sample " << p << " " << strain << "\n";
//	System::out() << "propagate_ Stress point " << npt << " sample " << p << " " << stress << "\n";
        System::out() << "propagate_ History point " << npt << " sample " << p << " " << hvals << "\n";
//	System::out() << "propagate_ Stiff point " << npt << " sample " << p << " " << stiff << "\n";
      }
     }
     
    data->activations ( hNeurons_, p ) = hall;          // Updating history of all int pts for each batch sample
  }

  if ( debug_ ) System::out() << "iNeurons: " << iNeurons_ << "\n";

  select ( data->activations, iNeurons_, ALL ) = netValues; // Propagating values (stresses)

  if ( debug_ ) System::out() << "Completed stress. Now propagate history \n";

  // Passing equivalent plastic deformation 
  //  System::out() << "eqepallpoints " << epeqallpoints << "\n";
  // data->history = epeqallpoints;


}

//-----------------------------------------------------------------------
//   backPropagate_
//-----------------------------------------------------------------------

void MatRecLayer::backPropagate_

  (    
    const Ref<NData>   data,
    const Ref<NData>   state,
          Vector&      grads,
    const Properties&  globdat  )

{
  if ( debug_ ) 
  {
     String msg = "";
     msg = (state == nullptr ) ? "BACKPROP. STATE NULL.\n": "BACKPROP. STATE NON NULL\n";
     System::out() << msg;
  }
  	
  Matrix derivs ( select ( data->values, iNeurons_, ALL ) );    // strains
  Matrix derivp ( select ( data->values, mpNeurons_, ALL ) );   // material properties
  Matrix derivh ( sSize_*nIntPts_, data->batchSize() );         // history 

  derivh = 0.0; 

  double dt = 0.0;                                              // default 
  double fdstep = 1.e-8;                                        // used for CFD

  // Auxiliary stuff
  // For numerical derivatives purposes
    
  Matrix dsigmadept ( sSize_, sSize_ );
  Matrix sdeltaspt ( sSize_, 1 );
  
  dsigmadept = 0.0; 

  Vector fstress ( sSize_ );
  Vector bstress ( sSize_ );

  fstress = 0.0; bstress = 0.0;

  Matrix stiff ( sSize_, sSize_ );
  Vector stress ( sSize_ );
  Matrix strain ( 3, 3 );

  stiff = 0.0; strain = 0.0; stress = 0.0;

  // Related to material properties (if enabled)

  Matrix dsigmadproppt ( learnProp_, sSize_ );
 
  // Get history (from previous time step) and deltas 
  
  Matrix hists ( select ( data->values, hNeurons_, ALL ) );
  Matrix sdeltas ( select ( data->deltas, iNeurons_, ALL ) );
  Matrix hdeltas ( select ( data->deltas, hNeurons_, ALL ) ); 

  if ( debug_ ) 
  {
    System::out() << "backPropagate. MatRecLayer. sdeltas: " << sdeltas << "\n";
    System::out() << "backPropagate. MatRecLayer. hdeltas: " << hdeltas << "\n"; 
    System::out() << "backPropagate. MatRecLayer. derivs (values): " << derivs << "\n";
  }

  for ( idx_t p = 0; p < data->batchSize(); ++p )
  {
     Vector phist ( hists( ALL, p ) );
     Vector hdeltasup ( sum ( hSizes_ ) );   
     hdeltasup = 0.0;
     
     // Getting dt on the fly or from data
    
     if ( frommodel_ )
     {
       dt = dt_;
     }
     else
     {
       dt = data->dts(0, p);
     }

     if ( debug_ ) System::out() << "batch " << p << " dt " << dt << " (backpropagate)\n";

     for ( idx_t npt = 0; npt < nIntPts_; ++npt )
     {
        Matrix hdeltaspt ( hSizes_[npt], 1 );
        Matrix dalphadept ( sSize_, hSizes_[npt] );
        Matrix dsigmadalphapt ( hSizes_[npt], sSize_ );
        Matrix dalphadalphapt ( hSizes_[npt], hSizes_[npt] ); 
        Matrix dalphadproppt ( learnProp_, hSizes_[npt] );
   	Vector fhist ( hSizes_[npt] );
  	Vector bhist ( hSizes_[npt] );
	Vector hist ( hSizes_[npt] );

        dsigmadalphapt = 0.0; dalphadalphapt = 0.0; dalphadept = 0.0;
	fhist = 0.0; bhist = 0.0; 
        
        hist = phist[slice(hSizesIdx_[npt], hSizesIdx_[npt+1])];

        for ( idx_t i = 0; i < 3; i++ )
	{
          strain(i, ALL) =  derivs(slice(npt*9+i*3, npt*9+i*3+3), p);
	}

	if ( debug_ )
	{
	 System::out() << "backPropagate_ History point " << npt << " sample " << p << 
		 " " << hist << "\n"; 
	 System::out() << "backPropagate_ Strain point " << npt << " sample " << p << 
		 " " << strain << "\n";
	}

	// Calculating dsigmade

	if ( npt < nIntPts1_ )
	{
	  child_[npt]->setHistory(hist, 0);           // Setting history of int point
	  child_[npt]->setDT(dt);
	}
	else
	{
	  hist = 0.0; 
	}

	if ( debug_ ) System::out() << "backPropagate_ updateMaterial.\n";

	child_[npt]->update ( stress, dsigmadept, strain, 0 );

	// Calculating dalphade using CFD

	Matrix dsigmadeptcdf ( sSize_, sSize_ );
	
	if ( debug_ ) System::out() << "Start calculating dalphade.\n"; 
	
	for ( idx_t nr = 0; nr < 3; ++nr )
	{
	  for ( idx_t nc = 0; nc < 3; ++nc )
	  {
	    strain(nr, nc) += fdstep;
	    child_[npt]->update ( fstress, stiff, strain, 0 ); 
            fstress = onl_->normalize(fstress);

	    if ( npt < nIntPts1_ )
	    {
	      child_[npt]->getHistory ( fhist, 0 );
	    }
	    else
	    {
	      fhist = 0.0;
	    }

	    strain(nr, nc) -= 2.0*fdstep;
	    child_[npt]->update ( bstress, stiff, strain, 0 );
            bstress = onl_->normalize(bstress);

	    if ( npt < nIntPts1_ )
	    {
	      child_[npt]->getHistory ( bhist, 0 );
	    }
	    else
	    {
	      bhist = 0.0;
	    }

	    dalphadept(nr*3+nc, ALL) = (fhist - bhist ) / 2.0 / fdstep;
	    dsigmadeptcdf(nr*3+nc, ALL) = (fstress - bstress ) / 2.0 / fdstep;  

	    strain(nr, nc) += fdstep;
 	  }
	}

	if ( debug_ )
	{
	  System::out() << "dsigmade:\n" << dsigmadept << "\n";
	  System::out() << "dsigmade cdf:\n" << dsigmadeptcdf << "\n";
	  System::out() << "dalphade:\n" << dalphadept << "\n";
	}
	
	// Calculating dalphadprop

	if ( learnProp_ > 0 )
	{
	Vector props(learnProp_);
	props = biasesProp_[slice(npt*learnProp_, npt*learnProp_+learnProp_)];

	for ( idx_t ns = 0; ns < learnProp_; ++ns )
	{
	  props[ns] += fdstep;	
	//	System::out() << "Props " << props << "\n";
	  if ( npt < nIntPts1_ )
  	  {
	    child_[npt]->updateYieldFunc ( props );
	    child_[npt]->update ( fstress, stiff, stiff, 0 );
	    child_[npt]->getHistory ( fhist, 0 );
	  }
	  else
	  {
	    fhist = 0.0; 
	  }

	  props[ns] -= 2.0*fdstep;

          if ( npt < nIntPts1_ )
	  {
	    child_[npt]->updateYieldFunc ( props );
	    child_[npt]->update ( bstress, stiff, stiff, 0 );
	    child_[npt]->getHistory ( bhist, 0 );
	  }
	  else
	  {
	    bhist = 0.0; 
 	  }

	dalphadproppt ( ns, ALL ) = ( fhist - bhist ) / 2.0 / fdstep;
	dsigmadproppt ( ns, ALL ) = ( fstress - bstress ) / 2.0 / fdstep;

	props[ns] += fdstep;
	}
	}

	// Calculating dsigmadalpha and dalphadalpha with Central differences

	if ( debug_ ) System::out() << "\nStart to calculate dsigmadalpha and dalphadalpha.\n";

       if ( hSizes_[npt] >= hSizes_[0] )
       {
	for ( idx_t alpha = 0; alpha < hSizes_[npt]; ++alpha )
	{
	   if ( alpha == 18 && hist[alpha] < 1e-5 )
	   {
	     fdstep = 1e-14;  
	   }
	   else
	   {
	     fdstep = 1e-8;
	   }


           double fdstepf = fdstep;
	   double fdstepb = fdstep;

	if ( alpha == 18)  
	{
	  if ( hist[alpha] + fdstep > 1.0 )
	  {
	   // System::out() << "Constraint on FD! Upper.\n";
	    fdstepf = 0.0;
	  }
	  else
	  {
	    fdstepf = fdstep;
	  }
	}

	hist[alpha] += fdstepf;

	if ( npt < nIntPts1_ )
	{
	  child_[npt]->setHistory ( hist, 0 );
	  child_[npt]->update ( fstress, stiff, strain, 0 );
	  child_[npt]->getHistory ( fhist, 0 ); 
	}
	else
	{
	  child_[npt]->update ( fstress, stiff, strain, 0 );
	  fhist = 0.0;
	}

	hist[alpha] -= fdstepf;
      
        if ( alpha == 18)
	{
	  if ( hist[alpha] - fdstep < 0.0 )
	  {
	    fdstepb = 0.0;
	//    System::out() << "Constraint on FD! Lower. hist[alpha] " << hist[alpha] << "\n";
  	  }
	  else 
	  {
	    fdstepb = fdstep;
	  }
	}

	hist[alpha] -= fdstepb;

	if ( npt < nIntPts1_ )
	{
	  child_[npt]->setHistory ( hist, 0 );
	  child_[npt]->update ( bstress, stiff, strain, 0 );
	  child_[npt]->getHistory ( bhist, 0 );
	}
	else
	{
	  child_[npt]->update ( bstress, stiff, strain, 0);
	  bhist = 0.0;
	}

        double step = fdstepf + fdstepb;
        hist[alpha] += fdstepb;
        
	//System::out() << "npt " << npt << " alpha " << alpha << " step " << step << "\n";
    
        fstress = onl_->normalize ( fstress );
	bstress = onl_->normalize ( bstress );

	dsigmadalphapt(alpha, ALL) = ( fstress - bstress ) / step;
	dalphadalphapt(alpha, ALL) = ( fhist - bhist) / step;

	if ( alpha == hSizes_[npt] - 1 ) child_[npt]->setHistory ( hist, 0 );
	//if ( alpha == 18 ) System::out() << "fhist - bhist " << fhist - bhist << "\n";
	}
	}

	if ( debug_ ) 
	{
	 System::out() << "dsigmadalpha:\n" << dsigmadalphapt << "\n";
	 System::out() << "dalphadalpha:\n" << dalphadalphapt << "\n";
	}

	sdeltaspt = 0.0; hdeltaspt = 0.0;  
	hdeltaspt(ALL, 0) = hdeltas(slice(hSizesIdx_[npt], hSizesIdx_[npt+1]), p);
	sdeltaspt(ALL, 0) = sdeltas(slice(npt*sSize_, (npt+1)*sSize_), p);

	// Computing dsigmade * sdeltas 

	Matrix aux = matmul ( dsigmadeptcdf, sdeltaspt );
	derivs (slice(npt*sSize_, (npt+1)*sSize_ ), p) = aux(ALL, 0);

	// Computing dalphade * hdeltas

	Matrix derivhpt ( matmul ( dalphadept, hdeltaspt ) );
	Matrix derivshpt ( matmul ( dsigmadalphapt, sdeltaspt  ) );
	Matrix derivhhpt ( matmul ( dalphadalphapt, hdeltaspt ) );

	hdeltasup[slice(hSizesIdx_[npt], hSizesIdx_[npt+1])] = derivshpt[0] + derivhhpt[0];

     // System::out() << "npt " << npt << ": derivshpt " << derivshpt[0] << " derivhhpt " << derivhhpt << "\n";
	derivh(slice(npt*sSize_, npt*sSize_ + sSize_), p) = derivhpt[0];

	// Computing dsigmadprops * sdeltas and dalphadprops * hdeltas (in case properties are variable)
	if ( learnProp_ > 0 )
	{
	  Matrix derivhppt ( matmul ( dalphadproppt, hdeltaspt ) ); 
	  Matrix derivsppt ( matmul ( dsigmadproppt, sdeltaspt ) ); 
	  derivp(slice(npt*learnProp_, (npt+1)*learnProp_), p) = derivhppt[0] + derivsppt[0];
        }
    }
  
   if ( debug_ )
   {
     System::out() << "sdeltas before update: " << select ( data->deltas, iNeurons_, ALL )
	     << "\n";
     System::out() << "hdeltas before update: " << select ( data->deltas, hNeurons_, ALL )
	     << "\n";
   }

    // Backpropagate through time
    if ( state != nullptr )
    {
      state->deltas ( hNeurons_, p ) = hdeltasup; 
      if ( debug_ )  System::out() << "hdeltas in the following iteration: " << state->deltas ( hNeurons_, p ) << "\n";
    }
  }
  
   // Update gradients
   
   Matrix acts ( select ( data->activations, inpNeurons_, ALL ) );
   Matrix gradmat ( wSize_, inpNeurons_.size() );
   gradmat = 0.0; 

   Matrix alphacontrib ( matmul ( derivh, acts.transpose()) );
   if ( debug_ ) System::out() << "Alpha contribution: " << alphacontrib << "\n";
   if ( debug_ ) System::out() << "Gradmat without history: " << matmul ( derivs, acts.transpose() );
  /* gradmat = matmul ( derivs, acts.transpose()) + alphacontrib;
   if ( debug_ ) System::out() << "With history: " << gradmat << "\n";

   for ( idx_t in = 0; in < wSize_; ++in )
   {
     if ( delconn_[in] == 0.0 ) grads[iInpWts_(in, ALL)] += gradmat (in, ALL);
     if ( useBias_ )
     {
        grads[iBiases_[in]] += sum (derivs (in, ALL));
        grads[iBiases_[in]] += sum (derivh (in, ALL));
     }
   }

  for ( idx_t in = 0; in < nIntPts_*learnProp_; ++in )
  {
     grads[iBiasesProp_[in]] += sum (derivp (in, ALL));
  }
*/
  // Backpropagate through the network

  select ( data->deltas, inpNeurons_, ALL) += derivs; 
//	      matmul ( inpWeights_.transpose(), derivs);
   
  select ( data->deltas, inpNeurons_, ALL) += derivh; 
//	      matmul ( inpWeights_.transpose(), derivh);
 
  if ( debug_ ) 
  {
    System::out() << "derivs (sig): " << derivs << "\n";
    System::out() << "derivh (his): " << derivh << "\n";
    System::out() << "deltas input: " << select ( data->deltas, inpNeurons_, ALL ) << "\n";
  }
}

//-----------------------------------------------------------------------
//   getHistory_
//-----------------------------------------------------------------------

void MatRecLayer::getHistory_

  ( const Ref<NData>  data,
    const Ref<NData>  state,
    const Properties& globdat,
    const Properties& params )

{
  if ( debug_ ) if ( state == nullptr ) System::out() << "STATE NULL\n"; 

  idx_t nIntPts = size_/sSize_;

  Matrix hists ( select ( data->values, hNeurons_, ALL ) );
  Vector aux ( data->batchSize());
  aux = 0.0;

  for ( idx_t p = 0; p < data->batchSize(); ++p )
  {
    Vector phist ( hists( ALL, p ) );

    for ( idx_t npt = 0; npt < nIntPts; ++npt )
    {

      Vector hist ( hSizes_[npt] );

      for ( idx_t nh = 0; nh < hSizes_[npt]; ++nh ) // History of int point
       {
 	hist[nh] = phist[hSizesIdx_[npt]+nh];
       }

       aux[p] += hist[6];
   
 //      System::out() << "History from getHistory: " << hist << "\n";
   }
  }

 aux /= (double) nIntPts;

// System::out() << "Average history: " << aux << "\n";

 params.set ( LearningParams::HISTORY, aux );

}


//-----------------------------------------------------------------------
//   getJacobian_
//-----------------------------------------------------------------------

void MatRecLayer::getJacobian_

  ( const Ref<NData>  data,
    const Properties& globdat )

{
  idx_t bsize = data->batchSize();
  idx_t jsize = data->outSize();
 
  if ( debug_ )
  {
    System::out() << "MatRecLayer. getJacobian.\n";
    System::out() << "MatRecLayer. bsize: " << bsize << " \n";
    System::out() << "MatRecLayer. jsize: " <<  jsize << "\n";
  }
   
  double fdstep = 1.e-8;

  if ( jacobian_.size(0) != jsize * bsize )
  {
    jacobian_.resize ( jsize*bsize, inpNeurons_.size() );
    jacobian_ = 0.0;
  }

  Vector normalized ( 9 );

  Matrix derivs ( select ( data->values, iNeurons_, ALL ) );

  Matrix stiff ( sSize_, sSize_ );
  Matrix strain ( 3, 3 ); 
  Vector stress ( sSize_ );
  Matrix jacpt ( sSize_, sSize_ );
  
  stiff = 0.0; stress = 0.0; strain = 0.0; jacpt = 0.0;
 
  double dt = 0.0;

  // Getting history
  
  Matrix hists ( select ( data->values, hNeurons_, ALL ) );

  for ( idx_t p = 0; p < data->batchSize(); ++p )
  {
    if ( frommodel_ )
    {
      dt = dt_;
    }
    else
    {
      dt = data->dts(0, p);
    }

    System::out() << "Batch " << p << " dt (getJac) " << dt << "\n";

    for ( idx_t npt = 0; npt < nIntPts_; ++npt )
    {
      Vector fhist ( hSizes_[npt] );
      Vector bhist ( hSizes_[npt] );
      Vector hist ( hSizes_[npt] );
      Matrix dalphade ( nIntPts_*sSize_, hSizes_[npt] );
 
      dalphade = 0.0; hist = 0.0; strain = 0.0;
            
      for ( idx_t i = 0; i < 3; i++ )
      { 
        strain(i, ALL) =  derivs(slice(npt*9+i*3, npt*9+(i+1)*3), p);
      }
            
      hist = hists(slice(hSizesIdx_[npt], hSizesIdx_[npt+1]), p);
      
      if ( debug_ )
      {
	 System::out() << "History point " << npt << " sample " << p << 
		 " " << hist << "\n"; 
	 System::out() << "Strain point " << npt << " sample " << p << 
		 " " << strain << "\n";
      }
     
      // Calculating dsigmade
      
      if ( npt < nIntPts1_ )
      {
        child_[npt]->setHistory(hist, 0);           // Setting history of int point
	child_[npt]->setDT(dt);
      }
      else
      {
	hist = 0.0; 
      }
      
      // Testing cfd stiff matrix
       Matrix dsigmadeptcdf ( 9, 9 );
       Matrix dsigmadept ( 9, 9 );
       
       Vector fstress ( sSize_ );
       Vector bstress ( sSize_ );
  
       fstress = 0.0; bstress = 0.0;
       
       child_[npt]->update ( stress, dsigmadept, strain, 0 ); 
       
     if ( debug_ ) System::out() << "Start calculating dalphade.\n"; 
      for ( idx_t nr = 0; nr < 3; ++nr )
      {
        for ( idx_t nc = 0; nc < 3; ++nc )
        {
          strain(nr, nc) += fdstep;
          child_[npt]->update ( fstress, stiff, strain, 0 );
        
          strain(nr, nc) -= 2.0*fdstep;
          child_[npt]->update ( bstress, stiff, strain, 0 );

	  fstress = onl_->normalize ( fstress );
	  bstress = onl_->normalize ( bstress );

          dsigmadeptcdf( ALL, nr*3+nc) = (fstress - bstress ) / 2.0 / fdstep;  

          strain(nr, nc) += fdstep;

// System::out() << "comp " << nr*3+nc << ": "<< dsigmadeptcdf(ALL,nr*3+nc) << "\n";
        }
      }

      if ( debug_ )
      {
        System::out() << "dsigmade:\n" << dsigmadept << "\n";
        System::out() << "dsigmade cdf:\n" << dsigmadeptcdf << "\n";
      }
      
      if ( hSizes_[npt] < 10 ) 
      {
        stiff = dsigmadeptcdf;
      }
      else
      {
        stiff = dsigmadeptcdf;
      }

//       stiff = dsigmadept;
      if ( debug_ ) System::out() << "MatRecLayer. getJacobian. stiffness: " << stiff << "\n";
      if ( debug_ ) System::out() << "jacobian " << data->jacobian.size(0) << " x " << data->jacobian.size(1) << "\n";

      jacpt = data->jacobian ( slice(p*jsize,(p+1)*jsize), slice( npt*sSize_, (npt+1)*sSize_ ) );   
     
//      for ( idx_t i = 0; i < 9; i++ ) jacpt(ALL, i) *= normalized[i];
      if ( debug_ ) System::out() << "normalization " << normalized << "\n";

      if ( debug_ ) System::out() << "jacpt: " << jacpt << "\n";
    
      jacpt = matmul ( jacpt, stiff );

      if ( debug_ ) System::out() << "jacpt multiplied by stiff: " << jacpt << "\n"; 

      // Save it in the final jacobian matrix 

     for ( idx_t i = 0; i < sSize_; ++i )
     {
	data->jacobian ( slice (p*jsize, (p+1)*jsize), npt*sSize_ + i ) = jacpt (ALL, i);
     }

     }
  } 

  if ( debug_ ) System::out() << "data->jacobian before being multiplied by the weights: " << data->jacobian << "\n";

 // System::out() << "data->jacobian: " << data->jacobian << "\n";
//  if ( debug_ ) System::out() << "inpWeights: " << inpWeights_ << "\n";

//jacobian_ = matmul ( data->jacobian, inpWeights_ ); // will be performed by block layer
 
//  if ( debug_ ) System::out() << "final jacobian: " << jacobian_ << "\n";
  //jacobian_ = data->jacobian;

 // data->jacobian.ref ( jacobian_ );
}

//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   newMatRecLayer
//-----------------------------------------------------------------------


Ref<Model>            newMatRecLayer

  ( const String&       name,
    const Properties&   conf,
    const Properties&   props,
    const Properties&   globdat )

{
  // Return an instance of the class.

  return newInstance<MatRecLayer> ( name, conf, props, globdat );
}

//-----------------------------------------------------------------------
//   declareMatRecLayer
//-----------------------------------------------------------------------

void declareMatRecLayer ()
{
  using jive::model::ModelFactory;

  // Register the StressModel with the ModelFactory.

  ModelFactory::declare ( "MatRec", newMatRecLayer );
}

