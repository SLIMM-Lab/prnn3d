/*
 * Copyright (C) 2019 TU Delft. All rights reserved.
 *
 * Base model for Gaussian Process (GP) regression
 *
 * Rasmussen, C. E. and Williams, C. K. I. Gaussian Processes
 * for Machine Learning. MIT Press, 2016. <www.gaussianprocess.org>
 *
 * Author: Iuri Barcelos, i.barceloscarneiromrocha@tudelft.nl
 * Date:   Oct 2019
 * 
 * Modified: Marina Maia, m.alvesmaia@tudelft.nl
 * Date:     Nov 2021
 *           Added samplePosterior with fixed seed 
 */

#include <jem/base/System.h>
#include <jem/base/Error.h>
#include <jem/base/array/operators.h>
#include <jem/base/array/select.h>
#include <jem/io/PrintWriter.h>
#include <jem/io/FileWriter.h>
#include <jem/numeric/algebra/utilities.h>
#include <jem/numeric/algebra/LUSolver.h>
#include <jem/numeric/algebra/matmul.h>
#include <jem/numeric/utilities.h>
#include <jem/numeric/func/UserFunc.h>
#include <jem/numeric/algebra/Cholesky.h>
#include <jem/util/ArrayBuffer.h>
#include <jem/util/Event.h>
#include <jive/util/error.h>
#include <jive/util/Printer.h>
#include <jive/util/utilities.h>
#include <jive/util/Constraints.h>
#include <jive/util/Globdat.h>
#include <jive/util/FuncUtils.h>
#include <jive/util/Random.h>
#include <jive/model/Actions.h>
#include <jive/model/ModelFactory.h>
#include <jive/implict/ArclenActions.h>
#include <jive/model/StateVector.h>
#include <jive/implict/SolverInfo.h>

#include <math.h>

#include "GPModel.h"
#include "LearningNames.h"
#include "TrainingData.h"
#include "NData.h"

using jem::util::ArrayBuffer;
using jem::io::endl;
using jem::numeric::UserFunc;
using jem::numeric::LUSolver;
using jem::numeric::Cholesky;
using jem::numeric::matmul;
using jem::io::PrintWriter;
using jem::io::FileWriter;
using jive::IntVector;
using jive::model::Actions;
using jive::model::ActionParams;
using jive::util::FuncUtils;
using jive::util::Globdat;
using jive::util::Random;
using jive::model::StateVector;
using jive::implict::ArclenActions;
using jive::implict::ArclenParams;
using jive::implict::SolverInfo;

//=======================================================================
//   class GPModel
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------

const char* GPModel::KERNEL = "kernel";
const char* GPModel::SEED   = "rseed";
const double PI             = 3.1415926535897932384626433;

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------

GPModel::GPModel

  ( const String&     name,
    const Properties& conf,
    const Properties& props,
    const Properties& globdat ):

    Model ( name )

{
  using jive::util::XDofSpace;
  using jive::util::newDummyItemSet;

  n_ = 0;

  newData_   = true;
  connected_ = false;

  // Initialize the RNG

  Properties       myConf  = conf .makeProps ( name );
  Properties       myProps = props.findProps ( name );

  idx_t seed;

  myProps.get ( seed, SEED );
  myConf.set  ( SEED, seed );

  Ref<Random> generator = Random::get ( globdat );
  
  generator->restart ( seed );

  // Create ItemSet and DofSpace 
  
  items_   = newDummyItemSet ( "hyperparameters", "param" );
  dofs_    = XDofSpace:: get ( items_,        globdat      );
  dofType_ = dofs_-> addType ( LearningNames::HYPERDOF     );

  items_->store ( LearningNames::HYPERSET, globdat );

  // Initialize kernel

  kernel_ = newKernel ( KERNEL, myConf, myProps, globdat );
}

GPModel::~GPModel ()
{}

//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------

void GPModel::configure

  ( const Properties& props,
    const Properties& globdat )

{
  System::out() << "GPModel::configure.\n" << props << "\n"; 
	
  Properties kProps = props;

  if ( props.contains ( myName_ ) )
  {
    kProps = props.findProps ( myName_ ).findProps ( KERNEL );
  }

  System::out() << "kernel props: " << kProps << "\n";

  kernel_->configure ( kProps, globdat );

  System::out() << "GPModel::configure ended.\n";

}

//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------

void GPModel::getConfig

  ( const Properties& conf,
    const Properties& globdat ) const

{
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------

bool GPModel::takeAction

  ( const String&     action,
    const Properties& params,
    const Properties& globdat )

{
  using jive::model::Actions;

//  System::out() << "GPModel. Action: " << action << "\n";

  if ( action == LearningActions::UPDATE )
  {
    double logp = update_ ( globdat );

    params.set ( LearningParams::OBJFUNCTION, logp );
    params.set ( LearningParams::NOISE, kernel_->getNoise() );

    return true;
  }

  if ( action == LearningActions::GETGRADS )
  {
    Vector grads;
    params.get ( grads, LearningParams::GRADS );

    getGrads_ ( grads, globdat );

    return true;
  }

  if ( action == LearningActions::WRITEPARAMS )
  {
    Vector vars = kernel_->getVars();

    params.set ( LearningParams::WEIGHTS, vars );

    return true;
  }

  if ( action == LearningActions::RECALL )
  {
    predict_ ( params, globdat );

    Vector vars = kernel_->getVars();

    params.set ( LearningParams::WEIGHTS, vars );

    return true; 
  }

  if ( action == LearningActions::BATCHRECALL )
  {
    batchPredict_ ( params, globdat );

    return true;
  }

  if ( action == LearningActions::SAMPLEPRIOR )
  {
    System::out() << "Sample prior\n";	  
    samplePrior_ ( params, globdat );

    return true;
  }

  if ( action == LearningActions::SAMPLEPOSTERIOR )
  {
    samplePosterior_ ( params, globdat );

    return true;
  }


  if ( action == LearningActions::SETEXTERNALNOISE )
  {
    params.get ( extNoise_, LearningParams::NOISE );

    return true;
  }

  return false;
}

//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------

void GPModel::init_

  ( const Properties& globdat )

{
  Ref<TrainingData> tdata = TrainingData::get ( globdat, getContext() );

  if ( !connected_ )
  {
    connect ( tdata->newDataEvent, this, &GPModel::invalidate_ );
    connected_ = true;
  }

  n_ = tdata->sampleSize();

//  System::out() << "Updating regression with " << n_ << " observations\n";

  IdxVector samples ( iarray ( n_ ) );

  Batch b = tdata->getData ( samples );

  if ( tdata->sequenceSize() > 1 )
  {
    throw IllegalInputException ( JEM_FUNC, 
      "GPModel can only handle time-independent data" );
  }

  if ( tdata->outputSize() > 1 )
  {
    throw IllegalInputException ( JEM_FUNC,
      "GPModel can only handle a single output variable at a time" );
  }

  x_.resize ( tdata->inputSize(), n_ );
  x_ = 0.0;

  y_.resize ( n_ );

  alpha_.resize ( n_ );
  alpha_ = 0.0;

  K_.resize ( n_, n_ );

  x_ = b[0]->inputs;
  y_ = b[0]->targets(0,ALL);

//  System::out() << "GP model init_ with x_ " << x_ << "\n";
//  System::out() << "GP model init_ with y_ " << y_ << "\n";

  newData_ = false;
}

//-----------------------------------------------------------------------
//   update_
//-----------------------------------------------------------------------

double GPModel::update_

  ( const Properties& globdat )

{
//  System::out() << "GPModel::update. Sample size: " << n_ << "\n";

  if ( newData_ )
  {
    init_ ( globdat );
  }

  // Update kernel hyperparameters

  kernel_->update ( globdat );

  // Get the covariance matrix of training cases

  K_ = kernel_->eval ( x_ );

  // Add noise coming from outside the kernel

  if ( extNoise_ != nullptr )
  {
    Matrix    blk  ( K_.shape() );
    blk = 0.0;

    IdxVector iblk ( iarray ( K_.size(0) ) );

    extNoise_->updateMatrix();
    extNoise_->getBlock ( blk, iblk, iblk );

    //System::out() << "GPModel: Got noise block " << blk << '\n';

    K_ += blk;
  }

  // Factor the matrix

  if ( !Cholesky::factor ( K_ ) )
  {
    throw Error ( JEM_FUNC, "GPModel: non-positive definite covariance matrix" );
  }

  alpha_ = y_;

  Cholesky::solve ( alpha_, K_ );

  // Compute the log marginal likelihood 

  double ltrace = 0.0;
  for ( idx_t s = 0; s < n_; s++ )
  {
    ltrace += log( K_(s,s) );
  }

  double logp = - 0.5 * dot ( y_, alpha_ ) -
                  ltrace - (double)n_ / 2.0 * log ( 2.0 * PI );

  return -logp;
}

//-----------------------------------------------------------------------
//   predict_
//-----------------------------------------------------------------------

void GPModel::predict_

  ( const Properties& params,
    const Properties& globdat )

{
  Vector input;
  
  params.get ( input, LearningParams::INPUT );

  System::out( ) << "Input: " << input << "\n";

  System::out( ) << "Input vector size: " << input.size() << " Matrix: " << x_ << "\n"; 

  JEM_PRECHECK ( input.size() == x_.size(0) );

  Vector kstar = kernel_->eval ( input, x_ );

  //System::out( ) << "Correlation vector kstar: " << kstar << "\n;

 // System::out( ) << "Weights (alpha): " << alpha_ << "\n"; 

  double mean = dot ( kstar, alpha_ );

  Vector v ( kstar.clone() );

  Cholesky::fsub ( v, K_ );

  double var  = sqrt(kernel_->eval ( input ) - dot ( v ) );

  Vector der ( input.shape() );

  Matrix derivs = kernel_->evalDerivs ( input, x_ );

  der = matmul ( derivs, alpha_ );

  params.set ( LearningParams::OUTPUT, mean     );
  params.set ( LearningParams::VARIANCE, var    );
  params.set ( LearningParams::DERIVATIVES, der );
}

//-----------------------------------------------------------------------
//   batchPredict_
//-----------------------------------------------------------------------

void GPModel::batchPredict_

  ( const Properties& params,
    const Properties& globdat )

{
  TensorIndex i, j;

  Ref<NData>  data;

  params.get ( data, LearningParams::DATA );

  if ( data->outSize() != 1 )
  {
    throw Error ( JEM_FUNC, "GPModel expects NData objects with scalar output" );
  }

  Matrix Kstar     = kernel_->eval ( data->inputs, x_ );
  Matrix Kstarstar = kernel_->eval ( data->inputs     );

  Vector mean ( matmul ( Kstar, alpha_ ) );

  Matrix V ( Kstar.transpose().clone() );

  Cholesky::solve ( V, K_ );

  data->covariance.resize ( Kstarstar.shape() );
  data->covariance = 0.0;

  data->outputs(0,ALL) = mean;
  data->covariance = Kstarstar - matmul ( Kstar, V );
}

//-----------------------------------------------------------------------
//   samplePrior_
//-----------------------------------------------------------------------

void GPModel::samplePrior_

  ( const Properties& params,
    const Properties& globdat )

{
  TensorIndex i, j;

//  System::out() << "Entered samplePrior_. Sample size: " << n_ << "\n";

  Ref<NData>  data;
  Ref<Random> rng = Random::get ( globdat );
  
  //int rseed = 0;
 
  //if ( globdat.find ( rseed, "rseed" ) )
  //{
  //   rng->restart ( rseed );
  //   System::out() << "seed: " << rseed << "\n";
  //}

  params.get ( data, LearningParams::INPUT );

  System::out() << "Input Prior " << data->inputs(ALL, ALL) << "\n";

  Matrix cov = kernel_->eval ( data->inputs );

  if ( !Cholesky::factor ( cov ) )
  {
    throw Error ( JEM_FUNC, "GPModel: non-positive definite covariance matrix" );
  }

  cov(i,j) = where ( j>i, 0.0, cov(i,j) );

  for ( idx_t o = 0; o < data->outSize(); ++o )
  {
    for ( idx_t b = 0; b < data->batchSize(); ++b )
    {
      data->outputs(o,b) = rng->nextGaussian();
    }

    data->outputs(o,ALL) = matmul ( cov, data->outputs(o,ALL) );
  }

 // System::out() << "Finished sampling prior\n";
}

//-----------------------------------------------------------------------
//   samplePosterior_
//-----------------------------------------------------------------------

void GPModel::samplePosterior_

  ( const Properties& params,
    const Properties& globdat )

{
  TensorIndex i, j;

  Ref<NData>  data;
  Ref<Random> rng = Random::get ( globdat );
 /* 
  int rseed = 0;
  
  if ( globdat.find ( rseed, "rseed" ) )
  {
     rng->restart ( rseed );
  }
*/
  params.get ( data, LearningParams::INPUT );

  Matrix Kstar = kernel_->eval ( data->inputs, x_ );

  Vector mean ( matmul ( Kstar, alpha_ ) );

  Matrix Kstarstar = kernel_->eval ( data->inputs );

  Matrix V ( Kstar.transpose().clone() );

  Cholesky::solve ( V, K_ );

  Matrix cov ( Kstarstar - matmul ( Kstar, V ) );

  if ( !Cholesky::factor ( cov ) )
  {
    throw Error ( JEM_FUNC, "GPModel: non-positive definite covariance matrix" );
  }

  cov(i,j) = where ( j>i, 0.0, cov(i,j) );

  for ( idx_t o = 0; o < data->outSize(); ++o )
  {
    for ( idx_t b = 0; b < data->batchSize(); ++b )
    {
      data->outputs(o,b) = rng->nextGaussian();
    //  System::out() << "samplePosterior random output " << data->outputs(o,b) << "\n";
    }

   // System::out() << "cov matrix " << cov << " mean " << mean << "\n";
    data->outputs(o,ALL) = mean + matmul ( cov, data->outputs(o,ALL) );
  }
}

//-----------------------------------------------------------------------
//   getGrads_
//-----------------------------------------------------------------------

void GPModel::getGrads_

  (       Vector&     grads,
    const Properties& globdat )

{
  JEM_ASSERT ( kernel_->varCount() == grads.size() );

  Matrix K = kernel_->eval ( x_ );

  if ( !Cholesky::invert ( K ) )
  {
    throw Error ( JEM_FUNC, 
      "GPModel: non-positive definite covariance matrix. This should not be happening!" );
  }

  Matrix alphas ( n_, n_ );
  alphas = matmul ( alpha_, alpha_ );

  alphas -= K;

  Cubix G; 
  kernel_->gradients ( G, x_ );

  grads = 0.0;

  for ( idx_t v = 0; v < kernel_->varCount(); ++v )
  {
    for ( idx_t i = 0; i < n_; ++i )
    {
      grads[v] += 0.5 * dot ( alphas(i,ALL), G(v,ALL,i) );
    }
  }

  grads *= -1.0;

  //System::out() << "Grads " << grads << "\n";
}

//-----------------------------------------------------------------------
//   invalidate_ 
//-----------------------------------------------------------------------

void GPModel::invalidate_ ()

{
  newData_ = true;
}

//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------

Ref<Model> GPModel::makeNew

  ( const String&     name,
    const Properties& conf,
    const Properties& props,
    const Properties& globdat )

{
  return newInstance<GPModel> ( name, conf, props, globdat );
}

//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   declareGPModel
//-----------------------------------------------------------------------

void declareGPModel ()
{
  using jive::model::ModelFactory;

  ModelFactory::declare ( "GP", 
                          & GPModel::makeNew );
}
