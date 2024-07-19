/*
 *  TU Delft 
 *
 *  Iuri Barcelos, May 2019
 *
 *  Stochastic gradient descent algorithm for neural network training
 *
 *  Kingma, D. P.; Ba, J. L. Adam: A method for stochastic optimization.
 *  In: Proceedings of the International Conference on Learning
 *  Representations (ICLR 2015), San Diego, 2015.
 *
 */

#include <cstdlib>
#include <random>

#include <jem/base/Error.h>
#include <jem/base/limits.h>
#include <jem/base/Float.h>
#include <jem/base/System.h>
#include <jem/base/Exception.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/array/operators.h>
#include <jem/base/Thread.h>
#include <jem/base/Monitor.h>
#include <jem/io/Writer.h>
#include <jem/io/PrintWriter.h>
#include <jem/io/FileWriter.h>
#include <jem/io/FileReader.h>
#include <jem/util/Event.h>
#include <jem/util/Flex.h>
#include <jem/numeric/algebra/utilities.h>
#include <jem/mp/MPException.h>
#include <jem/mp/Context.h>
#include <jem/mp/Buffer.h>
#include <jem/mp/Status.h>
#include <jive/util/utilities.h>
#include <jive/util/Globdat.h>
#include <jive/util/FuncUtils.h>
#include <jive/algebra/VectorSpace.h>
#include <jive/model/Actions.h>
#include <jive/model/StateVector.h>
#include <jive/app/ModuleFactory.h>
#include <jive/implict/Names.h>
#include <jive/implict/SolverInfo.h>
#include <jive/util/XDofSpace.h>
#include <jive/mp/Globdat.h>
#include <jive/util/Random.h>

#include "AdamModule.h"
#include "SolverNames.h"
#include "TrainingData.h"
#include "LearningNames.h"
#include "XNeuronSet.h"

JEM_DEFINE_CLASS( jive::implict::AdamModule );


JIVE_BEGIN_PACKAGE( implict )

using jive::util::Random;
using jem::max;
using jem::newInstance;
using jem::Float;
using jem::System;
using jem::Exception;
using jem::Error;
using jem::Thread;
using jem::Monitor;
using jem::io::endl;
using jem::io::Writer;
using jem::io::PrintWriter;
using jem::io::FileWriter;
using jem::io::FileReader;
using jem::numeric::axpy;
using jem::mp::MPException;
using jem::mp::SendBuffer;
using jem::mp::RecvBuffer;
using jem::mp::Status;

using jive::util::FuncUtils;
using jive::util::XDofSpace;
using jive::model::Actions;
using jive::model::StateVector;

//=======================================================================
//   class AdamModule
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------

const char* AdamModule::TYPE_NAME   = "Adam";
const char* AdamModule::SEED        = "rseed";
const char* AdamModule::ALPHA       = "alpha";
const char* AdamModule::BETA1       = "beta1";
const char* AdamModule::BETA2       = "beta2";
const char* AdamModule::EPSILON     = "epsilon";
const char* AdamModule::L2REG       = "l2reg";
const char* AdamModule::L1REG       = "l1reg";
const char* AdamModule::L1PL        = "l1pl";
const char* AdamModule::MINIBATCH   = "miniBatch";
const char* AdamModule::LOSSFUNC    = "loss";
const char* AdamModule::PRECISION   = "precision";
const char* AdamModule::VALSPLIT    = "valSplit";
const char* AdamModule::SKIPFIRST   = "skipFirst";
const char* AdamModule::JPROP       = "jProp";
const char* AdamModule::PRUNING     = "pruning";

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------

AdamModule::AdamModule ( const String& name ) :

  Super ( name )

{
  alpha_ = 0.001;
  beta1_ = 0.9;
  beta2_ = 0.999;
  eps_   = 1.e-8;
  lambda_ = 0.0;
  lambdaL1_ = 0.0;
  lambdaPl_ = 0.0;
  batchSize_ = 1;
  outSize_ = 1;
  epoch_     = 0;
  iiter_     = 1;
  precision_ = 1.e-3;
  valSplit_  = 0.0;
  skipFirst_ = 0;
  nip_ = 2;  
  prev_ = 0;
  subsetSize_ = -1;  
  rseed_ = 110; 

  pruning_ = false;
  dissipation_ = false;
  dehom_ = false;

  mpi_   = false;
  mpx_   = nullptr;

  jprop_ = 0.0;
}


AdamModule::~AdamModule ()
{}

//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------

Module::Status AdamModule::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  Ref<DofSpace>     dofs  = DofSpace::   get ( globdat, getContext() );
  Ref<Model>        model = Model::      get ( globdat, getContext() );
  Ref<TrainingData> data = TrainingData::get ( globdat, getContext() );

  // Parallelization stuff

  mpx_ = jive::mp::Globdat::getMPContext ( globdat );

  if ( mpx_ == nullptr )
  {
    throw Error ( JEM_FUNC, String::format (
       "MPContext has not been found" ) );
  }

  if ( mpx_->size() > 1 )
  {
    print ( System::info( myName_ ), getContext(), 
      ": Running in MP mode with ", mpx_->size(), " processes" );
    mpi_ = true;
  }

  if ( !mpi_ )
  {
    print ( System::info( myName_ ), getContext(), 
      ": Running in sequential mode" );
  }

  // Resize some stuff

  outSize_ = data->outputSize();

  if ( selComp_[0] < 0 )
  {
    selComp_.resize ( outSize_ );
    for ( idx_t i = 0; i < outSize_; i++ ) selComp_[i] = i;
  }

  System::out() << "# components in the loss function " << outSize_ << "\n";
  System::out() << "Index of components " << selComp_ << "\n";

  idx_t dc = dofs->dofCount();

  System::out() << "dofcount: " << dc << "\n";

  g_.resize     ( dc );
  g_ = 0.0;

  if ( lambda_ > 0.0 )
  {
    rg_.resize ( dc );
    rg_ = 0.0;
  }
  
  if ( lambdaL1_ > 0.0)
  {
    rg_.resize ( dc );
    rg_ = 0.0;
  }

  if ( lambdaPl_ > 0.0 )
  {
    rg_.resize ( dc );
    rg_ = 0.0;
  }

  if ( mpi_ && mpx_->myRank() == 0 )
  {
    gt_.resize ( dc );
    gt_ = 0.0;
  }

  if ( !mpi_ || mpx_->myRank() == 0 )
  {
    m_.resize     ( dc );
    v_.resize     ( dc );
    m0_.resize    ( dc );
    v0_.resize    ( dc );

    m_ = 0.0;  v_ = 0.0;
    m0_ = 0.0; v0_ = 0.0;
  }

  if ( valSplit_ > 0.0 && skipFirst_ == 0 )
  {
    skipFirst_ = valSplit_ * data->sampleSize();
  }

  return OK;
}

//-----------------------------------------------------------------------
//   shutdown
//-----------------------------------------------------------------------

void AdamModule::shutdown ( const Properties& globdat )
{
  bool root = ( !mpi_ || mpx_->myRank() == 0 );

  if ( root )
  {
    System::out() << "AdamModule statistics ..." 
      << "\n-- total # of epochs: " << epoch_ 
      << "\n-- total optimization time: " << total_ << ", of which"
      << "\n---- " << t5_ << " shuffling the dataset"
      << "\n---- " << t6_ << " updating weights"
      << "\n---- " << t1_ << " computing loss and grads, of which"
      << "\n------ " << t2_ << " allocating sample batches"
      << "\n------ " << t3_ << " propagating through the network"
      << "\n------ " << t4_ << " backpropagating through the network"
      << "\n\n";
  }
}

//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------

void AdamModule::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  using jem::maxOf;
 
  System::out() << "AdamModule configure\n";

  if ( props.contains( myName_ ) )
  {
    Properties  myProps = props.findProps ( myName_ );

    myProps.get  ( lossName_,  LOSSFUNC  );

    func_  = NeuralUtils::getLossFunc ( lossName_ );
    grad_  = NeuralUtils::getLossGrad ( lossName_ );

    selComp_.resize(1);
    selComp_[0] = -1;

    myProps.find ( alpha_,      ALPHA      );
    myProps.find ( eps_,        EPSILON    );
    myProps.find ( batchSize_,  MINIBATCH  );
    myProps.find ( precision_,  PRECISION  );
    myProps.find ( nip_,        "nip"      );
    myProps.find ( prev_,       "prev"     );
    myProps.find ( pruning_,    PRUNING    );
    myProps.find ( dehom_,      "dehom"    );
    myProps.find ( dissipation_, "dissipation" );
    myProps.find ( subsetSize_, "subset"   );
    myProps.find ( rseed_,      SEED       );
    myProps.find ( selComp_,    "selComp"  );  // Selected components only in
                                               // the loss function. Otherwise,
					       // all components are considered

    myProps.find ( beta1_,     BETA1,    0.0, 1.0            );
    myProps.find ( beta2_,     BETA2,    0.0, 1.0            );
    myProps.find ( lambda_,    L2REG,    0.0, maxOf(lambda_) );
    myProps.find ( lambdaL1_,  L1REG,    0.0, maxOf(lambdaL1_));
    myProps.find ( lambdaPl_,  L1PL,    0.0, maxOf(lambdaPl_));
    myProps.find ( valSplit_,  VALSPLIT, 0.0, 1.0            );

    if ( lambda_ > 0.0 ) System::out() << "lambda: " << lambda_ << "\n";
    if ( lambdaL1_ > 0.0 ) System::out() << "lambdaL1: " << lambdaL1_ << "\n";
    if ( lambdaPl_ > 0.0 ) System::out() << "lambdaPl: " << lambdaPl_ << "\n";

    if ( valSplit_ == 0.0 )
    {
      myProps.find ( skipFirst_, SKIPFIRST, 0, maxOf ( skipFirst_ ) );
    }

    globdat.set ( "subset", subsetSize_ );
    globdat.set ( "rseed", rseed_ );
    globdat.set ( "skipFirst", skipFirst_ );

    myProps.find ( jprop_, JPROP, 0.0, 1.0 );
  }
}

//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------

void AdamModule::getConfig

  ( const Properties&  conf,
    const Properties&  globdat ) const

{
  Properties  myConf = conf.makeProps ( myName_ );

  myConf.set ( LOSSFUNC,   lossName_   );
  myConf.set ( ALPHA,      alpha_      );
  myConf.set ( EPSILON,    eps_        );
  myConf.set ( MINIBATCH,  batchSize_  );
  myConf.set ( PRECISION,  precision_  );
  myConf.set ( BETA1,      beta1_      );
  myConf.set ( BETA2,      beta2_      );
  myConf.set ( VALSPLIT,   valSplit_   );
  myConf.set ( SKIPFIRST,  skipFirst_  );
  myConf.set ( "nip",      nip_        );
  myConf.set ( "prev",     prev_       );
  myConf.set ( "subset",   subsetSize_ );
  myConf.set ( "selComp",  selComp_    );
}

//-----------------------------------------------------------------------
//   advance
//-----------------------------------------------------------------------

void AdamModule::advance ( const Properties& globdat )
{
  epoch_++;
}

//-----------------------------------------------------------------------
//   solve
//-----------------------------------------------------------------------

void AdamModule::solve

  ( const Properties& info,
    const Properties& globdat )

{
  total_.start();

  if ( mpi_ )
  {
    mpSolve_ ( info, globdat );
  }
  else
  {
    solve_ ( info, globdat );
  }

  total_.stop();
}

//-----------------------------------------------------------------------
//   cancel
//-----------------------------------------------------------------------

void AdamModule::cancel ( const Properties& globdat )
{
}

//-----------------------------------------------------------------------
//   commit
//-----------------------------------------------------------------------

bool AdamModule::commit ( const Properties& globdat )
{
  return true;
}

//-----------------------------------------------------------------------
//   setPrecision
//-----------------------------------------------------------------------

void AdamModule::setPrecision ( double eps )
{
}

//-----------------------------------------------------------------------
//   getPrecision
//-----------------------------------------------------------------------

double AdamModule::getPrecision () const
{
  return precision_;
}

//-----------------------------------------------------------------------
//   mpSolve_
//-----------------------------------------------------------------------

void AdamModule::mpSolve_

  ( const Properties& info,
    const Properties& globdat )

{
  idx_t rank = mpx_->myRank();
  idx_t size = mpx_->size();
  bool  root = ( rank == 0 );
  bool  last = ( rank == size - 1 );

  Ref<TrainingData> data = TrainingData::get ( globdat, getContext() );
  Ref<DofSpace>     dofs = DofSpace::get     ( globdat, getContext() );
  Ref<Model>        model = Model::get       ( globdat, getContext() );

  idx_t dc = dofs->dofCount();

  Vector state;
  StateVector::get ( state, dofs, globdat );

  idx_t n = data->sampleSize();

  idx_t batch = 0, load = 0, end = 0, beg = 0;

  IdxVector valset;   
  IdxVector trainset;

  if ( skipFirst_ )
  {
    trainset.ref ( IdxVector ( iarray ( slice ( skipFirst_, n ) ) ) );
    n -= skipFirst_;
  }
  else
  {
    trainset.ref ( IdxVector ( iarray ( n ) ) );
  }

  t5_.start();
  NeuralUtils::shuffle ( trainset, globdat );
  t5_.stop();

  double trainloss = 0.0;

  while ( n > 0 )
  {
    batch = min ( n, batchSize_    );
    load  = max ( 1, batch / size  );
    end   = max ( n - rank*load, 0 );
    beg   = last ? n - batch : max ( end - load, 0_idx );

    n    -= batch;

    g_ = 0.0;
    gt_ = 0.0;

    double localloss = 0.0;
    
    if ( batch > rank )
    {
      localloss = eval_ ( trainset[slice(beg,end)], true, globdat );    
    }

    double accloss = 0.0;

    mpx_->reduce ( RecvBuffer ( &accloss,   1 ),
                   SendBuffer ( &localloss, 1 ),
		   0,
		   jem::mp::SUM );

    mpx_->reduce ( RecvBuffer ( gt_.addr(), dc ),
		   SendBuffer ( g_. addr(), dc ),
		   0,
		   jem::mp::SUM );

    t6_.start();
    if ( root )
    {
      trainloss += accloss;

      gt_ /= (double)batch;

      m_ = beta1_*m0_ + (1.-beta1_)*gt_;
      v_ = beta2_*v0_ + (1.-beta2_)*gt_*gt_;

      double alphat = alpha_ * (
		      sqrt ( 1. - pow( beta2_,(double)iiter_ ) ) /
			   ( 1. - pow( beta1_,(double)iiter_ ) ) );

      state -= alphat * m_ / ( sqrt(v_) + eps_ );

      m0_ = m_;
      v0_ = v_;


      mpx_->broadcast ( SendBuffer ( state.addr(), dc ) );
    }

    if ( !root )
    {
      mpx_->broadcast ( RecvBuffer ( state.addr(), dc ), 0 );
    }

    model->takeAction ( LearningActions::UPDATE, Properties(), globdat );

    if ( lambda_ > 0.0 )
    {
      IdxVector iwts ( dofs->dofCount() );
      IdxVector iaxs ( dofs->dofCount() );

      idx_t nw = dofs->getDofsForType ( iwts, iaxs,
		 dofs->findType ( LearningNames::WEIGHTDOF ) );

      iwts.reshape ( nw );

      r_        = 0.5 * lambda_ * dot ( state[iwts] );

//      System::out() << "iwt: " << iwts << "\n";
 //     System::out() << "========================\n";
  //    System::out() << "dot (state) " << dot ( state [iwts] ) << "\n";

      rg_[iwts] = lambda_ * state[iwts];
    }
    
    if ( lambdaL1_ > 0.0 )       // Lasso regression
    {
     IdxVector iwts ( dofs->dofCount() );
      IdxVector iaxs ( dofs->dofCount() );

      idx_t nw = dofs->getDofsForType ( iwts, iaxs,
		 dofs->findType ( LearningNames::WEIGHTDOF ) );

      iwts.reshape ( nw );
      
  // idx_t init = nip_*3*3 + nip_*3;
      idx_t init = prev_*3+prev_+nip_*3*prev_+nip_*3;

      Vector sig ( nw );
      sig = 0.0;
      double sum = 0.0;
      idx_t cont = 0;
      
      for (idx_t i = 0; i < nw; ++i )
      {
        if ( iwts[i] >= init ) 
        {
     //      System::out() << "iwts dof " << iwts[i] << " state[iwts[i]] " << state[iwts[i]] << "\n";
           if ( state[iwts[i]] < 0.0 )
           {
             sig[cont] = -1.0;
	     sum += abs(state[iwts[i]]);
           }
           else
           {
             sig[cont] = 0.0;
           //  sum += abs(state[iwts[i]]);
           }
           cont += 1;
        }
      }

      r_ = lambdaL1_ * sum;

   /*   System::out() << "AdamModule. idofs: " << iwts << "\n";
      System::out() << "AdamModule. state: " << state << "\n";
      System::out() << "AdamModule. sum: " << sum << "\n";
      System::out() << "AdamModule. signals: " << sig << "\n";
      System::out() << "sig size: " << sig.size() << " rg_[iwts] size " <<
	      rg_[iwts].size() << "\n";
*/
      rg_[iwts] = lambdaL1_ * sig;
    }

    if ( lambdaPl_ > 0.0 )
    {
      System::out() << "Not implemented for mpSolve!\n";
    }

    iiter_++;
    t6_.stop();
  }

  if ( root )
  {
    System::out() << "\n";
    print ( System::info( myName_ ), getContext(),
	    " : Epoch "          , epoch_,
	    ", training loss = ", trainloss / trainset.size(), endl );
    System::out() << "\n";
  }
}

//-----------------------------------------------------------------------
//   solve_
//-----------------------------------------------------------------------

void AdamModule::solve_

  ( const Properties&  info,
    const Properties&  globdat )

{
  Ref<TrainingData> data = TrainingData::get ( globdat, getContext() );
  Ref<DofSpace>     dofs = DofSpace::get     ( globdat, getContext() );
  Ref<Model>        model = Model::get       ( globdat, getContext() );

  Ref<Random> generator = Random::get ( globdat );
  generator->restart( rseed_ ); 

  Vector state;
  StateVector::get ( state, dofs, globdat );

  idx_t n = data->sampleSize();

  idx_t batch = 0;
  IdxVector valset, trainset;
  IdxVector fullset;

  double loss;

  if ( skipFirst_ )
  {
    print ( System::info( myName_ ), getContext(),
	    " : Epoch "          , epoch_,
	    ", skipping first ", skipFirst_, " samples" );

    fullset.ref ( IdxVector ( iarray ( slice ( skipFirst_, n ) ) ) );
    n -= skipFirst_; 
  }
  else
  {
    fullset.ref ( IdxVector ( iarray ( n ) ) );
  }


 /* if ( subsetSize_ > 0)
  { 
    IdxVector randsamp(subsetSize_);

    for ( idx_t r = 0; r < subsetSize_; r++ ) randsamp[r] = generator->next(n) + skipFirst_;

    trainset.ref( randsamp );
    n = subsetSize_;

    if ( epoch_  == 1 )
    {
     System::out() << "Subset " << trainset << "\n"; 
    }
  }
  else
  {*/
    trainset.ref ( fullset );
 // }

//  System::out() << "Shuffle training data. \n";

  t5_.start();
  NeuralUtils::shuffle ( trainset, globdat );
  t5_.stop();

  loss = 0.0;

  while ( n > 0 )
  {
    batch = min ( n, batchSize_ );

    g_ = 0.0;

    loss += eval_ ( trainset[slice(n-batch,n)], true, globdat );    
//    System::out() << "n batch " << n << "\n";

//    System::out() << "g_: " << g_[slice(0, 18)] << "\n";

    if ( computeBatch_ ) 
    {
    g_ /= (double)batch;

    t6_.start();

    m_ = beta1_*m0_ + (1.-beta1_)*g_;
    v_ = beta2_*v0_ + (1.-beta2_)*g_*g_;

    double alphat = alpha_ * (
		    sqrt ( 1. - pow( beta2_,(double)iiter_ ) ) /
			 ( 1. - pow( beta1_,(double)iiter_ ) ) );

    state -= alphat * m_ / ( sqrt(v_) + eps_ );
    
    m0_ = m_;
    v0_ = v_;
    t6_.stop();

    model->takeAction ( LearningActions::UPDATE, Properties(), globdat );
    }
    else
    {
      System::out() << "Skipping batch " << trainset[slice(n-batch, n)] << ".\n";
    }

    bool isDetNeg = false; //checkDet_ ( trainset[slice(n-batch, n)], true, globdat );

    idx_t maxIt = 5;
    idx_t nchanges = 0;

    while ( isDetNeg == true && nchanges < maxIt )
    {
      System::out() << "Negative determinant! >>>>> Changing weights for "
              "the " << nchanges+1 << "th time. <<<<<\n";
      for ( idx_t nb = 0; nb < negDetBlocks_.size(); nb++ )
      {
        if ( negDetBlocks_[nb] )
	{
          idx_t init = nb*6;
  	  idx_t end = (nb+1)*6;
//          System::out() << "Before " << state [ slice ( init, end ) ] << "\n";
          state [ slice ( init, end ) ] *= 0.99;
 //         System::out() << "After " << state [ slice ( init, end ) ] << "\n";
	}
      }

      // Updating with the new changes

      model->takeAction ( LearningActions::UPDATE, Properties(), globdat );

      nchanges += 1;

      isDetNeg = checkDet_ ( trainset[slice(n-batch, n)], true, globdat );
    }

    //Test Marina (only prints the last batch update)
    
  /*  if ( n == 1 )
    {

    System::out() << "Remaining predictions after updating weights (DEBUGGING).\n";
   
    Properties params;
    Ref<Model> model = Model::get ( globdat, getContext() );

    for (idx_t k = 0; k < data->sampleSize(); ++k )
    {
      params.erase ( LearningParams::STATE );

      IdxVector sp(1);
      sp[0] = k;

      Batch b = data->getData(sp);

      params.set ( LearningParams::DATA, b[0]);
  
     model->takeAction ( LearningActions::RECALL, params, globdat );
     
    System::out() << "output: " << b[0]->outputs << " target: " << b[0]->targets << "\n";
   }
   }
*/
    if ( lambda_ > 0.0 )
    {
      IdxVector iwts ( dofs->dofCount() );
      IdxVector iaxs ( dofs->dofCount() );

      idx_t nw = dofs->getDofsForType ( iwts, iaxs,
		 dofs->findType ( LearningNames::WEIGHTDOF ) );

      iwts.reshape ( nw );

      r_        = 0.5 * lambda_ * dot ( state[iwts] );
    /*  System::out() << "\nweights dofs: " << iwts << "\n";
      System::out() << "\nAdamOptimizer. weights: " << state[iwts] << "\n";
      System::out() << "AdamOptimizer. dot (weights): " << dot ( state[iwts] ) << "\n";
*/
      rg_[iwts] = lambda_ * state[iwts];
    }
    

    if ( lambdaL1_ > 0.0 )       // Lasso regression
    {
      IdxVector iwts ( dofs->dofCount() );
      IdxVector iaxs ( dofs->dofCount() );

      idx_t nw = dofs->getDofsForType ( iwts, iaxs,
		 dofs->findType ( LearningNames::WEIGHTDOF ) );

      iwts.reshape ( nw );
      idx_t init;
     // idx_t nip = 0; 
      if (prev_ <= 0)
      {
        init  = nip_*3*3;
      }
      else
      {
         init = prev_*3+nip_*3*prev_;
      }
      
      Vector sig ( rg_.size() );
      sig = 0.0;
      double sum = 0.0;
      idx_t cont = 0;
            
      for (idx_t i = 0; i < nw; ++i )
      {
        if ( iwts[i] >= init ) 
        {
           if ( state[iwts[i]] < 0.0 )
           {
        //     System::out() << "iwts dof " << iwts[i] << " state[iwts[i]] " << state[iwts[i]] << "\n";
             sig[iwts[i]] = -1.0;
	     sum += abs(state[iwts[i]]);
           }
           cont += 1;
        }
      }

      r_ = lambdaL1_ * sum;

     /* System::out() << "AdamModule. sum: " << sum << "\n";
      System::out() << "AdamModule. signals: " << sig << "\n";
      System::out() << "AdamModule. signals weights " << sig[iwts] << "\n";
      System::out() << "AdamModule. rg size: " << rg_.size() << "\n";
      System::out() << "AdamModule. rg_ " << rg_[iwts] <<"\n";*/

      rg_[iwts] = lambdaL1_ * sig[iwts];
    }
    
    if ( lambdaPl_ > 0.0 )
    {
        IdxVector iwts ( dofs->dofCount() );
        IdxVector iaxs ( dofs->dofCount() );

        idx_t nw = dofs->getDofsForType ( iwts, iaxs,
                 dofs->findType ( LearningNames::WEIGHTDOF ) );

        iwts.reshape ( nw );
        idx_t init = nip_*3*3;
        double sum = 0.0;
        for ( idx_t m = 0; m < init; m++ )
        {
          sum += abs(state[iwts[m]]);
          System::out() << "weights " << state[iwts[m]] << "\n";
        } 
        r_ = sum;   
    }

    iiter_++;

    n -= batch;
  }

  System::out() << "\n";
  print ( System::info( myName_ ), getContext(),
	  " : Epoch "          , epoch_,
	  ", training loss = ", loss / trainset.size(), endl );
  System::out() << "\n";

  Properties myVars = jive::mp::Globdat::getVariables ( myName_, globdat );
  
  myVars.set ( "loss", loss / trainset.size() );
  myVars.set ( "epoch", epoch_ );
}

//-----------------------------------------------------------------------
//   eval_ 
//-----------------------------------------------------------------------

double AdamModule::eval_

  ( const IdxVector&  samples,
    const bool        dograds,
    const Properties& globdat )

{
  t1_.start();

  Properties params;
 
  Ref<Model>        model = Model::get        ( globdat, getContext() );
  Ref<TrainingData> tdata = TrainingData::get ( globdat, getContext() );

  double loss = 0.0;

  t2_.start();
  Batch b = tdata->getData ( samples );
  t2_.stop();

  computeBatch_ = true; 

  IdxVector fullBatch ( b[0]->batchSize() );
  TensorIndex bIdx;
  fullBatch[bIdx] = bIdx;

 // System::out() << "AdamModule::eval_ tdata sequence size: " << tdata->sequenceSize() << "\n";
 // System::out() << "AdamModule::eval_ samples: " << samples << "\n";

  for ( idx_t t = 0; t < tdata->sequenceSize(); ++t )
  {
    params.erase ( LearningParams::STATE );

    if ( t > 0 )
    {
      params.set      ( LearningParams ::STATE,             b[t-1]  );
    }

    params.set        ( LearningParams ::DATA,              b[t]    );

    t3_.start();
    model->takeAction ( LearningActions::PROPAGATE, params, globdat );
   /* if ( t == 0 )
    {
      Ref<PrintWriter> out = newInstance<PrintWriter> ( &System::out() );
      out->nformat.setFractionDigits( 16 );

      model->takeAction ( LearningActions::GETJACOBIAN, params, globdat );
      *out << "jacobian at time step " << t << ":\n" << b[t]->jacobian << "\n";
    }*/
    t3_.stop();

//    System::out() << "Batch " << b[t]->outputs << "\n";

    Matrix selOut ( selComp_.size(), fullBatch.size() );
    Matrix selTar ( selComp_.size(), fullBatch.size() );

    if ( selComp_.size() < outSize_ )
    {
      for ( idx_t is = 0; is < selComp_.size(); is++ )
      {
        for ( idx_t js = 0; js < fullBatch.size(); js++ )
        {
          selOut(is, js) = b[t]->outputs( selComp_[is], fullBatch[js] );
          selTar(is, js) = b[t]->targets( selComp_[is], fullBatch[js] );
        }
      }
    
//      System::out() << "Selected part of batch " << selOut << "\n";
     
      loss += func_ ( selOut, selTar );
    }
    else
    {
      loss += func_ ( b[t]->outputs, b[t]->targets );
    }

    if ( dissipation_ )
    {
       Vector history;
       params.get ( history, LearningParams::HISTORY );
 //      System::out() << "History from propagate: " << history << "\n";
       
       idx_t ni = tdata->inputSize();
       idx_t no = 1; //tdata->outputSize();
       idx_t ns = samples.size();
       
    //   System::out() << "ni " << ni << " no " << no << " ns " << ns << "\n";
   //    System::out() << "Initiate data to store info from RVE\n";

	Ref<NData> rdata = newInstance<NData> ( ns, ni, no );

	double jloss = 0.0;

    //    System::out() << "Get jacobian from RVE\n";

	rdata->outputs = tdata->getHistory ( samples, t );

    //    System::out() << "rdata output " << rdata->outputs << "\n";

        Matrix aux ( no, ns );

	aux (0, ALL) =  history;

	jloss = func_ ( rdata->outputs, aux );

        //System::out() << "HF " << rdata->outputs << " NN " << aux << " \n";

        //System::out() << "jloss " << jloss << "\n";
	loss += jloss;
    }
  //  System::out() << "output: " << b[t]->outputs << "\ntarget: " << b[t]->targets << "\n";
  //  System::out() << "AdamModule::eval_ loss: " << func_ ( b[t]->outputs, b[t]-> targets ) << "\n";
   // System::out() << "AdamModule::eval_loss. Dograds: " << dograds << "\n";

    if ( lambda_ > 0.0 )
    {
      loss += r_;
    }

    if ( lambdaL1_ > 0.0 )
    {
      loss += r_;
    }

    // This is used to penalized fictitious points that are in the elastic regime (epeq = 0.0)

    if ( lambdaPl_ > 0.0 )  
    {
        double sum = 0.0;
        Matrix epeq ( b[t]->history );
	for (idx_t i = 0; i < epeq.size(0); i++ )
        {
           for (idx_t j = 0; j < epeq.size(1); j++)
	   {
	     if ( epeq(i, j) <= 1e-8 ) sum += lambdaPl_;
	   }
	} 

//	System::out() << "Loss " << loss << " sumweights " << r_ << " sum elastic points " << sum << "\n";
        loss += sum*r_;
    }

    // Getting dissipation from the fictitious material points

/*    if ( dissipation_ )
    {
     model->takeAction (LearningActions::GETHISTORY, params, globdat );
     
    // System::out() << "AdamModule. getHistory.\n";

    }
*/
  }

  if ( dograds )
  {

    //System::out() << "Start backpropagating.\n";
    for ( idx_t t = tdata->sequenceSize() - 1; t >= 0; --t )
    {
      grad_ ( b[t]->outputs, b[t]->targets );

      params.erase ( LearningParams::STATE );

      if ( t > 0 )
      {
	params.set      ( LearningParams ::STATE,                 b[t-1]  );
      }

      params.set        ( LearningParams ::GRADS,                 g_       );
      params.set        ( LearningParams ::DATA,                  b[t]     );

      //System::out() << "grad before: " << g_ << "\n";

      t4_.start();
      model->takeAction ( LearningActions::BACKPROPAGATE, params, globdat  );
      t4_.stop();
   //   System::out() << "Grad at time " << t << ": " << g_[slice(0,18)] << "\n";
    
      bool gradientClip_ = true;
      if ( gradientClip_ )
      {
        for ( idx_t ig = 0; ig < g_.size(); ig++ )
	{
// if ( abs ( g_[ig] ) > 1.e20 ) skipBatch_ = true;
	  if ( Float::isNaN ( g_[ig] ) || abs ( g_[ig] )  > 1.e20 ) 
	  {
     //        throw Error ( JEM_FUNC, String::format ("Gradient exploded." ) );
	      computeBatch_ = false;
/*	      g_[ig] = 1.e20; //Float::MAX_VALUE;
	  if ( g_[ig] > 1.e20 ) 
	  {
            g_[ig] = 1.e20;
	  }
	  else if ( g_[ig] < 1.e-16 )
	  {
	    g_[ig] = -1.e16;*/ 
	  }
	}
      }
    
 // System::out() << "size of g_: " << g_.size() << "\n";
     
   // Weight pruning
   if ( pruning_ )
   { 
    idx_t init;
    if ( dehom_ )
    {    
    idx_t cont = 0;
    for ( idx_t in = 0; in < nip_*3; ++in )
    {
      if ( cont == 0)
      {
        init = in*3+1;
        g_[init] = 0.0;
        g_[init+1] = 0.0;
        cont += 1;
      }
      else if ( cont == 1 )
      {
        init = in*3;
	g_[init] = 0.0;
	g_[init+2] = 0.0;
	cont += 1;
    }
      else
      {
      init = in*3;
      g_[init] = 0.0;
      g_[init+1] = 0.0;
      cont = 0;
      }
    }
    }
 
    idx_t ws;
    if (prev_ <= 0)
    {
       init  = nip_*3*3 + nip_*3 + 1;
    }
    else
    {
      init = prev_*3+prev_+nip_*3*prev_+nip_*3;
    }
    
    for ( idx_t k = 0; k < 3; ++k) // Number of components
     { 
       ws = 0;
       for ( idx_t p = 0; p < nip_-1; ++p ) // Number of int. points - 1
       {
         for ( idx_t j = 0; j < 2; ++j ) // Number of components erased
         {
           g_[init+j+ws] = 0.0;
         }	       
         ws += 3;
       }
       
       if ( k != 2 )
       {
	for ( idx_t i = 0; i < 3; ++i )
        {
         g_[init+i+ws] = 0.0;
        }
       }

       init += nip_*3+1;
      }
   
   //   System::out() << "\neval_ grad after pruning: " << g_ << "\n";
    //  System::out() << "size of g_: " << g_.size() << "\n";
    }
      if ( lambda_ > 0.0 )
      {
        g_ += rg_;
      }

      if ( lambdaL1_ > 0.0 )
      {
	g_ += rg_;
      }

   //   System::out() << "AdamModule. g_ before update: " << g_ << "\n";
      Ref<DofSpace> dofs = DofSpace::get ( globdat, getContext() );
      Vector state;
      StateVector::get ( state, dofs, globdat );
     // System::out() << "AdamModule. state before updateeee: " << state << "\n";
      
      if ( jprop_ > 0.0 )
      {
	idx_t ni = tdata->inputSize();
	idx_t no = tdata->outputSize();
	idx_t ns = samples.size();

	Ref<NData>  data = tdata->stretchData ( b[t] );
	Ref<NData> rdata = newInstance<NData> ( ns*ni, ni, no );

	Vector dummyg ( g_.size() );
	Vector jg     ( g_.size() );

	dummyg = 0.0; jg = 0.0;

	double jloss = 0.0;

	data->outputs = 0.0;

	for ( idx_t i = 0; i < ni; ++i )
	{
          data->outputs(i,slice(i*ns,(i+1)*ns)) = 1.0;
	}

	params.set ( LearningParams::DATA,  data   );
	params.set ( LearningParams::GRADS, dummyg );
	model->takeAction ( LearningActions::BACKPROPAGATE, params, globdat );

	rdata->inputs = -tdata->getJacobian ( samples );

	params.set ( LearningParams::RDATA, rdata );
	model->takeAction ( LearningActions::FORWARDJAC, params, globdat );

	jloss += 0.5 * dot ( rdata->inputs );

	params.set ( LearningParams::GRADS, jg );
	model->takeAction ( LearningActions::BACKWARDJAC, params, globdat );

	loss = (1.0-jprop_) * loss + jprop_ * jloss;
	g_   = (1.0-jprop_) * g_   + jprop_ * jg;

	//Ref<NData> rdata = newInstance<NData> ( samples.size(), 
	//                                        tdata->inputSize(),
	//					tdata->outputSize() );

	//Ref<Normalizer> inl = tdata->getInpNormalizer();

	//Vector dummyg ( g_.size() );
	//dummyg = 0.0;

	//Vector jg ( g_.size() );
	//jg = 0.0;

	//double jloss = 0.0;

        //for ( idx_t i = 0; i < tdata->inputSize(); ++i )
	//{
	//  b[t]->outputs = 0.0;
	//  b[t]->outputs(i,ALL) = 1.0;

	//  params.set ( NeuralParams::GRADS, dummyg );
	//  model->takeAction ( NeuralActions::BACKPROPAGATE, params, globdat );

	//  for ( idx_t s = 0; s < samples.size(); ++s )
	//  {
	//    Vector fac = inl->getJacobianFactor ( b[t]->inputs(ALL,s) );

	//    rdata->inputs(ALL,s) = - tdata->getJacobianRow ( i, samples[s] ) / fac;
	//  }

	//  params.set ( NeuralParams::RDATA, rdata );

	//  model->takeAction ( NeuralActions::FORWARDJAC, params, globdat );

	//  jloss += 0.5 * dot ( rdata->inputs );

	//  params.set ( NeuralParams::GRADS, jg );
	//  model->takeAction ( NeuralActions::BACKWARDJAC, params, globdat );
	//}

	//loss = (1.0-jprop_) * loss + jprop_ * jloss;
	//g_   = (1.0-jprop_) * g_   + jprop_ * jg;
      }
    }
   // dbg (FD check)

    double fdstep = 1.e-8;
    double floss = 0.0;
    double bloss = 0.0;
    
    double rauxf = 0.0;
    double rauxb = 0.0;
    bool check = false;

    if ( check )
    {
    
    IdxVector fullBatch ( b[0]->batchSize() );
    TensorIndex bIdx;
    fullBatch[bIdx] = bIdx;

    Ref<DofSpace>     dofs = DofSpace::get     ( globdat, getContext() );
    Vector state;
    StateVector::get ( state, dofs, globdat );
    
    idx_t w = 4;
    state[w] += fdstep;

    System::out() << "Start FD check for weight " << w << " worth " << state[w] << "\n";
    StateVector::store ( state, dofs, globdat );
    model->takeAction ( LearningActions::UPDATE, Properties(), globdat );

    for ( idx_t t = 0; t < tdata->sequenceSize(); ++t )
    {
      params.erase ( LearningParams::STATE );

      if ( t > 0 )
      {
        params.set      ( LearningParams ::STATE,             b[t-1]  );
      }

      params.set        ( LearningParams ::DATA,              b[t]    );

      t3_.start();
      model->takeAction ( LearningActions::PROPAGATE, params, globdat );
      t3_.stop();
      
      
      Matrix selOut ( selComp_.size(), fullBatch.size() );
      Matrix selTar ( selComp_.size(), fullBatch.size() );
      
      if ( selComp_.size() < outSize_ )
      {
      for ( idx_t is = 0; is < selComp_.size(); is++ )
      {
        for ( idx_t js = 0; js < fullBatch.size(); js++ )
        {
          selOut(is, js) = b[t]->outputs( selComp_[is], fullBatch[js] );
          selTar(is, js) = b[t]->targets( selComp_[is], fullBatch[js] );
        }
      }
        floss += func_ ( selOut, selTar );
      }
      else
      {
        floss += func_ ( b[t]->outputs, b[t]->targets );
      }

    if ( lambdaPl_ > 0.0 )
    {
        double sum = 0.0;
        Matrix epeq ( b[t]->history );
        for (idx_t i = 0; i < epeq.size(0); i++ )
        {
           for (idx_t j = 0; j < epeq.size(1); j++)
           {
             if ( epeq(i, j) <= 1e-8 ) sum += lambdaPl_;
           }
        }

        IdxVector iwts ( dofs->dofCount() );
        IdxVector iaxs ( dofs->dofCount() );

        idx_t nw = dofs->getDofsForType ( iwts, iaxs,
                 dofs->findType ( LearningNames::WEIGHTDOF ) );

        iwts.reshape ( nw );
        idx_t init = nip_*3*3;
       
        for ( idx_t m = 0; m < init; m++ )
        {
          rauxf += abs(state[iwts[m]]);
        //  System::out() << "weights " << state[iwts[m]] << "\n";
        }    

      //  System::out() << "Floss " << floss << " sumweights " << rauxf << " sum elastic points " << sum << "\n";
        floss += sum*rauxf;

    }
    }

    state[w] -= 2.0*fdstep;

    StateVector::store ( state, dofs, globdat );
    model->takeAction ( LearningActions::UPDATE, Properties(), globdat );

    for ( idx_t t = 0; t < tdata->sequenceSize(); ++t )
    {
      params.erase ( LearningParams::STATE );

      if ( t > 0 )
      {
        params.set      ( LearningParams ::STATE,             b[t-1]  );
      }

      params.set        ( LearningParams ::DATA,              b[t]    );

      t3_.start();
      model->takeAction ( LearningActions::PROPAGATE, params, globdat );
      t3_.stop();

      Matrix selOut ( selComp_.size(), fullBatch.size() );
      Matrix selTar ( selComp_.size(), fullBatch.size() );

      if ( selComp_.size() < outSize_ )
      {
      for ( idx_t is = 0; is < selComp_.size(); is++ )
      {
        for ( idx_t js = 0; js < fullBatch.size(); js++ )
        {
          selOut(is, js) = b[t]->outputs( selComp_[is], fullBatch[js] );
          selTar(is, js) = b[t]->targets( selComp_[is], fullBatch[js] );
        }
      }
        bloss += func_ ( selOut, selTar );
      }
      else
      {
        bloss += func_ ( b[t]->outputs, b[t]->targets );
      }

    if ( lambdaPl_ > 0.0 )
    {
        double sum = 0.0;
        Matrix epeq ( b[t]->history );
        for (idx_t i = 0; i < epeq.size(0); i++ )
        {
           for (idx_t j = 0; j < epeq.size(1); j++)
           {
             if ( epeq(i, j) <= 1e-8 ) sum += lambdaPl_;
           }
        }
        
        IdxVector iwts ( dofs->dofCount() );
        IdxVector iaxs ( dofs->dofCount() );

        idx_t nw = dofs->getDofsForType ( iwts, iaxs,
                 dofs->findType ( LearningNames::WEIGHTDOF ) );

        iwts.reshape ( nw );
        idx_t init = nip_*3*3;
       
        for ( idx_t m = 0; m < init; m++ )
        {
          rauxb += abs(state[iwts[m]]);
        }    

      //  System::out() << "Bloss " << bloss << " sumweights " << rauxb << " sum elastic points " << sum << "\n";
        bloss += sum*rauxb;

    }
    }

    state[w] += fdstep;
    StateVector::store ( state, dofs, globdat );
    model->takeAction ( LearningActions::UPDATE, Properties(), globdat );

    double fdgrad = ( floss - bloss ) / fdstep / 2.0;

//    System::out() << "Loss " << loss << " fdloss " << fdloss << '\n';
     System::out() << "NN gradient " << g_[w] << " finite diff grad " << fdgrad << " dif " << g_[w] - fdgrad << " Percentage: " << (g_[w]-fdgrad)*100.0/fdgrad << "\n";
     }
  }

  t1_.stop();
  
  return loss;
}


//-----------------------------------------------------------------------
//   checkDet_ 
//-----------------------------------------------------------------------

bool AdamModule::checkDet_

  ( const IdxVector&  samples,
    const bool        dograds,
    const Properties& globdat )

{
  Properties params;
  Vector dett;
  idx_t nBlocks = 1; // default
 
  Ref<Model>        model = Model::get        ( globdat, getContext() );
  Ref<TrainingData> tdata = TrainingData::get ( globdat, getContext() );

  double loss = 0.0;

  Batch b = tdata->getData ( samples );

  //System::out() << "AdamModule::checkDet_ tdata sequence size: " << tdata->sequenceSize() << "\n";
  //System::out() << "AdamModule::eval_ samples: " << samples << "\n";

  for ( idx_t t = 0; t < tdata->sequenceSize(); ++t )
  {
    params.erase ( LearningParams::STATE );

    if ( t > 0 )
    {
      params.set      ( LearningParams ::STATE,             b[t-1]  );
    }

    params.set        ( LearningParams ::DATA,              b[t]    );

    t3_.start();
    model->takeAction ( LearningActions::GETDETERMINANT, params, globdat );
    t3_.stop();

    params.get ( dett, LearningParams::DETERMINANT );

    if ( t == 0 ) 
    {
      nBlocks = dett.size(0);
      negDetBlocks_.resize ( nBlocks );
      negDetBlocks_ = false;
      detAll_.resize( tdata->sequenceSize(), nBlocks);
    }

    detAll_ ( t, ALL ) = dett; 
  }

//  System::out() << "All determinants " << detAll_ << "\n";

  for ( idx_t nb = 0; nb < nBlocks; nb++ )
  {
    if ( min ( detAll_(ALL, nb) ) <= 0.0 )
    {
      negDetBlocks_[nb] = true;
    }
  }

  if ( min ( detAll_ ) <= 0.0 )
  { 
    return true;
  }
  else
  {
    return false;
  }
}

//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Module> AdamModule::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  return newInstance<Self> ( name );
}

//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------

void AdamModule::declare ()
{
  using jive::app::ModuleFactory;

  ModuleFactory::declare ( TYPE_NAME, & makeNew );
  ModuleFactory::declare ( CLASS_NAME, & makeNew );
}

JIVE_END_PACKAGE( implict )

