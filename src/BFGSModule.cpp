/*
 *  TU Delft 
 *
 *  Iuri Barcelos, Oct 2019
 *
 *  Bounded BFGS algorithm for full-batch optimization
 *  of model parameters. Model should make sure the
 *  parameters are always in a [0,1] interval. Designed
 *  for use in a Bayesian training scheme (maximization
 *  of the log marginal likelihood).
 *
 */

#include <cstdlib>

#include <jem/base/System.h>
#include <jem/base/ClassTemplate.h>
#include <jive/util/Globdat.h>
#include <jem/base/Array.h>
#include <jem/base/Error.h>
#include <jem/base/Float.h>
#include <jem/base/array/tensor.h>
#include <jem/base/array/operators.h>
#include <jem/base/array/utilities.h>
#include <jive/app/ModuleFactory.h>
#include <jem/numeric/utilities.h>
#include <jem/numeric/algebra/utilities.h>
#include <jem/numeric/algebra/matmul.h>
#include <jem/numeric/algebra/MatmulChain.h>
#include <jem/numeric/algebra/LUSolver.h>
#include <jem/io/PrintWriter.h>
#include <jem/io/FileWriter.h>
#include <jive/util/XDofSpace.h>
#include <jive/util/Random.h>
#include <jive/model/StateVector.h>
#include <jive/implict/Names.h>
#include <jive/implict/SolverInfo.h>

#include "LearningNames.h"
#include "BFGSModule.h"

JEM_DEFINE_CLASS( jive::implict::BFGSModule );

JIVE_BEGIN_PACKAGE( implict )

using jem::numeric::matmul;
using jem::numeric::norm2;
using jem::numeric::MatmulChain;
using jem::System;
using jem::newInstance;
using jem::io::PrintWriter;
using jem::io::FileWriter;
using jive::Vector;
using jive::util::Globdat;
using jive::util::Random;
using jive::util::XDofSpace;
using jive::model::StateVector;

//=======================================================================
//   class BFGSModule
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------

const char* BFGSModule::TYPE_NAME        = "BFGS";
const char* BFGSModule::PRECISION        = "precision";
const char* BFGSModule::NRESTARTS        = "nRestarts";
const char* BFGSModule::MAXITER          = "maxIter";
const char* BFGSModule::OUTFILE          = "outFile";

//-----------------------------------------------------------------------
//  constructor and destructor 
//-----------------------------------------------------------------------

BFGSModule::BFGSModule

  ( const String& name ) :
      
      Super ( name )

{
  numEvals_ = 0;
  epoch_    = 0;
  nOpt_     = 0;
  maxOpt_   = 1;
  maxIter_  = 1000;
  precision_ = 1.e-6;

  outFile_  = "";

}

BFGSModule::~BFGSModule ()
{}

//-----------------------------------------------------------------------
//  init 
//-----------------------------------------------------------------------

Module::Status BFGSModule::init

  ( const Properties& conf,
    const Properties& props,
    const Properties& globdat )

{
  System::out() << "BFGSModule::init\n";  

  Ref<DofSpace> dofs  = DofSpace::get ( globdat, getContext() );

  idx_t size = dofs->dofCount();

  Properties myConf  = conf.makeProps  ( myName_ );
  Properties myProps = props.findProps ( myName_ );

  myConf.set   ( "type", TYPE_NAME );

  myProps.find ( precision_, PRECISION );
  myConf. set  ( PRECISION, precision_ );

  myProps.find ( maxOpt_, NRESTARTS );
  myConf. set  ( NRESTARTS, maxOpt_ );

  myProps.find ( maxIter_, MAXITER );
  myConf. set  ( MAXITER, maxIter_ );

  myProps.find ( outFile_, OUTFILE );
  myConf. set  ( OUTFILE, outFile_ );

  oldHessian_.resize ( size, size );
  hessian_.resize    ( size, size );
  oldGrads_.resize   ( size );
  grads_.resize      ( size );

  invH_.resize       ( size, size );
  oldInvH_.resize    ( size, size );
  pg_.resize         ( size );
  pg0_.resize        ( size );
  active_.resize     ( size );

  id_.resize         ( size, size );
  sSyST_.resize      ( size, size );
  ySsST_.resize      ( size, size );
  sSsST_.resize      ( size, size );

  best_.resize       ( size );
  bestReal_.resize   ( size );

  grads_      = 0.;
  oldGrads_   = 0.;
  oldHessian_ = 0.;
  hessian_    = 0.;
  invH_       = 0.0;
  oldInvH_    = 0.0;
  pg_         = 0.0;
  pg0_        = 0.0;
  active_     = false;

  id_         = 0.0;
  sSyST_      = 0.0;
  ySsST_      = 0.0;
  sSsST_      = 0.0;

  best_       = 0.0;
  bestReal_   = 0.0;
  bestObj_    = maxOf ( bestObj_ );

  for ( idx_t i = 0; i < size; ++i )
  {
    id_(i,i) = 1.0;
  }

  System::out() << "BFGSModule::init ended.\n";

  return OK;
}

//-----------------------------------------------------------------------
//   advance
//-----------------------------------------------------------------------

void BFGSModule::advance ( const Properties& globdat )

{

  System::out() << "BFGSModule::advance\n";	
  epoch_++;
}


//-----------------------------------------------------------------------
//  solve 
//-----------------------------------------------------------------------

void BFGSModule::solve

  ( const Properties& info,
    const Properties& globdat )

{ 
  System::out() << "BFGSModule::solve\n";

  TensorIndex i, j;

  Ref<DofSpace> dofs  = DofSpace::get ( globdat, getContext() );
  Ref<Model>    model = Model   ::get ( globdat, getContext() );

  info.clear();

  Properties params;

  Vector vars, oldvars;
  StateVector::get    ( vars,    dofs, globdat );
  StateVector::getOld ( oldvars, dofs, globdat );

  idx_t size = dofs->dofCount();

  if ( nOpt_ == maxOpt_ )
  {
    System::out() << getContext() << " Maximum number of restarts" <<
      " reached. Stopping optimization\n";
    System::out() << getContext() << " Setting variables to " << best_ << "\n";
    System::out() << getContext() << " Best objective function " << bestObj_ << "\n";

    info.set  ( "terminate", "sure" );

    vars = best_;

    model->takeAction ( LearningActions::UPDATE,       params, globdat );
    model->takeAction ( LearningActions::WRITEPARAMS,  params, globdat ); 

    params.get ( bestReal_, LearningParams::WEIGHTS );

    return;
  }

  if ( epoch_ == 1 )
  {
    model->takeAction ( LearningActions::UPDATE, params, globdat );
    params.get ( objfunc_, LearningParams::OBJFUNCTION );
    numEvals_++;

    params.set ( LearningParams::GRADS, grads_ );
    model->takeAction ( LearningActions::GETGRADS, params, globdat );

    pg0_ = vars - grads_;
    pg0_[i] = where ( pg0_[i] > 1.0, 1.0,
              where ( pg0_[i] < 0.0, 0.0,
	              pg0_[i]             ) );
    pg0_ = vars - pg0_;
    pg_ = pg0_;

    double eps = min ( 0.5, norm2 ( pg_ ) );

    active_ = 0;
    active_[i] = ( 1. - vars[i] <= eps || vars[i] <= eps );

    invH_(i,j) = where ( i == j && !active_[i] && !active_[j], 1., 0. );
    
    oldvars = vars;
  }

  // Check for convergence

  System::out() << getContext() << " : Epoch " << epoch_ << 
    ", current variables " << vars   << "\n";
  System::out() << getContext() << " : Epoch " << epoch_ << 
    ", current gradients " << grads_ << "\n";
  System::out() << getContext() << " : Epoch " << epoch_ << 
    ", norm of the projected gradient " << norm2(pg_) << "\n"; 
  System::out() << getContext() << " : Epoch " << epoch_ << 
    ", objective function " << objfunc_ << "\n"; 

  //if ( norm2 ( pg_ ) < precision_ )
  if ( false )
  {
    System::out() << getContext() << " : Epoch " << epoch_ << 
      ", tolerance level of " << precision_ << " has been reached!\n"; 

    restart_ ( globdat );

    return;
  }
  else if ( epoch_ > maxIter_ )
  {
    System::out() << getContext() << " : Epoch " << epoch_ << 
      ", maximum number of epochs " << maxIter_ << " has been reached!\n"; 

    restart_ ( globdat );
    
    return;
  }

  // Go to the next point

  Vector dir ( size );

  Matrix Pa ( size, size );

  Pa(i,j) = where ( i == j && ( active_[i] || active_[j] ), 1., 0. );

  Pa = Pa + invH_;

  dir = -1.*matmul ( Pa, grads_ );

  //System::out() << "vars before lineSearch " << vars << "\n";
  lineSearch_ ( dir, globdat );
  StateVector::get ( vars, dofs, globdat );
  //System::out() << "vars after lineSearch " << vars << "\n";

  // Compute new gradients

  oldGrads_ = grads_;

  params.set ( LearningParams::GRADS, grads_ );
  model->takeAction ( LearningActions::GETGRADS, params, globdat );

  
  // Update some stuff

  pg0_ = pg_;

  pg_ = vars - grads_;

  pg_[i] = where ( pg_[i] > 1.0, 1.0,
           where ( pg_[i] < 0.0, 0.0,
	           pg_[i]             ) );

  pg_ = vars - pg_;

  double eps = min ( 0.5, norm2 ( pg_ ) );

  active_ = 0;
  active_[i] = ( 1. - vars[i] <= eps || vars[i] <= eps );

  // Compute s# and y#

  Vector sSharp ( size );
  Vector ySharp ( size );

  sSharp[i] = where ( !active_[i], 0., vars[i]   - oldvars[i]   );
  ySharp[i] = where ( !active_[i], 0., grads_[i] - oldGrads_[i] );

  // Update hessian (or not)

  if ( dot ( ySharp, sSharp ) > 0.0 )
  {
    oldInvH_ = invH_;

    double den = dot ( ySharp, sSharp );

    sSyST_(i,j) = sSharp[i]*ySharp[j];
    sSyST_ = id_ + sSyST_/den;

    ySsST_(i,j) = ySharp[i]*sSharp[j];
    ySsST_ = id_ + ySsST_/den;

    sSsST_(i,j) = sSharp[i]*sSharp[j];
    sSsST_ = sSsST_/den;

    oldInvH_(i,j) = where ( !active_[i] && !active_[j], 0., oldInvH_(i,j) );

    invH_ = matmul ( sSyST_, oldInvH_ );

    invH_ = matmul ( invH_, ySsST_ );

    invH_ = invH_ - sSsST_;
  }
  else
  {
    invH_(i,j) = where ( i == j, 1., 0. );
  }

  System::out() << "BFGSModule::solve ended.\n";
}

//-----------------------------------------------------------------------
//  commit
//-----------------------------------------------------------------------

bool BFGSModule::commit
 
  ( const Properties& globdat )

{
  //Vector state, state0;

  System::out() << "BFGSModule::commit\n";

  Ref<DofSpace> dofs  = DofSpace::get ( globdat, getContext() );

  //StateVector::get ( state, dofs, globdat );
  //StateVector::getOld ( state0, dofs, globdat );

  StateVector::updateOld ( dofs, globdat );

  System::out() << "BFGSModule::commit ended.\n";
  
  return true;
}

//-----------------------------------------------------------------------
//  cancel
//-----------------------------------------------------------------------

void BFGSModule::cancel

  ( const Properties& globdat )

{
  Ref<DofSpace> dofs  = DofSpace::get ( globdat, getContext() );

  StateVector::restoreNew ( dofs, globdat );
}

//-----------------------------------------------------------------------
//   setPrecision
//-----------------------------------------------------------------------

void BFGSModule::setPrecision ( double eps )
{
  precision_ = eps;
}

//-----------------------------------------------------------------------
//   getPrecision
//-----------------------------------------------------------------------

double BFGSModule::getPrecision () const
{
  return precision_;
}

//-----------------------------------------------------------------------
//  shutdown 
//-----------------------------------------------------------------------

void BFGSModule::shutdown

  ( const Properties& globdat )

{
  if ( outFile_ != "" )
  {
    Properties params;

    Ref<PrintWriter> out = newInstance<PrintWriter> (
		      newInstance<FileWriter> ( outFile_ ) );
    
    // Print real variables

    for ( idx_t i = 0; i < bestReal_.size(); ++i )
    {
      *out << bestReal_[i] << " ";
    }
    *out << '\n';

    // Print normalized variables

    for ( idx_t i = 0; i < best_.size(); ++i )
    {
      *out << best_[i] << " ";
    }

    *out << "\n" << bestObj_;
  }
}

//-----------------------------------------------------------------------
//  configure 
//-----------------------------------------------------------------------

void BFGSModule::configure

  ( const Properties& props,
    const Properties& globdat )

{
  System::out() << "BFGSModule::configure\n";

  numEvals_ = 0;
  epoch_    = 0;
  nOpt_     = 0;

  best_    = 0.0;
  bestObj_ = maxOf ( bestObj_ );

  props.find ( precision_, PRECISION );
  props.find ( maxOpt_,    NRESTARTS );

  System::out() << "BFGSModule::configure ended.\n";
}

//-----------------------------------------------------------------------
//  getConfig 
//-----------------------------------------------------------------------

void BFGSModule::getConfig

  ( const Properties& conf,
    const Properties& globdat ) const

{
}

//-----------------------------------------------------------------------
//  makeNew 
//-----------------------------------------------------------------------

Ref<Module> BFGSModule::makeNew

  ( const String&     name,
    const Properties& conf,
    const Properties& props,
    const Properties& globdat )

{
  return newInstance<Self> ( name );
}

//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------

void BFGSModule::declare ()

{
  using jive::app::ModuleFactory;

  ModuleFactory::declare ( TYPE_NAME, & makeNew );
  ModuleFactory::declare ( CLASS_NAME, & makeNew );
}

//-----------------------------------------------------------------------
//  restart_ 
//-----------------------------------------------------------------------

void BFGSModule::restart_

  ( const Properties& globdat )

{
  Ref<DofSpace> dofs = DofSpace::get ( globdat, getContext() );
  Ref<Random>   rng  = Random::get ( globdat );

  Vector vars;
  StateVector::get ( vars, dofs, globdat );

  if ( objfunc_ < bestObj_ )
  {
    bestObj_ = objfunc_;
    best_    = vars;
  }

  for ( idx_t v = 0; v < vars.size(); ++v )
  {
    vars[v] = rng->next();
  }

  System::out() << "BFGS: Restarting with vars " << vars << '\n';

  grads_      = 0.;
  oldGrads_   = 0.;
  oldHessian_ = 0.;
  hessian_    = 0.;
  invH_       = 0.0;
  oldInvH_    = 0.0;
  pg_         = 0.0;
  pg0_        = 0.0;
  sSyST_      = 0.0;
  ySsST_      = 0.0;
  sSsST_      = 0.0;
  active_     = false;

  epoch_ = 0;
  nOpt_++;
}

//-----------------------------------------------------------------------
//  lineSearch_ 
//-----------------------------------------------------------------------

void BFGSModule::lineSearch_

  ( const Vector&     dir,
    const Properties& globdat )

{
  Ref<DofSpace> dofs  = DofSpace::get ( globdat, getContext() );
  Ref<Model>    model = Model   ::get ( globdat, getContext() );

  Vector vars;
  StateVector::get ( vars, dofs, globdat );

  // Armijo rule

  TensorIndex i;
  Properties params;

  double alpha = 1.e-4;
  double beta  = 0.5;

  double obj = Float::MAX_VALUE;

  double lambda = 1.;

  idx_t  m = 0;

  //beta = 0.7;

  Vector newvars ( dofs->dofCount() );

  newvars = vars + dir;
  newvars[i] = where ( newvars[i] > 1.0, 1.0,
	       where ( newvars[i] < 0., 0.,
		       newvars[i]             ) );

  StateVector::store ( newvars, dofs, globdat );

  model->takeAction ( LearningActions::UPDATE, params, globdat );
  params.get         ( obj, LearningParams::OBJFUNCTION         );

  while ( obj > objfunc_ - alpha / lambda * pow(norm2(vars-newvars),2.0) )
  {
    m++;
    lambda = pow(beta,m);

    newvars = vars + lambda*dir;
    newvars[i] = where ( newvars[i] > 1.0, 1.0,
		 where ( newvars[i] < 0., 0.,
			 newvars[i]             ) );

    model->takeAction ( LearningActions::UPDATE, params, globdat );
    params.get         ( obj, LearningParams::OBJFUNCTION         );
  }

  objfunc_ = obj;
}

JIVE_END_PACKAGE( implict )

// DBG

    //double step = 0.05;
    //double minobj = maxOf(minobj);
    //double obj;

    //idx_t nsteps = 1. / step;

    //Vector oldvars ( vars.clone() );

    //for ( idx_t s1 = 0; s1 < nsteps; ++s1 )
    //{
    //  for ( idx_t s2 = 0; s2 < nsteps; ++s2 )
    //  {
    //    for ( idx_t s3 = 0; s3 < nsteps; ++s3 )
    //    {
    //      vars[0] = s1 * step;
    //      vars[1] = s2 * step;
    //      vars[2] = s3 * step;

    //      model->takeAction ( LearningActions::UPDATE, params, globdat );
    //      params.get ( obj, LearningParams::OBJFUNCTION );

    //      //System::out() << vars[0] << " " << vars[1] << " " << vars[2] << " " << obj << "\n";
    //      if ( obj < minobj )
    //        minobj = obj;
    //    }
    //  }
    //}
    //System::out() << "Global minimum " << minobj << "\n";
    //vars = oldvars;

// DBG: Test gradients using FD
//  if ( true )
//  {
//    TensorIndex i;
//
//    double newobj;
//    vars[1] += 1.e-15;
//
//    vars[i] = where ( vars[i] > 1.0, 1.0,
//		 where ( vars[i] < 0., 0.,
//			 vars[i]             ) );
//
//
//    StateVector::store ( vars, dofs, globdat );
//    model->takeAction ( LearningActions::UPDATE, params, globdat );
//    params.get         ( newobj, LearningParams::OBJFUNCTION         );
//
//    System::out() << "old obj " << objfunc_ << " new obj " << newobj << "\n";
//    System::out() << "Analytical grad " << grads_[1] << "\n";
//    System::out() << "FD grad " << ( newobj - objfunc_ ) / 1.e-15 << "\n";
//
//    vars[1] -= 1.e-15;
//    StateVector::store ( vars, dofs, globdat );
//    model->takeAction ( LearningActions::UPDATE, params, globdat );
//  }

