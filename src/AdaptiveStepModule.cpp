/*
 * 
 *  Copyright (C) 2013 TU Delft. All rights reserved.
 *  
 *  This class implements a strategy for adaptive stepping 
 *  with a NonlinModule for tracing equilibrium paths without
 *  snapback behavior.
 *
 *  Author: F.P. van der Meer
 *  Date: May 2013
 *
 */

#include <jem/base/limits.h>
#include <jem/base/System.h>
#include <jem/base/Exception.h>
#include <jem/base/IllegalOperationException.h>
#include <jem/io/FileWriter.h>
#include <jem/util/Properties.h>
#include <jem/util/StringUtils.h>
#include <jem/numeric/algebra/utilities.h>
#include <jive/util/Globdat.h>
#include <jive/util/utilities.h>
#include <jive/util/DofSpace.h>
#include <jive/implict/SolverInfo.h>
#include <jive/implict/NonlinModule.h>
#include <jive/implict/ArclenModule.h>
#include <jive/app/ModuleFactory.h>

#include "AdaptiveStepModule.h"
#include "SolverNames.h"

using jem::IllegalOperationException;
using jem::System;
using jem::max;
using jem::maxOf;
using jem::newInstance;
using jem::io::endl;
using jem::io::FileWriter;
using jem::util::StringUtils;
using jive::Vector;
using jive::implict::SolverInfo;
using jive::util::Globdat;
using jive::util::joinNames;
using jive::util::DofSpace;

//=======================================================================
//   class AdaptiveStepModule
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------

const char* AdaptiveStepModule::TYPE_NAME       = "AdaptiveStep";
const char* AdaptiveStepModule::NONLIN          = "nonlin";
const char* AdaptiveStepModule::ARCLEN          = "arclen";
const char* AdaptiveStepModule::WRITE_STAT_PROP = "writeStats";
const char* AdaptiveStepModule::OPT_ITER_PROP   = "optIter";
const char* AdaptiveStepModule::MIN_INCR_PROP   = "minIncr";
const char* AdaptiveStepModule::MAX_INCR_PROP   = "maxIncr";
const char* AdaptiveStepModule::START_INCR_PROP = "startIncr";
const char* AdaptiveStepModule::REDUCTION_PROP  = "reduction";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------

AdaptiveStepModule::AdaptiveStepModule 

  ( const String&  name,
    Ref<SolverModule>  solver ) :

      Super    ( name   ),
      solver_  ( solver )

{
  istep_          = 0;
  istep0_         = 0;
  time_           = 0.;
  time0_          = 0.;
  nCancels_       = 0;
  nContinues_     = 0;
  maxNIter_       = 0;

  nRunTotal_      = 0;
  nCancTotal_     = 0;
  nContTotal_     = 0;
  nIterTotal_     = 0;
  nCancCont_      = 0;
  nCommTotal_     = 0;

  startIncr_      = 1.;
  minIncr_        = 1.e-5;
  maxIncr_        = 1.;
  reduction_      = 0.45;
  optIter_        = 5;
  increment_      = 0.;

  maxIncrTried_   = 0.;
  writeStats_     = false;
  triedSmall_     = false;
}

AdaptiveStepModule::~AdaptiveStepModule ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


Module::Status AdaptiveStepModule::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  model_ = Model::get ( globdat, getContext() );
  istep_ = istep0_ = 0;
  time_  = time0_  = 0.;

  globdat.set ( Globdat::TIME_STEP, istep_ );
  globdat.set ( Globdat::TIME, time_ );
  globdat.set ( "var.accepted", false );

  solver_->init ( conf, props, globdat );

  return OK;
}


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


Module::Status AdaptiveStepModule::run ( const Properties& globdat )

{
  using jem::Exception;

  Properties  info      = SolverInfo::get ( globdat );
  Properties  params;

  bool        accept    = true;
  bool        discard   = false;
  bool        converged = false;
  String      terminate;

  nRunTotal_  ++;

  info.clear ();

  // Setting random time increment

  setStepSize_( globdat );

  solver_->advance ( globdat );

  // Store maximum values of increment tried in this time step

  if ( nContinues_ == 0 )
  {
    maxIncrTried_ = max ( maxIncrTried_, increment_ );
  }
  
  // Advance the time/load step number

  istep_ = istep0_ + 1;
  time_  = time0_ + increment_;

  globdat.set ( Globdat::TIME_STEP, istep_ );
  globdat.set ( Globdat::TIME, time_ );

  // try to find a solution

  try
  {
    solver_->solve ( info, globdat );

    converged = true;
  }
  catch ( const jem::Exception& ex )
  {
    System::out() << "AdaptiveStepModule caught a solver exception: \n" 
                  << "   " << ex.where() << "\n"
                  << "   " << ex.what() << "\n\n";

    discard    = true;
  }

  // get # of iterations, NB: not set when error thrown by solver
  
  idx_t          iterCount = 0; 
  info.find   (  iterCount , SolverInfo::ITER_COUNT );

  maxNIter_    = max ( maxNIter_, iterCount );
  nIterTotal_ += iterCount;

  if ( nContinues_ > 0 ) nIterCont_ += iterCount;

  if ( converged )
  {
    // Ask the model whether to accept converged solution
    // (Model can ask for cancel via DISCARD)

    model_->takeAction ( SolverNames::CHECK_COMMIT, params, globdat );

    params.find ( discard, SolverNames::DISCARD );

    params.find ( accept,   SolverNames::ACCEPT    );
  }

  // set flag in globdat for output indicating solution quality

  globdat.set ( "var.accepted", ( accept && ! discard ) );

  if ( ! discard )
  {
    // take measures based on result of CHECK_COMMIT

    if ( accept )
    {
      // The final solution for this time step has been obtained

      commit_ ( globdat );

      // New random time increment will be generated

      if ( randomInc_ ) 
      {
        increment_ = generator_->next ( )*10;
	if ( increment_ < minIncr_ ) increment_ = minIncr_;
      }

      if ( params.find ( terminate, SolverNames::TERMINATE ) ) 
      {
        if ( terminate == "sure" )
        {
          System::out() << "Some model gave the terminate signal during "
            << "checkCommit,\nAdaptiveStepModule terminates run.\n"
            << "NB: output modules are not called for this time step\n";
          return EXIT;
        }
      }
    }
    else
    {
      // We have an equilibrium solution but it is not accepted:
      // continue with this time step

      continue_ ( params, globdat );
    }
  }
  else
  {
    // 1. discard solution

    cancel_ ( globdat );

    // 2. find proper strategy to try again
   
    // try to reduce the step size

    if ( reduceStep_ ( globdat ) )
    {
      System::out() << "Returning to the same load step "
                    << "with reduced step\n";
    }

    // try to increase the step size

    else if ( increaseStep_ ( globdat ) )
    {
      System::out() << "Returning to the same load step "
                    << "with increased step\n";
    }

    // give it up

    else
    {
      System::out() << "\n*** AdaptiveStepModule is out of inspiration,\n"
                    << "    can't find a winning strategy for this step."
                    << " Sorry.\n\n";

      return EXIT;
    }
  }
  return OK;
}


//-----------------------------------------------------------------------
//   shutdown
//-----------------------------------------------------------------------


void AdaptiveStepModule::shutdown ( const Properties& globdat )
{
  solver_->shutdown ( globdat );

  System::out() << "AdaptiveStepModule statistics ..." 
    << "\n... total # of runs: " << nRunTotal_ 
    << "\n... # of commits:    " << nCommTotal_
    << "\n... # of continues:  " << nContTotal_
    << "\n-------  ( of which     " << nCancCont_ << " were canceled.)"
    << "\n... # of cancels:    " << nCancTotal_
    << "\n..."
    << "\n... total # of iterations: " << nIterTotal_
    << "\n-------  ( of which " << nIterCont_ << " after continue.)"
    << endl;
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void AdaptiveStepModule::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  Properties myProps = props.findProps ( myName_ );

  myProps.find ( writeStats_ , WRITE_STAT_PROP );
  
  solver_->configure ( props, globdat );

  if ( writeStats_ )
  {
    cpuTimer_.start ();

    statOut_  = newInstance<PrintWriter>(
                newInstance<FileWriter> ( "flex.stats" ) );
    *statOut_ << "   nRun  nComm  nCanc  nCont  nCaCo  nIter  CPU" << endl;
  }

  randomInc_ = false;

  idx_t rseed = 0;
  myProps.find ( reduction_, REDUCTION_PROP  );
  myProps.find ( startIncr_, START_INCR_PROP );
  myProps.find ( randomInc_, "randomTimeIncr" );
  myProps.find ( rseed, "rseed" );

  generator_ = Random::get ( globdat );
  generator_->restart ( rseed );

  if ( randomInc_ ) 
  {
    startIncr_ = generator_->next ()*10;
    if ( increment_ < minIncr_ ) increment_ = minIncr_;
  }

  minIncr_ = startIncr_ * 1.e-3;
  maxIncr_ = startIncr_;

  increment_ = startIncr_;
    
  myProps.find ( minIncr_,   MIN_INCR_PROP   );
  myProps.find ( maxIncr_,   MAX_INCR_PROP   );
  myProps.find ( optIter_,   OPT_ITER_PROP   );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void AdaptiveStepModule::getConfig

  ( const Properties&  conf,
    const Properties&  globdat ) const

{
  Properties  myConf  = conf.makeProps ( myName_ );

  myConf.set ( "type", TYPE_NAME );

  myConf.set ( WRITE_STAT_PROP, writeStats_  );
  myConf.set ( REDUCTION_PROP,  reduction_   );
  myConf.set ( START_INCR_PROP, startIncr_   );
  myConf.set ( MAX_INCR_PROP,   maxIncr_     );
  myConf.set ( MIN_INCR_PROP,   minIncr_     );
  myConf.set ( OPT_ITER_PROP,   optIter_     );

  solver_->getConfig ( conf, globdat );
}

//-----------------------------------------------------------------------
//   commit_
//-----------------------------------------------------------------------

void AdaptiveStepModule::commit_

  ( const Properties&       globdat )

{
  // store this solution and proceed with next time step
  // model->takeAction(COMMIT) is called by solver!

  solver_->commit ( globdat );

  ++nCommTotal_;

  // do something to time step size comaring nIter4Adapt with optIter

  double fac  = ( maxNIter_ - optIter_ ) / 4.0; 
  if ( ( maxNIter_ > optIter_ && increment_ > minIncr_ ) || 
       ( maxNIter_ < optIter_ && increment_ < maxIncr_ ) )
  {
    increment_ *= ::pow ( 0.5, fac );
  }

  // reset variables

  istep0_ = istep_;
  time0_ = time_;

  maxNIter_       = 0;
  nCancels_       = 0;
  nContinues_     = 0;
  maxIncrTried_   = 0.;
  triedSmall_     = false;

  // write statistics (optional)

  if ( writeStats_ )
  {
    statOut_->nformat.setFractionDigits ( 4 );
    statOut_->nformat.setIntegerWidth   ( 6 );
    statOut_->nformat.setScientific     ( true );

    *statOut_ <<         nRunTotal_  << ' '  << nCommTotal_
              << ' '  << nCancTotal_ << ' '  << nContTotal_
              << ' '  << nCancCont_  << ' '  << nIterTotal_ 
              << "  " << cpuTimer_.toDouble() 
              << endl;

    statOut_->flush();
  }
}

//-----------------------------------------------------------------------
//   continue_
//-----------------------------------------------------------------------

void AdaptiveStepModule::continue_

  ( const Properties&       params,
    const Properties&       globdat )

{
  // the solver is not canceled, because the current solution
  // (converged but not accepted) is a good starting point
  
  System::out() << "Continuing with this load step\n\n";

  istep_ = istep0_;
  time_  = time0_;

  globdat.set ( Globdat::TIME_STEP, istep_ );
  globdat.set ( Globdat::TIME, time_ );

  ++nContinues_;
  ++nContTotal_;

  params.set ( SolverNames::N_CONTINUES, nContinues_ );

  model_->takeAction ( SolverNames::CONTINUE, params, globdat );
}

//-----------------------------------------------------------------------
//   cancel_
//-----------------------------------------------------------------------

void AdaptiveStepModule::cancel_

  ( const Properties&       globdat )

{
  // discard solution, go back to the beginning of the time step
  // model->takeAction(CANCEL) is called by solver!

  ++nCancels_;
  ++nCancTotal_;
  
  nCancCont_  += nContinues_;
  nContinues_  = 0;
  maxNIter_    = 0;

  solver_->cancel ( globdat );

  istep_ = istep0_;
  time_  = time0_;

  globdat.set ( Globdat::TIME_STEP, istep_ );
  globdat.set ( Globdat::TIME, time_ );
}

//-----------------------------------------------------------------------
//   reduceStep_
//-----------------------------------------------------------------------

bool AdaptiveStepModule::reduceStep_

  ( const Properties&       globdat )

{
  if ( !triedSmall_ && increment_ > minIncr_ )
  {
    increment_ *= reduction_;

    return true;
  }
  else
  {
    triedSmall_ = true;

    return false;
  }
}

//-----------------------------------------------------------------------
//   increaseStep_
//-----------------------------------------------------------------------

bool AdaptiveStepModule::increaseStep_

  ( const Properties&       globdat )

{
  if ( maxIncrTried_ > maxIncr_ )
  {
    return false;
  }
  else
  {
    maxIncrTried_ = increment_ = maxIncrTried_ / reduction_;

    return true;
  }
}

//-----------------------------------------------------------------------
//   setStepSize()
//-----------------------------------------------------------------------

void AdaptiveStepModule::setStepSize_

  ( const Properties& globdat ) const

{
  Properties params;

  params.set ( SolverNames::STEP_SIZE  , increment_ );
  params.set ( SolverNames::STEP_SIZE_0, startIncr_ );

  double percentage = increment_ / startIncr_ * 100.;

  System::out() << "Setting step size to " << increment_ << " ("
    << percentage << "\% of initial step size)" << endl;

  model_->takeAction ( SolverNames::SET_STEP_SIZE, params, globdat );
}

//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------

Ref<Module>  AdaptiveStepModule::makeNew

  ( const String&           name,
    const Properties&       conf,
    const Properties&       props,
    const Properties&       globdat )

{
  Properties  myConf  = conf .makeProps ( name );
  Properties  myProps = props.findProps ( name );

  Ref<SolverModule>  solver;
  String             type;

  myProps.get ( type, "solver" );
  myConf .set ( "solver", type );

  if      ( type == "Nonlin" )
  {
    solver = newInstance<NonlinModule> ( joinNames ( name, NONLIN ) );
  }
  else if ( type == "Arclen" )
  {
    solver = newInstance<ArclenModule> ( joinNames ( name, ARCLEN ) );
  }
  else
  {
    myProps.propertyError (
      name,
      "invalid solver type: " + type
    );
  }

  return newInstance<Self> ( name, solver );
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   declareAdaptiveStepModule
//-----------------------------------------------------------------------

void declareAdaptiveStepModule ()
{
  using jive::app::ModuleFactory;

  ModuleFactory::declare ( AdaptiveStepModule::TYPE_NAME,
                         & AdaptiveStepModule::makeNew );
}

