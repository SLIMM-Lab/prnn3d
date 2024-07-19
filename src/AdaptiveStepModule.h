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

#ifndef ADAPTIVE_STEP_MODULE_H
#define ADAPTIVE_STEP_MODULE_H

#include <jem/io/PrintWriter.h>
#include <jem/util/CPUTimer.h>
#include <jive/app/Module.h>
#include <jive/model/Model.h>

#include <jive/util/Random.h>

using jem::Ref;
using jem::String;
using jem::idx_t;
using jem::io::PrintWriter;
using jem::util::Properties;
using jem::util::CPUTimer;
using jive::app::Module;
using jive::model::Model;
using jive::implict::NonlinModule;
using jive::implict::ArclenModule;
using jive::implict::SolverModule;

using jive::util::Random;

//-----------------------------------------------------------------------
//   class AdaptiveStepModule
//-----------------------------------------------------------------------


class AdaptiveStepModule : public Module
{
 public:

  typedef AdaptiveStepModule  Self;
  typedef Module              Super;

  static const char*        TYPE_NAME;
  static const char*        NONLIN;
  static const char*        ARCLEN;
  static const char*        WRITE_STAT_PROP;
  static const char*        MIN_INCR_PROP;
  static const char*        MAX_INCR_PROP;
  static const char*        START_INCR_PROP;
  static const char*        REDUCTION_PROP;
  static const char*        OPT_ITER_PROP;

  explicit                  AdaptiveStepModule

    ( const String&           name = "AdaptiveStep",
      Ref<SolverModule>       solver = nullptr );

  virtual Status            init

    ( const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

  virtual Status            run

    ( const Properties&       globdat );

  virtual void              shutdown

    ( const Properties&       globdat );

  virtual void              configure

    ( const Properties&       props,
      const Properties&       globdat );

  virtual void              getConfig

    ( const Properties&       props,
      const Properties&       globdat )    const;

  static Ref<Module>        makeNew

    ( const String&           name,
      const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

 protected:

  virtual                  ~AdaptiveStepModule  ();

 private:

  bool                      reduceStep_

    ( const Properties&       globdat );

  bool                      increaseStep_

    ( const Properties&       globdat );

  void                      commit_ 
    
    ( const Properties&       globdat );

  void                      continue_ 
    
    ( const Properties&       params,
      const Properties&       globdat );

  void                      cancel_
    
    ( const Properties&       globdat );

  void                      setStepSize_

    ( const Properties&       globdat ) const;

 private:

  Ref<SolverModule>         solver_;

  Ref<Model>                model_;

  idx_t                     istep_;
  idx_t                     istep0_;

  double                    time_;
  double                    time0_;

  double                    maxIncrTried_;

  bool                      writeStats_;

  Ref<PrintWriter>          statOut_;

  // for adaptive time stepping 

  double                    startIncr_;
  double                    minIncr_;
  double                    maxIncr_;
  double                    reduction_;
  idx_t                     optIter_;
  double                    increment_;

  // for drawing random time increments

  bool                      randomInc_;
  Ref<Random>               generator_;

  // counters inside time step

  idx_t                     nCancels_;
  idx_t                     nContinues_;
  idx_t                     maxNIter_;
  bool                      triedSmall_;

  // cumulative counters (for shutdown statistics)

  idx_t                     nRunTotal_;
  idx_t                     nCancTotal_;
  idx_t                     nContTotal_;
  idx_t                     nIterTotal_;
  idx_t                     nIterCont_;
  idx_t                     nCommTotal_;
  idx_t                     nCancCont_;      // canceled continues

  CPUTimer                  cpuTimer_;
};


#endif
