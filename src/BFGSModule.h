/*
 *  TU Delft 
 *
 *  Iuri Barcelos, Oct 2019
 *
 *  Bounded BFGS algorithm for full-batch optimization
 *  of model parameters. The model should make sure the
 *  parameters are always in a [0,1] interval. Designed
 *  for use in a Bayesian training scheme (maximization
 *  of the log marginal likelihood).
 *
 */

#ifndef JIVE_IMPLICT_BFGSMODULE_H
#define JIVE_IMPLICT_BFGSMODULE_H

#include <jem/util/Properties.h>
#include <jive/implict/SolverModule.h>
#include <jive/model/Model.h>
#include <jive/Array.h>
#include <jive/fem/typedefs.h>

using namespace jem;

using jem::Ref;
using jem::idx_t;
using jem::String;
using jive::BoolVector;
using jem::util::Properties;
using jive::Vector;
using jive::Matrix;
using jive::app::Module;
using jive::model::Model;

JIVE_BEGIN_PACKAGE( implict )

//-----------------------------------------------------------------------
//   class BFGSModule
//-----------------------------------------------------------------------

class BFGSModule : public SolverModule
{
 public:

  JEM_DECLARE_CLASS       ( BFGSModule, SolverModule );

  static const char*         TYPE_NAME;
  static const char*         PRECISION;
  static const char*         NRESTARTS;
  static const char*         MAXITER;
  static const char*         OUTFILE;

  explicit                   BFGSModule

    ( const String&            name = "bfgs" );

  virtual Status             init

    ( const Properties&        conf,
      const Properties&        props,
      const Properties&        globdat );

  virtual void              advance

    ( const Properties&       globdat );

  virtual void               solve

    ( const Properties&        info,
      const Properties&        globdat );

  virtual void              cancel

    ( const Properties&       globdat );

  virtual bool              commit

    ( const Properties&       globdat );

  virtual void               shutdown

    ( const Properties&        globdat );

  virtual void               configure

    ( const Properties&        props,
      const Properties&        globdat );

  virtual void               getConfig

    ( const Properties&        props,
      const Properties&        globdat ) const;

  virtual void              setPrecision

    ( double                  eps );

  virtual double            getPrecision  () const;

  static Ref<Module>         makeNew

    ( const String&            name,
      const Properties&        conf,
      const Properties&        props,
      const Properties&        globdat );

  static void               declare       ();

 protected:

  virtual                   ~BFGSModule ();

 private:
  
  idx_t                      epoch_;
  double                     tolGrad_;
  double                     tolVar_;
  double                     tolSearch_;
  double                     searchAmp_;

  Ref<Model>                 model_;
  double                     objfunc_;
  Vector                     oldGrads_;
  Vector                     grads_;
  Vector                     vars_;
  Vector                     oldVars_;
  double                     fdStep_;
  Matrix                     oldHessian_;
  Matrix                     hessian_;

  Vector                     pg0_;
  Vector                     pg_;
  Matrix                     oldInvH_;
  Matrix                     invH_;
  BoolVector                 active_;
  double                     precision_;

  Matrix                     id_;
  Matrix                     sSyST_;
  Matrix                     ySsST_;
  Matrix                     sSsST_;

  idx_t                      maxOpt_;
  idx_t                      maxIter_;
  idx_t                      nOpt_;
  double                     bestObj_;
  Vector                     best_;
  Vector                     bestReal_;

  Properties                 globdat_;
  Properties                 params_;

  idx_t                      numEvals_;

  String                     outFile_;

  void                       lineSearch_

    ( const Vector&            dir,
      const Properties&        globdat );

  void                       restart_

    ( const Properties&        globdat );

};

JIVE_END_PACKAGE( implict )

#endif
