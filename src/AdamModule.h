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

#ifndef JIVE_IMPLICT_ADAMMODULE_H
#define JIVE_IMPLICT_ADAMMODULE_H

#include <jem/util/Flex.h>
#include <jem/util/Timer.h>
#include <jive/implict/SolverModule.h>
#include <jive/model/Model.h>
#include <jive/fem/typedefs.h>
#include <jem/io/PrintWriter.h>

#include "NeuralUtils.h"

using jive::model::Model;
using jem::util::Timer;
using jem::util::Flex;
using jive::MPContext;
using jem::io::PrintWriter;

using NeuralUtils::LossFunc;
using NeuralUtils::LossGrad;

JIVE_BEGIN_PACKAGE( implict )


//-----------------------------------------------------------------------
//   class AdamModule
//-----------------------------------------------------------------------


class AdamModule : public SolverModule
{
 public:

  JEM_DECLARE_CLASS       ( AdamModule, SolverModule );

  static const char*        TYPE_NAME;
  static const char*        SEED;
  static const char*        ALPHA;
  static const char*        BETA1;      
  static const char*        BETA2;      
  static const char*        EPSILON;    
  static const char*        L2REG;
  static const char*        L1REG;
  static const char*        L1PL;
  static const char*        MINIBATCH;  
  static const char*        LOSSFUNC;   
  static const char*        PRECISION;
  static const char*        THREADS;
  static const char*        VALSPLIT;
  static const char*        SKIPFIRST;
  static const char*        JPROP;
  static const char*        PRUNING;

  explicit                  AdamModule

    ( const String&           name = "adam" );

  virtual Status            init

    ( const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

  virtual void              shutdown

    ( const Properties&       globdat );

  virtual void              configure

    ( const Properties&       props,
      const Properties&       globdat );

  virtual void              getConfig

    ( const Properties&       conf,
      const Properties&       globdat )      const;

  virtual void              advance

    ( const Properties&       globdat );

  virtual void              solve

    ( const Properties&       info,
      const Properties&       globdat );

  virtual void              cancel

    ( const Properties&       globdat );

  virtual bool              commit

    ( const Properties&       globdat );

  virtual void              setPrecision

    ( double                  eps );

  virtual double            getPrecision  () const;

  static Ref<Module>        makeNew

    ( const String&           name,
      const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

  static void               declare       ();

 protected:

  virtual                  ~AdamModule  ();

  void                      mpSolve_

    ( const Properties&       info,
      const Properties&       globdat    );

  void                      solve_

    ( const Properties&       info,
      const Properties&       globdat    );

  double                    eval_

    ( const IdxVector&        samples,
      const bool              dograds,
      const Properties&       globdat    );

  bool                      checkDet_

    ( const IdxVector&        samples,
      const bool              dograds,
      const Properties&       globdat    );

 private:

  idx_t                     seed_;
  idx_t                     batchSize_;
  idx_t                     epoch_;
  idx_t                     iiter_;
  idx_t                     threads_;
  idx_t                     skipFirst_;
  idx_t                     nip_;
  idx_t                     prev_;
  idx_t                     subsetSize_;
  idx_t                     rseed_;
  idx_t                     outSize_;

  IdxVector                 selComp_;

  bool                      pruning_;
  bool                      dehom_;
  bool                      dissipation_;
 
  double                    alpha_;
  double                    beta1_;
  double                    beta2_;
  double                    eps_;
  double                    lambda_;
  double                    lambdaL1_;
  double                    lambdaPl_;
  double                    precision_;
  double                    valSplit_;

  double                    r_;

  Vector                    g_;
  Vector                    gt_;
  Vector                    rg_;
  Vector                    m_;
  Vector                    v_;
  Vector                    m0_;
  Vector                    v0_;
  Vector                    negDetBlocks_;

  Matrix                    detAll_;

  String                    lossName_;
  LossFunc                  func_;
  LossGrad                  grad_;

  Ref<MPContext>            mpx_;
  bool                      mpi_;
  bool                      computeBatch_;

  Timer                     total_;
  Timer                     t1_;
  Timer                     t2_;
  Timer                     t3_;
  Timer                     t4_;
  Timer                     t5_;
  Timer                     t6_;

  double                    jprop_;
};


JIVE_END_PACKAGE( implict )

#endif
