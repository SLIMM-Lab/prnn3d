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
 */

#ifndef GP_MODEL_H 
#define GP_MODEL_H

#include <jive/Array.h>
#include <jive/model/Model.h>
#include <jive/model/Actions.h>
#include <jive/model/ModelFactory.h>
#include <jive/util/Assignable.h>
#include <jive/util/XDofSpace.h>
#include <jive/util/ItemSet.h>
#include <jive/util/DummyItemSet.h>
#include <jive/algebra/MatrixBuilder.h>

#include "Kernel.h"

using namespace jem;

using jem::util::Properties;
using jive::Vector;
using jive::IntVector;
using jive::IdxVector;
using jive::StringVector;
using jive::model::Model;
using jive::util::XDofSpace;
using jive::util::DofSpace;
using jive::util::DummyItemSet;
using jive::algebra::MatrixBuilder;

class GPModel : public Model
{
 public:

  static const char* KERNEL;
  static const char* SEED;

                     GPModel

    ( const String&     name,
      const Properties& conf,
      const Properties& props,
      const Properties& globdat );

  virtual void       configure

    ( const Properties& props,
      const Properties& globdat );

  virtual void       getConfig

    ( const Properties& conf,
      const Properties& globdat )          const;

  virtual bool       takeAction

    ( const String&     action,
      const Properties& params,
      const Properties& globdat );

  static Ref<Model>  makeNew

    ( const String&     name,
      const Properties& conf,
      const Properties& props,
      const Properties& globdat );

 protected:
  
  virtual           ~GPModel ();

          void       init_

    ( const Properties& globdat );

          double     update_
	  
    ( const Properties& globdat );

          void       predict_

    ( const Properties& params,
      const Properties& globdat );

          void       batchPredict_

    ( const Properties& params,
      const Properties& globdat );

          void       samplePrior_

    ( const Properties& params,
      const Properties& globdat );

          void       samplePosterior_

    ( const Properties& params,
      const Properties& globdat );

          void       getGrads_
 
    (       Vector&     grads,
      const Properties& globdat );

          void       invalidate_ ();

 private:

  Ref<Kernel>         kernel_;

  Ref<DummyItemSet>   items_;
  Ref<XDofSpace>      dofs_;

  idx_t               dofType_;

  idx_t               n_;
  Matrix              x_;
  Vector              y_;

  Vector              alpha_;
  Matrix              K_;

  bool                newData_;
  bool                connected_;

  Ref<MatrixBuilder>  extNoise_;

};

#endif
