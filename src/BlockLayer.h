/*
 * Copyright (C) 2022 TU Delft. All rights reserved.
 *
 * Class that implements a layer connected by blocks. Each
 * of its blocks will connect themselves with one other block
 * in the previous layer (towards the input layer). Note that this
 * does not guarantee fully connectivity to the next layer.
 *
 * Author: Marina Maia, m.alvesmaia@tudelft.nl
 * Date:   Oct 2022
 * 
 */

#ifndef DENSELAYER_H
#define DENSELAYER_H

#include <jive/model/Model.h>
#include <jem/base/System.h>
#include <jem/util/Properties.h>
#include <jem/util/Timer.h>
#include <jive/util/Assignable.h>
#include <jive/util/XDofSpace.h>

#include "NeuralUtils.h"
#include "XAxonSet.h"
#include "XNeuronSet.h"

#include "NData.h"

using jem::String;
using jem::idx_t;
using jem::util::Properties;
using jem::util::Timer;

using jive::model::Model;
using jive::util::Assignable;
using jive::util::DofSpace;

using NeuralUtils::ActivationFunc;
using NeuralUtils::ActivationGrad;
using NeuralUtils::ActivationHess;
using NeuralUtils::InitFunc;

typedef Array<idx_t,2> IdxMatrix;

class BlockLayer : public Model
{
 public:
  
  static const char*   SIZE;
  static const char*   ACTIVATION;
  static const char*   INITIALIZATION;
  static const char*   USEBIAS;
  static const char*   DEBUG;
  static const char*   PRUNING;
  static const char*   POSWEIGHTS;

                       BlockLayer

    ( const String&      name,
      const Properties&  conf,
      const Properties&  props,
      const Properties&  globdat  );

  virtual void         configure

    ( const Properties&  props,
      const Properties&  globdat  );

  virtual void         getConfig

    ( const Properties&  conf,
      const Properties&  globdat  );

  virtual bool         takeAction

    ( const String&      action,
      const Properties&  params,
      const Properties&  globdat  );

  virtual void         getInpDofs 

    (       IdxMatrix&   dofs     );

 protected:

  virtual             ~BlockLayer  ();

  virtual void         update_

    ( const Properties&  globdat  );

  virtual void         propagate_

    ( const Ref<NData>   data,
      const Properties&  globdat  );
      
  virtual void         getDeterminant_

    ( const Ref<NData>   data,
      const Properties&  globdat,
      const Properties&  params );      

  virtual void         backPropagate_

    ( const Ref<NData>   data,
            Vector&      grads,
      const Properties&  globdat  );

  virtual void         backJacobian_

    ( const Ref<NData>   data,
      const Ref<NData>   rdata,
            Vector&      grads,
      const Properties&  globdat  );

  virtual void         forwardJacobian_

    ( const Ref<NData>   data,
      const Ref<NData>   rdata,
      const Properties&  globdat  );

  virtual void         getJacobian_

    ( const Ref<NData>   data,
      const Properties&  globdat  );

 private:
  
  idx_t                size_;
  idx_t                oSize_;
  idx_t                blockSize_;
  idx_t                blockDim_;
  idx_t                nBlocks_;
  idx_t                nBlocksPrev_;
  idx_t                weightType_;
  idx_t                biasType_;
  idx_t                prev;

  InitFunc             init_;
  ActivationFunc       actfuncweights_;
  ActivationFunc       func_;
  ActivationGrad       grad_;
  ActivationGrad       gradfuncweights_;
  ActivationHess       hess_;

  Ref<DofSpace>        dofs_;
  Ref<AxonSet>         axons_;
  Ref<NeuronSet>       neurons_;

  IdxVector            iNeurons_;
  IdxVector            inpNeurons_;

  IdxVector            iBiases_;
  IdxMatrix            iInpWts_;

  Vector               biases_;
  Matrix               weights_;
  Matrix               weightsDof_;
  Matrix               weightsDofcopy_;
  Matrix               weightsExp_;

  Matrix               jacobian_;

  bool                 inputLayer_;
  bool                 outputLayer_;
  bool                 mirrored_;
  bool                 useBias_;
  bool                 debug_;
  bool                 pruning_;
  bool                 posWeights_;
  bool                 actWeights_;
  bool                 postMult_;
  bool                 transpMat_;
  bool                 lower_; 

  Timer                pmmul_;
  Timer                pbias_;
  Timer                pfunc_;
  Timer                ptot_;
  Timer                palloc_;
  Timer                bpmmul_;
  Timer                bptot_;
  Timer                bpbias_;
  Timer                bpfunc_;
  Timer                bpalloc_;
};

#endif
