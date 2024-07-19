/*
 * Copyright (C) 2021 TU Delft. All rights reserved.
 *
 * Class that implements a fully-connected recurrent neural layer.
 * Neurons are connected with the previous and next layers as well
 * as with its own state at the time when forward propagation
 * occurs. This layer can be stacked with other MatRecLayers
 * as well as with DenseLayers.
 *
 * Author: Marina Maia, m.alvesmaia@tudelft.nl
 * Date:   May 2021
 * 
 */

#ifndef MATRECLAYER_H
#define MATRECLAYER_H

#include <jive/model/Model.h>
#include <jem/base/System.h>
#include <jem/util/Properties.h>
#include <jive/util/Assignable.h>
#include <jive/util/XDofSpace.h>

#include "NeuralUtils.h"
#include "XAxonSet.h"
#include "XNeuronSet.h"

#include "NData.h"

#include "Material.h"          //Added by Marina
#include <jem/util/ObjFlex.h>  //Added by Marina
#include "TrainingData.h"

using jem::String;
using jem::idx_t;
using jem::util::Properties;

using jive::model::Model;
using jive::util::Assignable;
using jive::util::DofSpace;

using NeuralUtils::ActivationFunc;
using NeuralUtils::ActivationGrad;

using NeuralUtils::InitFunc;

typedef Array<idx_t,2> IdxMatrix;
typedef Ref<Material>  MatRef;

class MatRecLayer : public Model
{
 public:
  
  static const char*   SIZE;
  static const char*   ACTIVATION;
  static const char*   INPINIT;
  static const char*   RECINIT;
  static const char*   MATERIAL_LIST;
  static const char*   MATERIAL_PROP;
  static const char*   RATIOMODEL;
  static const char*   USEBIAS; 
  static const char*   NMODELS;

                       MatRecLayer

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

  virtual             ~MatRecLayer  ();

  Array<MatRef>        child_;

  virtual void         update_

    ( const Properties&  globdat  );

  virtual void         propagate_

    ( const Ref<NData>   data,
      const Ref<NData>   state,
      const Properties&  globdat  );

  virtual void         backPropagate_

    ( const Ref<NData>   data,
      const Ref<NData>   state,
            Vector&      grads,
      const Properties&  globdat  );

  virtual void         getHistory_

    ( const Ref<NData>   data,
      const Ref<NData>   state,
      const Properties& globdat,
      const Properties& params);

  virtual void         getJacobian_

    ( const Ref<NData>   data,
      const Properties&  globdat  );

 private:
  
  Properties           myProps;
  Properties           globdat_;

  idx_t                size_;
  idx_t                weightType_;
  idx_t                biasType_;
  idx_t                sSize_;
  idx_t                hSize_;
  idx_t                mpSize_;
  idx_t                wSize_;

  IdxVector            hSizes_;
  IdxVector            hSizesIdx_;

  idx_t                nIntPts_;
  idx_t                nIntPts1_;
  idx_t                nIntPts2_;
  IdxVector            nModels_;
  IdxVector            modelType_;
  bool                 debug_;
  bool                 mirrored_;
  double               ratioModel_;
  bool                 pruning_;
  idx_t                learnProp_;
  bool                 useBias_;
  bool                 frommodel_;
  double               dt_;
  bool                 useBlocks_;

  InitFunc             inpInit_;
  InitFunc             recInit_;

  ActivationFunc       func_;
  ActivationGrad       grad_;

  Ref<DofSpace>        dofs_;
  Ref<AxonSet>         axons_;
  Ref<NeuronSet>       neurons_;

  IdxVector            iNeurons_;
  IdxVector            inpNeurons_;
  IdxVector            hNeurons_;
  IdxVector            mpNeurons_;

  IdxVector            iBiases_;
  IdxVector            iBiasesProp_;
  IdxVector            iInits_;
  IdxMatrix            iInpWts_;
  IdxMatrix            iMemWts_;

  Vector               delconn_;

  Vector               biases_;
  Vector               biasesProp_;
  Vector               inits_;
  Matrix               inpWeights_;
  Matrix               memWeights_;

  Matrix               jacobian_;

  Ref<Normalizer>      onl_;
  Vector               upper_;
  Vector               lower_;
};

#endif
