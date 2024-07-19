/*
 * Copyright (C) 2019 TU Delft. All rights reserved.
 *
 * Class with actions and parameters for machine learning
 *
 * Author: Iuri Barcelos, i.barceloscarneiromrocha@tudelft.nl
 * Date:   May 2019
 * 
 */

#include "LearningNames.h"

//-----------------------------------------------------------------------
//   class LearningActions
//-----------------------------------------------------------------------

const char* LearningActions::UPDATE           = "update";
const char* LearningActions::PROPAGATE        = "propagate";
const char* LearningActions::RECALL           = "recall";
const char* LearningActions::BATCHRECALL      = "batchRecall";
const char* LearningActions::MIDPROPAGATE     = "midpropagate";
const char* LearningActions::BACKPROPAGATE    = "backpropagate";
const char* LearningActions::GETJACOBIAN      = "getJacobianMatrix";
const char* LearningActions::GETMIDJACOBIAN   = "getMiddleJacobianMatrix";
const char* LearningActions::FORWARDJAC       = "doForwardJacobianPass";
const char* LearningActions::BACKWARDJAC      = "doBackwardJacobianPass";
const char* LearningActions::GETGRADS         = "getBayesianGradients";
const char* LearningActions::WRITEPARAMS      = "writeParams";
const char* LearningActions::SAMPLEPRIOR      = "samplePrior";
const char* LearningActions::SAMPLEPOSTERIOR  = "samplePosterior";
const char* LearningActions::SETEXTERNALNOISE = "setExternalNoise";
const char* LearningActions::COMPUTELIKELIHOOD = "computeLikelihood";
const char* LearningActions::OBTAINDATASET     = "obtainDataSet";
const char* LearningActions::SETNEWPARAMETERS  = "setNewParameters";
const char* LearningActions::GETHISTORY        = "getHistory";
const char* LearningActions::GETDETERMINANT    = "getDeterminant";

//-----------------------------------------------------------------------
//   class LearningParams
//-----------------------------------------------------------------------

const char* LearningParams::INPUT            = "neuralInput";
const char* LearningParams::OUTPUT           = "neuralOutput";
const char* LearningParams::SAMPLES          = "samples";
const char* LearningParams::DERIVATIVES      = "derivatives";
const char* LearningParams::VARIANCE         = "variance";
const char* LearningParams::NOISE            = "noise";
const char* LearningParams::VALUES           = "neuronValues";
const char* LearningParams::ACTIVATIONS      = "neuronActivations";
const char* LearningParams::WEIGHTS          = "weights";
const char* LearningParams::GRADS            = "gradients";
const char* LearningParams::LOSSGRAD         = "lossGrad";
const char* LearningParams::DELTAS           = "deltas";
const char* LearningParams::PREDECESSOR      = "predecessor";
const char* LearningParams::IMAGE            = "imageLayer";
const char* LearningParams::OBJFUNCTION      = "objFunction";
const char* LearningParams::DATA             = "neuralData";
const char* LearningParams::RDATA            = "RneuralData";
const char* LearningParams::STATE            = "neuralState";
const char* LearningParams::NEWSTATE         = "newState";
const char* LearningParams::HISTORY          = "history";
const char* LearningParams::DETERMINANT      = "determinant";


//-----------------------------------------------------------------------
//   class LearningNames
//-----------------------------------------------------------------------

const char* LearningNames::WEIGHTDOF         = "w";
const char* LearningNames::BIASDOF           = "b";
const char* LearningNames::HYPERDOF          = "theta";
const char* LearningNames::FIRSTLAYER        = "firstLayer";
const char* LearningNames::LASTLAYER         = "lastLayer";
const char* LearningNames::HYPERSET          = "hyperparameters";

