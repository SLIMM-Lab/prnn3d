/*
 * Copyright (C) 2019 TU Delft. All rights reserved.
 *
 * Class with actions and parameters for machine learning
 *
 * Author: Iuri Barcelos, i.barceloscarneiromrocha@tudelft.nl
 * Date:   May 2019
 * 
 */

#ifndef LEARNING_NAMES_H
#define LEARNING_NAMES_H

//-----------------------------------------------------------------------
//   class LearningActions 
//-----------------------------------------------------------------------

class LearningActions
{
 public:

  static const char* UPDATE;
  static const char* PROPAGATE;
  static const char* RECALL;
  static const char* BATCHRECALL;
  static const char* MIDPROPAGATE;
  static const char* BACKPROPAGATE;
  static const char* GETJACOBIAN;
  static const char* GETHISTORY;
  static const char* GETDETERMINANT;
  static const char* GETMIDJACOBIAN;
  static const char* FORWARDJAC;
  static const char* BACKWARDJAC;
  static const char* GETGRADS;
  static const char* WRITEPARAMS;
  static const char* SAMPLEPRIOR;
  static const char* SAMPLEPOSTERIOR;
  static const char* SETEXTERNALNOISE;
  static const char* COMPUTELIKELIHOOD;
  static const char* OBTAINDATASET;
  static const char* SETNEWPARAMETERS;
};

//-----------------------------------------------------------------------
//   class LearningParams
//-----------------------------------------------------------------------

class LearningParams
{
 public:

  static const char* INPUT;
  static const char* OUTPUT;
  static const char* SAMPLES;
  static const char* VARIANCE;
  static const char* DERIVATIVES;
  static const char* NOISE;
  static const char* VALUES;
  static const char* ACTIVATIONS;
  static const char* GRADS;
  static const char* WEIGHTS;
  static const char* LOSSGRAD;
  static const char* DELTAS;
  static const char* PREDECESSOR;
  static const char* IMAGE;
  static const char* OBJFUNCTION;

  static const char* DATA;
  static const char* RDATA;
  static const char* STATE;
  static const char* NEWSTATE;

  static const char* HISTORY;
  static const char* DETERMINANT;
};

//-----------------------------------------------------------------------
//   class LearningNames
//-----------------------------------------------------------------------

class LearningNames
{
 public:
  
  static const char* WEIGHTDOF;
  static const char* BIASDOF;
  static const char* HYPERDOF;
  static const char* FIRSTLAYER;
  static const char* LASTLAYER;
  static const char* HYPERSET;
};

#endif
