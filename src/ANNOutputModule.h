/*
 *  TU Delft 
 *
 *  Iuri Barcelos, July 2019
 *
 *  Output module for ANN data. Reads the TrainingData object, uses the
 *  trained network stored in the model to make predictions and prints
 *  the results to data files. It can also be used to print the network
 *  weights to a file.
 *
 *  This module should be used in the output UserconfModule. 
 *
 */

#ifndef ANNOUTPUTMODULE_H
#define ANNOUTPUTMODULE_H

#include <jem/util/Properties.h>
#include <jive/app/Module.h>
#include <jive/Array.h>
#include <jive/fem/typedefs.h>

#include "TrainingData.h"

using jem::Ref;
using jem::String;
using jem::util::Properties;
using jive::MPContext;

using jive::Vector;
using jive::app::Module;

//-----------------------------------------------------------------------
//   class ANNOutputModule
//-----------------------------------------------------------------------

class ANNOutputModule : public Module
{
 public:

  typedef ANNOutputModule   Self;
  typedef Module            Super;

  static const char*        TYPE_NAME;
  static const char*        FILENAME;
  static const char*        FORMAT;
  static const char*        PRINT;
  static const char*        WRITEEVERY;
  static const char*        SKIPWORSE;
  static const char*        RUNFIRST;
  static const char*        VALSPLIT;
  static const char*        PREDFILE;
  static const char*        BOUNDS;

  enum                      FormatType { LINES, COLUMNS };

  explicit                  ANNOutputModule

    ( const String&           name = "ANNOutput" );

  virtual Status            init

    ( const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

  virtual Status            run

    ( const Properties&       globdat );

  virtual void              shutdown

    ( const Properties&       globdat );

  static Ref<Module>        makeNew

    ( const String&           name,
      const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

  void                      read
	  ( const String& predName,
	    const Properties& globdat,
	    const String& fileName );

 protected:

  virtual                  ~ANNOutputModule  ();

	  void             runSamples_

    (       Vector&           errors, 
      const Batch&            samples,
      const Properties&       globdat         );

          void              writeWeights_

    ( const Properties&       globdat         );

          void              writeWeightsEpoch_

    ( const Properties&       globdat         );

          void              writeError_

    ( const Properties&       globdat         );


          void              writeLines_

    ( const Properties&       globdat         );

          void              writeCols_

    ( const Properties&       globdat         );

 private:

  String                    fname_;
  String                    predFile_;
  FormatType                format_;

  idx_t                     nsamples_;
  idx_t                     writeEvery_;
  bool                      mimicXOut_;
  bool                      printWts_;
  bool                      printOuts_;
  bool                      printInps_;

  idx_t                     epoch_;
  double                    best_;
  double                    relbest_;
  idx_t                     bestEpoch_;

  bool                      root_;

  Ref<MPContext>            mpx_;

  Vector                    bestWeights_;
  Batch                     bestSamples_;
  IdxVector                 sampIds_;
  IdxVector                 bounds_;
  IdxVector                 selComp_;  

  Vector                    stress;
  Vector                    strain;
};

#endif
