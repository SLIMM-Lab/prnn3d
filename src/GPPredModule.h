/*
 *  TU Delft 
 *
 *  Iuri Barcelos, Oct 2019
 *
 *  Input module for GP data. Creates and stores a TrainingData object.
 *  This module should be used in the input UserconfModule. Skip it if
 *  training data is coming from elsewhere (e.g. a nested or forked chain).
 *
 */

#ifndef GPPREDMODULE_H
#define GPPREDMODULE_H

#include <jem/util/Properties.h>
#include <jive/app/Module.h>
#include <jive/Array.h>

using jem::Ref;
using jem::String;
using jem::util::Properties;

using jive::idx_t;
using jive::Vector;
using jive::app::Module;

//-----------------------------------------------------------------------
//   class GPPredModule
//-----------------------------------------------------------------------

class GPPredModule : public Module
{
 public:

  typedef GPPredModule     Self;
  typedef Module             Super;

  static const char*        TYPE_NAME;
  static const char*        FILENAME;
  static const char*        MINVALUES;
  static const char*        MAXVALUES;
  static const char*        STEPSIZES;
  static const char*        CONFINTER;
  static const char*        PRIORFILE;
  static const char*        POSTFILE;
  static const char*        NPRIOR;
  static const char*        NPOST;
  static const char*        PREDFILE;
  static const char*        PREDOUTFILE;

  explicit                  GPPredModule

    ( const String&           name = "GPPred" );

  virtual Status            init

    ( const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

  virtual Status            run

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

  virtual                  ~GPPredModule  ();

 private:
 
  double                    conf_;

  Vector                    minValues_;
  Vector                    maxValues_;
  Vector                    stepSizes_;

  String                    fileName_;
  String                    priorFile_;
  String                    postFile_;
  String                    predFile_;
  String                    predOutFile_;

  idx_t                     nPrior_;
  idx_t                     nPost_;

};

#endif
