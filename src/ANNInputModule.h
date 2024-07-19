/*
 *  TU Delft 
 *
 *  Iuri Barcelos, May 2019
 *
 *  Input module for ANN data. Creates and stores a TrainingData object.
 *  This module should be used in the input UserconfModule. Skip it if
 *  training data is coming from elsewhere (e.g. a nested or forked chain).
 *
 */

#ifndef ANNINPUTMODULE_H
#define ANNINPUTMODULE_H

#include <jem/util/Properties.h>
#include <jive/app/Module.h>
#include <jive/Array.h>

using jem::Ref;
using jem::String;
using jem::util::Properties;

using jive::app::Module;

//-----------------------------------------------------------------------
//   class ANNInputModule
//-----------------------------------------------------------------------

class ANNInputModule : public Module
{
 public:

  typedef ANNInputModule     Self;
  typedef Module             Super;

  static const char*        TYPE_NAME;
  static const char*        FILENAME;
  static const char*        INPUT;
  static const char*        OUTPUT;

  explicit                  ANNInputModule

    ( const String&           name = "ANNInput" );

  virtual Status            init

    ( const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

  static Ref<Module>        makeNew

    ( const String&           name,
      const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

 protected:

  virtual                  ~ANNInputModule  ();

 private:
};

#endif
