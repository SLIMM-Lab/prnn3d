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

#include <jem/base/System.h>

#include <jem/base/IllegalInputException.h>
#include <jem/util/PropertyException.h>
#include <jem/io/FileReader.h>
#include <jem/io/FileInputStream.h>
#include <jem/io/InputStreamReader.h>
#include <jive/app/ModuleFactory.h>
#include <jive/util/DofSpace.h>
#include <jive/model/StateVector.h>

#include "GPInputModule.h"
#include "TrainingData.h"

using jem::newInstance;
using jem::util::PropertyException;
using jem::IllegalInputException;
using jem::io::FileReader;
using jem::io::FileInputStream;
using jem::io::InputStreamReader;

using jive::util::DofSpace;
using jive::model::StateVector;

//-----------------------------------------------------------------------
//   class GPInputModule
//-----------------------------------------------------------------------


//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------

const char*        GPInputModule::TYPE_NAME = "GPInput";
const char*        GPInputModule::FILENAME  = "file";
const char*        GPInputModule::INPUT     = "input";
const char*        GPInputModule::OUTPUT    = "output";

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------

GPInputModule::GPInputModule 
  ( const String& name ) : Super ( name )

{}

GPInputModule::~GPInputModule ()
{}

//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------

Module::Status GPInputModule::init

  ( const Properties& conf,
    const Properties& props,
    const Properties& globdat )

{
  using jem::System;

  Properties myConf  = conf. makeProps ( myName_ );
  Properties myProps = props.findProps ( myName_ );

  String fname;
  myProps.get ( fname, FILENAME );

  IdxVector input, output, dt;

  myProps.get ( input, INPUT );
  myProps.get ( output, OUTPUT );
  myProps.get ( dt, "dt" );

  System::out() << "GPInputModule. init " << fname << " input " << input <<
	  " output " << output << "\n";

  Ref<TrainingData> data = newInstance<TrainingData>
    ( fname, input, output, dt, globdat );

  data->configure ( myProps, globdat );

  myConf.set ( INPUT,    input   );
  myConf.set ( "dt",     dt      );
  myConf.set ( OUTPUT,   output  );

  myConf.set ( FILENAME, fname   );

  return OK;
}

//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------

Ref<Module> GPInputModule::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  return newInstance<Self> ( name );
}

//-----------------------------------------------------------------------
//   declareGPInputModule
//-----------------------------------------------------------------------

void declareGPInputModule ()
{
  using jive::app::ModuleFactory;

  ModuleFactory::declare ( GPInputModule::TYPE_NAME, & GPInputModule::makeNew );
}

