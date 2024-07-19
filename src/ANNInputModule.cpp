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

#include <jem/base/IllegalInputException.h>
#include <jem/util/PropertyException.h>
#include <jem/io/FileReader.h>
#include <jem/io/FileInputStream.h>
#include <jem/io/InputStreamReader.h>
#include <jive/app/ModuleFactory.h>
#include <jive/util/DofSpace.h>
#include <jive/model/StateVector.h>

#include "ANNInputModule.h"
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
//   class ANNInputModule
//-----------------------------------------------------------------------


//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------

const char*        ANNInputModule::TYPE_NAME = "ANNInput";
const char*        ANNInputModule::FILENAME  = "file";
const char*        ANNInputModule::INPUT     = "input";
const char*        ANNInputModule::OUTPUT    = "output";

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------

ANNInputModule::ANNInputModule 
  ( const String& name ) : Super ( name )

{}

ANNInputModule::~ANNInputModule ()
{}

//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------

Module::Status ANNInputModule::init

  ( const Properties& conf,
    const Properties& props,
    const Properties& globdat )

{

  Properties myConf  = conf. makeProps ( myName_ );
  Properties myProps = props.findProps ( myName_ );

  String fname;
  myProps.get ( fname, FILENAME );

  String all;

  try
  {
    myProps.get ( all, INPUT );

    if ( all.equalsIgnoreCase ( "all" ) )
    {
      Ref<TrainingData> data = newInstance<TrainingData>
	( fname, globdat );

      data->configure ( myProps, globdat );

      myConf.set ( INPUT, all );
    }
    else
    {
      throw IllegalInputException ( JEM_FUNC,
        "Input must be either a set of indices or the string 'all' (for autoencoders)" );
    }
  }
  catch ( const PropertyException& ex )
  {
    IdxVector input, output, dts;
    String normalizer;

    myProps.get ( input, INPUT );
    myProps.get ( output, OUTPUT );
    myProps.find ( dts, "dts" );
    myProps.find ( normalizer, "outNormalizer");

    Ref<TrainingData> data = newInstance<TrainingData>
      ( fname, input, output, dts, globdat );

    data->configure ( myProps, globdat );

    myConf.set ( INPUT,    input   );
    myConf.set ( OUTPUT,   output  );
    myConf.set ( "dts", dts );
    myConf.set ( "outNormalizer", normalizer ); 
  }

  myConf.set ( FILENAME, fname   );

  return OK;
}

//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------

Ref<Module> ANNInputModule::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  return newInstance<Self> ( name );
}

//-----------------------------------------------------------------------
//   declareANNInputModule
//-----------------------------------------------------------------------

void declareANNInputModule ()
{
  using jive::app::ModuleFactory;

  ModuleFactory::declare ( ANNInputModule::TYPE_NAME, & ANNInputModule::makeNew );
}

