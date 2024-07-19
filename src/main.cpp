
#include <jive/app/ChainModule.h>
#include <jive/app/ControlModule.h>
#include <jive/app/OutputModule.h>
#include <jive/app/ReportModule.h>
#include <jive/app/Application.h>
#include <jive/app/InfoModule.h>
#include <jive/app/SampleModule.h>
#include <jive/app/UserconfModule.h>
#include <jive/geom/declare.h>
#include <jive/model/declare.h>
#include <jive/femodel/declare.h>
#include <jive/fem/declare.h>
#include <jive/fem/InputModule.h>
#include <jive/fem/InitModule.h>
#include <jive/fem/ShapeModule.h>
#include <jive/implict/declare.h>
#include <jive/app/declare.h>
#include <jive/algebra/declare.h>
#include <jive/gl/declare.h>

#include "models.h"
#include "modules.h"
#include "XOutputModule.h"
#include "MyShapes.h"

using namespace jem;

using jive::app::Application;
using jive::app::Module;
using jive::app::ChainModule;
using jive::app::OutputModule;
using jive::app::InfoModule;
using jive::app::ControlModule;
using jive::app::ReportModule;
using jive::app::SampleModule;
using jive::app::UserconfModule;
using jive::fem::InputModule;
using jive::fem::InitModule;
using jive::fem::ShapeModule;

//-----------------------------------------------------------------------
//   mainModule
//-----------------------------------------------------------------------


Ref<Module> mainModule ()
{
  Ref<ChainModule>    chain = newInstance<ChainModule> ();

  // Declare internal shapes, models and matrix builders. These
  // functions essentially store pointers to construction functions
  // that are called when Jive needs to create a shape, model or
  // matrix builder of a particular type.

  jive::geom:: declareIShapes     ();
  jive::geom::declareShapes       ();

  declareMyShapes ();

  jive::model::declareModels      ();
  jive::fem::declareMBuilders     ();
  jive::implict::declareModels    ();
  jive::algebra::declareMBuilders ();
  jive::femodel::declareModels    ();

  declareModels                   ();

  // Declare all modules that you want to add dynamically.

  jive::app     ::declareModules  ();
  jive::implict ::declareModules  ();
  jive::gl      ::declareModules  ();

  declareModules                  ();

  // Set up the module chain. These modules will be called by Jive in
  // the order that they have been added to the chain.

  // User defined inputmodules

  chain->pushBack ( newInstance<UserconfModule> ( "userInput" ) );

  // The ShapeModule creates a ShapeTable that maps elements to shape
  // objects. This is needed, among others, by the FemViewModule
  // below.

  chain->pushBack ( newInstance<ShapeModule>() );

  // The InitModule creates the main model and initializes some other
  // stuff.

  chain->pushBack ( newInstance<InitModule>() );

  // The InfoModule prints information about the current calculation.

  chain->pushBack ( newInstance<InfoModule>() );

  // UserModules to specify solver and output

  chain->pushBack ( newInstance<UserconfModule> ( "userModules" ) );

  // The XOutputModule is used for user-defined output

  chain->pushBack ( newInstance<XOutputModule>( ) );

  // The SampleModule is used to generate simple numeric ouput 
  
  chain->pushBack ( newInstance<SampleModule> ( ) );

  chain->pushBack ( newInstance<ControlModule> ( "control" ) );

  // Finally, the chain is wrapped in a ReportModule that prints some
  // overall information about the current calculation.

  return newInstance<ReportModule> ( "report", chain );
}


//-----------------------------------------------------------------------
//   main
//-----------------------------------------------------------------------


int main ( int argc, char** argv )
{
  return Application::exec ( argc, argv, & mainModule );
}
