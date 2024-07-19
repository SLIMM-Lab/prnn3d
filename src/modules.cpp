
#include "modules.h"

#include <jive/fem/InputModule.h>
#include <jive/app/ModuleFactory.h>
#include "AdamModule.h"
#include "BFGSModule.h"

//-----------------------------------------------------------------------
//   declareModules
//-----------------------------------------------------------------------


void declareModules ()
{
  declareAdaptiveStepModule ();
  declareGmshInputModule    ();
  declareGroupInputModule   ();
  declareXOutputModule      ();
  declareInputModule        ();
  declareLaminateMeshModule ();
  declarePBCGroupInputModule ();
  declareANNInputModule     ();
  declareANNOutputModule    ();
  jive::implict::AdamModule::declare ();
  jive::implict::BFGSModule::declare ();
  declareGPInputModule      ();
  declareGPPredModule       ();
}

void declareInputModule ()
{
  using jive::app::ModuleFactory;
  using jive::fem::InputModule;

  ModuleFactory::declare ( "Input",
                         & InputModule::makeNew );
}

