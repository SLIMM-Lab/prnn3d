#include <jem/base/Once.h>
// #include "StdWedge12.h"
// #include "Wedge12.h"
#include "Wedge12AIO.h"


//-----------------------------------------------------------------------
//   declareMyShapes
//-----------------------------------------------------------------------

static void declareMyShapes_ ()
{
  jive::geom::StdWedge12:: declare();
  jive::geom::Wedge12   :: declare();
}

void declareMyShapes ()
{
  static jem::Once once = JEM_ONCE_INITIALIZER;

  jem::runOnce ( once, declareMyShapes_ );
}

