/*
 *  Copyright (C) 2015 TU Delft. All rights reserved.
 *  
 *  Frans van der Meer, January 2015
 *  
 *  Module to generate default NodeGroups for periodic boundary conditions
 *
 */

#ifndef PBC_GROUP_INPUT_MODULE_H
#define PBC_GROUP_INPUT_MODULE_H

#include <jive/app/Module.h>
#include <jive/fem/ElementGroup.h>
#include <jive/fem/NodeSet.h>
#include <jive/fem/ElementSet.h>
#include <jive/util/Constraints.h>
#include <jive/util/XDofSpace.h>
#include <jive/util/Assignable.h>

#include "GroupInputModule.h"

//-----------------------------------------------------------------------
//   class PBCGroupInputModule
//-----------------------------------------------------------------------

class PBCGroupInputModule : public GroupInputModule
{
 public:

  typedef PBCGroupInputModule   Self;
  typedef GroupInputModule      Super;

  static const char*            XMIN;
  static const char*            XMAX;
  static const char*            YMIN;
  static const char*            YMAX;
  static const char*            ZMIN;
  static const char*            ZMAX;
  static const char*            CORNER0;
  static const char*            CORNERX;
  static const char*            CORNERY;
  static const char*            CORNERZ;
  static const char*            TYPE_NAME;
  static const char*            EDGES[6];
  static const char*            CORNERS[4];
  static const char*            DUPEDNODES_PROP;

  explicit                  PBCGroupInputModule

    ( const String&           name   = "pbcGroupInput" );

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

  virtual                  ~PBCGroupInputModule  ();

  void                      prepareProps_

    ( const Properties&       myProps ) const;

  void                      sortBoundaryNodes_
    
    ( const IdxVector&        islaves,
      const IdxVector&        imasters,
      const NodeSet&          nodes,
      const Properties&       globdat,
      const idx_t             ix ) const;

 protected:

  idx_t                     rank_;
  double                    small_;

  String                    dupedNodeGroup_;
};



#endif
