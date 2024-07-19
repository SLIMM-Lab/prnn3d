/*
 *  Copyright (C) 2015 TU Delft. All rights reserved.
 *  
 *  Frans van der Meer, January 2015
 *  
 *  Module to generate default NodeGroups for periodic boundary conditions
 *  It basically runs the GroupInputModule with some predefined input
 *
 */

#include <jem/base/array/operators.h>
#include <jem/base/array/utilities.h>
#include <jem/base/array/select.h>
#include <jem/base/Error.h>
#include <jem/base/System.h>
#include <jem/numeric/algebra/utilities.h>
#include <jem/util/Properties.h>
#include <jem/util/ArrayBuffer.h>
#include <jive/app/ModuleFactory.h>
#include <jive/fem/NodeGroup.h>

#include "PBCGroupInputModule.h"


using jem::Error;
using jem::io::endl;
using jem::numeric::norm2;
using jem::util::ArrayBuffer;
using jive::fem::NodeSet;
using jive::fem::NodeGroup;

//=======================================================================
//   class PBCGroupInputModule
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------

const char* PBCGroupInputModule::TYPE_NAME = "PBCGroupInput";
const char* PBCGroupInputModule::EDGES[6] =
      { "xmin", "xmax", "ymin", "ymax", "zmin", "zmax" };
const char* PBCGroupInputModule::CORNERS[4] =
      { "corner0", "cornerx", "cornery", "cornerz" };
const char* PBCGroupInputModule::DUPEDNODES_PROP = "duplicatedNodes";

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------

PBCGroupInputModule::PBCGroupInputModule

  ( const String&  name ) :

      Super   ( name   )

{
  rank_ = -1;
  dupedNodeGroup_ = "";
}

PBCGroupInputModule::~PBCGroupInputModule ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


Module::Status PBCGroupInputModule::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  // set some variables

  NodeSet nodes = NodeSet::find    ( globdat );
  rank_ = nodes.rank();

  Matrix coords ( rank_, nodes.size() );

  nodes.getCoords ( coords );

  double dx = max(coords(0,ALL)) - min(coords(0,ALL));

  small_ = dx / 1.e6;

  // get duplicated nodes, if they exist

  props.findProps ( myName_ ).find
                  ( dupedNodeGroup_, DUPEDNODES_PROP );

  // make default Properties object for Super

  Properties myProps = props.makeProps ( myName_ );

  prepareProps_ ( myProps );

  // make NodeGroups

  Super::init ( conf, props, globdat );

  // sort NodeGroups such that ordering of opposite faces is matching

  for ( idx_t i = 0; i < rank_; ++i )
  {
    NodeGroup edge0 = NodeGroup::find ( EDGES[i*2  ], nodes, globdat );
    NodeGroup edge1 = NodeGroup::find ( EDGES[i*2+1], nodes, globdat );

    IdxVector inodes0 = edge0.getIndices();
    IdxVector inodes1 = edge1.getIndices();

    if ( inodes0.size() != inodes1.size() )
    {
      throw Error ( JEM_FUNC, String(i) + 
          "opposite edges do not have the same number of nodes" );
    }

    sortBoundaryNodes_ ( inodes1, inodes0, nodes, globdat, i );

    NodeGroup updated = newNodeGroup ( inodes1, nodes );

    updated.store ( EDGES[i*2+1], globdat );

    System::out() << "  ...Sorted NodeGroup `" << EDGES[i*2+1] <<
      "' wrt `" << EDGES[i*2] << "'\n";
  }

  return DONE;
}

//-----------------------------------------------------------------------
//   prepareProps_
//-----------------------------------------------------------------------

void PBCGroupInputModule::prepareProps_

  ( const Properties&  myProps ) const
  
{
  myProps.set ( "eps", small_ );

  StringVector nGroupNames ( 3*rank_+1 );

  idx_t j = 0;
  nGroupNames[j++] = CORNERS[0];
  for ( idx_t i = 0; i < rank_; ++i )
  {
    nGroupNames[j++] = CORNERS[i+1];
    nGroupNames[j++] = EDGES[i*2];
    nGroupNames[j++] = EDGES[i*2+1];
  }

  // read names of Groups that are to be created

  String XTYPE = ".xtype";
  String YTYPE = ".ytype";
  String ZTYPE = ".ztype";
  String MIN = "min";
  String MAX = "max";

  myProps.set  ( NODE_GROUPS, nGroupNames );

  myProps.set ( String(EDGES[0]) + XTYPE, MIN );
  myProps.set ( String(EDGES[1]) + XTYPE, MAX );
  myProps.set ( String(EDGES[2]) + YTYPE, MIN );
  myProps.set ( String(EDGES[3]) + YTYPE, MAX );

  myProps.set ( String(CORNERS[0]) + XTYPE, MIN );
  myProps.set ( String(CORNERS[0]) + YTYPE, MIN );

  myProps.set ( String(CORNERS[1]) + XTYPE, MAX );
  myProps.set ( String(CORNERS[1]) + YTYPE, MIN );

  myProps.set ( String(CORNERS[2]) + XTYPE, MIN );
  myProps.set ( String(CORNERS[2]) + YTYPE, MAX );

  if ( rank_ > 2 )
  {
    myProps.set ( String(EDGES[4]) + ZTYPE, MIN );
    myProps.set ( String(EDGES[5]) + ZTYPE, MAX );

    myProps.set ( String(CORNERS[0]) + ZTYPE, MIN );
    myProps.set ( String(CORNERS[1]) + ZTYPE, MIN );
    myProps.set ( String(CORNERS[2]) + ZTYPE, MIN );

    myProps.set ( String(CORNERS[3]) + XTYPE, MIN );
    myProps.set ( String(CORNERS[3]) + YTYPE, MIN );
    myProps.set ( String(CORNERS[3]) + ZTYPE, MAX );
  }
}

//-----------------------------------------------------------------------
//   sortBoundaryNodes_
//----------------------------------------------------------------------

void PBCGroupInputModule::sortBoundaryNodes_ 

  ( const IdxVector&  islaves,
    const IdxVector&  imasters,
    const NodeSet&    nodes,
    const Properties& globdat,
    const idx_t       ix ) const

{

  // make sure that ordering of islaves matches orderning of imasters

  JEM_ASSERT     ( islaves.size() == imasters.size() );
  const idx_t nn = islaves.size();

  Vector     mcoords ( rank_ );
  Vector     scoords ( rank_ );
  IdxVector  sorted  ( nn    );

  sorted = -1;

  // collect relevant coordinates 
  // e.g. for XMIN and XMAX plane, compare y and z coordinate

  ArrayBuffer<idx_t> ibuf;
  for ( idx_t jx = 0; jx < rank_; ++jx )
  {
    if ( jx != ix ) ibuf.pushBack ( jx );
  }
  IdxVector  irelevant ( ibuf.toArray() );


  for ( idx_t in = 0; in < nn; ++in )
  {
    nodes.getNodeCoords ( mcoords, imasters[in] );

    for ( idx_t jn = 0; jn < nn; ++jn )
    {
      nodes.getNodeCoords ( scoords, islaves[jn] );

      double dist = norm2 ( scoords[irelevant] - mcoords[irelevant] );

      if ( dist < small_ )
      {
        if ( dupedNodeGroup_ == "" )
        {
          sorted[in] = islaves[jn];
          break;
        }
        else
        {
          NodeGroup newNodes ( NodeGroup::get (
                               dupedNodeGroup_, nodes, globdat, "" ) );

          bool neither = !newNodes.contains ( imasters[in] ) &&
                         !newNodes.contains ( islaves [jn] );

          bool both    =  newNodes.contains ( imasters[in] ) &&
                          newNodes.contains ( islaves [jn] );

          if ( neither || both )
          {
            sorted[in] = islaves[jn];
            break;
          }
        }
      }
      
      if ( jn == nn-1 )
      {
        throw Error ( JEM_FUNC, "No match found for node " + String(in) );
      }
    }
  }
  JEM_ASSERT ( testall ( sorted >= 0 ) );
  islaves = sorted;
}

//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------

Ref<Module>  PBCGroupInputModule::makeNew

  ( const String&           name,
    const Properties&       conf,
    const Properties&       props,
    const Properties&       globdat )

{
  return newInstance<Self> ( name );
}

//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   declarePBCGroupInputModule
//-----------------------------------------------------------------------

void declarePBCGroupInputModule ()
{
  using jive::app::ModuleFactory;

  ModuleFactory::declare ( PBCGroupInputModule::TYPE_NAME,
                         & PBCGroupInputModule::makeNew );
}
