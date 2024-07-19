/*
 *  Copyright (C) 2014 TU Delft. All rights reserved.
 *  
 *  Frans van der Meer, September 2014
 *  
 *  Module to generate NodeGroups and ElementGroups from input file
 *
 */

#include <jem/base/array/operators.h>
#include <jem/base/Error.h>
#include <jem/base/System.h>
#include <jem/numeric/algebra/utilities.h>
#include <jem/util/ArrayBuffer.h>
#include <jem/util/Properties.h>
#include <jem/util/Dictionary.h>
#include <jive/util/utilities.h>
#include <jive/app/ModuleFactory.h>
#include <jive/fem/NodeGroup.h>
#include <jive/fem/ElementGroup.h>

#include "GroupInputModule.h"

using jem::io::endl;
using jem::util::ArrayBuffer;
using jive::Vector;
using jive::fem::NodeSet;
using jive::fem::NodeGroup;
using jive::fem::newNodeGroup;
using jive::util::makeUnique;


//=======================================================================
//   class GroupInputModule
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------

const char* GroupInputModule::TYPE_NAME       = "GroupInput";
const char* GroupInputModule::NODE_GROUPS     = "nodeGroups";
const char* GroupInputModule::ELEM_GROUPS     = "elemGroups";

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------

GroupInputModule::GroupInputModule

  ( const String&  name ) :

      Super   ( name   )

{}

GroupInputModule::~GroupInputModule ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


Module::Status GroupInputModule::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  Properties  myConf  = conf .makeProps ( myName_ );
  Properties  myProps = props.findProps ( myName_ );

  StringVector nGroupNames;
  StringVector eGroupNames;

  // read names of Groups that are to be created

  myProps.find ( nGroupNames, NODE_GROUPS );
  myConf.set   ( NODE_GROUPS, nGroupNames );

  myProps.find ( eGroupNames, ELEM_GROUPS );
  myConf.set   ( ELEM_GROUPS, eGroupNames );

  // make NodeGroups

  for ( idx_t ing = 0; ing < nGroupNames.size(); ++ing )
  {
    Ref<NGroup_> ngrInput = newInstance<NGroup_> ();

    ngrInput->configure( myConf, myProps, nGroupNames[ing] );
    ngrInput->makeGroup ( globdat );
  }

  // make ElementGroups

  for ( idx_t ing = 0; ing < eGroupNames.size(); ++ing )
  {
    Ref<EGroup_> egrInput = newInstance<EGroup_> ();

    egrInput->configure ( myConf, myProps, eGroupNames[ing] );
    egrInput->makeGroup ( globdat );
  }

  return DONE;
}

//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------

Ref<Module>  GroupInputModule::makeNew

  ( const String&           name,
    const Properties&       conf,
    const Properties&       props,
    const Properties&       globdat )

{
  return newInstance<Self> ( name );
}

//=======================================================================
//   class GroupInputModule::NGroup_
//=======================================================================

// ------------------------------------------------------------------------
//   static members
// ------------------------------------------------------------------------

const char*  GroupInputModule::NGroup_::X_NAMES[3] = {"x","y","z"};
const double GroupInputModule::NGroup_::PI         = 3.14159265358979323846;

//-----------------------------------------------------------------------
//   GroupInputModule::NGroup_::constructur,destructor
//-----------------------------------------------------------------------

GroupInputModule::NGroup_::NGroup_ 

  ( ) : Object ( )

{
  doX_     .resize ( 3 );

  xvals_[0].resize ( 1 );
  xvals_[1].resize ( 1 );
  xvals_[2].resize ( 1 );

  xbounds_ .resize ( 3, 2 );  // ( x/y/z , min/max )

  rank_   = 3;
  eps_    = 1.e-5;
  angle_  = 0.;
  radius_ = -1.;

  knownTypes.resize( 5 );

  knownTypes[0] = "min";
  knownTypes[1] = "max";
  knownTypes[2] = "firstHalf";
  knownTypes[3] = "secondHalf"; 
  knownTypes[4] = "mid";
}

GroupInputModule::NGroup_::~NGroup_ ()
{}

//-----------------------------------------------------------------------
//   GroupInputModule::NGroup_::configure
//-----------------------------------------------------------------------

void GroupInputModule::NGroup_::configure
 
  ( const Properties&  conf,
    const Properties&  props,
    const String&      name )

{
  rank_ = 3;
  all_  = false;

  String  xtypename; 
  Vector  bounds;

  // get properties

  myName_ = name;

  Properties  myConf  = conf .makeProps ( myName_ );
  Properties  myProps = props. getProps ( myName_ );

  // set nodes

  if ( myProps.find ( inodes_, "nodes" ) )
  {
    myConf.set ( "nodes", inodes_ );
    doX_ = NONE;
    return;
  }

  myProps.find ( all_, "all" );
  myConf.set   ( "all", all_ );

  // restrict to nodes belonging to an ElementGroup

  String restrictTo1;
  if ( myProps.find( restrictTo1, "restrictToGroup" ) )
  {
    restrictTo_.resize ( 1 );
    restrictTo_[0] = restrictTo1;

    myProps.find ( restrictPartial_, "restrictPartial" );
    myConf.set   ( "restrictPartial", restrictPartial_ );
  }
  else
  {
    myProps.find ( restrictTo_, "restrictToGroups" );
  }
  myConf.set ( "restrictToGroups", restrictTo_ );

  if ( all_ )
  {
    return;
  }

  // set tolerance

  myProps.find( eps_, "eps" );
  myConf.set  ( "eps", eps_ );

  // find coordinate types or values

  xtype_ = "";

  for ( idx_t ir = 0; ir < rank_ ; ++ir )
  {
    String thiss = X_NAMES[ir];

    String xtypename ( thiss + String("type") );
    String xvalname  ( thiss + "val"  );
    String xvalsname ( thiss + "vals" );
    String xinname   ( thiss + "bounds" );

    doX_[ir] = VALUE;

    if ( myProps.find ( xtype_[ir], xtypename ) )
    {
      myConf.set ( xtypename, xtype_[ir]);

      if ( ! testany ( knownTypes == xtype_[ir] ) )
      {
        throw Error ( JEM_FUNC, "Unknown xtype: " + xtype_[ir] );
      }
    }
    else if ( myProps.find ( xvals_[ir][0], xvalname ) )
    {
      myConf.set ( xvalname, xvals_[ir][0] );
    }
    else if ( myProps.find ( xvals_[ir], xvalsname ) )
    {
      myConf.set ( xvalsname, xvals_[ir] );
    }
    else if ( myProps.find ( bounds, xinname ) )
    {
      myConf.set ( xinname, bounds );

      if ( bounds.size() == 2 )
      {
        xbounds_(ir,0) = bounds[0] - eps_;
        xbounds_(ir,1) = bounds[1] + eps_;

        doX_[ir] = BOUNDS;
      }
      else
      {
        throw Error ( JEM_FUNC, xinname + " does not have length 2!" );
      }
    }
    else
    {
      doX_[ir] = NONE;
    }
  }

  if ( ! testany ( doX_ ) )
  {
    // line specified with x0, y0 and angle
    // or circle with x0, y0 and radius

    myProps.get ( xvals_[0][0], "x0" );
    myProps.get ( xvals_[1][0], "y0" );

    myConf.set  ( "x0", xvals_[0][0] );
    myConf.set  ( "y0", xvals_[1][0] );

    if ( myProps.find ( radius_, "radius", 0., maxOf(radius_) ) )
    {
      myConf.set ( "radius", radius_ );
    }
    else
    {
      myProps.get ( angle_, "angle" );
      myConf.set  ( "angle", angle_ );

      angle_ *= PI / 180.;
    }
  }
}

//-----------------------------------------------------------------------
//   GroupInputModule::NGroup_::makeGroup
//-----------------------------------------------------------------------

void GroupInputModule::NGroup_::makeGroup

  ( const Properties&  globdat )

{
  Assignable<NodeSet>      nodes;
  Assignable<ElementSet>   elems;

  nodes = NodeSet::get  ( globdat, "NGroup_::makeGroup" );
  elems = ElementSet::find ( globdat );

  findCandidates_ ( elems, globdat );
  findNodes_      ( nodes );
  store_          ( nodes, globdat );
}


//-----------------------------------------------------------------------
//   GroupInputModule::NGroup_::findCandidates_
//-----------------------------------------------------------------------

void GroupInputModule::NGroup_::findCandidates_
 
  ( const ElementSet& elems,
    const Properties& globdat )

{
  // store all candidate node numbers in inodes_
  // (restrict the candidate nodes if restrictToGroup is set)
  
  if ( restrictTo_.size() == 0 )
  {
    // default: use all 

    IdxVector iall ( iarray ( elems.getNodes().size() ) );
    inodes_. ref ( iall );
  }
  else
  {
    ArrayBuffer<idx_t> nodebuf;

    for ( idx_t igr = 0; igr < restrictTo_.size(); ++igr )
    {
      // get node numbers from this ElemGroup

      ElementGroup egroup ( ElementGroup::get 
        ( restrictTo_[igr], elems, globdat, "NGroup_::findCandidates" ) );

      IdxVector ielems = egroup.getIndices();

      if ( restrictPartial_.size() == 0 )
      {
        IdxVector inodes = elems.getNodesOf ( ielems );

        nodebuf.pushBack ( inodes.begin(), inodes.end() );
      }
      else
      {
        IdxVector inodes ( elems.maxElemNodeCountOf ( ielems ) );

        for ( idx_t ie = 0; ie < ielems.size(); ++ie )
        {
          elems.getElemNodes ( inodes, ielems[ie] );

          for ( idx_t ii = 0; ii < restrictPartial_.size(); ++ii )
          {
            nodebuf.pushBack ( inodes[restrictPartial_[ii]] );
          }
        }
      }
    }
    inodes_.ref ( makeUnique ( nodebuf.toArray() ) );
  }
}

//-----------------------------------------------------------------------
//   GroupInputModule::NGroup_::findNodes_
//-----------------------------------------------------------------------

void GroupInputModule::NGroup_::findNodes_
 
  ( const NodeSet&     nodes )

{
  Vector              xcoords ( nodes.size() );
  ArrayBuffer<idx_t>  nodebuf;
  Vector              coords  ( rank_ );

  bool                include;
  idx_t               nall = inodes_.size();

  if ( all_ )
  {
    // add all nodes

    nodebuf.reserve ( nodes.size() );

    for ( idx_t in = 0; in < nall; ++in )
    {
      idx_t inode = inodes_[in];
      nodebuf.pushBack ( inode );
    }
  }
  else if ( testany ( doX_ ) )
  {
    // find coordinate if specified with type

    for ( idx_t ir = 0; ir < nodes.rank(); ++ir )
    {
      if ( testany ( xtype_[ir] == knownTypes ) )
      {
        nodes.getData() -> getXCoords ( xcoords, ir );

        double minX = min ( xcoords );
        double maxX = max ( xcoords );

        if ( xtype_[ir] == "min" )
        {
          xvals_[ir] = minX;
        }
        else if ( xtype_[ir] == "max" )
        {
          xvals_[ir] = maxX;
        }
        else if ( xtype_[ir] == "firstHalf" )
        {
          xbounds_(ir,0) = minX - eps_;
          xbounds_(ir,1) = minX - eps_ + ( maxX - minX ) / 2.;

          doX_[ir] = BOUNDS;
        }
        else if ( xtype_[ir] == "secondHalf" )
        {
          xbounds_(ir,0) = minX + eps_ + ( maxX - minX ) / 2.;
          xbounds_(ir,1) = maxX + eps_;

          doX_[ir] = BOUNDS;
        }
        else if ( xtype_[ir] == "mid" )
        {
          xvals_[ir] = ( maxX + minX ) / 2.;
        }
      }
    }

    // Loop over nodes, node is included unless it violates the criterion
    // specified for one of the dimensions

    for ( idx_t in = 0; in < nall; ++in )
    {
      idx_t inode = inodes_[in];
      include = true;

      nodes.getNodeCoords ( coords, inode );

      // loop over dimensions

      for ( idx_t ir = 0; ir < nodes.rank(); ++ ir )
      {
        if ( doX_[ir] == VALUE )
        {
          // check if coordinate is close to one of specified values

          bool thisx = false;

          for ( idx_t ival = 0; ival < xvals_[ir].size(); ++ival )
          {
            thisx |= jem::numeric::abs( xvals_[ir][ival] - coords[ir] ) < eps_ ;
          }

          include &= thisx;
        }
        else if ( doX_[ir] == BOUNDS )
        {
          // check if coordinate is between specified bounds

          include &= ( coords[ir] > xbounds_(ir,0) &&
              coords[ir] < xbounds_(ir,1) );
        }
      }

      if ( include )
      {
        nodebuf.pushBack ( inode );
      }
    }
  }
  else if ( radius_ > 0. )
  {
    for ( idx_t in = 0; in < nall; ++in )
    {
      idx_t inode = inodes_[in];
      nodes.getNodeCoords ( coords, inode );

      double dx = coords[0] - xvals_[0][0];
      double dy = coords[1] - xvals_[1][0];

      double dist = sqrt ( dx * dx + dy * dy );

      if ( jem::numeric::abs ( dist - radius_ ) < eps_ )
      {
        nodebuf.pushBack ( inode );
      }
    }
  }
  else
  {
    // find nodes specified with x0, y0, angle

    for ( idx_t in = 0; in < nall; ++ in )
    {
      idx_t inode = inodes_[in];
      nodes.getNodeCoords ( coords, inode );

      double dx = coords[0] - xvals_[0][0];
      double dy = coords[1] - xvals_[1][0];

      double angle = ::atan2(dy,dx);
      double dist  = dx * dx + dy * dy ; 

      if ( jem::numeric::abs( angle - angle_ ) < eps_ || 
           jem::numeric::abs( angle - angle_ - PI ) < eps_ ||
           dist < eps_ )
      {
        nodebuf.pushBack ( inode );
      }
    }
  }
  // overwrite inodes_ to only store selected nodes

  inodes_.ref ( nodebuf.toArray() );
}

//-----------------------------------------------------------------------
//   GroupInputModule::NGroup_::store_
//-----------------------------------------------------------------------

void GroupInputModule::NGroup_::store_
 
  ( const NodeSet&    nodes,
    const Properties&  globdat )

{
  Assignable<NodeGroup> group;

  group = newNodeGroup ( inodes_, nodes );

  group.store ( myName_, globdat );

  System::out() << " ...Created NodeGroup `" << myName_ << 
    "' with " << inodes_.size() << " nodes.\n";
  
  if ( inodes_.size() < 1 )
  {
    System::warn() << "NodeGroup `" << myName_ << "' is empty.\n";
  }
}

//=======================================================================
//   class GroupInputModule::EGroup_
//=======================================================================

// ------------------------------------------------------------------------
//   static members
// ------------------------------------------------------------------------

const char*  GroupInputModule::EGroup_::X_NAMES[3] = {"x","y","z"};

//-----------------------------------------------------------------------
//   GroupInputModule::EGroup_::constructur,destructor
//-----------------------------------------------------------------------

GroupInputModule::EGroup_::EGroup_ 

  ( ) : Object ( )

{
  doX_     .resize ( 3 );

  xvals_[0].resize ( 1 );
  xvals_[1].resize ( 1 );
  xvals_[2].resize ( 1 );

  xbounds_ .resize ( 3, 2 );  // ( x/y/z , min/max )

  rank_   = 3;
  eps_    = 1.e-5;

  knownTypes.resize( 5 );

  knownTypes[0] = "min";
  knownTypes[1] = "max";
  knownTypes[2] = "firstHalf";
  knownTypes[3] = "secondHalf"; 
  knownTypes[4] = "mid";

  completely_ = false;
  parent_ = "all";
}

GroupInputModule::EGroup_::~EGroup_ ()
{}

//-----------------------------------------------------------------------
//   GroupInputModule::EGroup_::configure
//-----------------------------------------------------------------------

void GroupInputModule::EGroup_::configure
 
  ( const Properties&  conf,
    const Properties&  props,
    const String&      name )

{
  rank_ = 3;
  all_  = false;

  String  xtypename; 
  Vector  bounds;

  // get properties

  myName_ = name;

  Properties  myConf  = conf .makeProps ( myName_ );
  Properties  myProps = props. getProps ( myName_ );

  // set elems

  myProps.find ( parent_, "fromGroup" );
  myConf.set ( "fromGroup", parent_ );

  if ( myProps.find ( ielems_, "elems" ) )
  {
    myConf.set ( "elems", ielems_ );
    doX_ = NONE;
    return;
  }

  myProps.find ( all_, "all" );
  myConf.set   ( "all", all_ );

  if ( all_ )
  {
    return;
  }

  // set tolerance

  myProps.find( eps_, "eps" );
  myConf.set  ( "eps", eps_ );

  // find coordinate types or values

  xtype_ = "";

  for ( idx_t ir = 0; ir < rank_ ; ++ir )
  {
    String thiss = X_NAMES[ir];

    String xtypename ( thiss + String("type") );
    String xvalname  ( thiss + "val"  );
    String xvalsname ( thiss + "vals" );
    String xinname   ( thiss + "bounds" );
    String xcomplete ( thiss + "complete" );

    doX_[ir] = VALUE;

    if ( myProps.find ( xtype_[ir], xtypename ) )
    {
      myConf.set ( xtypename, xtype_[ir]);

      if ( ! testany ( knownTypes == xtype_[ir] ) )
      {
        throw Error ( JEM_FUNC, "Unknown xtype: " + xtype_[ir] );
      }
    }
    else if ( myProps.find ( xvals_[ir][0], xvalname ) )
    {
      myConf.set ( xvalname, xvals_[ir][0] );
    }
    else if ( myProps.find ( xvals_[ir], xvalsname ) )
    {
      myConf.set ( xvalsname, xvals_[ir] );
    }
    else if ( myProps.find ( bounds, xinname ) )
    {
      myConf.set ( xinname, bounds );

      if ( bounds.size() == 2 )
      {
        xbounds_(ir,0) = bounds[0] - eps_;
        xbounds_(ir,1) = bounds[1] + eps_;

        doX_[ir] = BOUNDS;

        myProps.find ( completely_[ir], xcomplete );
        myConf.set ( xcomplete, completely_[ir] );
      }
      else
      {
        throw Error ( JEM_FUNC, xinname + " does not have length 2!" );
      }
    }
    else
    {
      doX_[ir] = NONE;
    }
  }
}

//-----------------------------------------------------------------------
//   GroupInputModule::EGroup_::makeGroup
//-----------------------------------------------------------------------

void GroupInputModule::EGroup_::makeGroup

  ( const Properties&  globdat )

{
  Assignable<ElementSet>   elems;
  elems = ElementSet::get ( globdat, "EGroup_::makeGroup" );

  findElems_ ( elems, globdat );
  store_     ( elems, globdat );
}


//-----------------------------------------------------------------------
//   GroupInputModule::EGroup_::findElems_
//-----------------------------------------------------------------------

void GroupInputModule::EGroup_::findElems_
 
  ( const ElementSet&     elems,
    const Properties&     globdat )

{
  NodeSet             nodes   ( elems.getNodes() );
  idx_t  nodeCount  = elems.maxElemNodeCount();
  IdxVector           inodes  ( nodeCount    );
  Vector              xcoords ( elems.size() );
  IdxVector           ieparent;
  ArrayBuffer<idx_t>  elembuf;
  Matrix              coords  ( nodes.rank(), nodeCount);
  bool                include;

  if ( all_ )
  {
    // add all elems

    elembuf.reserve ( elems.size() );

    for ( idx_t ie = 0; ie < elems.size(); ++ie )
    {
      elembuf.pushBack ( ie );
    }
  }
  else if ( testany ( doX_ ) )
  {
    // find coordinate if specified with type

    for ( idx_t ir = 0; ir < nodes.rank(); ++ir )
    {
      if ( testany ( xtype_[ir] == knownTypes ) )
      {
        nodes.getData() -> getXCoords ( xcoords, ir );

        double minX = min ( xcoords );
        double maxX = max ( xcoords );

        if ( xtype_[ir] == "min" )
        {
          xvals_[ir] = minX;
        }
        else if ( xtype_[ir] == "max" )
        {
          xvals_[ir] = maxX;
        }
        else if ( xtype_[ir] == "firstHalf" )
        {
          xbounds_(ir,0) = minX - eps_;
          xbounds_(ir,1) = minX - eps_ + ( maxX - minX ) / 2.;

          doX_[ir] = BOUNDS;
        }
        else if ( xtype_[ir] == "secondHalf" )
        {
          xbounds_(ir,0) = minX + eps_ + ( maxX - minX ) / 2.;
          xbounds_(ir,1) = maxX + eps_;

          doX_[ir] = BOUNDS;
        }
        else if ( xtype_[ir] == "mid" )
        {
          xvals_[ir] = ( maxX + minX ) / 2.;
        }
      }
    }

    // Loop over elems, elem is included unless it violates the criterion
    // specified for one of the dimensions

    if ( parent_ == "all" )
    {
      ieparent.ref ( IdxVector ( iarray ( elems.size() ) ) );
    }
    else
    {
      ElementGroup egroup ( ElementGroup::get 
          ( parent_, elems, globdat, 
            "GroupInputModule search for parent" ) );
      ieparent.ref ( egroup.getIndices() );
    }

    for ( idx_t ie = 0; ie < ieparent.size(); ++ie )
    {
      include = true;

      idx_t ielem = ieparent[ie];

      elems.getElemNodes ( inodes, ielem );
      nodes.getSomeCoords ( coords, inodes );

      // loop over dimensions

      for ( idx_t ir = 0; ir < nodes.rank(); ++ ir )
      {
        double ixmin = min(coords(ir,ALL));
        double ixmax = max(coords(ir,ALL));

        if ( doX_[ir] == VALUE )
        {
          // check if one of specified values is inside element

          bool thisx = false;

          for ( idx_t ival = 0; ival < xvals_[ir].size(); ++ival )
          {
            thisx |= ( ixmin <= xvals_[ir][ival] && 
                       ixmax >= xvals_[ir][ival] );
          }
          include &= thisx;
        }
        else if ( doX_[ir] == BOUNDS )
        {
          if ( completely_[ir] )
          {
            // check if element is completely between specified bounds

            include &= ( ixmin > xbounds_(ir,0) &&
                         ixmax < xbounds_(ir,1) );
          }
          else
          {
            // check if element is partially between specified bounds

            include &= ( ixmin < xbounds_(ir,1) &&
                         ixmax > xbounds_(ir,0) );
          }
        }
      }

      if ( include )
      {
        elembuf.pushBack ( ielem );
      }
    }
  }
  ielems_.ref ( elembuf.toArray() );
}

//-----------------------------------------------------------------------
//   GroupInputModule::EGroup_::store_
//-----------------------------------------------------------------------

void GroupInputModule::EGroup_::store_
 
  ( const ElementSet&  elems,
    const Properties&  globdat )

{
  Assignable<ElementGroup> group;

  group = newElementGroup ( ielems_, elems );

  group.store ( myName_, globdat );

  System::out() << " ...Created ElementGroup `" << myName_ << 
    "' with " << ielems_.size() << " elements.\n";
  
  if ( ielems_.size() < 1 )
  {
    System::warn() << "ElementGroup `" << myName_ << "' is empty.\n";
  }
}

//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   declareGroupInputModule
//-----------------------------------------------------------------------

void declareGroupInputModule ()
{
  using jive::app::ModuleFactory;

  ModuleFactory::declare ( GroupInputModule::TYPE_NAME,
                         & GroupInputModule::makeNew );
}
