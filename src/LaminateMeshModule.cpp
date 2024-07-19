#include <jem/base/array/utilities.h>
#include <jem/base/array/operators.h>
#include <jem/base/array/select.h>
#include <jem/base/Exception.h>
#include <jem/base/IllegalInputException.h>
#include <jem/base/System.h>
#include <jem/io/PrintWriter.h>
#include <jem/io/FileWriter.h>
#include <jem/util/ArrayBuffer.h>
#include <jem/util/Properties.h>
#include <jem/util/Dictionary.h>
#include <jive/app/ModuleFactory.h>
#include <jive/fem/NodeGroup.h>
#include <jive/util/DofSpace.h>
#include <jive/util/Globdat.h>
#include <jive/util/XItemGroup.h>
#include <jive/util/ItemSet.h>
#include <jive/util/ConstraintsParser.h>

#include "LaminateMeshModule.h"

using jem::dynamicCast;
using jem::IllegalInputException;
using jem::io::endl;
using jem::io::FileWriter;
using jem::io::PrintWriter;
using jem::util::ArrayBuffer;
using jem::util::Dict;
using jem::util::DictEnum;
using jive::util::Globdat;
using jive::IdxVector;
using jive::IdxMatrix;
using jive::Matrix;
using jive::fem::toXElementSet;
using jive::fem::toXNodeSet;
using jive::fem::newXNodeSet;
using jive::fem::NodeGroup;
using jive::util::Constraints;
using jive::util::DofSpace;
using jive::util::ItemGroup;
using jive::util::XItemGroup;
using jive::util::ItemSet;


//=======================================================================
//   class LaminateMeshModule
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------

const char* LaminateMeshModule::DOF_NAMES[3]    = {"dx","dy","dz"};
const char* LaminateMeshModule::TYPE_NAME       = "LaminateMesh";
const char* LaminateMeshModule::N_LAYERS        = "nLayers";
const char* LaminateMeshModule::DIM             = "dim";
const char* LaminateMeshModule::THICKNESS       = "thickness";
const char* LaminateMeshModule::INTERFACES      = "interfaces";
const char* LaminateMeshModule::LAYER_NAMES     = "layerNames";
const char* LaminateMeshModule::INTERFACE_NAMES = "interfaceNames";
const char* LaminateMeshModule::NODE_GROUPS     = "nodeGroups";
const char* LaminateMeshModule::TIE_GROUPS      = "tieGroups";
const char* LaminateMeshModule::SKIP_NGROUPS    = "skipNGroups";
const char* LaminateMeshModule::SKIP_EGROUPS    = "skipEGroups";
const char* LaminateMeshModule::ELAS_EGROUPS    = "elasEGroups";
const char* LaminateMeshModule::NONE            = "none";
const char* LaminateMeshModule::NORMAL          = "normal";
const char* LaminateMeshModule::PERIODIC        = "periodic";
const char* LaminateMeshModule::TIE_ALL         = "tieAll";
const char* LaminateMeshModule::IELEMS_N        = "var.lamMsh.ielemsN";
const char* LaminateMeshModule::IELEMS_E        = "var.lamMsh.ielemsE";
const char* LaminateMeshModule::TRANSITION      = "transition";
const char* LaminateMeshModule::THICK_DIRECTION = "thickDir";
const char* LaminateMeshModule::WRITE_MESH      = "filename";

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------

LaminateMeshModule::LaminateMeshModule

  ( const String&  name ) :

      Super   ( name   )

{
  active_      = false;
  tieAll_      = false;
  interfaces_  = NORMAL;
  nElGroups_   = 0;
  rank_        = 0;
  numNodes_    = 0;
  numElems_    = 0;
  dz_          = 0.;
  doTieGroups_ = false;
  groupInput_  = newInstance<GroupInputModule> ( name );
  ofilename_   = "";
}

LaminateMeshModule::~LaminateMeshModule ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


Module::Status LaminateMeshModule::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  configure_ ( conf, props, globdat );

  if ( active_ )
  {
    System::out() << "Mesh extrusion with LaminateMeshModule \n";

    prepareMesh_ ( globdat );

    if ( rank_ == 3 )
    {
      createMesh3_ ( globdat );

      if ( doPermutate_ )
      {
        permutate3_ ( globdat );
      }
    }
    else
    {
      createMesh_ ( globdat );
    }

    if ( ! tieAll_ ) 
    {
      updateConstraints_ ( globdat );

      updateNodeGroups_ ( globdat );
    }
    groupInput_->init ( conf, props, globdat );

    doTieGroups_ &= tieGroupsVert_ ( globdat );

    if ( ofilename_.size() > 0 )
    {
      writeToFile_ ( ofilename_, globdat );
    }
  }

  // return OK when still need to tie nodegroups

  return doTieGroups_ ? OK : DONE;
}

//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------

Module::Status LaminateMeshModule::run ( const Properties& globdat )

{
  // Keep trying to tie the specified nodegroups until a DofSpace is found
  // Generally, that will be in the first 'run' 

  doTieGroups_ &= tieGroupsVert_ ( globdat );

  return doTieGroups_ ? OK : DONE;
}

//-----------------------------------------------------------------------
//   shutdown
//-----------------------------------------------------------------------


void LaminateMeshModule::shutdown ( const Properties& globdat )
{
}


//-----------------------------------------------------------------------
//   configure_
//-----------------------------------------------------------------------


void LaminateMeshModule::configure_

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  Properties  myConf  = conf .makeProps ( myName_ );
  Properties  myProps = props.findProps ( myName_ );

  // number of layers (if not specified: module inactive)

  if ( ! myProps.find( nLayers_, N_LAYERS ) )
  {
    active_ = false;
    myConf.set( "active", active_ );
    return;
  }
  myConf.set( N_LAYERS, nLayers_ );

  active_ = true;

  // plane elements or extruded volume elements

  rank_ = 2;

  if ( myProps.find( rank_, DIM ) )
  {
    JEM_ASSERT( rank_ == 2 || rank_ == 3 );
  }

  myConf.set( DIM, rank_ );

  // 3D has twice as many nodes, therefore often multiplications with
  // factor 2 when rank_==3; use private member rankFac_ for that.

  rankFac_ = rank_ - 1;

  // for 3D: get ply thickness 

  if ( rank_ == 3 )
  {
    myProps.get ( dz_, THICKNESS );
    myConf.set  ( THICKNESS, dz_ );
  }

  // option: generate interface elements between layers

  myProps.find( interfaces_, INTERFACES );

  myConf.set  ( INTERFACES, interfaces_ );

  if      ( interfaces_ == NORMAL )
  {
    nInterf_ = nLayers_ - 1;
  }
  else if ( interfaces_ == PERIODIC )
  {
    nInterf_ = nLayers_;
  }
  else if ( interfaces_ == NONE )
  {
    nInterf_ = 0;
  }
  else if ( interfaces_ == TIE_ALL )
  {
    nInterf_ = 0;
    tieAll_  = true;
  }
  else
  {
    throw IllegalInputException (
      JEM_FUNC,
      String (
        "Invalid interfaces type: `" + interfaces_ + "',\nshould be `" + 
        NORMAL  + "', `" + PERIODIC + "' or `" + NONE + "')\n"
      )
    );
  }

  nElGroups_ = nLayers_ + nInterf_;

  // names that will be given to element groups

  layerNames_    .resize( nLayers_ );
  interfaceNames_.resize( nInterf_ );

  if ( ! myProps.find( layerNames_, LAYER_NAMES ) )
  {
    for ( idx_t i = 0; i < nLayers_; ++i )
    {
      layerNames_[i] = String::format( "layer%i", i );
    }
  }

  if ( ! myProps.find( interfaceNames_, INTERFACE_NAMES ) )
  {
    for ( idx_t i = 0; i < nInterf_; ++i )
    {
      interfaceNames_[i] = String::format( "interface%i", i );
    }
  }

  myConf.set(     LAYER_NAMES,     layerNames_ );
  myConf.set( INTERFACE_NAMES, interfaceNames_ );

  // get element group "all"

  if ( ! myProps.contains ( "elements" ) )
  {
    myProps.set ( "elements", "all" );
  }

  const String context = getContext();

  egroup_ = ElemGroup::get ( myConf, myProps, globdat, context );

  // read names of NodeGroups that are to be created

  myProps.find ( groupNames_, NODE_GROUPS );
  myConf.set   ( NODE_GROUPS, groupNames_ );

  nng_ = groupNames_.size();

  // read names of NodeGroups that are to be tied vertically
  // (for uniform load boundary)
  
  if ( myProps.find ( tieGroups_, TIE_GROUPS ) )
  {
    doTieGroups_ = ( tieGroups_.size() > 0 );
    System::out() << "doTieGroups " << doTieGroups_ << endl;
    myConf.set   ( TIE_GROUPS, tieGroups_ );
  }

  // read names of ElemGroups which should not be duplicated

  myProps.find ( skipEGroups_, SKIP_EGROUPS );
  myConf.set   ( SKIP_EGROUPS, skipEGroups_ );

  // read names of ElemGroups which should be treated elastically by models
  
  myProps.find ( elasticEGroups_, ELAS_EGROUPS );
  myConf.set   ( ELAS_EGROUPS, elasticEGroups_ );

  if ( skipEGroups_.size() > 0 && elasticEGroups_.size() > 0 )
  {
    throw IllegalInputException ( JEM_FUNC,
        "combination of skipEGroups and elasticEGroups not allowed!\n" );
  }

  // read names of NodeGroups which should not be updated
  // (for partial boundary condition)

  myProps.find ( skipNGroups_, SKIP_NGROUPS );
  myConf.set   ( SKIP_NGROUPS, skipNGroups_ );
  
  // add nodeGroup "all" to groups that shouldn't be updated

  idx_t nInput = skipNGroups_.size();
  skipNGroups_.reshape ( nInput + 1 );
  skipNGroups_[nInput] = "all";

  // read permutation option (for orientation of extruded mesh)

  if ( rank_ == 3 )
  {
    thickDir_ = 2; // default: thickness in z-direction
    doPermutate_ = false;

    myProps.find ( thickDir_, THICK_DIRECTION );

    if ( thickDir_ != 2 )
    {
      doPermutate_ = true;

      iperm_.resize ( rank_ );

      for ( idx_t i = 0; i < rank_; ++i )
      {
        iperm_[i] = (i+2-thickDir_)%rank_;

        // thickdir_ = 0 --> iperm_ = [2,0,1]   (x<--z)
        // thickdir_ = 1 --> iperm_ = [1,2,0]   (y<--z)
        // thickdir_ = 2 --> iperm_ = [0,1,2]   (default)
      }
    }
    myConf.set ( THICK_DIRECTION, thickDir_ );
  }

  // read optional filename for writing mesh to file 
  // default: empty string --> no output written

  myProps.find ( ofilename_, WRITE_MESH );
  myConf.set   ( WRITE_MESH, ofilename_ );

}

//-----------------------------------------------------------------------
//   prepareMesh_
//-----------------------------------------------------------------------

void LaminateMeshModule::prepareMesh_

  ( const Properties&   globdat )

{
  // get mesh data (as generated by InputModule)

  elems_ = toXElementSet ( egroup_.getElements() );
  nodes_ = toXNodeSet    ( elems_.getNodes()     );

  JEM_ASSERT( nodes_.rank() == rank_ );

  numElems_ = elems_.size();
  numNodes_ = nodes_.size();

  doElem_.resize ( numElems_ );
  doNode_.resize ( numNodes_ );

  // prepare for skipping groups

  if ( skipEGroups_.size() > 0 )
  {
    prepareSkip_ ( globdat );
  }
  else
  {
    // default: do all elements

    doElem_ = true;
    doNode_ = true;

    ielemsY_ .   resize ( numElems_ );
    ielemsY_ = ( iarray ( numElems_ ) );

    inodesY_ .   resize ( numNodes_ );
    inodesY_ = ( iarray ( numNodes_ ) );

    if ( elasticEGroups_.size() > 0 ) 
    {
      // store element numbers for which elastic behavior is desired

      prepareElastic_ ( globdat );
    }
  }

  // tie all option overrules skipEGroups

  if ( tieAll_ ) doNode_ = false;

  numElemY_ = ielemsY_.size();
  numNodeY_ = inodesY_.size();

  eMatrix_.resize (            nLayers_, numElems_ );
  nMatrix_.resize ( rankFac_ * nLayers_, numNodes_ );

  eMatrix_(0,ALL) = iarray ( numElems_ );
  nMatrix_(0,ALL) = iarray ( numNodes_ );

  nodes_.reserve ( rankFac_ * nLayers_ * numNodes_ );
  elems_.reserve ( nElGroups_ * numElems_ );
}

//-----------------------------------------------------------------------
//   prepareSkip_
//-----------------------------------------------------------------------

void LaminateMeshModule::prepareSkip_

  ( const Properties&   globdat )

{
  ArrayBuffer<idx_t>  ieN;
  ArrayBuffer<idx_t>  ieY;

  IdxVector         ieThis;
  IdxVector         inodesT;

  Ref<Dict>         groups;
  Ref<DictEnum>     e;
  Ref<ItemGroup>    group;
  String            name;

  doElem_.resize ( numElems_ );
  doNode_.resize ( numNodes_ );

  groups = ItemGroup::getFor( elems_.getData(), globdat );

  // collect data in loop over ElemGroups

  for ( e = groups->enumerate(); ! e->atEnd(); e->toNext() )
  {
    group = dynamicCast<ItemGroup> ( e->getValue() );
    name  = e->getKey();

    ieThis.ref ( group->getIndices() );

    if ( name == "all" || name == "none" )
    {
      // do nothing
    }
    else if ( testany ( skipEGroups_ == name ) )
    {
      System::out() << "     " << name << " is skipped\n";

      ieN.pushBack ( ieThis.begin(), ieThis.end() );

      select ( doElem_, ieThis ) = false;
    }
    else
    {
      System::out() << "     " << name << " is extruded\n";

      ieY.pushBack ( ieThis.begin(), ieThis.end() );

      select ( doElem_, ieThis ) = true;
    }
  }

  ielemsY_.ref ( ieY.toArray() );
  inodesY_.ref ( elems_.getUniqueNodesOf ( ielemsY_ ) );

  ielemsN_.ref ( ieN.toArray() );
  inodesN_.ref ( elems_.getUniqueNodesOf ( ielemsN_ ) );

  if ( rank_ == 2 )
  {
    // 2D: double nodes are created on transition line and later tied
    // TODO: why not use the same as for 3D?

    doNode_ = false;

    select ( doNode_, inodesY_ ) = true;

    // find nodes on interface (to be constrained) 

    ArrayBuffer<idx_t>  inC;

    for ( idx_t in = 0; in < inodesY_.size(); ++in )
    {
      idx_t inode = inodesY_[in];

      if ( testany ( inode == inodesN_ ) )
      {
        inC.pushBack ( inode );
      }
    }

    // store nodeGroup for use in tieGroupVert_

    inodesT.ref ( inC.toArray() );

    Assignable<NodeGroup> nGroup;

    nGroup = newNodeGroup ( inodesT, nodes_ );

    nGroup.store ( TRANSITION, globdat );

    tieGroups_.reshape ( tieGroups_.size() + 1 );

    tieGroups_.back() = TRANSITION;

    doTieGroups_ = true;

    System::out() << " ...Created NodeGroup `" << TRANSITION << 
      "' with " << nGroup.size() << " nodes.\n";
  }
  else // if ( rank_ == 3 )
  {
    // 3D: don't make duplicate nodes on trasition line

    doNode_ = true;

    select ( doNode_, inodesN_ ) = false;
  }

  // store in globdat (used in XFEMModel)

  globdat.set ( IELEMS_N, ielemsN_ );

  // check if nodes are specified
  // (gmshInput.doElemGroups must be set to true!)

  if ( ielemsN_.size() + ielemsY_.size() != elems_.size() )
  {
    throw IllegalInputException ( JEM_FUNC, 
        String( "Not all elements are in an ElementGroup, ") +
        String("set gmshInput.doElemGroups to true") );
  }
}
  
//-----------------------------------------------------------------------
//   prepareElastic_
//-----------------------------------------------------------------------

void LaminateMeshModule::prepareElastic_

  ( const Properties&   globdat )

{
  ArrayBuffer<idx_t>  ieN;
  IdxVector         ieThis;

  Ref<Dict>         groups;
  Ref<DictEnum>     e;
  Ref<ItemGroup>    group;
  String            name;

  groups = ItemGroup::getFor( elems_.getData(), globdat );

  // collect data in loop over ElemGroups

  for ( e = groups->enumerate(); ! e->atEnd(); e->toNext() )
  {
    group = dynamicCast<ItemGroup> ( e->getValue() );
    name  = e->getKey();

    ieThis.ref ( group->getIndices() );

    if ( testany ( elasticEGroups_ == name ) )
    {
      System::out() << "     " << name << 
        " will have elastic interfaces\n";

      ieN.pushBack ( ieThis.begin(), ieThis.end() );
    }
  }

  ielemsN_.ref ( ieN.toArray() );

  // store in globdat (used in XFEMModel and NCInterfaceModel)

  globdat.set ( IELEMS_N, ielemsN_ );

  globdat.set ( IELEMS_E, ielemsN_ );
}

//-----------------------------------------------------------------------
//   createMesh_
//-----------------------------------------------------------------------

void LaminateMeshModule::createMesh_

  ( const Properties&   globdat )

{
  JEM_ASSERT ( rank_ == 2 );

  Matrix                coords;
  Assignable<ElemGroup> newGroup;

  idx_t nodeCount = elems_.maxElemNodeCount();

  IdxVector inodes  ( nodeCount );
  IdxVector jnodes  ( nodeCount * 2 );
  IdxVector knodes  ( nodeCount );

  coords.resize ( rank_, numNodes_ );

  nodes_.getCoords ( coords );

  System::out() << " ...Generating " << nLayers_ << " layers of "
    << "solid elements, with element group names \n    "
    << layerNames_ << "\n";

  if ( skipEGroups_.size() > 0 )
  {
    System::out() << "    Duplicating " << numElemY_ << " of " << 
      numElems_ << " elems, and " << numNodeY_ << " of " << 
      numNodes_ << " nodes.\n";
  }
  else
  {
    System::out() << "    Duplicating all elements and nodes\n";
  }

  // generate nodes (only for nodes for which doNode==true)
  
  for ( idx_t iLayer = 1; iLayer < nLayers_; ++iLayer )
  {
    // add node to set

    for ( idx_t in = 0; in < numNodes_; ++in )
    {
      if ( doNode_[in] )
      {
        nMatrix_(iLayer,in) = nodes_.addNode ( coords(ALL,in) );
      }
      else
      {
        nMatrix_(iLayer,in) = in;
      }
    }
  }
  
  // generate additional layers of solid elements (if doElem==true)

  for ( idx_t iLayer = 1; iLayer < nLayers_; ++iLayer )
  {
    for ( idx_t ie = 0; ie < numElems_; ++ie )
    {
      elems_.getElemNodes ( inodes, ie );

      knodes = select ( nMatrix_(iLayer,ALL), inodes );

      if ( doElem_[ie] )
      {
        eMatrix_(iLayer,ie) = elems_.addElement ( knodes );
      }
      else
      {
        eMatrix_(iLayer,ie) = ie;
      }
    }
  }

  // store solid element groups

  for ( idx_t iLayer = 0; iLayer < nLayers_; ++iLayer )
  {
    newGroup = newElementGroup( eMatrix_(iLayer,ALL), elems_ );

    newGroup.store( layerNames_[iLayer], globdat );
  }


  // generate interface elements (if doElem==true)

  if ( nInterf_ > 0 )
  {
    IdxVector jnodes0 ( jnodes [ slice ( 0, nodeCount  ) ] );  // shallow copy
    IdxVector jnodes1 ( jnodes [ slice ( nodeCount, END) ] );

    System::out() << " ...Generating " << nInterf_ << " layers of "
      << "interface elements (" << interfaces_ << "), with element group "
      << "names \n    " << interfaceNames_ << "\n";

    for ( idx_t iLayer = 0; iLayer < nInterf_; ++iLayer )
    {
      IdxVector jelems  ( numElemY_ );

      idx_t jLayer = ( iLayer + 1 ) % nLayers_;

      for ( idx_t ieY = 0; ieY < numElemY_; ++ieY )
      {
        idx_t ie = ielemsY_[ieY];

        idx_t ie0 = eMatrix_ ( iLayer, ie );
        idx_t ie1 = eMatrix_ ( jLayer, ie );

        elems_.getElemNodes ( jnodes0, ie0 );
        elems_.getElemNodes ( jnodes1, ie1 );

        jelems[ieY] = elems_.addElement ( jnodes );
      }

      // store group

      newGroup = newElementGroup( jelems, elems_ );

      newGroup.store( interfaceNames_[iLayer], globdat );
    }
  }
  else
  {
    System::out() << " ...Generating no interface elements.\n";
  }

  nodes_.trimToSize();
  elems_.trimToSize();
}

//-----------------------------------------------------------------------
//   createMesh3_
//-----------------------------------------------------------------------

void LaminateMeshModule::createMesh3_

  ( const Properties&   globdat )

{
  JEM_ASSERT ( rank_ == 3 );

  Assignable<ElemGroup> newGroup;

  idx_t nodeCount = elems_.maxElemNodeCount();

  IdxVector     inodes0;
  IdxVector     jnodes0;
  IdxVector     jnodes1;

  Matrix        coords ( rank_     , numNodes_ );
  IdxMatrix     ielems ( nElGroups_, numElems_ );
  IdxMatrix     jelems ( nInterf_  , numElemY_ );
  IdxVector     inodes ( nodeCount * 2 );
  IdxVector     jnodes ( nodeCount * 2 );

  inodes0.ref   ( inodes [ slice(  0,nodeCount) ] );
  jnodes0.ref   ( jnodes [ slice(  0,nodeCount) ] );
  jnodes1.ref   ( jnodes [ slice(nodeCount,END) ] );

  // get mesh data (as generated by InputModule)

  nodes_.getCoords ( coords );

  System::out() << " ...Generating " << nLayers_ << " layers of "
    << "solid elements, with element group names \n    "
    << layerNames_ << "\n";

  // generate nodes
  // for doNode==true: two nodes per layer
  // for doNode==false: skip duplicate nodes (related to interface)

  for ( idx_t iLayer = 1; iLayer < nLayers_ * 2; ++iLayer )
  {
    // set z-coords

    idx_t iz = ( iLayer + 1 ) / 2;

    bool odd = ( ( iLayer % 2 ) == 1 );  

    coords( 2, ALL ) = iz * dz_;

    // add node to set

    for ( idx_t in = 0; in < numNodes_; ++in )
    {
      if ( doNode_[in] || odd )
      {
        nMatrix_(iLayer,in) = nodes_.addNode ( coords(ALL,in) );
      }
      else
      {
        nMatrix_(iLayer,in) = nMatrix_(iLayer-1,in);
      }
    }
  }

  // adapt first layer of solid elements  
  // (irrespective of doElem)

  for ( idx_t ie = 0; ie < numElems_; ++ie )
  {
    elems_.getElemNodes ( inodes[slice(0,nodeCount)], ie );

    inodes [ slice(nodeCount,END) ] = 
      select ( nMatrix_(1,ALL), inodes[slice(0,nodeCount)] );
      // inodes [ slice(0,nodeCount) ] + numNodes_;

    elems_.setElemNodes ( ie, inodes );
  }

  // generate additional layers of solid elements
  // (irrespective of doElem)

  for ( idx_t iLayer = 1; iLayer < nLayers_; ++iLayer )
  {
    idx_t izNode0 = iLayer  * 2;
    idx_t izNode1 = izNode0 + 1;

    for ( idx_t ie = 0; ie < numElems_; ++ie )
    {
      elems_.getElemNodes ( inodes, ie );
      
      jnodes0 = select ( nMatrix_(izNode0,ALL), inodes0 );

      jnodes1 = select ( nMatrix_(izNode1,ALL), inodes0 );

      elems_.addElement ( jnodes );
    }
  }

  // store solid element groups

  for ( idx_t iLayer = 0; iLayer < nLayers_; ++iLayer )
  {
    ielems( iLayer, ALL ) = iarray( numElems_ ) + iLayer * numElems_;

    newGroup = newElementGroup( ielems(iLayer,ALL), elems_ );

    newGroup.store( layerNames_[iLayer], globdat );
  }

  // generate interface elements (if doElem==true)

  if ( nInterf_ > 0 )
  {
    System::out() << " ...Generating " << nInterf_ << " layers of "
      << "interface elements (" << interfaces_ << "), with element group "
      << "names \n    " << interfaceNames_ << "\n";

    for ( idx_t iLayer = 0; iLayer < nInterf_; ++iLayer )
    {
      idx_t izNode0 = iLayer  * 2 + 1;
      idx_t izNode1 = izNode0 + 1;

      for ( idx_t ieY = 0; ieY < numElemY_; ++ieY )
      {
        idx_t ie = ielemsY_[ieY];

        JEM_ASSERT ( doElem_[ie] );

        elems_.getElemNodes ( inodes, ie );

        jnodes0 = select ( nMatrix_(izNode0,ALL), inodes0 );

        jnodes1 = select ( nMatrix_(izNode1,ALL), inodes0 );

        jelems(iLayer,ieY) = elems_.addElement ( jnodes );
      }

      // store group
      
      newGroup = newElementGroup( jelems(iLayer,ALL), elems_ );

      newGroup.store( interfaceNames_[iLayer], globdat );
    }
  }
  else
  {
    System::out() << " ...Generating no interface elements.\n";
  }
}

//-----------------------------------------------------------------------
//   permutate3_
//-----------------------------------------------------------------------

void LaminateMeshModule::permutate3_

  ( const Properties&   globdat )

{
  JEM_ASSERT ( rank_ == 3 );

  // change ordering of coordinates 

  idx_t nn = nodes_.size();

  Matrix coords ( rank_, nn );
  Matrix reordered ( rank_, nn );

  nodes_.getCoords ( coords );

  reordered = select ( coords, iperm_, ALL );
  nodes_.setCoords ( reordered );
}

//-----------------------------------------------------------------------
//   updateConstraints_
//-----------------------------------------------------------------------


void LaminateMeshModule::updateConstraints_

  ( const Properties&   globdat )

{
  // This function doesn't do anything, except that it gives a warning
  // when constraints have been applied before the mesh is extracted.
  
  using jive::util::ConstraintsParser;

  StringVector            list = ItemSet::listAll ( globdat );

  Ref<ConstraintsParser>  conParser;
  Ref<Constraints>        cons;
  Ref<ItemSet>            items;

  idx_t                   i, n;

  for ( i = 0, n = list.size(); i < n; i++ )
  {
    items = ItemSet::find ( list[i], globdat );

    if ( items == nullptr )
    {
      continue;
    }

    conParser = ConstraintsParser::extract ( items, globdat );

    if ( conParser != nullptr )
    {
      if ( conParser->slaveDofCount() > 0 )
      {
        System::warn() << "Constraints are not copied to other layers\n";
      }
    }
  }
}

//-----------------------------------------------------------------------
//   updateNodeGroups_
//-----------------------------------------------------------------------


void LaminateMeshModule::updateNodeGroups_

  ( const Properties&   globdat )

{
  Ref<Dict>         groups;
  Ref<DictEnum>     e;
  Ref<XItemGroup>   group;
  String            name;

  groups = ItemGroup::getFor( nodes_.getData(), globdat );

  for ( e = groups->enumerate(); ! e->atEnd(); e->toNext() )
  {
    group = dynamicCast<XItemGroup> ( e->getValue() );
    name  = e->getKey();

    if ( ! ( group == nullptr || testany ( skipNGroups_ == name ) ) )
    {
      IdxVector inGr ( group->getIndices() );

      ArrayBuffer<idx_t> jnodeBuf;

      for ( idx_t i = 0; i < inGr.size(); ++i )
      {
        idx_t in = inGr[i];

        for ( idx_t iLayer = 1; iLayer < nMatrix_.size(0); ++iLayer )
        {
          idx_t candidate = nMatrix_(iLayer,in);

          if ( candidate != nMatrix_(iLayer-1,in) )
          {
            jnodeBuf.pushBack ( candidate );
          }
        }
      }
      group->append ( jnodeBuf.toArray() );

      System::out() << " ...Nodegroup `" << name <<
        "' now contains " << group->size() << " nodes.\n";
    }
  }
}

//-----------------------------------------------------------------------
//   tieGroupsVert_
//-----------------------------------------------------------------------

bool LaminateMeshModule::tieGroupsVert_

  ( const Properties&   globdat )

{
  Ref<DofSpace>     dofs = DofSpace::find ( nodes_.getData(), globdat );

  if ( dofs == nullptr ) return true;

  Ref<Constraints>  cons = Constraints::get ( dofs, globdat );

  idx_t             ntype  ( dofs->typeCount() );

  IdxVector         types  ( iarray ( ntype )  );
  IdxVector         idofs  ( ntype             );
  IdxVector         jdofs  ( ntype             );

  Ref<Dict>         groups;
  Ref<DictEnum>     e;
  Ref<ItemGroup>    group;

  groups = ItemGroup::getFor( nodes_.getData(), globdat );

  for ( e = groups->enumerate(); ! e->atEnd(); e->toNext() )
  {
    if ( testany ( e->getKey() == tieGroups_ ) )
    {
      System::out() << " ...Vertically tying nodes from nodeGroup `" 
        << e->getKey() << "'.\n";

      group = dynamicCast<ItemGroup> ( e->getValue() );

      JEM_PRECHECK ( group->size() > 0 );

      IdxVector inodes ( group->getIndices() );

      for ( idx_t in = 0; inodes[in] < numNodes_; ++in )
      {
        idx_t inode = inodes[in];

        dofs->getDofsForItem ( idofs, types, inode );

        for ( idx_t iLayer = 1; iLayer < nLayers_*rankFac_; ++iLayer )
        {
          idx_t jnode = nMatrix_ ( iLayer, inode );

          dofs->getDofsForItem ( jdofs, types, jnode );

          for ( idx_t id = 0; id < types.size(); ++ id )
          {
            try 
            {
              cons->addConstraint ( jdofs[id], idofs[id], 1. );
            }
            catch ( const jem::Exception& ex )
            {
              System::out() << "no constraint added for nodes " <<
                inode << " and " << jnode << " : " << id << endl;
            }
          }
        }
      }
    }
  }
  System::out() << endl;

  return false;
}

//-----------------------------------------------------------------------
//   writeToFile_
//-----------------------------------------------------------------------

void LaminateMeshModule::writeToFile_

  ( const String&      filename,
    const Properties&  globdat ) const

{
  Ref<PrintWriter> out = newInstance<PrintWriter> ( 
                         newInstance<FileWriter> ( filename ) );

  *out << "Nodes\n" << nodes_.size() << " " << rank_ << endl;
  print ( *out, nodes_ );
  *out << "\n\nElements\n" << elems_.size() << " " << 
    elems_.maxElemNodeCount() << endl;
  print ( *out, elems_ );
}

//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------

Ref<Module>  LaminateMeshModule::makeNew

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
//   declareLaminateMeshModule
//-----------------------------------------------------------------------

void declareLaminateMeshModule ()
{
  using jive::app::ModuleFactory;

  ModuleFactory::declare ( LaminateMeshModule::TYPE_NAME,
                         & LaminateMeshModule::makeNew );
}
