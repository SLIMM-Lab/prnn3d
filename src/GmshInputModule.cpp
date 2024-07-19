#include <jem/base/array/utilities.h>
#include <jem/base/array/operators.h>
#include <jem/base/array/select.h>
#include <jem/base/Error.h>
#include <jem/base/System.h>
#include <jem/numeric/utilities.h>
#include <jem/util/SparseArray.h>
#include <jem/util/ArrayBuffer.h>
#include <jive/app/ModuleFactory.h>
#include <jive/fem/NodeGroup.h>

#include "GmshInputModule.h"

using jem::io::endl;
using jem::util::SparseArray;
using jem::util::ArrayBuffer;
using jive::Vector;
using jive::Matrix;
using jive::fem::NodeGroup;
using jive::fem::NodeSet;
using jive::fem::ElementSet;
using jive::fem::newXElementSet;
using jive::fem::newXNodeSet;
using jive::fem::toXElementSet;
using jive::fem::toXNodeSet;

//=======================================================================
//   class GmshInputModule
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------

const char* GmshInputModule::TYPE_NAME       = "GmshInput";
const char* GmshInputModule::DIM             = "dim";
const char* GmshInputModule::FILE_NAME       = "file";
const char* GmshInputModule::DO_ELEM_GROUPS  = "doElemGroups";
const char* GmshInputModule::MAKE_LOW_ORDER  = "lowOrderCopy";
const char* GmshInputModule::GMSH_ELEMS      = "gmshElems";
const char* GmshInputModule::LOW_ORDER_ELEMS = "lowOrderElems";

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------

GmshInputModule::GmshInputModule

  ( const String&  name ) :

      Super   ( name   )

{
  rank_         = 0;
  numNodes_     = 0;
  numElems_     = 0;
  doElemGroups_ = false;
  lowOrderCopy_ = false;
  groupInput_   = newInstance<GroupInputModule> ( name );
}

GmshInputModule::~GmshInputModule ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


Module::Status GmshInputModule::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  configure_ ( conf, props, globdat );

  System::out() << "Reading input with " << myName_
    << " from file " << fileName_ << "\n";

  openFile_ ();

  initSets_ ( globdat );

  readMesh_ ( globdat );

  closeFile_ ();

  checkHighOrderMesh_ ();

  groupInput_->init ( conf, props, globdat );

  if ( lowOrderCopy_ )
  {
    makeLowOrderCopy_ ( globdat );
  }
  return DONE;
}

//-----------------------------------------------------------------------
//   configure_
//-----------------------------------------------------------------------


void GmshInputModule::configure_

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  Properties  myConf  = conf .makeProps ( myName_ );
  Properties  myProps = props.findProps ( myName_ );

  // 2D only (for 3D: combine with LaminateMeshModule)

  rank_ = 2;
  myProps.find ( rank_, DIM  );
  myConf.set   ( DIM , rank_ );

  // filename (if not specified: module inactive)

  myProps.get ( fileName_, FILE_NAME );
  myConf.set  ( FILE_NAME, fileName_ );

  // doElemGroups: make element groups based on physical surfaces

  myProps.find ( doElemGroups_, DO_ELEM_GROUPS );
  myConf.set   ( DO_ELEM_GROUPS, doElemGroups_ );

  // lowOrderCopy: make low order elements as well

  myProps.find ( lowOrderCopy_, MAKE_LOW_ORDER );
  myConf.set   ( MAKE_LOW_ORDER, lowOrderCopy_ );
}

//-----------------------------------------------------------------------
//   initSets_
//-----------------------------------------------------------------------

void GmshInputModule::initSets_

  ( const Properties&   globdat )

{
  // find or create NodeSet

  nodes_ = toXNodeSet ( NodeSet::find ( globdat ) );

  if ( nodes_ == nullptr )
  {
    // default name and storage mode!

    nodes_ = newXNodeSet ( );
  }

  nodes_.clear();

  // find or create ElementSet

  elems_ = toXElementSet ( ElementSet::find ( globdat ) );

  if ( elems_ == nullptr )
  {
    elems_ = newXElementSet ( nodes_ );
  }

  elems_.clear();
}

//-----------------------------------------------------------------------
//   readMesh_
//-----------------------------------------------------------------------

void GmshInputModule::readMesh_

  ( const Properties&   globdat )

{
  idx_t      iel;
  idx_t      igroup;
  idx_t      eltype;
  idx_t      nrtags;
  idx_t      nn;
  idx_t      nodeID;

  IdxVector  groupElems;
  IdxVector  inodes;
  Vector     coords( 3 );
  String     line;
  String     groupName;

  SparseArray< ArrayBuffer<idx_t> > egroups;
  Assignable < ElemGroup          > newGroup;
  Assignable < NodeGroup          > newNodeG;

  IdxVector  perm ( 20 ); // max # of nodes per element
    
  // jump over the first four lines of the input file

  iFile_->readLine();
  iFile_->readLine();
  iFile_->readLine();
  line = iFile_->readLine().stripWhite();

  if ( line != "$Nodes" )
  {
    System::out() << line.size() << line << endl;
    throw Error ( JEM_FUNC, "Unknown file format. "
      "Are any nodes specified in this Gmsh-file?" );
  }

  // read the number of nodes

  numNodes_ = iFile_->parseInt();

  nodes_.reserve( numNodes_ );

  System::out() << " ...Adding " << numNodes_ << " nodes.\n";
 
  // read and store the nodes

  for ( idx_t in = 0; in < numNodes_; ++in)
  {
    nodeID = iFile_->parseInt();;

    coords[0] = iFile_->parseFloat();
    coords[1] = iFile_->parseFloat();
    coords[2] = iFile_->parseFloat();

    nodes_.addNode( nodeID, coords[slice(BEGIN,rank_)] );
  }

  nodes_.store( globdat );

  // read current line and jump over next two

  iFile_->readLine();
  iFile_->readLine();
  line = iFile_->readLine().stripWhite();

  if ( line != "$Elements" )
  {
    throw Error ( JEM_FUNC, "Unknown file format. "
      "Are any elements specified in this Gmsh-file?" );
  }

  // read the number of elements

  numElems_ = iFile_->parseInt();

  elems_.reserve( numElems_ );

  System::out() << " ...Adding " << numElems_ << " elements.\n";

  for( idx_t ie = 0; ie < numElems_; ++ie)
  {
    iel    = iFile_->parseInt();
    eltype = iFile_->parseInt();
    nrtags = iFile_->parseInt();
    igroup = iFile_->parseInt();

    for ( idx_t j = 1; j < nrtags; ++j )
    {
             iFile_->parseInt();
    }

    switch(eltype)
    {
      case  1:   nn = 2;   break;
      case  2:   nn = 3;   break;
      case  3:   nn = 4;   break;
      case  4:   nn = 4;   break;
      case  8:   nn = 3;   break;
      case  9:   nn = 6;   break;
      case 10:   nn = 9;   break;

      default:

      throw Error ( JEM_FUNC, String::format ( "Unknown element type."
            " ( iel = %i, eltype = %i )", iel, eltype ) );
    }

    // get permutation vector (for quadratic elements)

    if ( eltype == 9 )
    {
      perm[0] = 0;
      perm[1] = 2;
      perm[2] = 4;
      perm[3] = 1;
      perm[4] = 3;
      perm[5] = 5;
    }
    else if ( eltype == 10 )
    {
      perm[0] = 0;
      perm[1] = 2;
      perm[2] = 4;
      perm[3] = 6;
      perm[4] = 1;
      perm[5] = 3;
      perm[6] = 5;
      perm[7] = 7;
      perm[8] = 8;
    }
    else
    {
      perm[slice(0,nn)] = iarray ( nn );
    }

    // read nodes
    
    inodes.resize( nn );

    for ( idx_t in = 0; in < nn; ++ in )
    {
      inodes[ perm[in] ] = nodes_.findNode( iFile_->parseInt() );
    }
    elems_.addElement ( inodes );

    // store index for ElementGroup

    if ( doElemGroups_ )
    {
      egroups[igroup].pushBack ( ie );
    }
  }

  // store ElementSet

  elems_.store( globdat );

  // create ElementGroups

  if ( doElemGroups_ )
  {
    SparseArray< ArrayBuffer<idx_t> >::Iterator  it; 
    SparseArray< ArrayBuffer<idx_t> >::Iterator end; 

    for ( it = egroups.begin(); it != egroups.end(); ++it )
    {
      groupElems.ref ( it->value.toArray() );

      newGroup  = newElementGroup( groupElems, elems_ );

      groupName = "gmsh" + String(it->index[0]);

      newGroup.store( groupName, globdat );

      System::out() << " ... stored ElementGroup " << groupName << 
        " with " << groupElems.size() << " elements.\n";
    }
  }

  // create ElementGroup for all input elements

  groupElems.ref ( IdxVector ( iarray ( numElems_ ) ) );

  newGroup = newElementGroup ( groupElems, elems_  );
  newGroup . store           ( GMSH_ELEMS, globdat );

  groupElems.ref ( IdxVector ( iarray ( numNodes_ ) ) );
}

//-----------------------------------------------------------------------
//   openFile
//-----------------------------------------------------------------------

void         GmshInputModule::openFile_ ()

{
  iFile_ = newInstance<FileReader> ( fileName_ );
}

//-----------------------------------------------------------------------
//   closeFile
//-----------------------------------------------------------------------

void         GmshInputModule::closeFile_ ()

{
  iFile_->close();
}


//-----------------------------------------------------------------------
//   checkHighOrderMesh
//-----------------------------------------------------------------------

void         GmshInputModule::checkHighOrderMesh_ ()

{
  // when ( maxx > tiny ) or ( maxy > tiny ) there are curved element 
  // boundaries. 

  idx_t nn = elems_.maxElemNodeCount( ); 

  IdxVector inodes( nn );
  Matrix    coords( rank_, nn );
  Vector    c0;
  Vector    c1;
  Vector    cm;
  Vector    dd ( rank_ );

  double    maxx = 0.;
  double    maxy = 0.;

  if ( nn == 6 || nn == 8 )
  {
    idx_t nb = nn / 2;

    for ( idx_t ie = 0; ie < numElems_; ++ie )
    {
      elems_.getElemNodes  ( inodes, ie );
      nodes_.getSomeCoords ( coords, inodes );

      for ( idx_t ib = 0; ib < nb; ++ib )
      {
        c0.ref ( coords ( ALL, ib * 2 ) );
        c1.ref ( coords ( ALL, ( ib * 2 + 2 ) % nn ) );
        cm.ref ( coords ( ALL, ib * 2 + 1) );

        dd = (c1-cm) - (cm-c0); 

        maxx = max ( maxx, jem::numeric::abs(dd[0]) );
        maxy = max ( maxy, jem::numeric::abs(dd[1]) );
      }
    }
  }

  if ( max ( maxx, maxy ) > 1.e-10 )
  {
    System::warn() << "The mesh seems to have curved element boundaries.\n "
      << "Consider setting Mesh.SecondOrderLinear = 1 in .geo-file" << endl;
  }
}

//-----------------------------------------------------------------------
//   makeLowOrderCopy_
//-----------------------------------------------------------------------

void GmshInputModule::makeLowOrderCopy_

  ( const Properties&   globdat )

{
  System::out() << " ...Making low order copy." << endl;
  // make duplicate low order elements and store as an elementgroup

  IdxVector               lowElems ( numElems_ );
  Assignable<ElemGroup>   newGroup;

  idx_t nn0 = elems_.maxElemNodeCount ( ); 

  IdxVector inodes ( nn0 );

  for ( idx_t ie = 0; ie < numElems_; ++ie )
  {
    idx_t nn = elems_.getElemNodes ( inodes, ie );

    if ( nn != 6 && nn != 8 )
    {
      throw Error ( JEM_FUNC, MAKE_LOW_ORDER + String::format 
          ( " not implemented for nn = %i!", nn ) ); 
    }

    // make new element with only the corner nodes 
    // (i.e. the nodes with even numbers)

    elems_.addElement ( inodes[slice(0,nn,2)] );
  }

  JEM_ASSERT ( elems_.size() == 2*numElems_ );

  lowElems = iarray ( slice ( numElems_, 2*numElems_ ) );

  newGroup = newElementGroup ( lowElems, elems_  );
  newGroup . store           ( LOW_ORDER_ELEMS, globdat );
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------

Ref<Module>  GmshInputModule::makeNew

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
//   declareGmshInputModule
//-----------------------------------------------------------------------

void declareGmshInputModule ()
{
  using jive::app::ModuleFactory;

  ModuleFactory::declare ( GmshInputModule::TYPE_NAME,
                         & GmshInputModule::makeNew );
}
