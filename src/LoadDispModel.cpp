//=======================================================================
//
// Model that prints load-displacement data for nodegroup to file:
// The sum of the nodal loads and the average of the nodal displacements
//
// Adapted from: MonitorModel, 
// FPM, 29-1-2008
// 
// Data is computed in GET_MATRIX0 and GET_INT_VECTOR
//         and optionally written to file in COMMIT
//         (rather using the SampleModule to write to file recommended)
//
//=======================================================================

#include <jem/io/PrintWriter.h>
#include <jem/io/FileWriter.h>
#include <jem/base/array/select.h>
#include <jem/base/array/utilities.h>
#include <jem/base/System.h>
#include <jem/util/Properties.h>
#include <jive/util/Globdat.h>
#include <jive/util/ItemSet.h>
#include <jive/util/DofSpace.h>
#include <jive/util/Assignable.h>
#include <jive/model/Actions.h>
#include <jive/model/Model.h>
#include <jive/model/ModelFactory.h>
#include <jive/model/StateVector.h>
#include <jive/fem/ElementSet.h>
#include <jive/fem/NodeGroup.h>

#include "models.h"


using namespace jem;
using jem::io::endl;

using jem::util::Properties;
using jem::io::PrintWriter;
using jem::io::FileWriter;
using jive::Vector;
using jive::IdxVector;
using jive::IdxMatrix;
using jive::StringVector;
using jive::util::DofSpace;
using jive::util::Assignable;
using jive::model::Model;
using jive::fem::NodeSet;
using jive::fem::NodeGroup;
using jive::fem::ElementSet;


//=======================================================================
//   class LoadDispModel
//=======================================================================

class LoadDispModel : public Model
{
 public:

  static const char*        NODES;
  static const char*        GROUP;

                            LoadDispModel

    ( const String&           name,
      const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

  virtual void              configure

    ( const Properties&       props,
      const Properties&       globdat );

  virtual void              getConfig

    ( const Properties&       conf,
      const Properties&       globdat )      const;

  virtual bool              takeAction

    ( const String&           action,
      const Properties&       params,
      const Properties&       globdat );


 protected:

  virtual                  ~LoadDispModel  ();


 private:

  Ref<DofSpace>             dofs_;
  Assignable<ElementSet>    elems_;
  Assignable<NodeSet>       nodes_;

  IdxVector                 inodes_;
  IdxMatrix                 idofs_;
  idx_t                     nn_;
  idx_t                     ndofTypes_;
};


//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  LoadDispModel::NODES = "nodes";
const char*  LoadDispModel::GROUP = "group";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


LoadDispModel::LoadDispModel

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat ) :

    Model ( name )

{
  const String       context = getContext ();

  elems_ = ElementSet::get ( globdat, context );
  nodes_ = elems_.getNodes ();
  dofs_  = DofSpace::get ( nodes_.getData(), globdat, context );
  nn_ = -1;
  ndofTypes_ = dofs_->typeCount();
}


LoadDispModel::~LoadDispModel ()
{}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void LoadDispModel::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  using jive::util::Globdat;

  Properties    myProps = props.findProps ( myName_ );

  const String  context = getContext ();

  IdxVector     nodeIDs;
  IdxVector     jtypes ( ndofTypes_ );
  String        groupName;

  Properties    myVars = Globdat::getVariables ( myName_, globdat );

  if ( myProps.find( nodeIDs, NODES ) )
  {
    // Get node and dof numbers specified with 'nodes = IdxVector'
    
    nn_ = nodeIDs.size();

    inodes_.resize ( nn_ );
    idofs_.resize ( nn_ , ndofTypes_ );

    for ( idx_t i = 0; i < nn_; i++ )
    {
      inodes_[i] = nodes_.findNode ( nodeIDs[i] );
      ndofTypes_ = dofs_->getDofsForItem ( idofs_(i,ALL), jtypes, inodes_[i] );
    }
  }

  if ( myProps.find( groupName, GROUP ) )
  {
    // Get node and dof numbers specified with 'group = String'
    
    NodeGroup nGroup = NodeGroup::get ( groupName, nodes_, globdat, context );
    nn_ = nGroup.size();

    inodes_.resize ( nn_ );
    idofs_.resize ( nn_ , ndofTypes_ );

    inodes_ = nGroup.getIndices();

    for ( idx_t i = 0; i < nn_; i++ )
    {
      ndofTypes_ = dofs_->getDofsForItem ( idofs_(i,ALL), jtypes, inodes_[i] );
    }
  }
}

//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void LoadDispModel::getConfig

  ( const Properties&  conf,
    const Properties&  globdat ) const

{
  Properties  myConf = conf.makeProps ( myName_ );

  myConf.set ( NODES, inodes_ );
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool LoadDispModel::takeAction

  ( const String&      action,
    const Properties&  params,
    const Properties&  globdat )

{
  using jive::util::Globdat;
  using jive::model::Actions;
  using jive::model::ActionParams;
  using jive::model::StateVector;

  if ( nn_ < 0 )
  {
    return false;
  }

  if ( action == Actions::GET_MATRIX0 ||
       action == Actions::GET_INT_VECTOR )
  {
    Vector      load ( ndofTypes_ ) ;
    Vector      disp ( ndofTypes_ ) ;

    Properties  myVars = Globdat::getVariables ( myName_, globdat );

    // get global data

    Vector      fint;
    Vector      state;

    params.get ( fint, ActionParams::INT_VECTOR );
    StateVector::get ( state, dofs_, globdat );

    // compute cumulative load and average displacement

    for ( idx_t i = 0; i < ndofTypes_; i++ )
    {
      load[i] = sum ( select ( fint , idofs_(ALL,i) ) );
      disp[i] = sum ( select ( state , idofs_(ALL,i) ) ) / nn_;
    }

    // store in globdat

    myVars.set ( "load" , load );
    myVars.set ( "disp" , disp );

    return true;
  }
  return false;
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   newLoadDispModel
//-----------------------------------------------------------------------


static Ref<Model>     newLoadDispModel

  ( const String&       name,
    const Properties&   conf,
    const Properties&   props,
    const Properties&   globdat )

{
  return newInstance<LoadDispModel> ( name, conf, props, globdat );
}


//-----------------------------------------------------------------------
//   declareLoadDispModel
//-----------------------------------------------------------------------


void declareLoadDispModel ()
{
  using jive::model::ModelFactory;

  ModelFactory::declare ( "LoadDisp", & newLoadDispModel );
}
