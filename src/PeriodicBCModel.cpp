/*
 *
 * A model to impose periodic boundary conditions on
 * the outer boundary of unit cells.
 *
 * Frans van der Meer, November 2014
 *
 */

#include <jem/base/array/utilities.h>
#include <jem/base/array/operators.h>
#include <jem/base/limits.h>
#include <jem/base/System.h>
#include <jem/base/Error.h>
#include <jem/numeric/func/UserFunc.h>
#include <jem/util/Properties.h>

#include <jive/fem/NodeGroup.h>
#include <jive/model/Actions.h>
#include <jive/model/Model.h>
#include <jive/model/ModelFactory.h>
#include <jive/util/utilities.h>
#include <jive/util/Assignable.h>
#include <jive/util/Constraints.h>
#include <jive/util/FuncUtils.h>
#include <jive/util/Printer.h>
#include <jive/util/XDofSpace.h>

#include "models.h"
#include "SolverNames.h"
#include "utilities.h"
#include "voigtUtilities.h"
#include "PBCGroupInputModule.h"

using namespace jem;

using jem::Error;
using jem::numeric::Function;
using jem::numeric::UserFunc;
using jem::util::Properties;
using jem::io::endl;
using jive::Vector;
using jive::IdxVector;
using jive::BoolVector;
using jive::fem::NodeGroup;
using jive::fem::NodeSet;
using jive::model::Model;
using jive::util::Assignable;
using jive::util::Constraints;
using jive::util::XDofSpace;
using jive::util::FuncUtils;

// =========================================================
//    class PeriodicBCModel
// =========================================================

class PeriodicBCModel : public Model
{
 public:

  typedef PeriodicBCModel    Self;
  typedef Model              Super;
  typedef Array<Ref<Function>,1>  FuncVector;

  static const char*         NODE_GROUPS[6];
  static const char*         STRAINRATE_PROP;
  static const char*         STRAINPATH_PROP;
  static const char*         MAXTIME_PROP;
  static const char*         ACTIVE_PROP;
  static const char*         DUPEDNODES_PROP;

  enum StrainType { Free, ConstantRate, Path };

                       PeriodicBCModel

    ( const String&       name,
      const Properties&   conf,
      const Properties&   props,
      const Properties&   globdat );

  virtual void         configure

    ( const Properties&   props,
      const Properties&   globdat );

  virtual void         getConfig

    ( const Properties&   conf,
      const Properties&   globdat )      const;

  virtual bool         takeAction

    ( const String&       action,
      const Properties&   params,
      const Properties&   globdat );

 protected:

  virtual              ~PeriodicBCModel ();

 private:

  void                 init_

    ( const Properties&   globdat );

  void                 setPeriodicCons_() const;

  void                 advance_() const;

  void                 checkCommit_ 
    
    ( const Properties&  params,
      const Properties&  globdat ) const;

  void                 fixCorner_() const;
  
  void                 applyStrain_

    ( const Vector&      strain ) const;

  FuncVector           makeStrainFuncs_ 
    
    ( const Vector&      strainRate ) const;

  FuncVector           getStrainFuncs_ 
    
    ( const Properties&  globdat ) const;

 private:

  IdxVector               dofTypes_;

  Assignable<NodeSet>     nodes_;
  BoolVector              active_;

  Ref<XDofSpace>          dofs_;
  Ref<Constraints>        cons_;

  IdxVector               bndNodes_[6]; // indices of boundary nodes
  Tuple<idx_t,3>          masters_; // master corner nodes
  idx_t                   ifixed_;

  Vector                  imposedStrain_; //total applied strain

  double                  time_;
  double                  stepSize_;
  double                  maxTime_;
  
  Vector                  dx_;
  idx_t                   rank_;

  String                  strainFile_;
  StrainType              strainType_;
  FuncVector              strainFunc_;

  String                  dupedNodeGroup_;
};


//-----------------------------------------------------------------------
//   static constants
//-----------------------------------------------------------------------

const char* PeriodicBCModel::STRAINRATE_PROP  = "strainRate";
const char* PeriodicBCModel::STRAINPATH_PROP  = "strainPath";
const char* PeriodicBCModel::MAXTIME_PROP     = "maxTime";
const char* PeriodicBCModel::ACTIVE_PROP      = "active";
const char* PeriodicBCModel::DUPEDNODES_PROP  = "duplicatedNodes";

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------


PeriodicBCModel::PeriodicBCModel

   ( const String&       name,
     const Properties&   conf,
     const Properties&   props,
     const Properties&   globdat ) : Super(name)
{
  maxTime_ = jem::maxOf ( maxTime_ );

  Properties  myProps = props.getProps ( myName_ );
  Properties  myConf  = conf.makeProps ( myName_ );
  Vector      dStrain;

  myProps.find ( dupedNodeGroup_, DUPEDNODES_PROP );
  myConf.set   ( DUPEDNODES_PROP, dupedNodeGroup_ );

  nodes_ = NodeSet::find    ( globdat );
  rank_ = nodes_.rank();

  idx_t strCount = STRAIN_COUNTS[rank_];

  imposedStrain_ . resize ( strCount );
  imposedStrain_ = 0.;

  strainFunc_    . resize ( strCount );
  strainType_    = Free;

  myProps.find( dStrain, STRAINRATE_PROP );
  myConf.set  ( STRAINRATE_PROP, dStrain );

  bool        strainPath = false;
  myProps.find ( strainPath, STRAINPATH_PROP );
  myConf.set ( STRAINPATH_PROP, strainPath );

  JEM_ASSERT  ( dStrain.size() == 0 || dStrain.size() == strCount );

  if ( dStrain.size() == strCount )
  {
    strainType_    = ConstantRate;
    // oldStrain_     . resize ( strCount );
    // oldStrain_     = 0.;

    strainFunc_ = makeStrainFuncs_ ( dStrain );

    if ( strainPath )
    {
      System::warn() << "strainPath input ignored, because strainRate "
        << "has already been defined." <<  endl;
    }
  }
  else
  {
    if ( strainPath )
    {
      strainType_ = Path;
      strainFunc_ = getStrainFuncs_ ( globdat );

      myProps.get ( maxTime_, MAXTIME_PROP );
      myConf.set ( MAXTIME_PROP, maxTime_ );
    }
  }

  if ( myProps.find ( active_, ACTIVE_PROP ) )
  {
    JEM_PRECHECK ( active_.size() == rank_ );
  }
  else
  {
    active_ . resize ( rank_ );
    active_ = true;
  }

  time_          = 0.;
  stepSize_      = 1.;
  masters_       = -1;
  ifixed_        = -1;
}

PeriodicBCModel::~PeriodicBCModel()
{}

//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------

void PeriodicBCModel::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  using jive::util::joinNames; 

  System::out() << "Making NodeGroups from PeriodicBCModel" << endl;
  Ref<Module> ngroupMaker = newInstance<PBCGroupInputModule> ( "pBC" );

  Properties tmp;
  Properties conf;

  if ( dupedNodeGroup_ != "" )
  {
    tmp.set ( joinNames ( "pBC", DUPEDNODES_PROP ), dupedNodeGroup_ );
  }

  ngroupMaker->init ( conf, tmp, globdat );
  ngroupMaker->shutdown ( globdat );
}

//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void PeriodicBCModel::getConfig 

  ( const Properties& conf,
    const Properties& globdat ) const

{
}

//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool PeriodicBCModel::takeAction

  ( const String&      action,
    const Properties&  params,
    const Properties&  globdat )

{
  using jive::model::Actions;
  using jive::model::ActionParams;

  if ( action == Actions::INIT )
  {
    init_ ( globdat );

    return true;
  }

  if ( action == Actions::ADVANCE )
  {
    advance_();

    return true;
  }

  if ( action == SolverNames::SET_STEP_SIZE )
  {
    double dt, dt0;

    params.get ( dt,  SolverNames::STEP_SIZE   );
    params.get ( dt0, SolverNames::STEP_SIZE_0 );

    stepSize_ = dt / dt0;

    return true;
  }

  if ( action == Actions::GET_CONSTRAINTS )
  {
    fixCorner_();

    // apply strain (on corner nodes)

    if ( strainType_ != Free )
    {
      applyStrain_ ( imposedStrain_ );
    }
  
    // apply periodic conditions

    setPeriodicCons_ (  );

    return true;
  }

  if ( action == SolverNames::CHECK_COMMIT )
  {
    checkCommit_ ( params, globdat );

    return true;
  }

  if ( action == Actions::COMMIT )
  {
    time_ += stepSize_;

    return true;
  }

  return false;
}

//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------

void PeriodicBCModel::init_

  ( const Properties&  globdat )

{
  dofs_  = XDofSpace::get   ( nodes_.getData(), globdat );
  cons_  = Constraints::get ( dofs_, globdat );

  dofTypes_ . resize ( rank_ );
  dx_       . resize ( rank_ );

  dofTypes_[0] = dofs_->addType ( "dx" );
  dofTypes_[1] = dofs_->addType ( "dy" );
  if ( rank_ > 2 )
  {
    dofTypes_[2] = dofs_->addType ( "dz" );
  }

  // get nodes on boundaries
  
  for ( idx_t i = 0; i < 2*rank_; ++i )
  {
    NodeGroup edge = NodeGroup::get 
      ( PBCGroupInputModule::EDGES[i], nodes_, globdat, getContext() );
    bndNodes_[i] . ref ( edge.getIndices() );
  }
  for ( idx_t i = 0; i < rank_; ++i )
  {
    if ( bndNodes_[i*2]. size () != bndNodes_[i*2+1].size () )
    {
      throw Error ( JEM_FUNC, String(i) +
        " opposite edges do not have the same number of nodes!!!" );
    }
  }

  // get specimen dimensions
  // NB: assuming rectangular or hexahedral orientation
  //     this assumption is not checked in the code!

  Matrix coords    ( nodes_.toMatrix() );
  Vector box       ( rank_ * 2 );
  for ( idx_t ix = 0; ix < rank_; ++ix )
  {
    box[ix*2]   = coords ( ix, bndNodes_[ix*2][0] );
    box[ix*2+1] = coords ( ix, bndNodes_[ix*2+1][0] );

    dx_[ix] = box[ix*2+1] - box[ix*2];
  }

  // find which nodes are in the corners

  ifixed_ = NodeGroup::get 
      ( PBCGroupInputModule::CORNERS[0], nodes_, globdat, getContext() ).
      getIndices()[0];

  for ( idx_t i = 0; i < rank_; ++i )
  {
    masters_[i] = NodeGroup::get
      ( PBCGroupInputModule::CORNERS[i+1], nodes_, globdat, getContext() ).
      getIndices()[0];
  }
}

//-----------------------------------------------------------------------
//   advance_
//-----------------------------------------------------------------------

void PeriodicBCModel::advance_ () const

{
  for ( idx_t i = 0; i < imposedStrain_.size(); ++i )
  {
    if ( strainFunc_[i] != nullptr )
    {
      imposedStrain_[i] = strainFunc_[i]->eval ( time_ );
    }
  }
}

//-----------------------------------------------------------------------
//   checkCommit_
//-----------------------------------------------------------------------

void PeriodicBCModel::checkCommit_

  ( const Properties&  params,
    const Properties&  globdat ) const

{
  // terminate the computation if displacement exceeds maximum.

  if ( time_ > maxTime_ ) 
  {
    System::out() << myName_ << " says: TERMINATE because "
      << " time > maxTime." << endl;

    params.set ( SolverNames::TERMINATE, "sure" );
  }
}

//-----------------------------------------------------------------------
//   fixCorner_
//-----------------------------------------------------------------------

void PeriodicBCModel::fixCorner_ () const
{
  // apply zero displacement on first corner 

  for ( idx_t i = 0; i < rank_; ++i )
  {
    idx_t idof = dofs_->getDofIndex ( ifixed_, dofTypes_[i] );
    cons_->addConstraint ( idof );
  }
}

//-----------------------------------------------------------------------
//   applyStrain_
//-----------------------------------------------------------------------

void PeriodicBCModel::applyStrain_ ( const Vector& strain ) const
{
  // prescribe displacement on the master corner nodes

  Matrix eps ( rank_, rank_ );
  voigtUtilities::voigt2TensorStrain ( eps, strain );

  for ( idx_t i = 0; i < rank_; ++i )
  {
    for ( idx_t j = 0; j < rank_; ++j )
    {
      idx_t ivoigt = voigtUtilities::voigtIndex(i,j,rank_);
      if ( strainFunc_[ivoigt] != nullptr )
      {
        idx_t idof = dofs_->getDofIndex ( masters_[i], dofTypes_[j] );
        cons_->addConstraint ( idof, dx_[i]*eps(i,j) );
      }
    }
  }
}

//-----------------------------------------------------------------------
//   setPeriodicCons_
//-----------------------------------------------------------------------

void PeriodicBCModel::setPeriodicCons_ ( ) const

{
  // loop over right and top edge: connect to opposite edges
  // relative to corner movement

  IdxVector   idofs(2);
  Vector      coefs(2);

  coefs = 1.;

  // loop over faces

  for ( idx_t ix = 0; ix < rank_; ix++ )
  {
    if ( active_[ix] )
    {
      IdxVector  inodes ( bndNodes_[ix*2] );
      IdxVector  jnodes ( bndNodes_[ix*2+1] );

      // loop over nodes of edges

      for ( idx_t  in = 0; in < inodes.size(); in++ )
      {
        if ( jnodes[in] != masters_[ix] )
        {
          // System::out() << "constraining node " << jnodes[in] << 
          // " to  " << inodes[in] << " and " << masters_[ix] << endl;

          // loop over dof types

          for ( idx_t jx = 0; jx < rank_; ++jx )
          {
            // master dofs: opposite node and relevant corner node

            idofs[0] = dofs_->getDofIndex ( inodes[in], dofTypes_[jx] );
            idofs[1] = dofs_->getDofIndex ( masters_[ix], dofTypes_[jx] );

            // dofs of dependent (slave) node jd

            idx_t jdof = dofs_->getDofIndex ( jnodes[in], dofTypes_[jx] );

            // add constraint

            cons_->addConstraint ( jdof, idofs, coefs );
          }
        }
      }
    }
  }
  // cons_->printTo(jive::util::Printer::get());
}

//-----------------------------------------------------------------------
//   makeStrainFuncs_
//-----------------------------------------------------------------------

Array<Ref<Function>,1>  PeriodicBCModel::makeStrainFuncs_ 
  
  ( const Vector&      strainRate ) const

{
  // make Function objects that define strain components as linear
  // function of time

  FuncVector ret ( strainRate.size() );

  for ( idx_t i = 0; i < strainRate.size(); ++i )
  {
    String args = "t";
    String expr = String(strainRate[i]) + String("*t");
    ret[i] = newInstance<UserFunc> ( args, expr );
  }

  return ret;
}

//-----------------------------------------------------------------------
//   getStrainFuncs_
//-----------------------------------------------------------------------

Array<Ref<Function>,1>  PeriodicBCModel::getStrainFuncs_ 
  
  ( const Properties&  globdat ) const

{
  System::out() << myName_ << " gets strain path from globdat\n" 
    << "supposing UserFuncs have been defined elsewhere." << endl;

  FuncVector ret ( strainFunc_.size() );

  String context = "getStrainFuncs_(globdat)";
  
  // normal strain components are optional to allow for uniaxial load
  // shear strain components are required to exclude rigid body modes

  if ( ret.size() == 3 )
  {
    ret[0] = FuncUtils::findFunc ( "strain_xx", globdat );
    ret[1] = FuncUtils::findFunc ( "strain_yy", globdat );
    ret[2] = FuncUtils::getFunc  ( "strain_xy", globdat, context );
  }
  else
  {
    ret[0] = FuncUtils::findFunc ( "strain_xx", globdat );
    ret[1] = FuncUtils::findFunc ( "strain_yy", globdat );
    ret[2] = FuncUtils::findFunc ( "strain_zz", globdat );
    ret[3] = FuncUtils::getFunc  ( "strain_xy", globdat, context );
    ret[4] = FuncUtils::getFunc  ( "strain_yz", globdat, context );
    ret[5] = FuncUtils::getFunc  ( "strain_zx", globdat, context );
  }

  return ret;
}

//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   newPeriodicBCModel
//-----------------------------------------------------------------------


static Ref<Model>     newPeriodicBCModel

  ( const String&       name,
    const Properties&   conf,
    const Properties&   props,
    const Properties&   globdat )

{
  return newInstance<PeriodicBCModel> ( name, conf, props, globdat );
}


//-----------------------------------------------------------------------
//   declarePeriodicBCModel
//-----------------------------------------------------------------------


void declarePeriodicBCModel ()

{
  using jive::model::ModelFactory;

  ModelFactory::declare ( "PeriodicBC", & newPeriodicBCModel );
}

