/*
 * 
 *  Copyright (C) 2010 TU Delft. All rights reserved.
 *  
 *  This class implements a model for neumann boundary conditions.
 *  As opposed to the PointLoadModel, this can apply loads on changing
 *  NodeGroups
 *
 *  Author:  F.P. van der Meer, F.P.vanderMeer@tudelft.nl
 *  Date:    April 2011
 *
 */

#include <jem/base/array/operators.h>
#include <jem/base/array/select.h>
#include <jem/base/System.h>
#include <jem/base/Float.h>
#include <jem/numeric/algebra/utilities.h>
#include <jem/numeric/utilities.h>
#include <jem/util/Event.h>
#include <jem/util/StringUtils.h>
#include <jive/util/error.h>
#include <jive/util/utilities.h>
#include <jive/util/Globdat.h>
#include <jive/algebra/VectorSpace.h>
#include <jive/model/Actions.h>
#include <jive/model/StateVector.h>
#include <jive/model/ModelFactory.h>
#include <jive/implict/SolverInfo.h>

#include "NeumannModel.h"
// #include "SolverNames.h"

using jem::io::endl;
using jem::util::StringUtils;
using jive::IdxVector;
using jive::model::Actions;
using jive::model::ActionParams;
using jive::model::StateVector;
using jive::implict::SolverInfo;


//=======================================================================
//   class NeumannModel
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  NeumannModel::TYPE_NAME       = "Neumann";

const char*  NeumannModel::LOAD_INCR_PROP  = "loadIncr";
const char*  NeumannModel::INIT_LOAD_PROP  = "initLoad";
const char*  NeumannModel::MIN_LOAD_PROP   = "minLoadIncr";
const char*  NeumannModel::MAX_LOAD_PROP   = "maxLoad";
const char*  NeumannModel::REDUCTION_PROP  = "reduction";
const char*  NeumannModel::NODES_PROP      = "nodeGroups";
const char*  NeumannModel::DOF_PROP        = "dofs";
const char*  NeumannModel::FACTORS_PROP    = "factors";

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


NeumannModel::NeumannModel

  ( const String&      name ) :

    Super  ( name  )

{
  loadScale0_  = 0.;
  loadIncr0_   = 1.0;
  initLoad_    = 0.0;
  reduction_   = .55;
  minLoadIncr_ = 0.0;
  maxLoadVal_  = Float::MAX_VALUE;
}


NeumannModel::~NeumannModel ()
{}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool NeumannModel::takeAction

  ( const String&      action,
    const Properties&  params,
    const Properties&  globdat )

{
  // initialization

  if ( action == Actions::INIT )
  {
    init_ ( globdat );

    return true;
  }

  // compute the external force vector
  // (only necessary with load control)

  if ( action == Actions::GET_EXT_VECTOR  )
  {
    Vector f;

    params.get ( f, ActionParams::EXT_VECTOR );

    getExtVector_ ( f, globdat );

    return true;
  }

  // proceed to next time step

  if ( action == Actions::COMMIT )
  {
    commit_ ( params, globdat );

    return true;
  }

  // advance to next time step

  if ( action == Actions::ADVANCE )
  {
    globdat.set ( "var.accepted", true );

    advance_ ( globdat );

    return true;
  }

  // adapt step size

  return false;
}

//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void NeumannModel::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  Properties  myProps = props.findProps ( myName_ );

  double maxD = Float::MAX_VALUE;

  myProps.find ( reduction_, REDUCTION_PROP,   0.0, 1.0 );

  myProps.get  ( loadIncr0_, LOAD_INCR_PROP ); 
  myProps.find ( initLoad_,  INIT_LOAD_PROP );

  loadIncr_  = loadIncr0_;
  loadScale0_ = loadScale_ = initLoad_;

  minLoadIncr_ = jem::numeric::abs ( loadIncr0_ );

  myProps.find ( minLoadIncr_, MIN_LOAD_PROP, 0.0, minLoadIncr_    );
  myProps.find ( maxLoadVal_,  MAX_LOAD_PROP, 0.0, maxD            );

  myProps.get( nodeGroups_, NODES_PROP );
  ngroups_ = nodeGroups_.size ( );

  myProps.get( dofTypes_, DOF_PROP );

  if ( dofTypes_.size() != ngroups_ )
  {
    throw IllegalInputException ( JEM_FUNC,
          "dofTypes must have the same length as nodeGroups" );
  }

  myProps.get ( factors_, FACTORS_PROP );

  if ( factors_.size() != ngroups_ )
  {
    throw IllegalInputException ( JEM_FUNC,
          "dofTypes must have the same length as nodeGroups" );
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void NeumannModel::getConfig

  ( const Properties&  conf,
    const Properties&  globdat ) const

{
  Properties  myConf = conf.makeProps ( myName_ );

  myConf.set ( REDUCTION_PROP, reduction_    );
  myConf.set ( LOAD_INCR_PROP, loadIncr0_    );
  myConf.set ( INIT_LOAD_PROP, initLoad_     );
  myConf.set ( MIN_LOAD_PROP,  minLoadIncr_  );
  myConf.set ( MAX_LOAD_PROP,  maxLoadVal_   );

  myConf.set ( NODES_PROP,    nodeGroups_ );
  myConf.set ( DOF_PROP,      dofTypes_   );
  myConf.set ( FACTORS_PROP,  factors_    );
}



//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Model> NeumannModel::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  return newInstance<Self> ( name );
}

//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void NeumannModel::init_ ( const Properties& globdat )
{
  // Get nodes, then dofs of nodes, and constraints of dofs

  nodes_ = NodeSet::find    ( globdat );
  dofs_  = XDofSpace::get   ( nodes_.getData(), globdat );
}

//-----------------------------------------------------------------------
//   init_
//-----------------------------------------------------------------------


void NeumannModel::getExtVector_ 

  ( const Vector&     fext,
    const Properties& globdat ) const
{
  idx_t                 nn;
  IdxVector             itype ( 1 );
  Assignable<NodeGroup> group;
  IdxVector             inodes;
  IdxVector             idofs;
  String                context = getContext();

  for ( idx_t ig = 0; ig < ngroups_; ++ig )
  {
    group  = NodeGroup::get ( nodeGroups_[ig], nodes_, globdat, context );
    nn     = group.size();

    inodes . resize ( nn );
    idofs  . resize ( nn );
    inodes = group.getIndices ();

    itype[0] = dofs_->findType ( dofTypes_[ig] );

    dofs_->findDofIndices ( idofs, inodes, itype );

    select ( fext, idofs ) += loadScale_ * factors_[ig];
  }
}

//-----------------------------------------------------------------------
//   advance_
//-----------------------------------------------------------------------

void NeumannModel::advance_

  ( const Properties&  globdat )

{
  bool accepted;

  globdat.get ( accepted, "var.accepted" );

  if ( accepted )
  {
    loadScale_ = loadScale0_ + loadIncr_;
    System::out() << "new load factor " << loadScale_ << endl;
  }
}

//-----------------------------------------------------------------------
//   commit_
//-----------------------------------------------------------------------


void NeumannModel::commit_

  ( const Properties&  params,
    const Properties&  globdat )

{
  // store converged boundary quantities

  loadScale0_  = loadScale_;
}

//-----------------------------------------------------------------------
//   reduceStep_
//-----------------------------------------------------------------------

void NeumannModel::reduceStep_

  ( const Properties&  params,
    const Properties&  globdat )

{
  // reduce the load increment and use the new increment
  // to update the prescribed value for this time step

  loadIncr_ *= reduction_;

  loadScale_ = loadScale0_ + loadIncr_;
}

//-----------------------------------------------------------------------
//   increaseStep_
//-----------------------------------------------------------------------

void NeumannModel::increaseStep_

  ( const Properties&  params,
    const Properties&  globdat )

{
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   declareNeumannModel
//-----------------------------------------------------------------------


void declareNeumannModel ()
{
  using jive::model::ModelFactory;

  ModelFactory::declare ( NeumannModel::TYPE_NAME,
                          & NeumannModel::makeNew );
}
