/*
 * 
 *  Copyright (C) 2010 TU Delft. All rights reserved.
 *  
 *  This class implements a model for dirichlet boundary conditions.
 *
 *  Author:  F.P. van der Meer, F.P.vanderMeer@tudelft.nl
 *  Date:    May 2010
 *
 */

#include <jem/base/System.h>
#include <jem/base/Float.h>
#include <jem/numeric/utilities.h>
#include <jem/io/PrintWriter.h>
#include <jive/fem/NodeGroup.h>
#include <jive/model/Actions.h>
#include <jive/model/ModelFactory.h>
#include <jive/util/XDofSpace.h>
#include <jive/util/Globdat.h>

#include <jive/model/StateVector.h>
#include <jem/util/ArrayBuffer.h>

#include <jem/io/FileWriter.h>
#include <jem/io/PrintWriter.h>
#include <jem/util/StringUtils.h>

#include "DirichletModel.h"
#include "SolverNames.h"

using jem::io::endl;

using jive::Matrix;
using jive::fem::NodeGroup;
using jive::model::Actions;
using jive::util::XDofSpace;
using jive::model::StateVector;
using jem::util::ArrayBuffer;

using jem::io::FileWriter;
using jem::io::PrintWriter;
using jem::util::StringUtils;

//=======================================================================
//   class DirichletModel
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  DirichletModel::TYPE_NAME       = "Dirichlet";

const char*  DirichletModel::DISP_INCR_PROP  = "dispIncr";
const char*  DirichletModel::DISP_RATE_PROP  = "dispRate";
const char*  DirichletModel::INIT_DISP_PROP  = "initDisp";
const char*  DirichletModel::MAX_DISP_PROP   = "maxDisp";
const char*  DirichletModel::NODES_PROP      = "nodeGroups";
const char*  DirichletModel::DOF_PROP        = "dofs";
const char*  DirichletModel::FACTORS_PROP    = "factors";
const char*  DirichletModel::LOADED_PROP     = "loaded";

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


DirichletModel::DirichletModel

  ( const String&      name,
    const Ref<Model>&  child ) :

    Super  ( name  )

{
  dispScale0_  = 0.;
  dispIncr0_   = 0.0;
  initDisp_    = 0.0;
  maxDispVal_  = Float::MAX_VALUE;
  method_      = INCREMENT;
  l1_ = 1.e-2;
  l2_ = 1.e-2;
  l3_ = 1.e-2;
}


DirichletModel::~DirichletModel ()
{}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool DirichletModel::takeAction

  ( const String&      action,
    const Properties&  params,
    const Properties&  globdat )

{
  System::out () << "@DirichletModel::takeAction(), action... " << action << "\n";


  // initialization

  if ( action == Actions::INIT )
  {
    init_ ( globdat );

    return true;
  }

  // apply displacement increment

  if ( action == Actions::GET_CONSTRAINTS )
  {
    applyConstraints_ ( params, globdat );

    return true;
  }

  // check state

  if ( action == SolverNames::CHECK_COMMIT )
  {
    checkCommit_ ( params, globdat );

    return true;
  }

  // proceed to next time step

  if ( action == Actions::COMMIT )
  {
    commit_ ( params, globdat );

 /*   Matrix Floc ( 3, 3);
    Floc = 0.0;

    updateDefGrad_ ( Floc, globdat );
*/ 
//    printStrains_ ( params, globdat );

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

  else if ( action == SolverNames::SET_STEP_SIZE )
  {
    setDT_ ( params );
  }
  return false;
}

//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void DirichletModel::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  Properties  myProps = props.findProps ( myName_ );

  double maxD = Float::MAX_VALUE;

  if ( myProps.find ( dispRate_, DISP_RATE_PROP ) )
  {
    myProps.find ( initDisp_,  INIT_DISP_PROP );

    method_     = RATE;
    dispScale0_ = dispScale_ = 0.;
  }
  else
  {
    myProps.find ( initDisp_,  INIT_DISP_PROP );
    myProps.get ( dispIncr0_, DISP_INCR_PROP );

    method_     = INCREMENT;
    dispIncr_   = dispIncr0_;
    dispScale0_ = dispScale_ = initDisp_ - dispIncr0_;  // cancel first increment
  }

   System::out() << "check1 ok\n";

  myProps.find ( maxDispVal_,  MAX_DISP_PROP, 0.0, maxD );

  myProps.get( nodeGroups_, NODES_PROP );
  ngroups_ = nodeGroups_.size ( );

  myProps.get( dofTypes_, DOF_PROP );

  if ( dofTypes_.size() != ngroups_ )
  {
    throw IllegalInputException ( JEM_FUNC,
          "dofTypes must have the same length as nodeGroups" );
  }

  if ( myProps.find ( factors_, FACTORS_PROP ) )
  { 
    if ( factors_.size() != ngroups_ )
    {
      throw IllegalInputException ( JEM_FUNC,
            "dofTypes must have the same length as nodeGroups" );
    }
  }
  else
  {
    idx_t loaded;

    factors_.resize ( ngroups_ );

    factors_ = 0.;

    if ( myProps.find( loaded, LOADED_PROP, -1, ngroups_-1 ) )
    {
      factors_[loaded] = 1.;
    }
  }

  const String  context = getContext ();

  nodes_ = NodeSet::find    ( globdat );
  dofs_  = XDofSpace::get   ( nodes_.getData(), globdat );

  ndofTypes_ = dofs_->typeCount();

  IdxVector     jtypes ( ndofTypes_ );

  IdxVector                 xinodes;
  IdxVector                 yinodes;
  IdxVector                 zinodes;
  idx_t                     nn;

  NodeGroup xgroup = NodeGroup::get ( "cornerx", nodes_, globdat, context );

  nn = xgroup.size();

  xinodes.resize ( nn );
  xinodes = xgroup.getIndices ();

  NodeGroup ygroup = NodeGroup::get ( "cornery", nodes_, globdat, context );

  nn = ygroup.size();

  yinodes.resize ( nn );
  yinodes = ygroup.getIndices ();

  NodeGroup zgroup = NodeGroup::get ( "cornerz", nodes_, globdat, context );

  nn = zgroup.size();

  zinodes.resize ( nn );
  zinodes = zgroup.getIndices ();

  // configure lengths 

  Matrix corxcoords ( ndofTypes_, xinodes.size() );
  Matrix corycoords ( ndofTypes_, yinodes.size() );
  Matrix corzcoords ( ndofTypes_, zinodes.size() );

  nodes_.getSomeCoords ( corxcoords, xinodes );
  nodes_.getSomeCoords ( corycoords, yinodes );
  nodes_.getSomeCoords ( corzcoords, zinodes );

  l1_ = corxcoords(0,0) - corycoords(0,0);
  l2_ = corycoords(1,0) - corzcoords(1,0);
  l3_ = corzcoords(2,0) - corycoords(2,0);

  System::out() << "l1 " << l1_ << "\n";
  System::out() << "l2 " << l2_ << "\n";
  System::out() << "l3 " << l3_ << "\n";

  initMasters_ ( globdat );
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void DirichletModel::getConfig

  ( const Properties&  conf,
    const Properties&  globdat ) const

{
  Properties  myConf = conf.makeProps ( myName_ );

  if ( method_ == INCREMENT )
  {
    myConf.set ( DISP_INCR_PROP, dispIncr0_  );
    myConf.set ( INIT_DISP_PROP, initDisp_   );
  }
  else
  {
    myConf.set ( DISP_RATE_PROP, dispRate_   );
  }

  myConf.set ( MAX_DISP_PROP,  maxDispVal_   );

  myConf.set ( NODES_PROP,     nodeGroups_   );
  myConf.set ( DOF_PROP,       dofTypes_     );
  myConf.set ( FACTORS_PROP,   factors_      );
}



//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Model> DirichletModel::makeNew

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


void DirichletModel::init_ ( const Properties& globdat )
{
  // Get nodes, then dofs of nodes, and constraints of dofs

  nodes_ = NodeSet::find    ( globdat );
  dofs_  = XDofSpace::get   ( nodes_.getData(), globdat );
  cons_  = Constraints::get ( dofs_, globdat );
}

//-----------------------------------------------------------------------
//   advance_
//-----------------------------------------------------------------------

void DirichletModel::advance_

  ( const Properties&  globdat )

{
  if ( method_ == RATE && jem::numeric::abs(dispIncr_) < Float::EPSILON )
  {
    System::warn() << myName_ << " zero increment in RATE mode."
      << " It seems the time increment has not been set." << "\n";
  }

  dispScale_   = dispScale0_ + dispIncr_;

  System::out() << "New displacement factor " << dispScale_ << "\n";
}

//-----------------------------------------------------------------------
//   applyConstraints_
//-----------------------------------------------------------------------

void DirichletModel::applyConstraints_

  ( const Properties&  params,
    const Properties&  globdat )

{
  idx_t                 nn;
  Assignable<NodeGroup> group;
  IdxVector             inodes;
  String                context;

  // loop over node groups

  for ( idx_t ig = 0; ig < ngroups_; ig++ )
  {
    group  = NodeGroup::get ( nodeGroups_[ig], nodes_, globdat, context );

    nn     = group.size();

    inodes . resize ( nn );
    inodes = group.getIndices ();

    idx_t itype  = dofs_->findType ( dofTypes_[ig] );

    double val = dispScale_ * factors_[ig];

    // apply constraint

    for ( idx_t in = 0; in < nn; in++ )
    {
      idx_t idof = dofs_->getDofIndex ( inodes[in], itype );

      cons_->addConstraint ( idof, val );
      System::out() << "Setting " << idof << " to " << val << '\n';
    }
  }

  // compress for more efficient storage

  cons_->compress();
}

//-----------------------------------------------------------------------
//   updateDefGrad_
//-----------------------------------------------------------------------

void DirichletModel::updateDefGrad_

  (       Matrix&        F,
    const Properties& globdat )

{
  Matrix F1loc ( 3, 3 );

  F1loc = 0.;

  Vector  disp1;

  StateVector::get    ( disp1, dofs_, globdat );

  /*Ref<PrintWriter> out = newInstance<PrintWriter> ( &System::out() );
  out->nformat.setFractionDigits( 10 );
  
  *out << "updateDefGrad elemDisp1 " << disp1 << "\n";
  for ( idx_t i = 0; i < masters_.size(0); i++ ) *out << "updateDefGrad masters " << masters_[i] << ": " << disp1[masters_[i]] << "\n";
*/

  F1loc(0,0) = 1. + disp1[masters_[3]]/l3_;
  F1loc(0,1) =      disp1[masters_[1]]/l1_;
  F1loc(0,2) =      disp1[masters_[7]]/l2_;
  F1loc(1,0) =      disp1[masters_[2]]/l3_;
  F1loc(1,1) = 1. + disp1[masters_[0]]/l1_;
  F1loc(1,2) =      disp1[masters_[8]]/l2_;
  F1loc(2,0) =      disp1[masters_[5]]/l3_;
  F1loc(2,1) =      disp1[masters_[6]]/l1_;
  F1loc(2,2) = 1. + disp1[masters_[4]]/l2_;

//  *out << "updateDefGrad F " << F1loc << "\n"; 

  F = F1loc;
}

//-----------------------------------------------------------------------
//   initMasters_
//-----------------------------------------------------------------------

void DirichletModel::initMasters_

  ( const Properties& globdat )

{
  idx_t                 nn;
  Assignable<NodeGroup> group;
  IdxVector             inodes;
  String                context;
  ArrayBuffer<idx_t>    mbuf;

  for ( idx_t ig = 0; ig < nodeGroups_.size(); ig++ )
  {
    group = NodeGroup::get ( nodeGroups_[ig], nodes_, globdat, context );

    nn = group.size();

    inodes.resize ( nn );
    inodes = group.getIndices ();

    idx_t itype = dofs_->findType ( dofTypes_[ig] );
    idx_t idof  = dofs_->getDofIndex ( inodes[0], itype );

    System::out() << "inodes " << inodes << "\n";
    System::out() << "initMasters itype " << itype << " idof " << idof << "\n"; 

    mbuf.pushBack ( idof );
  }

  masters_.ref ( mbuf.toArray() );

  System::out() << "initMasters " << masters_ << "\n"; 

  dofs_->resetEvents ();
}

//-----------------------------------------------------------------------
//   checkCommit_
//-----------------------------------------------------------------------

void DirichletModel::checkCommit_

  ( const Properties&  params,
    const Properties&  globdat )

{
  // terminate the computation if displacement exceeds maximum.
  // be careful with this!

  if ( jem::numeric::abs ( dispScale_ ) > maxDispVal_ ) 
  {
    System::out() << myName_ << " says: TERMINATE because "
      << " disp > maxDispVal." << "\n";

    params.set ( SolverNames::TERMINATE, "sure" );
  }
}

//-----------------------------------------------------------------------
//   commit_
//-----------------------------------------------------------------------


void DirichletModel::commit_

  ( const Properties&  params,
    const Properties&  globdat )

{
  // store converged boundary quantities

  dispScale0_  = dispScale_;
}

//-----------------------------------------------------------------------
//   setDT_
//-----------------------------------------------------------------------

void DirichletModel::setDT_

  ( const Properties&  params )

{
  double       dt;
  double       dt0;

  params.get ( dt,  SolverNames::STEP_SIZE   );
  params.get ( dt0, SolverNames::STEP_SIZE_0 );

  System::out() << "DirichletModel. dt " << dt << "\n";

  dt_ = dt;

  if ( method_ == RATE )
  {
    dispIncr_ = dispRate_ * dt;
  }
  else
  {
    // rate dependent with an initial increment is also supported
    // the displacement rate is then implicit input as dispIncr0/dt0

    dispIncr_  = dispIncr0_ * dt / dt0;
  }
}

//-----------------------------------------------------------------------
//    initWriter_
//-----------------------------------------------------------------------

Ref<PrintWriter>  DirichletModel::initWriter_

  ( const Properties&  params,
    const String       name )  const

{
  // Open file for output

  StringVector fileName;
  String       prepend;

  if ( params.find( prepend, "prepend" ) )
  {
    fileName.resize(3);

    fileName[0] = prepend;
    fileName[1] = "homogenized";
    fileName[2] = name;
  }
  else
  {
    fileName.resize(2);

    fileName[0] = "homogenized";
    fileName[1] = name;
  }

  System::out() << "filename for printing " << fileName << "\n";
  return newInstance<PrintWriter>( newInstance<FileWriter> (
         StringUtils::join( fileName, "." ) ) );
}

//-----------------------------------------------------------------------
//    printStrains_
//-----------------------------------------------------------------------

void DirichletModel::printStrains_

  ( const Properties&  params,
    const Properties&  globdat )

{
  using jive::util::Globdat;

  Matrix Floc ( 3, 3);
  Floc = 0.0;

  updateDefGrad_ ( Floc, globdat );

  idx_t       it;

  if ( strainOut_ == nullptr )
  {
    strainOut_ = initWriter_ ( params, "defgrad" );
  }

  strainOut_->nformat.setFractionDigits(8);

  globdat.get ( it, Globdat::TIME_STEP    );

  for ( idx_t r = 0; r < 3; ++r )
  {
      for ( idx_t k = 0; k < 3; ++k )
      {
        *strainOut_ << Floc (r,k) << " ";
      }
  }

  for ( idx_t r = 0; r < 3; ++r )
  {
      for ( idx_t k = 0; k < 3; ++k )
      {
        *strainOut_ << Floc (r,k) << " ";
      }
  }

  *strainOut_ << dt_ << "\n";

  strainOut_->flush();
}

//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   declareDirichletModel
//-----------------------------------------------------------------------


void declareDirichletModel ()
{
  using jive::model::ModelFactory;

  ModelFactory::declare ( DirichletModel::TYPE_NAME,
                          & DirichletModel::makeNew );
}
