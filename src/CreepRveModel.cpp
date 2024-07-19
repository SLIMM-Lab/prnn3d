/*
 *  TU Delft; Dragan Kovacevic 
 *
 *  2022
 *
 *  finite deformation model to apply off-axis creep stress on RVE
 * 
 *  model derived from strain-rate based arclength model
 *  SRArclenModel.cpp 
 *
 */

#include <jem/base/System.h>
#include <jem/base/Error.h>
#include <jem/base/array/operators.h>
#include <jem/base/array/select.h>
#include <jem/numeric/algebra/utilities.h>
#include <jem/numeric/utilities.h>
#include <jem/numeric/algebra/MatmulChain.h>
#include <jem/numeric/algebra/EigenUtils.h>
#include <jem/numeric/algebra/LUSolver.h>
#include <jem/util/ArrayBuffer.h>
#include <jem/util/Event.h>
#include <jive/util/error.h>
#include <jive/util/Printer.h>
#include <jive/util/utilities.h>
#include <jive/util/Constraints.h>
#include <jive/util/Globdat.h>
#include <jive/model/Actions.h>
#include <jive/model/ModelFactory.h>
#include <jive/implict/ArclenActions.h>
#include <jive/model/StateVector.h>
#include <jive/implict/SolverInfo.h>

#include <jem/io/FileWriter.h>
#include <jem/io/PrintWriter.h>
#include <jem/util/StringUtils.h>

#include <math.h>

#include "CreepRveModel.h"
#include "SolverNames.h"

using jem::util::ArrayBuffer;
using jem::io::endl;
using jem::numeric::MatmulChain;
using jem::numeric::inverse;
using jive::IntVector;
using jive::Matrix;
using jive::model::Actions;
using jive::model::ActionParams;
using jive::util::Globdat;
using jive::model::StateVector;
using jive::implict::ArclenActions;
using jive::implict::ArclenParams;
using jive::implict::SolverInfo;

using jem::io::FileWriter;
using jem::io::PrintWriter;
using jem::util::StringUtils;

typedef MatmulChain<double,3>   MChain3;

//-----------------------------------------------------------------------
//   constants
//-----------------------------------------------------------------------

const double PI               = 3.14159265;

//=======================================================================
//   class CreepRveModel
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------

const char* CreepRveModel::NODEGROUPS_PROP  = "nodeGroups";
const char* CreepRveModel::LOADGROUPS_PROP  = "loadGroups";
const char* CreepRveModel::DOFS_PROP        = "dofs";
const char* CreepRveModel::LOADDOFS_PROP    = "loadDofs";
const char* CreepRveModel::STRESSRATE_PROP  = "stressRate";
const char* CreepRveModel::OFFANGLE_PROP    = "offAngle";
const char* CreepRveModel::MAXSTRESS_PROP   = "maxStress";
const char* CreepRveModel::MAXSTRAIN_PROP   = "maxStrain";

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------

CreepRveModel::CreepRveModel

  ( const String&     name,
    const Properties& conf,
    const Properties& props,
    const Properties& globdat ):

    Model ( name )

{
  time_  = 0.0;
  time0_ = 0.0;
  dt_    = 1.;

  strain_    = 0.;
  strain0_   = 0.;
  maxStrain_ = jem::maxOf ( maxStrain_ );

  stressRate_ = 1.;
  stress_     = 0.;
  stress0_    = 0.;
  maxStress_  = jem::maxOf ( maxStress_ );;

  nodes_ = NodeSet::find    ( globdat );
  dofs_  = XDofSpace::get   ( nodes_.getData(), globdat );
  cons_  = Constraints::get ( dofs_, globdat );

  l1_ = 1.e-2;
  l2_ = 1.e-2;
  l3_ = 1.e-2;

  offAngle_   = 90.;
  theta0_     = 0.;
  c0_         = 1.;
  s0_         = 0.;
  phi_        = 0.;

  ndofTypes_ = dofs_->typeCount();

  F1loc_ = 0.;

  // configure ( props, globdat );
}

CreepRveModel::~CreepRveModel ()
{}

//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------

void CreepRveModel::configure

  ( const Properties& props,
    const Properties& globdat )

{
  if ( props.contains( myName_ ) )
  {
    Properties myProps = props.getProps ( myName_ );

    myProps.get  ( nodeGroups_, NODEGROUPS_PROP );
    numgroups_ = nodeGroups_.size ( );

    myProps.get  ( loadGroups_, LOADGROUPS_PROP );
    numloadgroups_ = loadGroups_.size ( );

    myProps.get  ( dofTypes_,  DOFS_PROP      );
    myProps.get  ( ldofTypes_, LOADDOFS_PROP  );

    myProps.find ( maxStrain_, MAXSTRAIN_PROP );

    if ( dofTypes_.size() != numgroups_ )
      throw IllegalInputException ( JEM_FUNC,
      "nodeGroups and dofTypes must have the same size." );

    if ( ldofTypes_.size() != numloadgroups_ )
      throw IllegalInputException ( JEM_FUNC,
      "loadGroups and ldofTypes must have the same size." );

    myProps.get  ( offAngle_,   OFFANGLE_PROP,   0., 90.  );
    myProps.get  ( stressRate_, STRESSRATE_PROP  );
    myProps.get  ( maxStress_,  MAXSTRESS_PROP );
  }

  const String  context = getContext ();

  IdxVector     jtypes ( ndofTypes_ );

  // get idofY_; needed to update length in the local y-direction
  // necessary because cornery is not in the masters_ list (no load on this node)

  IdxVector                 xinodes;
  IdxVector                 yinodes;
  IdxVector                 zinodes;
  idx_t                     nn;

  NodeGroup xgroup = NodeGroup::get ( "cornerx", nodes_, globdat, context );

  nn = xgroup.size();

  xinodes.resize ( nn );
  xinodes = xgroup.getIndices ();


  IdxMatrix  idofsY;

  NodeGroup ygroup = NodeGroup::get ( "cornery", nodes_, globdat, context );

  nn = ygroup.size();

  yinodes.resize ( nn );
  yinodes = ygroup.getIndices ();

  idofsY.resize ( nn , ndofTypes_ );

  ndofTypes_ = dofs_->getDofsForItem ( idofsY(0,ALL), jtypes, yinodes[0] );

  idofY_ = dofs_->getDofIndex (yinodes[0], jtypes[1] );

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

  System::out() << "l11 " << l1_ << endl;
  System::out() << "l22 " << l2_ << endl;
  System::out() << "l33 " << l3_ << endl;

  initMasters_ (globdat);

  JEM_PRECHECK ( masters_.size() == 4 );

  transform0_      .resize(masters_.size());
  transformForce_  .resize(masters_.size());

  F1loc_(0,0) = F1loc_(1,1) = F1loc_(2,2) = 1.;

  initDirection_ (); // init theta0_, transform0_
}

//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------

void CreepRveModel::getConfig

  ( const Properties& conf,
    const Properties& globdat ) const

{
  Properties myConf = conf.makeProps ( myName_ );

  myConf.set ( NODEGROUPS_PROP, nodeGroups_ );
  myConf.set ( DOFS_PROP, dofTypes_         );

  myConf.set ( LOADGROUPS_PROP, loadGroups_  );
  myConf.set ( LOADDOFS_PROP, ldofTypes_     );

  myConf.set ( MAXSTRAIN_PROP, maxStrain_   );
  myConf.set ( OFFANGLE_PROP, offAngle_     );
  myConf.set ( STRESSRATE_PROP, stressRate_ );
  myConf.set ( MAXSTRESS_PROP,   maxStress_ );
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------

bool CreepRveModel::takeAction

  ( const String&     action,
    const Properties& params,
    const Properties& globdat )

{
  // System::out () << "@CreepRveModel::takeAction(), action... " << action << endl;

  if ( action == Actions::GET_EXT_VECTOR  )
  {
    getExtVector_ ( params, globdat );

    return true;
  }

  if ( action == SolverNames::SET_STEP_SIZE )
  {
    params.get ( dt_, SolverNames::STEP_SIZE );

    return true;
  }

  // if ( action == SolverNames::CONTINUE )
  // {
  //   getExtVector_ ( params, globdat );

  //   return true;
  // }

  if ( action == Actions::GET_CONSTRAINTS )
  {
    applyDisps_ ( params, globdat );

    return true;
  }

  if ( action == SolverNames::CHECK_COMMIT )       
  {
    updateDefGrad_ ( globdat );

    updateTransform_ ( );

    updateStrain_ ( );

    if ( strain_ > maxStrain_ )
    {
      System::out() << "CreepRveModel: Maximum strain reached. Terminating analysis.\n";

      params.set ( SolverNames::TERMINATE, "sure" );
    }

    return true;
  }

  if ( action == Actions::COMMIT )
  {
    Properties  myVars = Globdat::getVariables ( myName_, globdat ); 

    double dedt = (strain_ - strain0_) / dt_; 

    myVars.set ( "strain" , strain_ );
    myVars.set ( "stress" , stress_ );
    myVars.set ( "dedt"   , dedt    );
    myVars.set ( "time"   , time_   );

    System::out() << "strain " << strain_ << endl; 
    System::out() << "stress " << stress_ << endl;  

    time0_   = time_;
    stress0_ = stress_;
    strain0_ = strain_;

    // ----------------------------------------
    // print homogenized def. grad. and homogenized
    // Cauchy stress

    System::out() << "homogenized deformation gradient: \n" << F1loc_ << endl;

    // homogenized deformation gradient rotated for the angle phi

    M33 Rot;
    M33 Fhat;

    Rot = 0.;

    Rot(0,0) =   ::cos(phi_);
    Rot(0,1) = - ::sin(phi_);
    Rot(1,0) =   ::sin(phi_);
    Rot(1,1) =   ::cos(phi_);
    Rot(2,2) =   1.;

    Fhat = 0.;

    Fhat = matmul (Rot, F1loc_);

    System::out() << "homogenized deformation gradient rotated for the angle phi\n" << 

                  Fhat << endl;

    double theta1 = theta0_ + phi_;

    double c1 = std::cos (theta1);
    double s1 = std::sin (theta1);

    Sbar1_ = 0.;

    Sbar1_(0,0) = s1*s1 * stress_;
    Sbar1_(0,1) = c1*s1 * stress_; 
    Sbar1_(1,0) = c1*s1 * stress_; 
    Sbar1_(1,1) = c1*c1 * stress_;

    System::out() << "homogenized Cauchy stress: \n" << Sbar1_ << endl;

    printStrains_ ( params, globdat );
    
    return true;
  }

  if ( action == Actions::ADVANCE )
  {
    time_ = time0_ + dt_;

    globdat.set ( Globdat::TIME, time_ );

    if ( stress_ < maxStress_ )
    {
      double dsigma = stressRate_ * dt_;

      stress_ = stress0_ + dsigma;

      if ( stress_ > maxStress_ ) { stress_ = maxStress_; }
    }

    return true;
  }

  if ( action == Actions::CANCEL )
  {
    time_   = time0_;
    stress_ = stress0_;

    return true;
  }

  return false;
}

//-----------------------------------------------------------------------
//   applyDisps_
//-----------------------------------------------------------------------

void CreepRveModel::applyDisps_

  ( const Properties& params,
    const Properties& globdat )

{
  idx_t                 nn;
  Assignable<NodeGroup> group;
  IntVector             inodes;
  String                context;

  for ( idx_t ig = 0; ig < numgroups_; ig++ )
  {
    group = NodeGroup::get ( nodeGroups_[ig], nodes_, globdat, context );

    nn = group.size();

    inodes.resize ( nn );
    inodes = group.getIndices ();

    idx_t itype = dofs_->findType ( dofTypes_[ig] );
    idx_t idof  = dofs_->getDofIndex ( inodes[0], itype );

    cons_->addConstraint ( idof );    

    for ( idx_t in = 1; in < nn; in++ )
    {
      idx_t jdof = dofs_->getDofIndex ( inodes[in], itype );

      cons_->addConstraint( jdof, idof, 1. );
    }
  }

  cons_->compress();
}

//-----------------------------------------------------------------------
//   connect_
//-----------------------------------------------------------------------

void CreepRveModel::connect_ ()

{
  using jem::util::connect;

  connect ( dofs_->newSizeEvent,  this, &CreepRveModel::dofsChanged_ );
  connect ( dofs_->newOrderEvent, this, &CreepRveModel::dofsChanged_ );

  dofs_->resetEvents ();
}

//-----------------------------------------------------------------------
//   dofsChanged_
//-----------------------------------------------------------------------

void CreepRveModel::dofsChanged_ ()

{
  // ulUpd_ = false;
}

//-----------------------------------------------------------------------
//   getExtVector_
//-----------------------------------------------------------------------


void CreepRveModel::getExtVector_ 

  ( const Properties& params,
    const Properties& globdat )
{
  Vector fext;

  params.get ( fext, ActionParams::EXT_VECTOR );

  idx_t                 nn;
  Assignable<NodeGroup> group;
  IntVector             inodes;
  String                context;

  for ( idx_t ig = 0; ig < numloadgroups_; ig++ )
  {
    group = NodeGroup::get ( loadGroups_[ig], nodes_, globdat, context );

    nn = group.size();

    inodes.resize ( nn );
    inodes = group.getIndices ();

    idx_t itype = dofs_->findType ( ldofTypes_[ig] );
    idx_t idof  = dofs_->getDofIndex ( inodes[0], itype );
    
    fext[idof] = stress_ * transformForce_[ig];     
  }

  // System::out() << fext << endl;

  dofs_->resetEvents ();
}

//-----------------------------------------------------------------------
//   updateTransform_
//-----------------------------------------------------------------------

void CreepRveModel::updateTransform_

    ( )
{
  double nom, denom, theta1;

  if ( F1loc_(1,0) == 0. )
  {
    nom = -F1loc_(0,0)*transform0_[1] + F1loc_(0,1)*transform0_[3];
    nom += F1loc_(1,1)*transform0_[1];

    denom  = F1loc_(0,0)*transform0_[0] - F1loc_(0,1)*transform0_[1];
    denom += F1loc_(1,1)*transform0_[3];
  }
  else if ( F1loc_(0,1) == 0. )
  {
    nom = -F1loc_(0,0)*transform0_[1] - F1loc_(1,0)*transform0_[0];
    nom += F1loc_(1,1)*transform0_[1];

    denom  = F1loc_(0,0)*transform0_[0] - F1loc_(1,0)*transform0_[1];
    denom += F1loc_(1,1)*transform0_[3];
  }
  else
  {
    throw Error ( JEM_FUNC, "Either cornerz or cornerx must be restrained \
     in shearing direction" );    
  }
  
  phi_   = atan(nom/denom);

  // phi_ = 0.;

  theta1 = theta0_ + phi_;

  double c1 = std::cos (theta1);
  double s1 = std::sin (theta1);

  double jac = jem::numeric::det (F1loc_);

  transformForce_[0] = jac * l2_*l3_* ( c1*c1 / F1loc_(1,1) - 

                                c1*s1 * F1loc_(1,0) / ( F1loc_(0,0)*F1loc_(1,1) ) );

  transformForce_[1] = jac * l2_*l3_ * c1*s1 / F1loc_(1,1);

  transformForce_[2] = jac * l1_*l2_ * c1*s1 / F1loc_(0,0);

  transformForce_[3] = jac * l1_*l2_* ( s1*s1 / F1loc_(0,0) - 

                                c1*s1 * F1loc_(0,1) / ( F1loc_(0,0)*F1loc_(1,1) ) );

  // System::out() << "force: " << transformForce_ << endl;

  // System::out() << "theta1: " << theta1 << endl;
}



//-----------------------------------------------------------------------
//   initMasters_
//-----------------------------------------------------------------------

void CreepRveModel::initMasters_  

  ( const Properties& globdat )

{
  idx_t                 nn;
  Assignable<NodeGroup> group;
  IntVector             inodes;
  String                context;
  ArrayBuffer<idx_t>    mbuf;

  for ( idx_t ig = 0; ig < numloadgroups_; ig++ )
  {
    group = NodeGroup::get ( loadGroups_[ig], nodes_, globdat, context );

    nn = group.size();

    inodes.resize ( nn );
    inodes = group.getIndices ();

    idx_t itype = dofs_->findType ( ldofTypes_[ig] );
    idx_t idof  = dofs_->getDofIndex ( inodes[0], itype );

    mbuf.pushBack ( idof );
  }   

  masters_.ref ( mbuf.toArray() );

  // System::out() << "Test \n" << masters_ << endl; 

  dofs_->resetEvents ();
}

//-----------------------------------------------------------------------
//   initDirection_
//-----------------------------------------------------------------------

void CreepRveModel::initDirection_  

  ( )

{
  // angle between global x-axis and carbon fiber (local 3-axis)

  theta0_ = (90. - offAngle_) * PI / 180.;

  c0_ = std::cos (theta0_);
  s0_ = std::sin (theta0_);

  transform0_[0] = c0_*c0_;
  transform0_[1] = c0_*s0_;
  transform0_[2] = c0_*s0_;
  transform0_[3] = s0_*s0_;

  double theta1 = theta0_ + phi_;

  double c1 = std::cos (theta1);
  double s1 = std::sin (theta1);

  double jac = jem::numeric::det (F1loc_);

  // either F1loc_(0,1) or F1loc_(1,0) must be zero, depending on BCs

  transformForce_[0] = jac * l2_*l3_* ( c1*c1 / F1loc_(1,1) - 

                                c1*s1 * F1loc_(1,0) / ( F1loc_(0,0)*F1loc_(1,1) ) );

  transformForce_[1] = jac * l2_*l3_ * c1*s1 / F1loc_(1,1);

  transformForce_[2] = jac * l1_*l2_ * c1*s1 / F1loc_(0,0);

  transformForce_[3] = jac * l1_*l2_* ( s1*s1 / F1loc_(0,0) - 

                                c1*s1 * F1loc_(0,1) / ( F1loc_(0,0)*F1loc_(1,1) ) );
}

//-----------------------------------------------------------------------
//   updateDefGrad_
//-----------------------------------------------------------------------

void CreepRveModel::updateDefGrad_  

  ( const Properties& globdat )

{
  Vector  disp1;

  StateVector::get    ( disp1, dofs_, globdat ); 

  F1loc_(0,0) = 1. + disp1[masters_[3]]/l3_;
  F1loc_(0,1) =      disp1[masters_[1]]/l1_;
  F1loc_(1,0) =      disp1[masters_[2]]/l3_;
  F1loc_(1,1) = 1. + disp1[masters_[0]]/l1_;
  F1loc_(2,2) = 1. + disp1[idofY_]     /l2_;
}

//-----------------------------------------------------------------------
//   updateStrain_
//-----------------------------------------------------------------------

void CreepRveModel::updateStrain_ 

  ( )

{
  M33 Q;
  M33 Qt;
  M33 R;
  M33 F1glob;
  M33 Temp;

  Q = 0.;
  R = 0.;

  Q(0,0) = sqrt(transform0_[0]); // cos
  Q(0,1) = sqrt(transform0_[3]); // sin
  Q(1,0) =-sqrt(transform0_[3]); 
  Q(1,1) = sqrt(transform0_[0]); 
  Q(2,2) = 1.;

  R(0,0) = cos(phi_);
  R(0,1) =-sin(phi_);
  R(1,0) = sin(phi_);
  R(1,1) = cos(phi_);
  R(2,2) = 1.;

  Qt = Q.transpose();

  Temp = matmul(R,F1loc_);

  F1glob = matmul(Qt, matmul(Temp,Q));

  strain_ = std::log( F1glob(1,1) );

  // System::out() << "F1glob: \n" << F1glob << endl;
}

//-----------------------------------------------------------------------
//    initWriter_
//-----------------------------------------------------------------------

Ref<PrintWriter>  CreepRveModel::initWriter_

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

void CreepRveModel::printStrains_

  ( const Properties&  params,
    const Properties&  globdat )

{
  using jive::util::Globdat;

  System::out() << "Printing strains!!\n";

  idx_t       it;

  if ( strainOut_ == nullptr )
  {
    strainOut_ = initWriter_ ( params, "defgrad" );
  }

  strainOut_->nformat.setFractionDigits(8);

  globdat.get ( it, Globdat::TIME_STEP    );

//  *strainOut_ << "Time step " << it << "\n";

  for ( idx_t r = 0; r < 3; ++r )
  {
      for ( idx_t k = 0; k < 3; ++k )
      {
        *strainOut_ << F1loc_ (r,k) << " ";
      }
  }

  for ( idx_t r = 0; r < 3; ++r )
  {
      for ( idx_t k = 0; k < 3; ++k )
      {
        *strainOut_ << Sbar1_ (r,k) << " ";
      }
  }

  *strainOut_ << time_ << "\n";

  strainOut_->flush();
}

//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------

Ref<Model> CreepRveModel::makeNew

  ( const String&     name,
    const Properties& conf,
    const Properties& props,
    const Properties& globdat )

{
  return newInstance<CreepRveModel> ( name, conf, props, globdat );
}

//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   declareCreepRveModel
//-----------------------------------------------------------------------

void declareCreepRveModel ()
{
  using jive::model::ModelFactory;

  ModelFactory::declare ( "CreepRve", 
                          & CreepRveModel::makeNew );
}
