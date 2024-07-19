/*
 *  TU Delft; Dragan Kovacevic 
 *
 *  2021
 *
 *  Arclength model with constraint equation based 
 *  on the global strain rate in the loading direction
 * 
 *  Loading direction assumed along global y-axis 
 *  model partially derived from Iuri's BCModel 
 *
 */

#include <jem/base/System.h>
#include <jem/base/Error.h>
#include <jem/base/array/operators.h>
#include <jem/base/array/select.h>
#include <jem/numeric/algebra/utilities.h>
#include <jem/numeric/utilities.h>
#include <jem/numeric/func/UserFunc.h>
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
#include <jive/util/FuncUtils.h>
#include <jive/model/Actions.h>
#include <jive/model/ModelFactory.h>
#include <jive/implict/ArclenActions.h>
#include <jive/model/StateVector.h>
#include <jive/implict/SolverInfo.h>

#include <jem/io/FileWriter.h>
#include <jem/io/PrintWriter.h>
#include <jem/util/StringUtils.h>

#include <math.h>

#include "SRArclenModel.h"
#include "SolverNames.h"

using jem::util::ArrayBuffer;
using jem::io::endl;
using jem::numeric::UserFunc;
using jem::numeric::MatmulChain;
using jem::numeric::inverse;
using jive::IdxVector;
using jive::Matrix;
using jive::model::Actions;
using jive::model::ActionParams;
using jive::util::FuncUtils;
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
//   class SRArclenModel
//=======================================================================


//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------

const char* SRArclenModel::NODEGROUPS_PROP  = "nodeGroups";
const char* SRArclenModel::LOADGROUPS_PROP  = "loadGroups";
const char* SRArclenModel::DOFS_PROP        = "dofs";
const char* SRArclenModel::LOADDOFS_PROP    = "loadDofs";
const char* SRArclenModel::STRAINRATE_PROP  = "strainRate";
const char* SRArclenModel::OFFANGLE_PROP    = "offAngle";
const char* SRArclenModel::MAXSTRAIN_PROP   = "maxStrain";

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------

SRArclenModel::SRArclenModel

  ( const String&     name,
    const Properties& conf,
    const Properties& props,
    const Properties& globdat ):

    Model ( name )

{
  time_  = 0.0;
  time0_ = 0.0;
  dt_    = 1.;

  strain0_ = 0.;
  strain_  = 0.;

  maxStrain_ = jem::maxOf ( maxStrain_ );

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
  strainRate_ = 1.e-4;

  ndofTypes_ = dofs_->typeCount();

  F0loc_ = 0.;

  configure ( props, globdat );
}

SRArclenModel::~SRArclenModel ()
{}

//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------

void SRArclenModel::configure

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
    myProps.get  ( strainRate_, STRAINRATE_PROP  );
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

  System::out() << "l1 " << l1_ << endl;
  System::out() << "l2 " << l2_ << endl;
  System::out() << "l3 " << l3_ << endl;

  initMasters_ (globdat);

  JEM_PRECHECK ( masters_.size() == 4 );

  transform0_      .resize(masters_.size());
  transformDefGrad_.resize(masters_.size());
  transformForce_  .resize(masters_.size());

  initDirection_ (); // init theta0_, transform0_

  F0loc_(0,0) = F0loc_(1,1) = F0loc_(2,2) = 1.;
}

//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------

void SRArclenModel::getConfig

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
  myConf.set ( STRAINRATE_PROP, strainRate_ );
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------

bool SRArclenModel::takeAction

  ( const String&     action,
    const Properties& params,
    const Properties& globdat )

{
  System::out () << "@SRArclenModel::takeAction(), action... " << action << endl;
  
  if ( action == ArclenActions::GET_UNIT_LOAD )
  {
    getUnitLoad_ ( params, globdat );

    return true;
  }

  if ( action == ArclenActions::GET_ARC_FUNC )
  {
    getArcFunc_ ( params, globdat );

    return true;
  }

  if ( action == SolverNames::SET_STEP_SIZE )
  {
    params.get ( dt_, SolverNames::STEP_SIZE );
    
    return true;
  }

  // if ( action == SolverNames::CONTINUE )
  // {
  //   initUnitLoad_ ( globdat );

  //   return true;
  // }

  if ( action == Actions::GET_CONSTRAINTS )
  {
    applyDisps_ ( params, globdat );

    return true;
  }

  if ( action == SolverNames::CHECK_COMMIT )       
  {
    double depsilon = strainRate_ * dt_;

    strain_ = strain0_ + depsilon;

    if ( fabs(strain_) > maxStrain_ )
    {
      System::out() << "SRArclenModel: Maximum strain reached. Terminating analysis.\n";

      params.set ( SolverNames::TERMINATE, "sure" );
    }

    return true;
  }

  if ( action == Actions::COMMIT )
  {
    Properties  myVars = Globdat::getVariables ( myName_, globdat );

    double depsilon = strainRate_ * dt_;

    strain_ = strain0_ + depsilon;

    time0_   = time_;
    strain0_ = strain_;

    myVars.set ( "strain" , strain_ );

    System::out() << "strain " << strain_ << endl;

    double lambda;

    Properties info = SolverInfo::get ( globdat );

    info.find ( lambda, SolverInfo::LOAD_SCALE ); 
    
    myVars.set ( "lambda", lambda );


    M33 F1;

    F1 = 0.;

    updateDefGrad_ (F1, globdat);

    F0loc_ = F1;

    // ----------------------------------------
    // print homogenized def. grad. and homogenized
    // Cauchy stress

    System::out() << "homogenized deformation gradient: \n" << F1 << endl;

    double c1 = transformDefGrad_[0] / c0_;
    double s1 = transformDefGrad_[1] / c0_;

    Sbar1_ = 0.;

    Sbar1_(0,0) = s1*s1 * lambda;
    Sbar1_(0,1) = c1*s1 * lambda;
    Sbar1_(1,0) = c1*s1 * lambda;
    Sbar1_(1,1) = c1*c1 * lambda;

    System::out() << "homogenized Cauchy stress: \n" << Sbar1_ << endl;

    printStrains_ ( params, globdat );
   // printStresses_ ( params, globdat );
    
    return true;
  }

  if ( action == Actions::ADVANCE )
  {
    time_ = time0_ + dt_;

    globdat.set ( Globdat::TIME, time_ );

    initTransform_ (globdat);

    return true;
  }

  if ( action == Actions::CANCEL )
  {
    time_   = time0_;

    return true;
  }

  return false;
}

//-----------------------------------------------------------------------
//   applyDisps_
//-----------------------------------------------------------------------

void SRArclenModel::applyDisps_

  ( const Properties& params,
    const Properties& globdat )

{
  idx_t                 nn;
  Assignable<NodeGroup> group;
  IdxVector             inodes;
  String                context;

  System::out() << "applyDisps " << numgroups_ << "\n";

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

void SRArclenModel::connect_ ()

{
  using jem::util::connect;

  connect ( dofs_->newSizeEvent,  this, &SRArclenModel::dofsChanged_ );
  connect ( dofs_->newOrderEvent, this, &SRArclenModel::dofsChanged_ );

  dofs_->resetEvents ();
}

//-----------------------------------------------------------------------
//   dofsChanged_
//-----------------------------------------------------------------------

void SRArclenModel::dofsChanged_ ()

{
  // ulUpd_ = false;
}

//-----------------------------------------------------------------------
//   initUnitLoad_
//-----------------------------------------------------------------------

void SRArclenModel::initUnitLoad_  

  ( const Properties& globdat )

{
  unitLoad_.resize ( dofs_->dofCount() );
  unitLoad_ = 0.; 

  idx_t                 nn;
  Assignable<NodeGroup> group;
  IdxVector             inodes;
  String                context;

  System::out() << "initUnitLoad_ " << numloadgroups_ << "\n";

  for ( idx_t ig = 0; ig < numloadgroups_; ig++ )
  {
    group = NodeGroup::get ( loadGroups_[ig], nodes_, globdat, context );

    nn = group.size();

    inodes.resize ( nn );
    inodes = group.getIndices ();

    idx_t itype = dofs_->findType ( ldofTypes_[ig] );
    idx_t idof  = dofs_->getDofIndex ( inodes[0], itype );
    
    unitLoad_[idof] = transformForce_[ig];    

    System::out() << "unitLoad_[idof] " <<  unitLoad_[idof] << " loadgroup " << loadGroups_[ig] << "\n";

  //   else
  //   {
  //     throw Error ( JEM_FUNC, "load should be specified either on cornerx or cornerz" );
  //   }
  }

  dofs_->resetEvents ();
}

//-----------------------------------------------------------------------
//   getUnitLoad_
//-----------------------------------------------------------------------

void SRArclenModel::getUnitLoad_
  
  ( const Properties& params,
    const Properties& globdat )

{
  initUnitLoad_ ( globdat );

  Vector f;

  params.get ( f, ArclenParams::UNIT_LOAD );

  if ( f.size() != unitLoad_.size() )
    throw Error ( JEM_FUNC, "unit load vector mismatch." );

  f = unitLoad_;
}

//-----------------------------------------------------------------------
//   getArcFunc_
//-----------------------------------------------------------------------

void SRArclenModel::getArcFunc_

  ( const Properties& params,
    const Properties& globdat )

{
  double omega, jac11;

  Vector jac10;

  Vector  disp1;
  Vector  disp  ( masters_.size() );
  Vector  coeff ( masters_.size() );

  StateVector::get    ( disp1, dofs_, globdat );

  disp = select ( disp1, masters_ );

  System::out() << "getArcFunc_. disp " << disp << " dt " << dt_ << endl;

  // M33 F1;

  // F1 = 0.;

  // not updating deg.grad. since the loading and transformations
  // are calculated with values from previous converged time step

  // updateDefGrad_ (F1, globdat);

  coeff[0] = transformDefGrad_[0] / l1_;
  coeff[1] = transformDefGrad_[1] / l1_;
  coeff[2] = transformDefGrad_[2] / l3_;
  coeff[3] = transformDefGrad_[3] / l3_;

  params.get ( jac10, ArclenParams::JACOBIAN10 );
 
  jac11 = 0.;
  jac10 = 0.;

  double depsilon = strainRate_ * dt_;

  double epsilonY = strain0_ + depsilon;

  omega  = dot (disp, coeff) + transformDefGrad_[0] + transformDefGrad_[3];
  omega -= exp(epsilonY);

  jac10[masters_] = coeff;

  System::out() << "omega " << omega << "\n";

  params.set ( ArclenParams::ARC_FUNC,   omega );
  params.set ( ArclenParams::JACOBIAN11, jac11 );
}

//-----------------------------------------------------------------------
//   initTransform_
//-----------------------------------------------------------------------

void SRArclenModel::initTransform_

    ( const Properties& globdat )
{
  double nom, denom, theta1;

  if ( F0loc_(1,0) == 0. )
  {
    nom = -F0loc_(0,0)*transform0_[1] + F0loc_(0,1)*transform0_[3];
    nom += F0loc_(1,1)*transform0_[1];

    denom  = F0loc_(0,0)*transform0_[0] - F0loc_(0,1)*transform0_[1];
    denom += F0loc_(1,1)*transform0_[3];
  }
  else if ( F0loc_(0,1) == 0. )
  {
    nom = -F0loc_(0,0)*transform0_[1] - F0loc_(1,0)*transform0_[0];
    nom += F0loc_(1,1)*transform0_[1];

    denom  = F0loc_(0,0)*transform0_[0] - F0loc_(1,0)*transform0_[1];
    denom += F0loc_(1,1)*transform0_[3];
  }
  else
  {
    throw Error ( JEM_FUNC, "Either cornerz or cornerx must be restrained \
     in shearing direction" );    
  }
  
  phi_   = atan(nom/denom);

//   phi_ = 0.;

  theta1 = theta0_ + phi_;

  updateTransform_ ( theta1 );

  System::out() << "theta1: " << theta0_ << " phi " << phi_ << endl;
}

//-----------------------------------------------------------------------
//   initMasters_
//-----------------------------------------------------------------------

void SRArclenModel::initMasters_  

  ( const Properties& globdat )

{
  idx_t                 nn;
  Assignable<NodeGroup> group;
  IdxVector             inodes;
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

//   System::out() << "Test \n" << masters_ << endl; 

  dofs_->resetEvents ();
}

//-----------------------------------------------------------------------
//   initDirection_
//-----------------------------------------------------------------------

void SRArclenModel::initDirection_  

  ( )

{
  // angle between global x-axis and carbon fiber (local 3-axis)

  theta0_ = (90. - offAngle_) * PI / 180.;

  double phi = 0.;

  c0_ = std::cos (theta0_);
  s0_ = std::sin (theta0_);

  double theta1 = theta0_ + phi;

  double c1 = std::cos (theta1);
  double s1 = std::sin (theta1);

  transform0_[0] = c0_*c0_;
  transform0_[1] = c0_*s0_;
  transform0_[2] = c0_*s0_;
  transform0_[3] = s0_*s0_;

  transformDefGrad_[0] = c0_*c1;
  transformDefGrad_[1] = c0_*s1;
  transformDefGrad_[2] = s0_*c1;
  transformDefGrad_[3] = s0_*s1;

  transformForce_[0] = c0_*c0_;
  transformForce_[1] = c0_*s0_;
  transformForce_[2] = c0_*s0_;
  transformForce_[3] = s0_*s0_;

 // System::out() << "initDirection. transformForce " << transformForce_ << "\n";
}

//-----------------------------------------------------------------------
//   updateDefGrad_
//-----------------------------------------------------------------------

void SRArclenModel::updateDefGrad_  

  (       M33&        F,
    const Properties& globdat )

{
  M33 F1loc;

  F1loc = 0.;

  Vector  disp1;

  StateVector::get    ( disp1, dofs_, globdat ); 

  F1loc(0,0) = 1. + disp1[masters_[3]]/l3_;
  F1loc(0,1) =      disp1[masters_[1]]/l1_;
  F1loc(1,0) =      disp1[masters_[2]]/l3_;
  F1loc(1,1) = 1. + disp1[masters_[0]]/l1_;
  F1loc(2,2) = 1. + disp1[idofY_]     /l2_;

  F = F1loc;

  System::out() << "@SRArclen. updateDefGrad " << F << "\n";
}

//-----------------------------------------------------------------------
//   updateTransform_
//-----------------------------------------------------------------------

void SRArclenModel::updateTransform_  

  ( const double&     theta1 )

{
  double c1 = std::cos (theta1);
  double s1 = std::sin (theta1);

  transformDefGrad_[0] = c0_*c1;
  transformDefGrad_[1] = c0_*s1;
  transformDefGrad_[2] = s0_*c1;
  transformDefGrad_[3] = s0_*s1;

  // either F0loc_(0,1) or F0loc_(1,0) must be zero, depending on BCs

  double jac = jem::numeric::det (F0loc_);

  transformForce_[0] = jac * l2_*l3_* ( c1*c1 / F0loc_(1,1) - 

                                c1*s1 * F0loc_(1,0) / ( F0loc_(0,0)*F0loc_(1,1) ) );

  transformForce_[1] = jac * l2_*l3_ * c1*s1 / F0loc_(1,1);

  transformForce_[2] = jac * l1_*l2_ * c1*s1 / F0loc_(0,0);

  transformForce_[3] = jac * l1_*l2_* ( s1*s1 / F0loc_(0,0) - 

                                c1*s1 * F0loc_(0,1) / ( F0loc_(0,0)*F0loc_(1,1) ) );

//  System::out() << "updateTransform. transformForce " << transformForce_ << endl;
}

//-----------------------------------------------------------------------
//    initWriter_
//-----------------------------------------------------------------------

Ref<PrintWriter>  SRArclenModel::initWriter_

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

void SRArclenModel::printStrains_

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

  strainOut_->nformat.setFractionDigits( 8 );

  globdat.get ( it, Globdat::TIME_STEP    );

//  *strainOut_ << "Time step " << it << "\n";

  for ( idx_t r = 0; r < 3; ++r )
  {
      for ( idx_t k = 0; k < 3; ++k )
      {
        *strainOut_ << F0loc_ (r,k) << " ";
      }
  }

  for ( idx_t r = 0; r < 3; ++r )
  {
      for ( idx_t k = 0; k < 3; ++k )
      {
        *strainOut_ << Sbar1_ (r,k) << " ";
      }
  }

  *strainOut_ << dt_ << "\n";

  strainOut_->flush();
}

//-----------------------------------------------------------------------
//    printStresses_
//-----------------------------------------------------------------------

void SRArclenModel::printStresses_

  ( const Properties&  params,
    const Properties&  globdat )

{
  using jive::util::Globdat;

  System::out() << "Printing stresses!!\n";

  idx_t       it;

  if ( stressOut_ == nullptr )
  {
    stressOut_ = initWriter_ ( params, "stresses" );
  }

  globdat.get ( it, Globdat::TIME_STEP    );

//  *stressOut_ << "Time step " << it << "\n";

  for ( idx_t r = 0; r < 3; ++r )
  {
      for ( idx_t k = 0; k < 3; ++k )
      {
        *stressOut_ << Sbar1_ (r,k) << " ";
      }
  }

  *stressOut_ << "\n";

  stressOut_->flush();
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------

Ref<Model> SRArclenModel::makeNew

  ( const String&     name,
    const Properties& conf,
    const Properties& props,
    const Properties& globdat )

{
  return newInstance<SRArclenModel> ( name, conf, props, globdat );
}

//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   declareSRArclenModel
//-----------------------------------------------------------------------

void declareSRArclenModel ()
{
  using jive::model::ModelFactory;

  ModelFactory::declare ( "SRArclen", 
                          & SRArclenModel::makeNew );
}
