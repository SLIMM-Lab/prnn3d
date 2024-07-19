/*
 *  TU Delft 
 *
 *  Iuri Barcelos, August 2018
 *
 *  BC model adapted from VEVPLoadModel. 
 *
 */

#include <jem/base/System.h>
#include <jem/base/Error.h>
#include <jem/base/array/operators.h>
#include <jem/base/array/select.h>
#include <jem/numeric/algebra/utilities.h>
#include <jem/numeric/utilities.h>
#include <jem/numeric/func/UserFunc.h>
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

#include <jive/app/ChainModule.h>
#include <jive/app/InfoModule.h>
#include <jive/fem/InitModule.h>
#include <jem/io/FileReader.h>
#include <jive/model/Model.h>

#include <math.h>
#include "utilities.h"

#include "BCModel.h"
#include "SolverNames.h"
#include "BFGSModule.h"
#include "GPInputModule.h"
#include "LearningNames.h"
#include "IsoRBFKernel.h"

#include <jem/io/FileWriter.h>
#include <jem/io/PrintWriter.h>
#include <jem/util/StringUtils.h>

using jem::util::ArrayBuffer;
using jem::io::endl;
using jem::numeric::UserFunc;
using jive::IntVector;
using jive::model::Actions;
using jive::model::ActionParams;
using jive::util::FuncUtils;
using jive::util::Globdat;
using jive::model::StateVector;
using jive::implict::ArclenActions;
using jive::implict::ArclenParams;
using jive::implict::SolverInfo;

using jive::app::ChainModule;
using jive::app::InitModule;
using jive::app::InfoModule;
using jive::model::Model;

using jem::io::FileReader;

using jive::implict::BFGSModule;

using jem::io::FileWriter;
using jem::io::PrintWriter;
using jem::util::StringUtils;

//=======================================================================
//   class BCModel
//=======================================================================


//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------

const char* BCModel::MODE_PROP        = "mode";
const char* BCModel::NODEGROUPS_PROP  = "nodeGroups";
const char* BCModel::DOFS_PROP        = "dofs";
const char* BCModel::UNITVEC_PROP     = "unitVec";
const char* BCModel::SHAPE_PROP       = "shape";
const char* BCModel::STEP_PROP        = "step";
const char* BCModel::ALENGROUP_PROP   = "arclenGroup";
const char* BCModel::LOAD_PROP        = "loads";      
const char* BCModel::MAXNORM_PROP     = "maxNorm";
const char* BCModel::SHAPE_GP         = "shapeGP";
const char* BCModel::INIT_HYPERS      = "initHypers";

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------

BCModel::BCModel

  ( const String&     name,
    const Properties& conf,
    const Properties& props,
    const Properties& globdat ):

    Model ( name )

{
  rseed_ = 110;
   
  l1_ = 1.e-2;
  l2_ = 1.e-2;
  l3_ = 1.e-2;
   
  time_  = 0.0;
  time0_ = 0.0;

  dt_   = 1.;
  stepF_ = 1.;
  
  unitVec_    = 0.;
  initVals_   = 0.;

  mode_      = LOAD;

  useShapeGP_ = false;
  updateGP_ = true;
  numGP_ = 6;

  ulUpd_     = false;
  master_    = 0;
  alenGroup_ = 0;	  

  maxNorm_   = jem::maxOf ( maxNorm_ );

  pLoad_     = nullptr;

  nodes_ = NodeSet::find    ( globdat );
  dofs_  = XDofSpace::get   ( nodes_.getData(), globdat );
  cons_  = Constraints::get ( dofs_, globdat );

  ndofTypes_ = dofs_->typeCount();

  configure ( props, globdat );

  meanprev_.resize ( numGP_ );
  meanprev_ = 1e10; 

  Properties myProps = props.getProps ( myName_ );
  
  myProps.find ( useShapeGP_ , SHAPE_GP );
  myProps.find ( rseed_, "rseed" );

  String mode;
  myProps.find ( mode, MODE_PROP );
  
  if ( useShapeGP_ )
  {
     if ( mode == "arclen" ) numGP_ = 1;

     if ( myProps.find ( hpFiles_, INIT_HYPERS ) )
     {
        if ( hpFiles_.size() <= 0 )
        {
          throw Error ( JEM_FUNC,
          "Please specify the initial hyperparameters of the GP" );
        }
     }

     gp_        .resize ( numGP_ );
     gpGlobdat_ .resize ( numGP_ );
     gpData_    .resize ( numGP_ );

     double precision = 1e-6;
     int nrestarts  = 1;
     int maxiter = 2;

     Properties gpProps, gpConf;

     gpProps.set ( "model.type", "GP");
     
     IdxVector idxinput(1); IdxVector idxoutput(1);
     IdxVector idxdt(1); 
     idxinput = 0; idxdt = 1; idxoutput = 2;
  
     gpProps.set ( "model.kernel.type", "IsoRBF" );
     gpProps.set ( "solver.precision", precision );
     gpProps.set ( "solver.nRestarts", nrestarts );
     gpProps.set ( "solver.maxIter", maxiter );
     gpProps.set ( "userinput.file", "sampleprior.data" );
     gpProps.set ( "userinput.input", idxinput );
     gpProps.set ( "userinput.output", idxoutput );
     gpProps.set ( "userinput.dt", idxdt );
 
     for ( idx_t i = 0; i < numGP_; ++i )
     {
 	System::out() << "Creating GP " << i << "\n";
        Ref<ChainModule> chain;
	gpProps.set ( "model.rseed", rseed_*(i+1));

	gpGlobdat_[i] = Globdat::newInstance ( "GP" + String(i) );
	gpGlobdat_[i].set("rseed", rseed_*(i+1));

	Globdat::getVariables ( gpGlobdat_[i] );

	chain = newInstance<ChainModule> ();

	chain->pushBack ( newInstance<GPInputModule> ( "userinput" ) ); 
	chain->pushBack ( newInstance<InitModule> ( "init" ));
	chain->pushBack ( newInstance<InfoModule> ( "info" ));
        chain->pushBack ( newInstance<BFGSModule> ( "solver" ));         

        System::out() << "Created GP chain.\n";

	chain->configure ( gpProps, gpGlobdat_[i] );
        chain->getConfig ( gpConf, gpGlobdat_[i] );
	chain->init ( gpConf, gpProps, gpGlobdat_[i] );
	
	gp_[i] = chain;
	
	gpData_[i] = TrainingData::get ( gpGlobdat_[i], getContext() ); 
        gpData_[i]->configure ( gpProps, gpGlobdat_[i] );
       
        // Read input info to define the GP

        Vector nbounds ( 2 );
        Vector sbounds ( 2 );
        Vector lbounds ( 2 );

        double variance, length, noise;

	System::out() << "Read HP file "<< hpFiles_ << "\n";
        Ref<FileReader> in = newInstance<FileReader> ( hpFiles_[0] );
	System::out() << "Finished reading HP\n";

        variance = in->parseFloat();
        length   = in->parseFloat();
        noise    = in->parseFloat();

	if ( i == 2 ) variance = variance / 5.;
//	if ( i > 2 ) variance = variance / 2.;

        System::out() << "Variance: " << variance << " length: " << length << 
		" noise: " << noise << ".\n";

        sbounds[0] = 0.1 * variance; sbounds[1] = 10.0 * variance;
        lbounds[0] = 0.1 * length;   lbounds[1] = 10.0 * length;
        nbounds[0] = 0.1 * noise;    nbounds[1] = 10.0 * noise;

        Properties params;

	gpProps.set ( "model.kernel.noise", noise );
        gpProps.set ( "model.kernel.variance", variance );
        gpProps.set ( "model.kernel.lengthScale", length   );

	gpProps.set ( "model.kernel.nBounds", nbounds   );
        gpProps.set ( "model.kernel.sBounds", sbounds);
        gpProps.set ( "model.kernel.lBounds", lbounds);

        // Configure GP with info for the Kernel

	gp_[i]->configure ( gpProps, gpGlobdat_[i] );

	System::out() << "GP is configured.\n";

	Ref<Model> model = Model::get ( gpGlobdat_[i], "" );

	// Updating GP with data in the prior file

	System::out() << "Updating GP\n";
	
	model->takeAction ( LearningActions::UPDATE, params, gpGlobdat_[i] );
       }
    }

  System::out() << "BCModel constructor ended.\n";
}

BCModel::~BCModel ()
{}

//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------

void BCModel::configure

  ( const Properties& props,
    const Properties& globdat )

{
  System::out() << "BCModel::configure.\n";
 
  const String  context = getContext ();

  System::out() << "ndofTypes_ " << ndofTypes_ << "\n";
  IdxVector     jtypes ( ndofTypes_ );

  if ( props.contains( myName_ ) )
  {
    Properties myProps = props.getProps ( myName_ );

    String mode;
    myProps.get ( mode, MODE_PROP );

    if ( mode == "load" )
      mode_ = LOAD;
    else if ( mode == "disp" )
      mode_ = DISP;
    else if ( mode == "arclen" )
      mode_ = ALEN;
    else
      throw IllegalInputException ( JEM_FUNC,
	    "Invalid BC mode. Choose either 'load', 'disp' or 'arclen'." );

    myProps.get  ( nodeGroups_, NODEGROUPS_PROP );
    numgroups_ = nodeGroups_.size ( );

    System::out() << "node groups " << numgroups_ << "\n";

    myProps.get  ( dofTypes_,  DOFS_PROP      );
    myProps.get  ( unitVec_,   UNITVEC_PROP   );

    symmetric_ = false;
    myProps.find ( symmetric_, "symmetric" );
    myProps.find ( maxNorm_, MAXNORM_PROP );

    if ( mode_ == ALEN )
    {
      myProps.get ( alenGroup_, ALENGROUP_PROP );
      myProps.find ( maxNorm_, MAXNORM_PROP );

      if ( alenGroup_ == -1 )
      {
	Properties loadProps;

	if ( myProps.find ( loadProps, LOAD_PROP ) )
	{
	  System::out() << "BCModel::configure. Find loadprops.\n";
	  if ( pLoad_ == nullptr )
	  {
            System::out() << "BCModel::configure. Create pload.\n";
	    pLoad_ = newInstance<PointLoadModel> ( LOAD_PROP );
	    pLoad_->configure ( myProps, globdat );
	  }
	}
	else
	{
          throw IllegalInputException ( JEM_FUNC,
	    "Invalid arclength mode. Either specify a group or give PointLoadModel data." );
	}
      }
    }

    String args = "t";
 
    shape_  = FuncUtils::newFunc ( args, SHAPE_PROP, myProps, globdat );
    FuncUtils::resolve ( *shape_, globdat );

    initVals_.resize ( numgroups_ );
    initVals_ = 0.0;

    if ( !useShapeGP_ ) 
    {
      meanprev_.resize(numgroups_);
      meanprev_ = 0.0;
    }

    if ( dofTypes_.size() != numgroups_ || unitVec_.size() != numgroups_ )
      throw IllegalInputException ( JEM_FUNC,
	    "nodeGroups, dofTypes and unitVector must have the same size." );

    myProps.find  ( dt_,  STEP_PROP  );
    myProps.get  ( stepF_, "stepF"     );
  
    // Get all dofs for describing the deformation gradient F
    
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

  NodeGroup zgroup = NodeGroup::get ( "cornerz", nodes_, globdat, context );

  nn = zgroup.size();

  zinodes.resize ( nn );
  zinodes = zgroup.getIndices ();

  idofsY.resize ( nn , ndofTypes_ );

  ndofTypes_ = dofs_->getDofsForItem ( idofsY(0,ALL), jtypes, yinodes[0] );

  idofXX_ = dofs_->getDofIndex (xinodes[0], jtypes[0] );
  idofXY_ = dofs_->getDofIndex (xinodes[0], jtypes[1] );
  idofXZ_ = dofs_->getDofIndex (xinodes[0], jtypes[2] );
  idofYX_ = dofs_->getDofIndex (yinodes[0], jtypes[0] );
  idofYY_ = dofs_->getDofIndex (yinodes[0], jtypes[1] );
  idofYZ_ = dofs_->getDofIndex (yinodes[0], jtypes[2] );
  idofZX_ = dofs_->getDofIndex (zinodes[0], jtypes[0] );
  idofZY_ = dofs_->getDofIndex (zinodes[0], jtypes[1] );
  idofZZ_ = dofs_->getDofIndex (zinodes[0], jtypes[2] );

  System::out() << "idofXX " << idofXX_ << " idofXY " << idofXY_ << " idofXZ " << idofXZ_ << "\n";
  System::out() << "idofYX " << idofYX_ << " idofYY " << idofYY_ << " idofYZ " << idofYZ_ << "\n";
  System::out() << "idofZX " << idofZX_ << " idofZY " << idofZY_ << " idofZZ " << idofZZ_ << "\n";

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
  
  Ref<PrintWriter> out = newInstance<PrintWriter> ( &System::out() );
  out->nformat.setFractionDigits( 10 );

  *out << "l1 " << l1_ << " l2 " << l2_ << " l3 " << l3_ << "\n";
  //*out << "corxcoords " << corxcoords(0,0) << " " << corycoords(0,0) << "\n";
  // Make deformation gradient symmetric

  if ( symmetric_ ) //&& mode_ == DISP )
  {
    symFactors_.resize( 9 );
    symFactors_ = 1.;
    symFactors_[3] = l2_/l1_;
    symFactors_[6] = l3_/l1_;
    symFactors_[7] = l3_/l2_;
  }
  
  System::out() << "BCModel::configure ended.\n";
  }
}

//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------

void BCModel::getConfig

  ( const Properties& conf,
    const Properties& globdat ) const

{
  Properties myConf = conf.makeProps ( myName_ );

  String mode;

  if ( mode_ == LOAD )
    mode = "load";
  else if ( mode_ == DISP )
    mode = "disp";
  else
    mode = "undefined";

  myConf.set ( MODE_PROP, mode              );
  myConf.set ( NODEGROUPS_PROP, nodeGroups_ );
  myConf.set ( DOFS_PROP, dofTypes_         );
  myConf.set ( UNITVEC_PROP, unitVec_       );
  myConf.set ( STEP_PROP, dt_               );
  myConf.set ( "stepF", stepF_              );

  if ( pLoad_ != nullptr )
  {
    pLoad_->getConfig ( myConf, globdat );
  }

  myConf.set ( MAXNORM_PROP, maxNorm_ );

  FuncUtils::getConfig ( myConf, shape_, SHAPE_PROP   ); 
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------

bool BCModel::takeAction

  ( const String&     action,
    const Properties& params,
    const Properties& globdat )

{
  System::out() << "@BCModel::takeAction. Action: " << action << ".\n";

  if ( action == ArclenActions::GET_UNIT_LOAD && mode_ == ALEN )
  {
    getUnitLoad_ ( params, globdat );

    return true;
  }

  if ( action == ArclenActions::GET_ARC_FUNC && mode_ == ALEN )
  {
    getArcFunc_ ( params, globdat );

    return true;
  }

  if ( action == SolverNames::SET_STEP_SIZE )
  {
    params.get ( dt_, SolverNames::STEP_SIZE );

    return true;
  }

  if ( action == Actions::GET_EXT_VECTOR && mode_ == LOAD )
  {
    Vector f;

    params.get ( f, ActionParams::EXT_VECTOR );

    applyLoads_ ( f, globdat );

    return true;
  }

  if ( action == Actions::GET_CONSTRAINTS )
  {
    if ( mode_ == LOAD )
      return false;

    applyDisps_ ( params, globdat );

    return true;
  }


  if ( action == SolverNames::CHECK_COMMIT && 
        mode_ == ALEN                          )
       
  {
    Vector u;

    StateVector::get ( u, dofs_, globdat );

    double norm = (pLoad_ == nullptr) ? fabs(u[master_]) : jem::numeric::norm2 ( u[masters_] );

    System::out() << "BCModel: Current displacement norm " << norm << "\n";
    System::out() << "BCModel: Max norm " << maxNorm_ << '\n';

    if ( norm > maxNorm_ )
    {
      System::out() << "BCModel: Maximum displacement norm reached. Terminating analysis.\n";

      params.set ( SolverNames::TERMINATE, "sure" );
    }

    printStrains_ ( params, globdat );

    return true;
  }


  if ( action == SolverNames::CHECK_COMMIT && 
        mode_ == DISP                          )
       
  {
    Vector u;

    StateVector::get ( u, dofs_, globdat );

    double norm = jem::numeric::norm2 ( 0.01*meanprev_ );

    System::out() << "BCModel: Current displacement norm " << norm << "\n";
    System::out() << "BCModel: Max norm " << maxNorm_ << '\n';

    if ( norm > maxNorm_ )
    {
      System::out() << "BCModel: Maximum displacement norm reached. Terminating analysis.\n";

      params.set ( SolverNames::TERMINATE, "sure" );
    }

    printStrains_ ( params, globdat );

    return true;
  }

  if ( action == Actions::COMMIT )
  {
    time0_ = time_;

    double lambda;

    Properties info = SolverInfo::get ( globdat );

    if ( info.find ( lambda, SolverInfo::LOAD_SCALE ) )
    {
      Properties  myVars = Globdat::getVariables ( myName_, globdat );

      myVars.set ( "lambda", lambda );
    }
    
    if ( useShapeGP_ )
    {    
      idx_t time;
      globdat.get ( time, Globdat::TIME_STEP );
      Vector inputs(1);  Vector outputs(1);
      Properties params; String context;
      
      for ( idx_t comp = 0; comp < numGP_ ;  ++comp )
      {
        inputs[0] = (double)(time);
        outputs[0] = meanprev_[comp];
        Ref<Model> model = Model::get ( gpGlobdat_[comp], context );
        gpData_[comp]->addData ( inputs, outputs );
        model->takeAction ( LearningActions::UPDATE, params, gpGlobdat_[comp]);   
      } 

      updateGP_ = true;
    }

    //System::out() << "commit. unitLoad " << unitLoad_[masters_] << "\n";
    
    return true;
  }

  if ( action == Actions::ADVANCE )
  {
    time_ = time0_ + dt_;

    globdat.set ( Globdat::TIME, time_ );

    return true;
  }

  if ( action == Actions::CANCEL )
  {
    time_ = time0_;

    return true;
  }

  return false;
}

//-----------------------------------------------------------------------
//   applyLoads_
//-----------------------------------------------------------------------

void BCModel::applyLoads_

  ( const Vector&     fext,
    const Properties& globdat )

{
  System::out() << "BCModel::applyLoads_.\n";
  idx_t                 nn;
  Assignable<NodeGroup> group;
  IdxVector             itype ( 1 );
  IdxVector             inodes;
  IdxVector             idofs;
  String                context;

  for ( idx_t ig = 0; ig < numgroups_; ig++ )
  {
    group = NodeGroup::get ( nodeGroups_[ig], nodes_, globdat, context );

    nn = group.size();

    inodes.resize ( nn );
    idofs.resize ( nn );
    inodes = group.getIndices ();

    itype[0] = dofs_->findType ( dofTypes_[ig] );

    double loadfunc = 0.;
    double time = 0.;
    idx_t timeStep;
    
    globdat.get ( timeStep, Globdat::TIME_STEP );

//    loadfunc = shape_->eval ( time );
    
    loadfunc = shape_->eval ( (timeStep-1)*stepF_ );

    double val = unitVec_[ig] * loadfunc / nn;

    dofs_->findDofIndices ( idofs, inodes, itype );

   /* System::out() << "time: " << time << " loadfunc: " << loadfunc << ".\n";
    System::out() << " val: " << val << ".\n";
    System::out() << "fext: " << fext << ".\n";
*/
    select ( fext, idofs ) += val;

 //   System::out() << "fext up: " << fext << ".\n";

    // Set load variable (for printing)

    Properties  myVars = Globdat::getVariables ( myName_, globdat );

    myVars.set ( "load", loadfunc );

  }
}

//-----------------------------------------------------------------------
//   applyDisps_
//-----------------------------------------------------------------------

void BCModel::applyDisps_

  ( const Properties& params,
    const Properties& globdat )

{
  System::out() << "applyDisps.\n";
  idx_t                 nn;
  Assignable<NodeGroup> group;
  IntVector             inodes;
  String                context;

  Vector state;
  Vector gpUnitVec ( numgroups_ );
  Vector mean( numGP_ );

  if ( useShapeGP_ && mode_ == DISP )
  {
    Properties params;
    String context ( "Evaluation of GP for shape function ");

    idx_t time;
    globdat.get ( time, Globdat::TIME_STEP );
    Ref<NData> data = newInstance<NData> ( 1, 1, 1 );
    System::out() << "time " << time << " timestep size " << dt_ << "\n";
    for ( idx_t t = 0; t < 1; ++t ) data->inputs(0, t) = (double)(time); 
    params.set ( LearningParams::INPUT, data );	
  
    // Evaluate GPs for each displacement dof

    for ( idx_t comp = 0; comp < numGP_;  ++comp )
    {
      Ref<Model> model = Model::get ( gpGlobdat_[comp], context );

      model->takeAction ( LearningActions::SAMPLEPOSTERIOR, params, gpGlobdat_[comp]);     
     //if ( comp == 2 )
     if ( false )
     {
       mean[comp+1] = 0.0;
     }
     else 
     {
      /* if ( time > 1 )
       {
         if ( abs(data->outputs(0,0) - meanprev_[comp+1]) < dt_ )
         {*/
           mean[comp] = data->outputs(0,0);
       /*  }
         else
         {
           if ( data->outputs(0,0) - meanprev_[comp+1] < 0.0 )
	   {  
	     mean[comp+1] = meanprev_[comp+1] - 1e-5; 
	   }
	   else
	   {
	     mean[comp+1] = meanprev_[comp+1] + 1e-5;
	   }

           System::out() << "GP prev " << meanprev_[comp+1] << " GP current " << data->outputs(0,0) << " GP corrected " << mean[comp+1] << "\n";
         }
       }
       else
       {
          mean[comp+1] = data->outputs(0,0);
       }*/
    }
    //  System::out() << "Posterior GP " << comp << " Input " << data->inputs(ALL, ALL) << " Outputs " << data->outputs(ALL, ALL) << "\n";

    }
     
     // Normalize unitVector
      
   /*  double norm = sqrt ( dot (mean) );
           
     gpUnitVec = mean/norm;
   */ 
   
    gpUnitVec[slice(0, 3)] = mean[slice(0,3)]; // Diagonal
    gpUnitVec[3] = gpUnitVec[4] = mean[3];     // Off-diagonal
    gpUnitVec[5] = gpUnitVec[6] = mean[4];
    gpUnitVec[7] = gpUnitVec[8] = mean[5];

    meanprev_ = mean;
    System::out() << "Normalized gpUnitVec " << gpUnitVec << "\n";       
    System::out() << "GP mean " << mean << "\n";       
     

/*	 params.set ( LearningParams::INPUT, inpGP );
         model->takeAction ( LearningActions::RECALL, params, gpGlobdat_[0] );
         params.get ( mean,  LearningParams::OUTPUT );
	 System::out() << "RECALL. input " << data->inputs(ALL, 0) << "mean " << mean << "\n"; //<< " variance " << var <<  "\n";*/
	 
  }

  System::out() << "BCModel. numgroups " << numgroups_ << "\n";

  for ( idx_t ig = 0; ig < numgroups_; ig++ )
  {
    group = NodeGroup::get ( nodeGroups_[ig], nodes_, globdat, context );

    nn = group.size();

    inodes.resize ( nn );
    inodes = group.getIndices ();

    idx_t itype = dofs_->findType ( dofTypes_[ig] );
    idx_t idof  = dofs_->getDofIndex ( inodes[0], itype );

    if ( mode_ == DISP || ig != alenGroup_ )
    {
      idx_t timeStep = 0;
      globdat.get ( timeStep, Globdat::TIME_STEP );

      double actualStep = stepF_;

      double loadfunc = shape_->eval ( (timeStep-1)* actualStep );
     
      if ( symmetric_ )
      {
        actualStep = stepF_*symFactors_[ig];
      }

      //System::out() << "BCModel. Node: " << inodes[0] << "\n";

      double val = 0.0;
      
      if ( useShapeGP_ ) 
      {
        val = unitVec_[ig] * gpUnitVec[ig]; // * loadfunc; 
   //     System::out() << "BCModel. unitVec_[ig] " << unitVec_[ig] << " gpUnitVec " <<  gpUnitVec[ig] << " loadfunc " << loadfunc << "\n";
      }
      else
      {    
        val = unitVec_[ig] * loadfunc + initVals_[ig];
	meanprev_[ig] = val;
        //System::out() << "BCModel. unitVec_[ig] " << unitVec_[ig] << " timeStep " << timeStep << "\n";
       // System::out() << "BCModel. loadfunc " <<  loadfunc << " val " << val << "\n";
      }
 
     // System::out() << "BCModel. idof " << idof << " val " << val << "\n";

      if ( unitVec_[ig] == 0.0 )
	cons_->addConstraint ( idof );
      else
	cons_->addConstraint ( idof, val );

    }

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

void BCModel::connect_ ()

{
  using jem::util::connect;

  connect ( dofs_->newSizeEvent,  this, &BCModel::dofsChanged_ );
  connect ( dofs_->newOrderEvent, this, &BCModel::dofsChanged_ );

  dofs_->resetEvents ();
}

//-----------------------------------------------------------------------
//   dofsChanged_
//-----------------------------------------------------------------------

void BCModel::dofsChanged_ ()

{
  ulUpd_ = false;
}

//-----------------------------------------------------------------------
//   initUnitLoad_
//-----------------------------------------------------------------------

void BCModel::initUnitLoad_  

  ( const Properties& globdat )

{
  String                context;

  unitLoad_.resize ( dofs_->dofCount() );
  unitLoad_ = 0.0;

  if ( pLoad_ == nullptr )
  {
    Assignable<NodeGroup> group = 
      NodeGroup::get ( nodeGroups_[alenGroup_], nodes_, globdat, context );

    idx_t nn = group.size();

    IntVector inodes;
    inodes.resize ( nn );
    inodes = group.getIndices ();

    idx_t itype = dofs_->findType ( dofTypes_[alenGroup_] );

    master_  = dofs_->getDofIndex ( inodes[0], itype );

    unitLoad_[master_] = 1.0;
  }
  else
  {
    Properties params;
    params.set ( ActionParams::SCALE_FACTOR, 1.      );
    params.set ( ActionParams::EXT_VECTOR, unitLoad_ );

    pLoad_->takeAction ( Actions::GET_EXT_VECTOR, params, globdat );

    ArrayBuffer<idx_t>  mbuf;
    ArrayBuffer<double> sbuf;

    for ( idx_t idof = 0; idof < dofs_->dofCount(); ++idof )
    {
      if ( jem::numeric::abs ( unitLoad_[idof] ) > 0.0 )
      {
        mbuf.pushBack ( idof );
	sbuf.pushBack ( unitLoad_[idof] / jem::numeric::abs ( unitLoad_[idof] ) );

//	System::out() << "array buffer " << mbuf << "\n"; 
      }
    }

    masters_.ref ( mbuf.toArray() );
    signs_.  ref ( sbuf.toArray() );

    //double max = 0.0;

    //for ( idx_t idof = 0; idof < dofs_->dofCount(); ++idof )
    //{
    //  double val = jem::numeric::abs ( unitLoad_[idof] );

    //  if ( val > max )
    //  {
    //    max = val;
    //    master_ = idof;
    //  }
    //}
  }

  dofs_->resetEvents ();

//  System::out() << "initUnitLoad " << unitLoad_[masters_] << "\n";

  ulUpd_ = true;
}

//-----------------------------------------------------------------------
//   getUnitLoad_
//-----------------------------------------------------------------------

void BCModel::getUnitLoad_
  
  ( const Properties& params,
    const Properties& globdat )

{
  if ( !ulUpd_ )
    initUnitLoad_ ( globdat );

  Vector f;

  params.get ( f, ArclenParams::UNIT_LOAD );

  if ( f.size() != unitLoad_.size() )
    throw Error ( JEM_FUNC, "unit load vector mismatch." );

  System::out() << "getUnitLoad " << unitLoad_[masters_] << "\n";

  f = unitLoad_;
}

//-----------------------------------------------------------------------
//   getArcFunc_
//-----------------------------------------------------------------------

void BCModel::getArcFunc_

  ( const Properties& params,
    const Properties& globdat )

{
  double phi, jac11, loadfunc, val;

  idx_t timeStep;

  Vector u, jac10;

  StateVector::get ( u, dofs_, globdat );

  globdat.get ( timeStep, Globdat::TIME_STEP );

  // Evaluate loading function

  if ( useShapeGP_ )
  {
    Properties params;
    String context ( "Evaluation of GP for shape function ");

    Ref<NData> data = newInstance<NData> ( 1, 1, 1 );
    data->inputs  = (double)(timeStep);

    // Evaluate GP for the shape function

    params.set ( LearningParams::INPUT, data );
    Ref<Model> model = Model::get ( gpGlobdat_[0], context );
    model->takeAction ( LearningActions::SAMPLEPOSTERIOR, params, gpGlobdat_[0]);
    if ( updateGP_ == false )
    {
      loadfunc = meanprev_[0];
    }
    else
    {
      loadfunc = data->outputs(0,0); 
      meanprev_[0] = loadfunc;
      updateGP_ = false;
    }
  }
  else
  {
    loadfunc = shape_->eval ( (timeStep-1)*stepF_ );
  }

  System::out() << "getArcFunc. time globdat: " << timeStep << " loadfunc: " << loadfunc << ".\n";

  params.get ( jac10, ArclenParams::JACOBIAN10 );

  jac11 = 0.0;
  jac10 = 0.0;

  if ( alenGroup_ != -1 )
  {
    val = unitVec_[alenGroup_] * loadfunc + initVals_[alenGroup_];

    phi   = u[master_] - val;

    jac10[master_] = 1.0;
  }
  else
  {
    double norm = jem::numeric::norm2 ( u[masters_] );

    phi = dot ( u[masters_], signs_ ) - loadfunc;
    jac10[masters_] = signs_;

    //phi = sum ( u[masters_] ) - loadfunc;
    //jac10[masters_] = 1.0;

    System::out() << "u[masters] = " << u[masters_] << "\n"; // added by marina
    System::out() << "signs = " << signs_ << ".\n";
    System::out() << "dot (u, signs) " << dot ( u[masters_], signs_ ) << "\n";
    System::out() << "loadfunc " << loadfunc << "\n";
     System::out() << "phi (arc func):" << phi << ".\n"; //added by marina 

    //phi = norm - loadfunc; 

    //for ( idx_t im = 0; im < masters_.size(); ++im )
    //{
    //  idx_t idof = masters_[im];

    //  if ( norm > 0.0 )
    //    jac10[idof] = u[idof] / norm;
    //  else
    //    jac10[idof] = 1.0;
    //}

    // dbg

    //phi = u[masters_[0]] - loadfunc;
    //jac10 = 0.0;
    //jac10[masters_[0]] = 1.0;
  }

  params.set ( ArclenParams::ARC_FUNC,   phi   );
  params.set ( ArclenParams::JACOBIAN11, jac11 );
}

//-----------------------------------------------------------------------
//   updateDefGrad_
//-----------------------------------------------------------------------

void BCModel::updateDefGrad_

  (       Matrix&        F,
    const Properties& globdat )

{
  Matrix F1loc ( 3, 3 );

  F1loc = 0.;

  Vector  disp1;

  StateVector::get    ( disp1, dofs_, globdat );
  
  Ref<PrintWriter> out = newInstance<PrintWriter> ( &System::out() );
  out->nformat.setFractionDigits( 10 );

/*  *out << "updateDefGrad elemDisp1 XZ "  << disp1[idofXZ_] << "\n";
  *out << "updateDefGrad elemDisp1 ZX "  << disp1[idofZX_] << "\n";*/
  *out << "l1 " << l1_ << " l2 " << l2_ << " l3 " << l3_ << "\n";
  
  F1loc(0,0) = 1. + disp1[idofXX_]/l1_;
  F1loc(0,1) =      disp1[idofYX_]/l2_;
  F1loc(0,2) =      disp1[idofZX_]/l3_;
  F1loc(1,0) =      disp1[idofXY_]/l1_;
  F1loc(1,1) = 1. + disp1[idofYY_]/l2_;
  F1loc(1,2) =      disp1[idofZY_]/l3_;
  F1loc(2,0) =      disp1[idofXZ_]/l1_;
  F1loc(2,1) =      disp1[idofYZ_]/l2_;
  F1loc(2,2) = 1. + disp1[idofZZ_]/l3_;

  F = F1loc;
}

//-----------------------------------------------------------------------
//    initWriter_
//-----------------------------------------------------------------------

Ref<PrintWriter>  BCModel::initWriter_

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

//  System::out() << "filename for printing " << fileName << "\n";
  return newInstance<PrintWriter>( newInstance<FileWriter> (
         StringUtils::join( fileName, "." ) ) );
}


//-----------------------------------------------------------------------
//    printStrains_
//-----------------------------------------------------------------------

void BCModel::printStrains_

  ( const Properties&  params,
    const Properties&  globdat )

{
  using jive::util::Globdat;

  Matrix Floc ( 3, 3);
  Floc = 0.0;

  updateDefGrad_ ( Floc, globdat );
  
  Ref<PrintWriter> out = newInstance<PrintWriter> ( &System::out() );
  out->nformat.setFractionDigits( 8 );
  *out << "F " << Floc << "\n"; 

  Matrix R, U;
  polarDecomposition ( R, U, Floc );

 /* *out << "Rotation tensor " << R << "\n"; 
  *out << "Strech tensor " << U << "\n"; 
*/
  idx_t       it;

  if ( strainOut_ == nullptr || rotationOut_ == nullptr || stretchOut_ == nullptr )
  {
    strainOut_ = initWriter_ ( params, "defgrad" );
    rotationOut_ = initWriter_ ( params, "rotation" );
    stretchOut_ = initWriter_ ( params, "stretch" );
  }

  strainOut_->nformat.setFractionDigits(10);
  rotationOut_->nformat.setFractionDigits(10);
  stretchOut_->nformat.setFractionDigits(10);

  globdat.get ( it, Globdat::TIME_STEP    );

  for ( idx_t r = 0; r < 3; ++r )
  {
      for ( idx_t k = 0; k < 3; ++k )
      {
        *strainOut_ << Floc (r,k) << " ";
        *rotationOut_ << R (r,k) << " ";
	*stretchOut_ << U (r,k) << " ";
      }
  }

  *strainOut_ << dt_ << "\n";
  *rotationOut_ << dt_ << "\n";
  *stretchOut_ << dt_ << "\n";

  strainOut_->flush();
  rotationOut_->flush();
  stretchOut_->flush();
}

//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------

Ref<Model> BCModel::makeNew

  ( const String&     name,
    const Properties& conf,
    const Properties& props,
    const Properties& globdat )

{
  return newInstance<BCModel> ( name, conf, props, globdat );
}

//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   declareBCModel
//-----------------------------------------------------------------------

void declareBCModel ()
{
  using jive::model::ModelFactory;

  ModelFactory::declare ( "BC", 
                          & BCModel::makeNew );

}
