/*
 *  TU Delft 
 *
 *  Iuri Barcelos, Oct 2019
 *
 *  Input module for GP data. Creates and stores a TrainingData object.
 *  This module should be used in the input UserconfModule. Skip it if
 *  training data is coming from elsewhere (e.g. a nested or forked chain).
 *
 */

#include <jem/base/System.h>
#include <jem/base/IllegalInputException.h>
#include <jem/util/PropertyException.h>
#include <jem/io/FileReader.h>
#include <jem/io/FileInputStream.h>
#include <jem/io/InputStreamReader.h>
#include <jem/io/PrintWriter.h>
#include <jem/io/FileWriter.h>
#include <jive/Array.h>
#include <jive/app/ModuleFactory.h>
#include <jive/util/DofSpace.h>
#include <jive/model/Model.h>
#include <jive/model/StateVector.h>
#include <jive/implict/SolverInfo.h>

#include "GPPredModule.h"
#include "TrainingData.h"
#include "LearningNames.h"

using namespace jem;

using jem::util::PropertyException;
using jem::IllegalInputException;
using jem::io::FileReader;
using jem::io::FileInputStream;
using jem::io::InputStreamReader;
using jem::io::PrintWriter;
using jem::io::FileWriter;

using jive::model::Model;
using jive::util::DofSpace;
using jive::model::StateVector;
using jive::implict::SolverInfo;

//-----------------------------------------------------------------------
//   class GPPredModule
//-----------------------------------------------------------------------


//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------

const char*        GPPredModule::TYPE_NAME = "GPPred";
const char*        GPPredModule::FILENAME  = "file";
const char*        GPPredModule::MINVALUES = "minValues";
const char*        GPPredModule::MAXVALUES = "maxValues";
const char*        GPPredModule::STEPSIZES = "stepSizes";
const char*        GPPredModule::CONFINTER = "confInterval";
const char*        GPPredModule::PRIORFILE = "priorFile";
const char*        GPPredModule::POSTFILE  = "posteriorFile";
const char*        GPPredModule::NPRIOR    = "nPrior";
const char*        GPPredModule::NPOST     = "nPosterior";
const char*        GPPredModule::PREDFILE  = "predFile";
const char*        GPPredModule::PREDOUTFILE = "predOutFile";

//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------

GPPredModule::GPPredModule 
  ( const String& name ) : Super ( name )

{
  priorFile_  = "";
  postFile_   = "";
  nPrior_     = 1;
  nPost_      = 1;
  predFile_   = "";
  predOutFile_ = "";
}

GPPredModule::~GPPredModule ()
{}

//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------

Module::Status GPPredModule::init

  ( const Properties& conf,
    const Properties& props,
    const Properties& globdat )

{
  System::out() << "GPPredModule::init.\n";

  Properties myConf  = conf. makeProps ( myName_ );
  Properties myProps = props.findProps ( myName_ );

  myProps.get  ( fileName_,  FILENAME  );
  myProps.get  ( minValues_, MINVALUES );
  myProps.get  ( maxValues_, MAXVALUES );
  myProps.get  ( stepSizes_, STEPSIZES );
  myProps.find ( priorFile_, PRIORFILE );
  myProps.find ( postFile_,  POSTFILE  );
  myProps.find ( nPrior_,    NPRIOR    );
  myProps.find ( nPost_,     NPOST     );
  myProps.find ( predFile_,  PREDFILE  );
  myProps.find ( predOutFile_, PREDOUTFILE );

  myConf. set ( FILENAME,  fileName_  );
  myConf. set ( MINVALUES, minValues_ );
  myConf. set ( MAXVALUES, maxValues_ );
  myConf. set ( STEPSIZES, stepSizes_ );
  myConf. set ( PREDOUTFILE, predOutFile_ );

  conf_ = 1.96; // 2 std deviations

  myProps.find ( conf_, CONFINTER );
  myConf .set  ( CONFINTER, conf_ );

  if ( minValues_.size() != maxValues_.size() ||
       minValues_.size() != stepSizes_.size() ||
       stepSizes_.size() != maxValues_.size()   )
  {
    throw IllegalInputException ( JEM_FUNC, "Sizes do not match" );
  }

  System::out() << "GPPredModule::init ended.\n";

  return OK;
}

//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------

Module::Status GPPredModule::run

  ( const Properties& globdat )

{
  System::out() << "GPPredModule::run. Predfile: " << predFile_ << ".\n";

  Properties  info = SolverInfo::get ( globdat );

  if ( !info.contains ( "terminate" ) )
  {
    System::out() << "Terminate.\n";
    return OK;
  }

  Properties params;

  Ref<Model> model = Model   ::get ( globdat, getContext() );

  if ( priorFile_ != "" )
  {
    Ref<PrintWriter> out = newInstance<PrintWriter> (
		      newInstance<FileWriter> ( priorFile_ ) );

    out->setPageWidth ( 1000000 );
    idx_t nb = ( maxValues_[0] - minValues_[0] ) / stepSizes_[0];
    idx_t ni = minValues_.size();
    idx_t no = nPrior_;

    Ref<NData> data = newInstance<NData> ( nb, ni, no );

    for ( idx_t s = 0; s < nb; ++s )
    {
      data->inputs(ALL,s) = minValues_ + (double)s * stepSizes_;
    }

    params.set ( LearningParams::INPUT, data );

    model->takeAction ( LearningActions::SAMPLEPRIOR, params, globdat );

    for ( idx_t s = 0; s < nb; ++s )
    {
      for ( idx_t i = 0; i < ni; ++i )
      {
	*out << data->inputs(i,s) << " ";
      }

      for ( idx_t o = 0; o < no; ++o )
      {
	*out << data->outputs(o,s) << " ";
      }

      *out << '\n';
    }
  }

  if ( postFile_ != "" )
  {
    Ref<PrintWriter> out = newInstance<PrintWriter> (
		      newInstance<FileWriter> ( postFile_ ) );

    out->setPageWidth ( 1000000 );
    idx_t nb = ( maxValues_[0] - minValues_[0] ) / stepSizes_[0];
    idx_t ni = minValues_.size();
    idx_t no = nPost_;

    Ref<NData> data = newInstance<NData> ( nb, ni, no );

    for ( idx_t s = 0; s < nb; ++s )
    {
      data->inputs(ALL,s) = minValues_ + (double)s * stepSizes_;
    }

    params.set ( LearningParams::INPUT, data );

    model->takeAction ( LearningActions::SAMPLEPOSTERIOR, params, globdat );

    for ( idx_t s = 0; s < nb; ++s )
    {
      for ( idx_t i = 0; i < ni; ++i )
      {
	*out << data->inputs(i,s) << " ";
      }

      for ( idx_t o = 0; o < no; ++o )
      {
	*out << data->outputs(o,s) << " ";
      }

      *out << '\n';
    }
  }

  if ( predFile_ != "")
  {
     read ( predFile_, globdat, predOutFile_ );
  }
  else
  {
  Ref<PrintWriter> out = newInstance<PrintWriter> (
		    newInstance<FileWriter> ( fileName_ ) );

  System::out() << "Input min values:.\n";
  System::out() << minValues_ << "\n";

  Vector input ( minValues_.clone() );

  System::out() << "Input: " << input << ".\n";

  while ( true )
  {
    if ( testany ( input > maxValues_ ) )
    {
      break;
    }
    
    System::out() << "Input loop: " << input << ".\n";

    params.set ( LearningParams::INPUT, input );

    Vector samples;

    model->takeAction ( LearningActions::RECALL, params, globdat );

    Vector mean, var;

    params.get ( mean, LearningParams::OUTPUT   );
    params.get ( var,  LearningParams::VARIANCE );

    for ( idx_t i = 0; i < input.size(); ++i )
    {
      *out << input[i] << " ";
    }

    for ( idx_t o = 0; o < mean.size(); ++o )
    {
      *out << mean[o] << " " << mean[o]-conf_*var[o] << " " << mean[o]+conf_*var[o] << " ";
    }
    *out << '\n';

    input += stepSizes_;
  }
  }

  System::out() << "GPPredModule::run ended.\n";

  return EXIT;
}

//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------

Ref<Module> GPPredModule::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  return newInstance<Self> ( name );
}

//-----------------------------------------------------------------------
//   read
//-----------------------------------------------------------------------

void GPPredModule::read

  ( const String& predFile, const Properties& globdat, const String& fileName)

{
  Properties params;

  Ref<Model> model = Model ::get ( globdat, getContext() );

  Ref<FileReader> in = newInstance<FileReader> ( predFile );
  Ref<PrintWriter> out = newInstance<PrintWriter> ( newInstance<FileWriter> ( fileName ) );

  idx_t ns = in-> parseInt();
  idx_t nc = in-> parseInt();
  idx_t nv = in-> parseInt();

  IdxVector ins(nv);
  IdxVector ncv(nc);
  ncv = 0.0;

  for (idx_t i = 0; i < nv; ++i)
  {
	  ins[i] = in-> parseInt() - 1.0;
  }

  for (idx_t i = 0; i < nc; ++i)
  {
	  for (idx_t j = 0; j < nv; ++j)
	  {
		if (i == ins[j])
      		{
			ncv[i] = 1.0;
      		}
	  }
  }

  System::out() << "Teste: " << ncv << ".\n";

  Vector input ( nv );

  Vector mean, var;

  int v;
  double aux;

  for (idx_t i = 0; i < ns; ++i )
  {
	  v = 0.0;

	  for (idx_t j = 0; j < nc; ++j)
	  {
		 aux = in-> parseDouble();

		 if (ncv[j] == 1)
		 {
			 input[v] = aux;
			 *out << input[v] << " ";
			 v += 1.0;
		 }
	  }

	  System::out() << "Input: " << input << "\n";

	  params.set ( LearningParams::INPUT, input);

	  model->takeAction (LearningActions::RECALL, params, globdat );

	  Vector mean, var;

	  params.get ( mean, LearningParams::OUTPUT );
	  params.get ( var, LearningParams::VARIANCE );

         for (idx_t o = 0; o < mean.size(); ++o)
	 {
		 *out << mean[o] << " " << mean[o]-conf_*var[o] << " " << mean[0]+conf_*var[o];
	 }
	 
	 *out << "\n";
  }

  System::out() << "GPPredModule::Read finished.\n";
}


//-----------------------------------------------------------------------
//   declareGPPredModule
//-----------------------------------------------------------------------

void declareGPPredModule ()
{
  using jive::app::ModuleFactory;

  ModuleFactory::declare ( GPPredModule::TYPE_NAME, & GPPredModule::makeNew );
}

