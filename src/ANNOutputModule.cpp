/*
 *  TU Delft 
 *
 *  Iuri Barcelos, July 2019
 *
 *  Output module for ANN data. Reads the TrainingData object, uses the
 *  trained network stored in the model to make predictions and prints
 *  the results to data files. It can also be used to print the network
 *  weights to a file.
 *
 *  This module should be used in the output UserconfModule. 
 *
 */

#include <jem/base/Error.h>
#include <jem/base/System.h>
#include <jem/base/limits.h>
#include <jem/base/Exception.h>
#include <jem/base/array/operators.h>
#include <jem/base/IllegalInputException.h>
#include <jem/base/array/utilities.h>
#include <jem/util/PropertyException.h>
#include <jem/util/StringUtils.h>
#include <jem/io/Writer.h>
#include <jem/io/PrintWriter.h>
#include <jem/io/FileWriter.h>
#include <jem/io/FileFlags.h>
#include <jem/numeric/algebra/utilities.h>
#include <jem/mp/MPException.h>
#include <jem/mp/MPIContext.h>
#include <jem/mp/Buffer.h>
#include <jem/mp/Status.h>
#include <jive/util/utilities.h>
#include <jive/util/Globdat.h>
#include <jive/model/Model.h>
#include <jive/model/Actions.h>
#include <jive/model/StateVector.h>
#include <jive/app/ModuleFactory.h>
#include <jive/util/DofSpace.h>
#include <jive/mp/Globdat.h>

#include <jem/io/FileReader.h>
#include <jem/io/FileInputStream.h>
#include <jem/io/InputStreamReader.h>

#include "LearningNames.h"

#include "ANNOutputModule.h"

using namespace jem;

using jem::util::PropertyException;
using jem::util::StringUtils;
using jem::IllegalInputException;
using jem::io::Writer;
using jem::io::PrintWriter;
using jem::io::FileWriter;
using jem::io::FileReader;
using jem::io::InputStreamReader;
using jem::io::FileInputStream;
using jem::io::FileFlags;
using jem::mp::SendBuffer;
using jem::mp::RecvBuffer;

using jive::StringVector;
using jive::model::Model;
using jive::model::StateVector;
using jive::util::DofSpace;

//=======================================================================
//   class ANNOutputModule
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------

const char* ANNOutputModule::TYPE_NAME  = "ANNOutput";
const char* ANNOutputModule::FILENAME   = "filename";
const char* ANNOutputModule::FORMAT     = "format";
const char* ANNOutputModule::PRINT      = "print";
const char* ANNOutputModule::WRITEEVERY = "writeEvery";
const char* ANNOutputModule::RUNFIRST   = "runFirst";
const char* ANNOutputModule::VALSPLIT   = "valSplit";
const char* ANNOutputModule::PREDFILE   = "predFile";
const char* ANNOutputModule::BOUNDS     = "bounds"; 

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------

ANNOutputModule::ANNOutputModule

  ( const String& name ) : Super ( name )

{
  predFile_   = "";
}

ANNOutputModule::~ANNOutputModule ()
{}

//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------

Module::Status ANNOutputModule::init

  ( const Properties& conf,
    const Properties& props,
    const Properties& globdat )

{
  Properties myConf  = conf. makeProps ( myName_ );
  Properties myProps = props.findProps ( myName_ );

  Ref<TrainingData> data = TrainingData::get ( globdat, getContext() );
  Ref<DofSpace>     dofs = DofSpace::get     ( globdat, getContext() );

  bestWeights_.resize ( dofs->dofCount() );
  bestWeights_ = 0.0;

  epoch_        = 0;
  best_         = jem::maxOf ( best_ );
  relbest_      = jem::maxOf ( relbest_ );
  nsamples_     = data->sampleSize();
  writeEvery_   = 100;
  mimicXOut_    = false;
  printWts_     = false;
  printOuts_    = false;
  printInps_     = false;

  String format ( "lines" );
  String print  ( "" );
 
  selComp_.resize(1);
  selComp_[0] = -1;

  IdxVector temp(2);
  temp = -1_idx;
  
  bounds_.ref ( temp );

  myProps.find ( format,      FORMAT     );
  myProps.find ( print,       PRINT      );
  myProps.find ( writeEvery_, WRITEEVERY );
  myProps.find ( predFile_,  PREDFILE  );
  myProps.get  ( fname_,      FILENAME   );
  myProps.find ( bounds_,     BOUNDS     );
  myProps.find ( selComp_,   "selComp"   );

  System::out() << "Bounds for error validation: " << bounds_ << "\n";

  if ( !myProps.find ( nsamples_, RUNFIRST, 0, data->sampleSize() ) )
  {
    double valsplit = 0.0;

    if ( myProps.find ( valsplit, VALSPLIT, 0.0, 1.0 ) )
    {
      nsamples_ = valsplit * data->sampleSize();
    }
  }

  if ( format.equalsIgnoreCase ( "columns" ) )
  {
    format_ = COLUMNS;
  }
  else if ( format.equalsIgnoreCase ( "lines" ) )
  {
    format_ = LINES;
  }
  else
  {
    throw IllegalInputException ( getContext(),
      "Unknown format type." );
  }

  StringVector prints = StringUtils::split ( print, '|' );

  for ( idx_t i = 0; i < prints.size(); ++i )
  {
    String str = prints[i].stripWhite();

    if ( str.equalsIgnoreCase ( "inputs" ) )
    {
      printInps_ = true;
    }
    if ( str.equalsIgnoreCase ( "outputs" ) )
    {
      printOuts_ = true;
    }
    if ( str.equalsIgnoreCase ( "xoutputs" ) )
    {
      printOuts_ = true;
      mimicXOut_ = true;
    }
    if  ( str.equalsIgnoreCase ( "weights" ) )
    {
      printWts_ = true;
    }
  }

  if ( format_ == COLUMNS &&
       printInps_ && printOuts_ &&
       data->inputSize() != data->outputSize() )
  {
    throw IllegalInputException ( getContext(),
      "Column-wise printing of both inputs and outputs is only possible if inputSize == outputSize" );
  }

  mpx_ = jive::mp::Globdat::getMPContext ( globdat );

  if ( mpx_ == nullptr )
  {
    throw Error ( JEM_FUNC, String::format (
       "MPContext has not been found" ) );
  }

  idx_t rank = mpx_->myRank();
  idx_t size = mpx_->size();
  bool  mp   = ( size > 1 );

  if ( mp )
  {
    System::out() << "MP is true\n";
    bool  last = ( rank == size - 1 );

    idx_t load = max ( 1, nsamples_ / size );
    idx_t end  = max ( nsamples_ - rank*load, 0 );
    idx_t beg  = last ? 0 : max ( end - load, 0 );

    sampIds_.ref ( IdxVector ( iarray ( slice ( beg, end ) ) ) );
  }
  else
  {
    sampIds_.ref ( IdxVector ( iarray ( nsamples_ ) ) );
  }

  return OK;
}

//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------

Module::Status ANNOutputModule::run

  ( const Properties& globdat )

{
  epoch_++;

  idx_t rank = mpx_->myRank();
  idx_t size = mpx_->size();
  bool  root = ( rank == 0 );
  bool  mp   = ( size > 1 );

//  System::out() << "ANNOutputModule::run.\n";

  if ( !(epoch_ % writeEvery_) )
  {
 //   System::out() << "ANNOutputModule::run. Loop.\n";
    Ref<TrainingData> data = TrainingData::get ( globdat, getContext() );

    double lerr = 0.0, err = 0.0, rerr = 0.0;
    Vector errors;

    Batch samples = data->getData ( sampIds_ );
   
    if ( nsamples_ > rank )
    {
      runSamples_ ( errors, samples, globdat );
    }

    if ( mp )
    {
      lerr = errors[0];
      mpx_->reduce ( RecvBuffer ( &err, 1 ),
		     SendBuffer ( &lerr, 1 ),
		     0,
		     jem::mp::SUM );
    }
    else
    {
      err = errors[0];
      rerr = errors[1];
    }

    System::out() << "ANNOutputModule::run. err: " << err << " best: " << best_ << ".\n";

    if ( err < best_ )
    {
      best_      = err;
      relbest_   = rerr;
      bestEpoch_ = epoch_;

      Ref<DofSpace> dofs = DofSpace::get     ( globdat, getContext() );

      Vector state;
      StateVector::get ( state, dofs, globdat );

      bestWeights_ = state;
      bestSamples_.ref ( samples );

      writeWeights_ ( globdat );
      writeError_ ( globdat );
      writeLines_ ( globdat );
    }

    writeWeightsEpoch_ ( globdat );

    if ( root )
    {
      print ( System::info( myName_ ), getContext(), 
	": Epoch ", epoch_, ", computing error for the first ", nsamples_, " samples\n" );
      print ( System::info( myName_ ), getContext(), 
	": Epoch ", epoch_, ", mean error = ", err, "\n" );
      print ( System::info( myName_ ), getContext(), 
	": Lowest error up until now = ", best_, " (from epoch ", bestEpoch_, ")\n\n" );  
    }

    Properties myVars = jive::mp::Globdat::getVariables ( myName_, globdat );

    myVars.set ( "error", err );
    myVars.set ("epoch", epoch_ );

  }

  if ( predFile_ != "")
  {
     read ( predFile_, globdat, "annext.out" );
  }

  return OK;
}

//-----------------------------------------------------------------------
//   shutdown
//-----------------------------------------------------------------------

void ANNOutputModule::shutdown

  ( const Properties& globdat )

{
  if ( mpx_->myRank() == 0 )
  {
    print ( System::info( myName_ ), getContext(), 
      ", writing results ...\n" );

    if ( printWts_ )
    { 
      writeWeights_ ( globdat );
    }
  }

  if ( printInps_ || printOuts_ )
  {
    if ( format_ == LINES )
    {
      writeLines_   ( globdat );
    }
    else if ( format_ == COLUMNS )
    {
      writeCols_    ( globdat );
    }
  }

  writeError_ ( globdat );
}

//-----------------------------------------------------------------------
//   runSamples_
//-----------------------------------------------------------------------

void ANNOutputModule::runSamples_

  (       Vector&     errors, 
    const Batch&      samples,
    const Properties& globdat )

{
  Properties params;
  double error = 0.0, totalerror = 0.0;
  double relerror = 0.0, totalrelerror = 0.0;

  Ref<Model>        model = Model::get        ( globdat, getContext() );
  Ref<TrainingData> data  = TrainingData::get ( globdat, getContext() );
  Ref<Normalizer>   inl   = data->getInpNormalizer();
  Ref<Normalizer>   onl   = data->getOutNormalizer();

  idx_t ns   = samples[0]->batchSize();
  idx_t seq  = data->sequenceSize();

//  System::out() << "seq: " << seq << " ns: " << ns << "\n";

  for ( idx_t t = 0; t < seq; ++t )
  {
    params.erase ( LearningParams::STATE );

    if ( t > 0 )
    {
      params.set      ( LearningParams ::STATE,            samples[t-1]   );
    }

    params.set        ( LearningParams ::DATA,             samples[t]     );

    model->takeAction ( LearningActions::RECALL, params, globdat );

    //// DEBUG
    //if ( false )
    //{
    //  model->takeAction ( NeuralActions::GETJACOBIAN, params, globdat );
    //  for ( idx_t s = 0; s < ns; ++s )
    //  {
    //    idx_t os = data->outputSize();

    //    Matrix jac ( samples[t]->jacobian ( slice(s*os,(s+1)*os), ALL ) );

    //    //for ( idx_t row = 0; row < os; ++row )
    //    //{
    //    //  jac(row,ALL) *= inl->getJacobianFactor ( Vector() );
    //    //}

    //    System::out() << jac << "\n";
    //  }
    //}
  }

  idx_t seqsize = seq;
  idx_t seqrelsize = seq;
  for ( idx_t s = 0; s < ns; ++s )
  {
    error = 0.0; relerror = 0.0;
    Vector noise ( samples[0]->outputs(ALL, s).size());
    noise = 1e-12;

    if ( selComp_[0] > -1 )
    {
       noise.resize( selComp_.size() );
       noise = 1e-12;
    }

    for ( idx_t t = 0; t < seq; ++t )
    {
      Vector out = onl->denormalize ( samples[t]->outputs(ALL,s) );
      Vector tar = onl->denormalize ( samples[t]->targets(ALL,s) );

      Vector selOut ( selComp_.size() );
      Vector selTar ( selComp_.size() );
     
      if ( selComp_[0] > -1 )
      {
        for ( idx_t i = 0; i < selComp_.size(); i++ ) 
        {
          selOut[i] = out[selComp_[i]];
          selTar[i] = tar[selComp_[i]];
        }
      }
      
      if (bounds_[0] > -1_idx )
      {
       if ( t >= bounds_[0] && t <= bounds_[1] )
       {
        if ( selComp_[0] > - 1 )
        {
          relerror += jem::numeric::norm2 ( selTar - selOut ) / (jem::numeric::norm2 ( selTar + noise));
          error += jem::numeric::norm2 ( selTar - selOut );
        }
        else
        {      
          error += jem::numeric::norm2 ( tar - out );
          relerror += jem::numeric::norm2 ( tar - out )/ (jem::numeric::norm2 ( selTar + noise ));
        }
//        System::out() << "step " << t << " tar: " << tar << " out: " << out << "\n";
       }
       seqrelsize = bounds_[1] - bounds_[0];
     }
     else
     {
        if ( selComp_[0] > -1 )
        {
  //        System::out() << "Step " << t << " selTar: " << selTar << " selOut: " << selOut << "\n";
   //       System::out() << "Step " << t << " tar: " << tar << " out: " << out << "\n";

          error += jem::numeric::norm2 ( selTar - selOut );
	  relerror += jem::numeric::norm2( selTar - selOut ) / ( jem::numeric::norm2( selTar + noise ) );
        }
        else
        {
          error += jem::numeric::norm2 ( tar - out );
	  relerror += jem::numeric::norm2( tar - out ) / ( jem::numeric::norm2( tar + noise ) );
        }
     }
    }
    totalerror += error / seqsize;
    totalrelerror += relerror / seqrelsize;
  }
  
  errors.resize(2);
  errors[0] = totalerror / nsamples_;
  errors[1] = totalrelerror / nsamples_; 

//  return totalerror / nsamples_;
}

//-----------------------------------------------------------------------
//   writeWeights_
//-----------------------------------------------------------------------

void ANNOutputModule::writeWeights_

  ( const Properties& globdat )

{
  String name = fname_ + ".net";

  Ref<TrainingData> data  = TrainingData::get ( globdat, getContext() );
  Ref<Normalizer>   inl   = data->getInpNormalizer();
  Ref<Normalizer>   onl   = data->getOutNormalizer();

  Ref<PrintWriter> fout = newInstance<PrintWriter> (
    newInstance<FileWriter> ( name, FileFlags::WRITE ) );

  fout->nformat.setFractionDigits ( 16 );

  for ( idx_t i = 0; i < bestWeights_.size(); ++i )
  {
    *fout << bestWeights_[i] << "\n";
  }

  inl->write ( "input"  );
  onl->write ( "output" );
}


//-----------------------------------------------------------------------
//   writeWeightsEpoch_
//-----------------------------------------------------------------------

void ANNOutputModule::writeWeightsEpoch_

  ( const Properties& globdat )

{
  String name = fname_ + String(epoch_) + ".net";

  Ref<DofSpace> dofs = DofSpace::get     ( globdat, getContext() );

  Vector state;
  StateVector::get ( state, dofs, globdat );

  Ref<TrainingData> data  = TrainingData::get ( globdat, getContext() );
  Ref<Normalizer>   inl   = data->getInpNormalizer();
  Ref<Normalizer>   onl   = data->getOutNormalizer();

  Ref<PrintWriter> fout = newInstance<PrintWriter> (
    newInstance<FileWriter> ( name, FileFlags::WRITE ) );

  fout->nformat.setFractionDigits ( 16 );

  for ( idx_t i = 0; i < state.size(); ++i )
  {
    *fout << state[i] << "\n";
  }

  inl->write ( "input"  );
  onl->write ( "output" );
}

//----------------------------------------------------------------------
//   writeError_
//----------------------------------------------------------------------

void ANNOutputModule::writeError_
   
   ( const Properties & globdat )
{
  
//  System::out() << "writeError_ began\n";

  String name = fname_ + ".err";

  Ref<TrainingData> data  = TrainingData::get ( globdat, getContext() );

  Ref<PrintWriter> fout = newInstance<PrintWriter> (
    newInstance<FileWriter> ( name, FileFlags::WRITE ) );

  fout->nformat.setFractionDigits ( 5 );

//  print ( System::info( myName_ ), getContext(), ", writing error...\n" );

//  Properties myVars = jive::mp::Globdat::getVariables ( myName_, globdat );

 // double err = 0.0;

 // myVars.find ( err, "error");

  *fout << best_ << " " << bestEpoch_ << " " << relbest_ << "\n";  


}

//-----------------------------------------------------------------------
//   writeLines_
//-----------------------------------------------------------------------

void ANNOutputModule::writeLines_

  ( const Properties& globdat  )

{
  idx_t rank =   mpx_->myRank();
  bool  mp   = ( mpx_->size() > 1 );
 // System::out() << "ANNOutputModule::writeLines.\n";

  String name = mp ? fname_ + String ( rank ) + ".out" : fname_ + ".out";

  Ref<TrainingData> data  = TrainingData::get ( globdat, getContext() );
  Ref<Normalizer>   inl   = data->getInpNormalizer();
  Ref<Normalizer>   onl   = data->getOutNormalizer();

  idx_t isz  = data->inputSize();
  idx_t osz  = data->outputSize();
  idx_t seq  = data->sequenceSize();
  idx_t ns   = sampIds_.size();

  Ref<PrintWriter> fout = newInstance<PrintWriter> (
    newInstance<FileWriter> ( name, FileFlags::WRITE ) );

  fout->nformat.setFractionDigits ( 10 );

  System::out() << "ANNOutputModule::writeLines. ns: " << ns << " seq: " << seq << ".\n";

  for ( idx_t s = 0; s < ns; ++s )
  {
    for ( idx_t t = 0; t < seq; ++t )
    {
      Vector inp = inl->denormalize ( bestSamples_[t]->inputs (ALL,s) );
      Vector out = onl->denormalize ( bestSamples_[t]->outputs(ALL,s) );
    
    //  System::out() << "ANNOutputModule::writeLines. inp: " << inp << ".\n";
    //  System::out() << "ANNOutputModule::writeLines. out: " << out << ".\n";  

      if ( printInps_ )
      {
        for ( idx_t i = 0; i < isz; ++i )
	{
          *fout << inp[i] << " ";
	}
      }

      if ( printOuts_ )
      {
	for ( idx_t o = 0; o < osz; ++o )
	{
	  *fout << out[o] << " ";
	}
      }

      *fout << "\n";
    }

    *fout << "\n";
  }
}

//-----------------------------------------------------------------------
//   writeCols_
//-----------------------------------------------------------------------

void ANNOutputModule::writeCols_

  ( const Properties& globdat  )

{
  String name;

  Ref<TrainingData> data  = TrainingData::get ( globdat, getContext() );
  Ref<Normalizer>   inl   = data->getInpNormalizer();
  Ref<Normalizer>   onl   = data->getOutNormalizer();

  idx_t osz  = data->outputSize();
  idx_t seq  = data->sequenceSize();
  idx_t ns   = sampIds_.size();

  for ( idx_t s = 0; s < ns; ++s )
  {
    name = fname_ + String ( sampIds_[s] ) + ".out";

    Ref<PrintWriter> fout = newInstance<PrintWriter> (
      newInstance<FileWriter> ( name, FileFlags::WRITE ) );

    fout->nformat.setFractionDigits ( 5 );

    for ( idx_t t = 0; t < seq; ++t )
    {
      Vector inp = inl->denormalize ( bestSamples_[t]->inputs (ALL,s) );
      Vector out = onl->denormalize ( bestSamples_[t]->outputs(ALL,s) );

      if ( mimicXOut_ )
      {
        *fout << "newXOutput" << String ( t ) << "\n";
      }

      for ( idx_t i = 0; i < osz; ++i )
      {
        if ( printInps_ )
	{
          *fout << inp[i] << " ";
	}

        if ( printOuts_ )
	{
	  *fout << out[i] << "\n";
	}
      }

      *fout << "\n";
    }
  }
}

//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------

Ref<Module> ANNOutputModule::makeNew

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

void ANNOutputModule::read

  ( const String& predFile, const Properties& globdat, const String& fileName)

{
/*  Properties params;

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

  int v;
  double aux;

  idx_t seq = ns;
  Batch samples ( seq );

  for (idx_t i = 0; i < ns; ++i )
  {
	  params.erase ( LearningParams::STATE );

          if ( i > 0 )
          {
             params.set ( LearningParams::STATE, samples[i-1] );
          }
        
          v = 0.0;
	  for (idx_t j = 0; j < nc; ++j)
	  {
		 aux = in-> parseDouble();
		 if (ncv[j] == 1)
		 {
			 samples[i]->inputs(ALL, v) = aux;
			 v += 1.0;
		 }
	  }
	 // System::out() << "Input: " << samples[i] << "\n";

	  params.set ( LearningParams::DATA, samples);

	  model->takeAction (LearningActions::RECALL, params, globdat );

          System::out() << "Recalled.\n";
  }

 for ( idx_t s = 0; s < ns; ++s )
  {
    error = 0.0;

    for ( idx_t t = 0; t < seq; ++t )
    {
      Vector out = onl->denormalize ( samples[t]->outputs(ALL,s) );
      Vector tar = onl->denormalize ( samples[t]->targets(ALL,s) );
 
      System::out() << "tar: " << tar << " out: " << out << "\n";
      error += jem::numeric::norm2 ( tar - out );
    }
    totalerror += error / seq;
  }

  System::out() << "ANNOutputModule::Read finished.\n";*/
}

//-----------------------------------------------------------------------
//   declareANNOutputModule
//-----------------------------------------------------------------------

void declareANNOutputModule ()
{
  using jive::app::ModuleFactory;

  ModuleFactory::declare ( ANNOutputModule::TYPE_NAME, & ANNOutputModule::makeNew );
}

