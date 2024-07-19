/*
 * Copyright (C) 2021 TU Delft. All rights reserved.
 *
 *  Material shell that takes stress/strain/history observations
 *  from anchor models and prints them to a file.
 *
 * Author: Iuri Rocha, i.rocha@tudelft.nl
 * Date:   Apr 2021
 *
 */

#include <jem/base/Error.h>
#include <jem/base/System.h>
#include <jem/base/Array.h>
#include <jem/util/Properties.h>
#include <jem/numeric/algebra/matmul.h>
#include <jem/numeric/algebra/utilities.h>
#include <jem/numeric/algebra/LUSolver.h>
#include <jem/numeric/algebra/Cholesky.h>
#include <jem/io/FileReader.h>
#include <jem/io/PrintWriter.h>
#include <jem/io/FileWriter.h>
#include <jem/io/FileFlags.h>
#include <jem/io/PatternLogger.h>
#include <jive/model/Model.h>
#include <jive/model/StateVector.h>
#include <jive/app/UserconfModule.h>
#include <jive/app/InfoModule.h>
#include <jive/fem/InitModule.h>
#include <jive/fem/ElementSet.h>
#include <jive/fem/NodeSet.h>
#include <jive/util/Random.h>
#include <jive/util/Globdat.h>
#include <jive/util/utilities.h>
#include <jive/util/DofSpace.h>
#include <jive/util/Assignable.h>
#include <jive/implict/SolverInfo.h>
#include <jive/algebra/FlexMatrixBuilder.h>
#include <jive/algebra/SparseMatrixObject.h>

#include "ObserverMaterial.h"

#include "utilities.h"
#include "LearningNames.h"
#include "SolverNames.h"
#include "NData.h"
#include "NeuralUtils.h"

using namespace jem;

using jem::numeric::norm2;
using jem::numeric::matmul;
using jem::numeric::LUSolver;
using jem::numeric::Cholesky;
using jem::io::FileReader;
using jem::io::FileWriter;
using jem::io::PrintWriter;
using jem::io::PatternLogger;
using jem::io::FileFlags;

using jive::Cubix;

using jive::model::Model;
using jive::model::StateVector;
using jive::app::ChainModule;
using jive::app::UserconfModule;
using jive::app::InitModule;
using jive::app::InfoModule;
using jive::util::Random;
using jive::util::Globdat;
using jive::util::joinNames;
using jive::util::DofSpace;
using jive::util::Assignable;
using jive::fem::ElementSet;
using jive::fem::NodeSet;
using jive::algebra::FlexMatrixBuilder;

//=================================================================================================
//   class ObserverMaterial
//=================================================================================================

//-------------------------------------------------------------------------------------------------
//   static data
//-------------------------------------------------------------------------------------------------

const char* ObserverMaterial::MATERIAL         = "material";
//const char* ObserverMaterial::FEATUREEXTRACTOR = "featureExtractor";
const char* ObserverMaterial::OUTFILE          = "outFile";
const char* ObserverMaterial::RSEED            = "rseed";
const char* ObserverMaterial::SHUFFLE          = "shuffle";

//-------------------------------------------------------------------------------------------------
//   constructor and destructor
//-------------------------------------------------------------------------------------------------

ObserverMaterial::ObserverMaterial

  ( const Properties& props,
    const Properties& conf,
    const idx_t       rank,
    const Properties& globdat )

  : Material ( rank, globdat )

{
  JEM_PRECHECK ( rank >= 1 && rank <= 3 );

  const idx_t  STRAIN_COUNTS[4] = { 0, 1, 3, 6 };

  strCount_ = STRAIN_COUNTS[rank];
  rank_     = rank;
  props_    = props;
  conf_     = conf;
  globdat_  = globdat;
  file_     = "";
  out_      = nullptr;
  shuffle_  = false;

  child_    = newMaterial ( MATERIAL, conf, props, globdat );
}

ObserverMaterial::~ObserverMaterial ()

{
  IdxVector points ( iarray ( hist_.size() ) );

  if ( shuffle_ )
  {
    NeuralUtils::shuffle ( points, globdat_ );
  }

  for ( idx_t p = 0_idx; p < hist_.size(); ++p )
  {
    idx_t point = points[p];

    for ( idx_t t = 0_idx; t < hist_[point].strains.size(); ++t )
    {
   /*   for ( idx_t f = 0_idx; f < hist_[point].features[t].size(); ++f )
      {
        *out_ << hist_[point].features[t][f] << " ";
      }
*/
      for ( idx_t s = 0_idx; s < strCount_; ++s )
      {
        *out_ << hist_[point].strains[t][s] << " ";
      }

      for ( idx_t s = 0_idx; s < strCount_; ++s )
      {
        *out_ << hist_[point].stresses[t][s] << " ";
      }

      if ( writeHistory_ )
        *out_ << hist_[point].history[t][0] << " ";

      *out_ << '\n';
    }
    *out_ << '\n';
  }
}

//-------------------------------------------------------------------------------------------------
//   configure
//-------------------------------------------------------------------------------------------------

void ObserverMaterial::configure

  ( const Properties& props )

{
  // Printing
  writeHistory_ = false;

  props.get ( file_, OUTFILE );
  props.find ( writeHistory_, "writeHistory");

  out_ = newInstance<PrintWriter> ( newInstance<FileWriter> ( file_ ) );

  out_->setPageWidth ( 1000000000 );

  // Shuffling

/*  props.find ( shuffle_, SHUFFLE );

  if ( shuffle_ )
  {
    props.get ( rseed_, RSEED );

    Ref<Random> rng = Random::get ( globdat_ );

    rng->restart ( rseed_ );
  }
*/
  // Configure child

  child_->configure ( props_.findProps ( MATERIAL )); 
}

//-------------------------------------------------------------------------------------------------
//   getConfig
//-------------------------------------------------------------------------------------------------

void ObserverMaterial::getConfig

  ( const Properties& conf ) const 
    
{
  conf.set ( OUTFILE,          file_                                 );
 // conf.set ( FEATUREEXTRACTOR, props_.findProps ( FEATUREEXTRACTOR ) );

  // Get child config

  child_->getConfig ( conf.makeProps ( MATERIAL ) );
}

//-------------------------------------------------------------------------------------------------
//  allocPoints 
//-------------------------------------------------------------------------------------------------

void ObserverMaterial::allocPoints

  ( const idx_t       npoints )

{
  child_->allocPoints ( npoints );

  //strains_. pushBack ( Vector ( strCount_ ), npoints );
  //stresses_.pushBack ( Vector ( strCount_ ), npoints );

  for ( idx_t p = 0_idx; p < npoints; ++p )
  {
    strains_ .pushBack ( Matrix ( strCount_, strCount_ ) );
    stresses_.pushBack ( Vector ( strCount_ ) );          
    history_.pushBack ( Vector ( 1 ) );    

    hist_.pushBack ( Hist_() );

  /*  hist_[p].extractor = newMaterial ( FEATUREEXTRACTOR, Properties(), props_, globdat_ );
    hist_[p].extractor->configure ( props_.findProps ( FEATUREEXTRACTOR ), globdat_ );
    hist_[p].extractor->createIntPoints ( 1 );*/
  }
}

//-------------------------------------------------------------------------------------------------
//  update 
//-------------------------------------------------------------------------------------------------

void ObserverMaterial::update

  ( const Vector& stress,
    const Matrix& stiff,
    const Matrix& df,
    idx_t   ipoint )
{
  // Update using nested material

  System::out() << "ObserverMaterial. update. \n";
  child_->update ( stress, stiff, df, ipoint );
  Vector diss ( 1 );
  diss[0] = 0.0;
  diss[0] =  child_->giveDissipation ( ipoint );
//  System::out() << "ObserverMaterial. dissipation " << child_->getDissipation(ipoint) << "\n";

  // Store strains and stresses for observation purposes

  strains_ [ipoint] = df;
  stresses_[ipoint] = stress;
  history_[ipoint] = diss;
}

//-------------------------------------------------------------------------------------------------
//  cancel
//-------------------------------------------------------------------------------------------------

void ObserverMaterial::cancel ()

{
  child_->cancel();
}

//-------------------------------------------------------------------------------------------------
//  commit
//-------------------------------------------------------------------------------------------------

void ObserverMaterial::commit ()

{
 
    System::out() << "ObserverMaterial. Commit.\n"; 
    child_->commit();

    hist_[0_idx].strains .pushBack ( strains_ [0_idx].clone() );
    hist_[0_idx].stresses.pushBack ( stresses_[0_idx].clone() );
//    hist_[p].history.pushBack ( history_[p].clone() );

System::out() << "ObserverMaterial. Commited.\n"; 

// Store converged observations

  Matrix stiff  ( strCount_, strCount_ );
  Vector stress ( strCount_            );

  stiff = 0.0; stress = 0.0;

  for ( idx_t p = 0; p < hist_.size(); ++p )
  {
/*    Vector aux(3);
    aux[0] = -0.00166700;
    aux[1] = 0.000487161;
    aux[2] = 3.80682e-05;*/
    Matrix df ( 3, 3);
    df = 0.0;
    System::out() << "ObserverMaterial. Update after commit.\n"; 
    child_->update ( stress, stiff, df, p );
    stresses_[p] = stress;
    Vector diss (1);    
    diss[0] = child_->giveDissipation ( p );
    history_[p] = diss;

 //   hist_[p].strains .pushBack ( strains_ [p].clone() );
 //   hist_[p].stresses.pushBack ( stresses_.clone() );
    hist_[p].history.pushBack ( history_[p].clone() );

    System::out() << "Observer. Strains. " << strains_[p] <<"\n";
    System::out() << "Observer. Stresses. " << stresses_[p] << "\n";
    System::out() << "Observer. Stiff. " << stiff << "\n";
  
  /*  hist_[p].extractor->update ( stiff, stress, strains_[p], 0_idx );
    hist_[p].extractor->commit ();
*/
  /*  BayRef bay = dynamicCast<Bayesian> ( hist_[p].extractor );
    hist_[p].features.pushBack ( bay->getFeatures ( 0_idx ).clone() );*/
  }
}

//-------------------------------------------------------------------------------------------------
//  CheckCommit
//-------------------------------------------------------------------------------------------------

void ObserverMaterial::checkCommit 

  ( const Properties& params )

{
  child_->checkCommit ( params );
}

//-------------------------------------------------------------------------------------------------
//  clone 
//-------------------------------------------------------------------------------------------------

Ref<Material> ObserverMaterial::clone ( ) const

{
  return newInstance<ObserverMaterial> ( *this );
}

//-----------------------------------------------------------------------
//  updateWriteTable
//-----------------------------------------------------------------------

void ObserverMaterial::updateWriteTable

    ( const Vector&         stress,
      const Matrix&         stiff,
      const Matrix&         df,
      idx_t                 ipoint )

{
}

//-------------------------------------------------------------------------------------------------
//  getHistory
//-------------------------------------------------------------------------------------------------

void ObserverMaterial::getHistory

  ( Vector&           hvals,
    const idx_t       mpoint ) 

{
  child_->getHistory ( hvals, mpoint );
}

//-------------------------------------------------------------------------------------------------
//  Hist_
//-------------------------------------------------------------------------------------------------

ObserverMaterial::Hist_::Hist_ ()
{
}

ObserverMaterial::Hist_::~Hist_ ()
{
}

  
