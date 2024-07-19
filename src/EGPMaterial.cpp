/*
 * 
 *  Copyright (C) 2019 TU Delft. All rights reserved.
 *  
 *  This class implements the Eindhoven Glassy Polymer
 *  material model.
 *
 *  reference: "Extending the EGP constitutive model for polymer 
 *  glasses to multiple relaxation times" Van Breemen et al. 2011
 * 
 *
 */

#include <jem/base/limits.h>
#include <jem/base/Error.h>
#include <jem/base/System.h>
#include <jem/base/array/operators.h>
#include <jem/base/array/tensor.h>
#include <jem/base/array/intrinsics.h>
#include <jem/io/Writer.h>
#include <jem/numeric/algebra/matmul.h>
#include <jem/numeric/algebra/LUSolver.h>
#include <jem/numeric/algebra/EigenUtils.h>
#include <jem/numeric/algebra/utilities.h>
#include <jem/io/FileWriter.h>


using jem::ALL;
using jem::END;
using jem::BEGIN;
using jem::slice;

#include "utilities.h"
#include "voigtUtilities.h"
#include "EGPMaterial.h"
#include "mcmlib.h"

using namespace voigtUtilities;
using namespace mcmlib;

using jem::io::FileWriter;
using jem::maxOf;
using jem::Error;
using jem::TensorIndex;
using jem::System;
using jem::io::endl;
using jem::io::Writer;
using jem::numeric::matmul;
using jem::numeric::LUSolver;
using jem::numeric::inverse;
using jem::numeric::norm2;
using jem::numeric::invert;

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------

const char*  EGPMaterial::STATE_PROP   = "state";

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------


EGPMaterial::EGPMaterial 

  ( idx_t rank, const Properties& globdat )
    : Material ( rank, globdat )
{
  JEM_PRECHECK ( rank >= 1 && rank <= 3 );

  t0a_    = 1.;
  t0b_    = 1.;
  Gr_     = 1.;
  T_      = 0.;      
  dH0a_   = 0.;   
  dH0b_   = 0.;   
  SHa_    = 1.;    
  SHb_    = 1.;    
  evNskT_ = 1.; 
  eva_    = 1.;    
  evh_    = 1.;    
  Cr1a_   = 1.;   
  Cr2a_   = 1.;   
  Cr1b_   = 1.;   
  Cr2b_   = 1.;   
  SSa_    = 1.;    
  SSb_    = 1.;   
  r0a_    = 1.;   
  r1a_    = 1.;
  r2a_    = 1.;
  r0b_    = 1.;
  r1b_    = 1.;
  r2b_    = 1.;
  Vacta_  = 1.;  
  Vactb_  = 1.;
  ma_     = 1.;    
  mb_     = 1.;
  k_      = 1.e3;      
  mode_   = 1;   
  nom_    = 1;    
  nam_    = 1;    
  G_      = 1.;       
  h0_     = 1.;     
  R_      = 8.3144598e3;;      
  kb_     = 1.38064852e-20; 

  strCount_ = 1;
}


EGPMaterial::~EGPMaterial ()
{}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void EGPMaterial::configure ( const Properties& props )
{
  props.get( strHardening_, "strHardening" );
  props.get( stateString_, STATE_PROP );

  if ( strHardening_ == "NEOHOOKEAN")
  {
    props.get ( t0a_, "t0a" );
    props.get ( t0b_, "t0b" );
    props.get ( Gr_, "Gr" );
  }
  else
  {
    props.get ( Vacta_, "Vacta" );
    props.get ( Vactb_, "Vactb" );
    props.get ( T_, "T" );
    props.get ( R_, "R" );
    props.get ( kb_, "kb" );
    props.get ( dH0a_, "dH0a" );
    props.get ( dH0b_, "dH0b" );
    props.get ( SHa_, "SHa" );
    props.get ( SHb_, "SHb" );
    props.get ( evNskT_, "evNskT" );
    props.get ( eva_, "eva" );
    props.get ( evh_, "evh" );
    props.get ( Cr1a_, "Cr1a" );
    props.get ( Cr2a_, "Cr2a" );
    props.get ( Cr1b_, "Cr1b" );
    props.get ( Cr2b_, "Cr2b" );
  }

  // universal
  props.get ( SSa_, "SSa" );
  props.get ( SSb_, "SSb" );
  props.get ( r0a_, "r0a" );
  props.get ( r1a_, "r1a" );
  props.get ( r2a_, "r2a" );
  props.get ( r0b_, "r0b" );
  props.get ( r1b_, "r1b" );
  props.get ( r2b_, "r2b" );
  props.get ( ma_, "ma" );
  props.get ( mb_, "mb" );
  props.get ( k_, "k" );
  props.get ( mode_, "mode" );
  props.get ( nom_, "nom" );
  props.get ( nam_, "nam" );
  props.get ( G_, "G" );
  props.get ( h0_, "h0" );

  historyNames_.resize ( 1 );
  historyNames_[0] = "eqps";

  strCount_ = STRAIN_COUNTS[rank_];

  if ( stateString_ == "AXISYMMETRIC" ) { strCount_ += 1; }

  nn_ = false;
  first_ = true;
  props.find ( nn_, "nn" );

  System::out() << "Test " << strCount_ << endl;
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void EGPMaterial::getConfig ( const Properties& conf ) const
{
  if ( strHardening_ == "NEOHOOKEAN")
  {
    conf.set ( "t0a", t0a_ );
    conf.set ( "t0b", t0b_ );
    conf.set ( "Gr", Gr_ );
  }
  else
  {
    conf.set ( "dH0a", dH0a_ );
    conf.set ( "dH0b", dH0b_ );
    conf.set ( "SHa", SHa_ );
    conf.set ( "SHb", SHb_ );
    conf.set ( "evNskT", evNskT_ );
    conf.set ( "eva", eva_ );
    conf.set ( "evh", evh_ );
    conf.set ( "Cr1a", Cr1a_ );
    conf.set ( "Cr2a", Cr2a_ );
    conf.set ( "Cr1b", Cr1b_ );
    conf.set ( "Cr2b", Cr2b_ );
    conf.set ( "Vacta", Vacta_ );
    conf.set ( "Vactb", Vactb_ );
    conf.set ( "R", R_);
    conf.set ( "kb", kb_);
    conf.set ( "T", T_ );
  }

  // universal
  conf.set ( "SSa", SSa_ );
  conf.set ( "SSb", SSb_ );
  conf.set ( "r0a", r0a_ );
  conf.set ( "r1a", r1a_ );
  conf.set ( "r2a", r2a_ );
  conf.set ( "r0b", r0b_ );
  conf.set ( "r1b", r1b_ );
  conf.set ( "r2b", r2b_ );
  conf.set ( "ma", ma_ );
  conf.set ( "mb", mb_ );
  conf.set ( "k", k_ );
  conf.set ( "mode", mode_ );
  conf.set ( "nom", nom_ );
  conf.set ( "nam", nam_ );
  conf.set ( "G", G_ );
  conf.set ( "h0", h0_ );

  
  conf.set ( "strHardening", strHardening_ );
  conf.set ( STATE_PROP, stateString_ );

  conf.set ( "strCount", strCount_ );
}


//-----------------------------------------------------------------------
//   getHistory
//-----------------------------------------------------------------------

void EGPMaterial::getHistory

  ( const Vector&  hvals,
    const idx_t    mpoint ) const

{
  (*latestHist_)[mpoint].toVector ( hvals );
 // System::out() << "getHistory: " << hvals << "\n";
}

//-----------------------------------------------------------------------
//   getInitHistory
//-----------------------------------------------------------------------

void EGPMaterial::getInitHistory

  ( const Vector&  hvals,
    const idx_t    mpoint ) const

{
  initHist_[0].toVector ( hvals );
 // System::out() << "getInitHistory.\n"; // << preHist_[mpoint].F << "\n";
}

//-----------------------------------------------------------------------
//   getHistoryCount
//-----------------------------------------------------------------------

idx_t EGPMaterial::getHistoryCount

  ( ) const

{
  return 10+9*nom_+nom_*3+mode_*2; //+strCount_*strCount_;
}

//-----------------------------------------------------------------------
//   setHistory
//-----------------------------------------------------------------------

void EGPMaterial::setHistory

  ( const Vector&  hvals,
    const idx_t    mpoint )

{
//  System::out() << "Setting history\n";
  preHist_[mpoint].F(0,0)     = hvals[0];     // Total deformation gradient
  preHist_[mpoint].F(0,1)     = hvals[1];
  preHist_[mpoint].F(0,2)     = hvals[2];
  preHist_[mpoint].F(1,0)     = hvals[3];
  preHist_[mpoint].F(1,1)     = hvals[4];
  preHist_[mpoint].F(1,2)     = hvals[5];
  preHist_[mpoint].F(2,0)     = hvals[6];
  preHist_[mpoint].F(2,1)     = hvals[7];
  preHist_[mpoint].F(2,2)     = hvals[8];

  idx_t current = 9;

  for ( idx_t imode = 0; imode < nom_; imode++ )
  {
    for ( idx_t i = 0; i < 3; i++ )
    {
      for ( idx_t j = 0; j < 3; j++ )
      {
        preHist_[mpoint].mtBe[imode](i, j) = hvals[current];
        current += 1;
      }
    }
  }

  for ( idx_t imode = 0; imode < nom_; imode++ )
  {
    preHist_[mpoint].lmbd     = hvals[current];
    current += 1;
  }

  for ( idx_t imode = 0; imode < nom_; imode++ )
  {
    preHist_[mpoint].visc     = hvals[current];
    current += 1;
  }

  for ( idx_t imode = 0; imode < mode_; imode++ )
  {
    preHist_[mpoint].taueq    = hvals[current];
    current += 1;
  }


  for ( idx_t imode = 0; imode < nom_; imode++ )
  {
    preHist_[mpoint].mtaueq    = hvals[current];
    current += 1;
  }


  for ( idx_t imode = 0; imode < mode_; imode++ )
  {
    preHist_[mpoint].agepar    = hvals[current];
    current += 1;
  }

  preHist_[mpoint].eqps = hvals[current];
  current += 1;
/*
  for ( idx_t i = 0; i < strCount_; i++ )
  {
    for ( idx_t j = 0; j < strCount_; j++ )
    {
      preHist_[mpoint].Stiff(i, j) = hvals[current];
      current += 1;
    }
  }
*/

  latestHist_      = &preHist_;
  newHist_[mpoint] = preHist_[mpoint];
}

//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------

void EGPMaterial::update

    ( const Vector&         stress,
      const Matrix&         stiff,
      const Matrix&         df,
      idx_t                 ipoint )

{
  // this version implements update function using tuples
  // this is done to gain computational speed

  M33 mI, mtF;
  M33 mtB, mS, mtBd;

  Vector gamma (nom_);

  // Writer&     dbgOut     ( System::debug ( "ul" ) );

  // get history variables from the previous (converged) time step
  // and initialize them for the current time step

  // get defor. grad. at previous and current time step

  M33&     mFB      (preHist_[ipoint].F);
  M33&     mF       (newHist_[ipoint].F);

  Vector lambdaB  (preHist_[ipoint].lmbd); // plasticity parameters
  Vector lambda   (newHist_[ipoint].lmbd);

  Vector hB       (preHist_[ipoint].visc); // viscosities
  Vector h        (newHist_[ipoint].visc);

  Vector teqB     (preHist_[ipoint].taueq); // eq. stress for process a/b
  Vector teq      (newHist_[ipoint].taueq);

  Vector mteqB    (preHist_[ipoint].mtaueq); // eq.stress for each mode
  Vector mteq     (newHist_[ipoint].mtaueq); 

  Vector SB       (preHist_[ipoint].agepar); // aging parameters a/b
  Vector S        (newHist_[ipoint].agepar);

  // call isochoric, elastic, left Cauchy Green deformation tensor mtBe 

  Array<M33>    mtBeB    (preHist_[ipoint].mtBe);
  Array<M33>    mtBe     (newHist_[ipoint].mtBe);
 
  double& eqpsB    (preHist_[ipoint].eqps); // equiv. plastic strain
  double& eqps     (newHist_[ipoint].eqps);

  // update mF with given df
  // first it is needed to translate matrix df to corresp. tuple DF

  M33 DF;

  Matrix& Stiff (newHist_[ipoint].Stiff);

    Ref<PrintWriter> out = newInstance<PrintWriter> ( &System::out() );
    out->nformat.setFractionDigits( 8 );

 if ( nn_ )
  {
  
    // When NN is enabled, the total def. grad is passed
    mF(0,0) = df(0,0); mF(0,1) = df(0,1); mF(0,2) = df(0,2);
    mF(1,0) = df(1,0); mF(1,1) = df(1,1); mF(1,2) = df(1,2);
    mF(2,0) = df(2,0); mF(2,1) = df(2,1); mF(2,2) = df(2,2);

 //   *out << "preHist F " << preHist_[ipoint].F << "\n";
  
    M33 mFBinv ( preHist_[ipoint].F );

    invert ( mFBinv );

    DF = matmul ( mF, mFBinv );
   
  //  out << "pre lamba " << preHist_[ipoint].lmbd << " eqps " << preHist_[ipoint].eqps << "\n";
 /*   *out << "preHist F " << preHist_[ipoint].F << "\n";
    *out << "Fprev inverse: " << mFBinv << "\n";
    *out << "F " << mF << "\n";
    *out << "def incremental " << DF << "\n";
    System::out() << "det total def " << jem::numeric::det(mF) << "\n";
    System::out() << "det incremental " << jem::numeric::det(DF) << "\n";*/
  }
  else
  {
    DF(0,0) = df(0,0); DF(0,1) = df(0,1); DF(0,2) = df(0,2);
    DF(1,0) = df(1,0); DF(1,1) = df(1,1); DF(1,2) = df(1,2);
    DF(2,0) = df(2,0); DF(2,1) = df(2,1); DF(2,2) = df(2,2);
  
    mF = matmul ( DF, mFB );

 //   System::out() << "def total " << mF << "\n";
  }

  // initialize other history variables for the current time step
  // these variables will be updated in Newton-Raphson procedure
  // as a part of deviatoric stress calculation

  lambda = lambdaB.clone();
  h      = hB     .clone();
  teq    = teqB   .clone();
  mteq   = mteqB  .clone();
  S      = SB     .clone();
  mtBe   = mtBeB  .clone();
  eqps   = eqpsB          ;

  if ( ipoint == 1000000 )
  {
     System::out() << "preHist F " << preHist_[ipoint].F << "\n";
     System::out() << "preHist mtBe " << preHist_[ipoint].mtBe << "\n";
     System::out() << "preHist lambda " << preHist_[ipoint].lmbd << "\n";
     System::out() << "preHist visc " << preHist_[ipoint].visc << "\n";
     System::out() << "preHist taueq" << preHist_[ipoint].taueq << "\n";
     System::out() << "preHist mtaueq " << preHist_[ipoint].mtaueq << "\n";
     System::out() << "preHist agepar " << preHist_[ipoint].agepar << "\n";
     System::out() << "preHist eqps " << preHist_[ipoint].eqps << "\n";
  } 

  // determine Jacobian for current and previous time step

  double J  = jem::numeric::det(mF);
  double JB = jem::numeric::det(mFB);

  double trmtB;

  // isochoric deformation gradient

  mtF = mF / pow( J, 1.0/3.0 );


  // isochoric left Cauchy Green deformation tensor (t: ~)

  mtB = matmul ( mtF, mtF.transpose() );

//  if ( nn_ ) System::out() << "mtB " << mtB << "\n";

  // form unit tensor

  mcmlib::inimI( mI );

  //   calculate hydrostatic stress

  M33 mSh; // initialize hydrostatic stress
  
  mSh = k_ * ( J - 1.0 ) * mI;
 
  //   calculate strain hardening stress  

  M33 mSr; // initialize hardening stress

  if ( strHardening_ == "NEOHOOKEAN" )
  {
    trace ( mtB, trmtB );

    mtBd = mtB - 1.0 / 3.0 * trmtB * mI;

    mSr = Gr_ * mtBd;
  }
  else
  {
    edvilStress_ ( mSr, mI, J, mtB);
  }
  
  //   calculate deviatoric stress

  M33 mSs; // initialize deviatoric stress

  deviatoricStress_ (mSs, DF, mtB, mtBe, mtBeB, lambda, h, teq, 
                          mteq, S, gamma, eqps, JB, eqpsB);

  //   calculate total stress tensor   

  mS = mSh + mSs + mSr;

//  *out << "total mS " << mS << "\n";

  // calculate column array equivalent to the total strss tensor

  if ( stateString_ == "PLANE_STRAIN" )
  {
    stress[0] = mS(0,0);
    stress[1] = mS(1,1);

    stress[2] = ( mS(0,1) + mS(1,0) ) / 2.;
  }

  else if ( stateString_ == "AXISYMMETRIC" )
  {
    stress[0] = mS(0,0);
    stress[1] = mS(1,1);

    stress[2] = ( mS(0,1) + mS(1,0) ) / 2.;

    stress[3] = mS(2,2);
  }

  else if ( stateString_ == "3D")
  {
    stress[0] = mS(0,0);
    stress[1] = mS(1,1);
    stress[2] = mS(2,2);

    stress[3] = ( mS(0,1) + mS(1,0) ) / 2.;
    stress[4] = ( mS(1,2) + mS(2,1) ) / 2.;
    stress[5] = ( mS(2,0) + mS(0,2) ) / 2.;
  }

  else
  {
  	throw Error ( JEM_FUNC, "unknown state problem" );
  }

  
  //   stiffness update 

  // -------------------
  // full Newton-Raphson
  // -------------------

  // stiffness_ ( stiff, mF, mFB, mtB, 

  //           mS, mtBe, mtBeB,

  //           h, mteq, lambda,

  //           teq, gamma, J, JB );

  // Stiff = stiff;

  // -----------------------
  // modified Newton-Raphson
  // -----------------------


  Matrix StiffFull ( 9, 9 );
  StiffFull = 0.0;

 // if ( nn_ ) //jem::numeric::det(DF) == 1.0  )
 // {
//    System::out() << "using it for NN? " << nn_ << "\n";
   // System::out() << "Stress " << stress << "\n";
/*    System::out() << "Stiff " << Stiff << " mf " << mF << "mFB " << mFB << "\n";
    System::out() << "mtB " << mtB << " mS " << mS << " mtBe " << mtBe << "\n";
    System::out() << "mtBeB " << mtBeB << " h " << h << " mteq " << mteq << "\n";
    System::out() << "lambda " << lambda << " teq " << teq << " gamma " << gamma << "\n";
    System::out() << "J " << J << " JB " << JB << "\n";
*/
  stiffness_ ( nn_, StiffFull, Stiff, mF, mFB, mtB, 
  		mS, mtBe, mtBeB,
  		h, mteq, lambda,
  		teq, gamma, J, JB );

  if ( nn_ ) 
  {
    stiff = 0.0;
    // stiff ( slice (0, 6), slice (0,6) ) = Stiff.clone();

    // Changing to finite differences format
    IdxVector idc ( 9 );
    idc[0] = 0;    // Getting normal components
    idc[1] = 3;
    idc[2] = 8;

    idc[3] = 4; idc[4] = 1;
    idc[5] = 5; idc[6] = 7;
    idc[7] = 6; idc[8] = 2;

    stiff = 0.0;
    Matrix stiffGen ( 9, 9 );
    stiffGen = 0.0;

    for ( idx_t i = 0; i < 9; i++ )
    {
      for ( idx_t j = 0; j < 9; j++ )
      {
        stiffGen ( i, j ) = StiffFull (idc[i], idc[j] );
      }
    }      
     
/*      stiff = StiffFull;*/
     Vector stressGen ( 9 );
     for ( idx_t i = 0; i < 3; i++ )
     {
       for ( idx_t j = 0; j < 3; j++ )
       {
         stressGen[i*3+j] = 0.5 * ( mS(i,j) + mS(j,i) );
    //     stressGen[i*3+j] = mS(i,j);
	}
     }

  //   System::out() << "mS " << mS << "\n";
     stress = stressGen;

     // Making the stiffness tensor symmetric

 /*    stiff(0,ALL) = stiffGen(0,ALL);
     stiff(4,ALL) = stiffGen(4,ALL);
     stiff(8,ALL) = stiffGen(8,ALL);
     stiff(1,ALL) = stiff(3,ALL) = 0.5 * ( stiffGen(1,ALL) + stiffGen(3,ALL) );
     stiff(2,ALL) = stiff(6,ALL) = 0.5 * ( stiffGen(2,ALL) + stiffGen(6,ALL) );
     stiff(5,ALL) = stiff(7,ALL) = 0.5 * ( stiffGen(5,ALL) + stiffGen(7,ALL) );*/
   stiff = stiffGen;
  }
  else
  {
    stiff = Stiff;
  }

  latestHist_ = &newHist_;
}

//-----------------------------------------------------------------------
//   updateWriteTable
//-----------------------------------------------------------------------

void EGPMaterial::updateWriteTable

    ( const Vector&         stress,
      const Matrix&         stiff,
      const Matrix&         df,
      idx_t                 ipoint )

{
  M33 mI, mtF, DF;
  M33 mtB, mS, mtBd;

  Vector gamma (nom_);

  // get defor. grad. at previous and current time step

  M33&       mF      (preHist_[ipoint].F);
  M33&       mFB     (newHist_[ipoint].F);
  Vector     lambda  (preHist_[ipoint].lmbd); 
  Array<M33> mtBeB   (newHist_[ipoint].mtBe);

  // determine Jacobian for current time step

  double J   = jem::numeric::det(mF);

  DF = matmul ( mF, inverse (mFB) );

  double trmtB;

  // isochoric deformation gradient

  mtF = mF / pow( J, 1.0/3.0 );

  // isochoric left Cauchy Green deformation tensor (t: ~)

  mtB = matmul ( mtF, mtF.transpose() );

  // form unit tensor

  mcmlib::inimI( mI );

  //-----------------------------------------------------------------------
  //   calculate hydrostatic stress
  //-----------------------------------------------------------------------

  M33 mSh; // initialize hydrostatic stress
  
  mSh = k_ * ( J - 1.0 ) * mI;

  //-----------------------------------------------------------------------
  //   calculate strain hardening stress
  //-----------------------------------------------------------------------

  M33 mSr; // initialize hardening stress

  if ( strHardening_ == "NEOHOOKEAN" )
  {
    trace ( mtB, trmtB );

    mtBd = mtB - 1.0 / 3.0 * trmtB * mI;

    mSr = Gr_ * mtBd;
  }
  else
  {
    edvilStress_ ( mSr, mI, J, mtB);
  }

  //-----------------------------------------------------------------------
  //   calculate deviatoric stress
  //-----------------------------------------------------------------------

  M33 mSs; // initialize deviatoric stress

  double Jn; 

  Tuple <double,3> eigval, leig;  

  M33 mtFn, mtCn, evecs, mtUn;
  M33 mtUni, mRn, mbSs;
  
  Array <M33> mCpnB (nom_), mCpn (nom_), mbtBen (nom_);


  for (idx_t i = 0; i < nom_; i++)
  {
    mCpnB [i] = inverse (mtBeB [i]);
  }

  // calculate incremental deformation variables

  Jn = jem::numeric::det( DF ); 

  mtFn = DF / pow( Jn, 1.0 / 3.0 );
  
  mtCn = matmul ( mtFn.transpose(), mtFn );

  // calculate eigenvalues and eigenvectors of mtCn
  // this will provide lambda^2 (not plasticity parameters)

  jem::numeric::EigenUtils::symSolve (eigval, evecs, mtCn);

  // eigsum  = eigval[0] + eigval[1] + eigval[2];

  leig[0] = sqrt (eigval[0]); // vector of eigenvalues, lambda
  leig[1] = sqrt (eigval[1]); // do not confuse with plsticity parameters
  leig[2] = sqrt (eigval[2]);


  for (idx_t k = 0; k < 3; k++)
  {
    for (idx_t l = 0; l < 3; l++)
    {
      mtUn(k,l) = 0.0;
      // mtCn(k,l) = 0.0; // already calculated 

      for (idx_t m = 0; m < 3; m++)
      {
        mtUn(k,l) += leig[m] * evecs(k,m) * evecs(l,m);
        // mtCn(k,l) = mtCn(k,l) + lsq[m] * evecs(k,m) * evecs(l,m);
      }
    }
  }

  M33 mCpni, mbtBend, mbsdi; 
  M33 mbsda, mbsdb;

  double trmbtBen;

  mbsda = 0.0;
  mbsdb = 0.0;

  for (idx_t i = 0; i < nom_; i++)
  {
    // calculate kinematic variables

    mCpn[i] = (1.0 - lambda[i]) * mtCn +
                                lambda[i] * mCpnB[i];

    mCpni = inverse(mCpn[i]);

    mbtBen[i] = matmul(matmul(mtUn, mCpni), mtUn.transpose());

    trace (mbtBen[i], trmbtBen);


    mbtBend = mbtBen[i] - 1.0 / 3.0 * trmbtBen * mI;


    // calculate rotation neutralized stress 
    // and determine coressponding process (alpha or beta)

    mbsdi = G_[i] * mbtBend;

    if (i <= nam_ - 1)
    {
      mbsda += mbsdi;
    }
    else
    {
      mbsdb += mbsdi;
    }
  }

  mbSs = mbsda + mbsdb;

  // calculate rotation tensor

  mtUni = inverse (mtUn);

  mRn = matmul (mtFn, mtUni);

  // calculate increment of deviatoric stress

  mSs = matmul(matmul(mRn,mbSs), mRn.transpose()); 


  //-----------------------------------------------------------------------
  //   calculate total stress tensor 
  //-----------------------------------------------------------------------

  mS = mSh + mSs + mSr;

  // calculate column array equivalent to the total strss tensor

  if ( stateString_ == "PLANE_STRAIN" )
  {
    
    // all non-zero components are stored in the table
    // this is different compared to update_ function which has only in-plane 
    // components; this is beacuse tangent stiff for plane strain is 3x3
    // for the output, out-of-plane normal stress component might be relevant

    stress[0] = mS(0,0);
    stress[1] = mS(1,1);

    stress[2] = ( mS(0,1) + mS(1,0) ) / 2.;

    stress[3] = mS(2,2);
  }

  else if ( stateString_ == "AXISYMMETRIC" )
  {
    stress[0] = mS(0,0);
    stress[1] = mS(1,1);

    stress[2] = ( mS(0,1) + mS(1,0) ) / 2.;

    stress[3] = mS(2,2);
  }

  else
  {
    stress[0] = mS(0,0);
    stress[1] = mS(1,1);
    stress[2] = mS(2,2);

    stress[3] = ( mS(0,1) + mS(1,0) ) / 2.;
    stress[4] = ( mS(1,2) + mS(2,1) ) / 2.;
    stress[5] = ( mS(2,0) + mS(0,2) ) / 2.;
  }

  // because only stresses are printed to the table
  // stiffness calculction is not relevant here

  stiff = 0.0;

  stiff ( 0, 0) = J;

}

//-----------------------------------------------------------------------
//   fill3DStress
//-----------------------------------------------------------------------

// Tuple<double,6> EGPMaterial::fill3DStress

//   ( const Vector&    v3 ) const

// {
//   if ( v3.size() == 3 )
//   {
//     // double poisson = lambda_ / ( 2 * (lambda_ + mu_));

//     // double sig_zz = state_ == PlaneStress
//     //               ? 0.
//     //               : poisson * ( v3[0] + v3[1] );

//     // return fillFrom2D_ ( v3, sig_zz );

//     throw Error ( JEM_FUNC, "Implementation with 3 stress components not supportd in EGP." );
//   }
//   else if ( v3.size() == 4 )
//   {
//     return fillFrom2D_ ( v3[slice(0,3)], v3[3] );
//   }
//   else
//   {
//     return fillFrom3D_ ( v3 );
//   }
// }

// --------------------------------------------------------------------
//  commit
// --------------------------------------------------------------------

void  EGPMaterial::commit()

{
  newHist_.swap ( preHist_ );

  latestHist_ = &preHist_;
}

// --------------------------------------------------------------------
//  allocPoints
// --------------------------------------------------------------------

void  EGPMaterial::allocPoints

    ( const idx_t   count )

{
  Hist_ hist ( nom_, mode_, strCount_ );
 
  initHist_.pushBack ( hist, 1 );
  preHist_.pushBack ( hist, count );
  newHist_.pushBack ( hist, count );
}

// --------------------------------------------------------------------
//  Hist_ constructor
// --------------------------------------------------------------------

EGPMaterial::Hist_::Hist_ 

  ( const idx_t nom, const idx_t mode, const idx_t strCount )

  : mtBe(nom), Stiff(strCount,strCount), lmbd(nom), visc(nom), 

    taueq(mode), mtaueq(nom), agepar(mode)

{
  F = 0.0;

  F(0,0) = 1.0; F(1,1) = 1.0; F(2,2) = 1.0;

  for (idx_t k = 0; k < nom; k++)
  {
    M33 mtemp; mtemp = 0.0;

    mtemp (0,0) = 1.0; mtemp (1,1) = 1.0; mtemp (2,2) = 1.0;

    mtBe[k] = mtemp;
  }

  Stiff = 0.0;


  for (idx_t i = 0; i < strCount; i++)
  {
    Stiff(i,i) = 1.0;
  }

  lmbd = 1.0; // corresponds to purely elastic behavior

  visc = 1.0e10; 

  taueq = 0.; mtaueq = 0.; agepar = 0.0; eqps = 0.0;

}

// --------------------------------------------------------------------
//  Hist_ copy constructor 
// --------------------------------------------------------------------

EGPMaterial::Hist_::Hist_ 

  ( const Hist_&  h )

  : mtBe ( h.mtBe.clone() ), Stiff ( h.Stiff.clone() ),

    lmbd ( h.lmbd.clone() ), visc ( h.visc.clone() ),

    taueq ( h.taueq.clone() ), mtaueq ( h.mtaueq.clone() ),

    agepar ( h.agepar.clone() ), eqps ( h.eqps )

{
  F = h.F;
}

// -------------------------------------------------------------------
//  Hist_::toVector
// -------------------------------------------------------------------

inline void EGPMaterial::Hist_::toVector

 ( const Vector&  vec ) const

{
//  System::out() << "vec size " << vec.size(0) << "\n";

  if ( vec.size(0) == 1 )
  {
    vec[0] = eqps; 
  }
  else
  {

  // Define sizes of matrices

  idx_t nom, mode, strCount;
  nom = mtBe.size(0);
  mode = taueq.size(0);
//  strCount = Stiff.size(0);
 // System::out() << "nom " << nom << " mode " << mode << " strCount " << strCount << "\n";

  idx_t id = 0;

  for ( idx_t i = 0; i < 3; i++ )
  {
    for ( idx_t j = 0; j < 3; j++ )
    {
      vec[id]  = F(i, j);
      id += 1;
    }
  }

  for ( idx_t k = 0; k < nom; k++ )
  {
  //  System::out() << "mtBe toVector " << mtBe[k] << "\n";
    for ( idx_t i = 0; i < 3; i++ )
    {
      for ( idx_t j = 0; j < 3; j++ )
      {
        vec[id]  = mtBe[k](i, j);
        id += 1;
      }
    }
  }
 
  for ( idx_t k = 0; k < nom; k++ )
  {
    vec[id] = lmbd[k];
    id += 1;
  }
 
  for ( idx_t k = 0; k < nom; k++ )
  {
    vec[id] = visc[k];
    id += 1;
  } 

  for ( idx_t k = 0; k < mode; k++ )
  {
    vec[id] = taueq[k];
    id += 1;
  }

  for ( idx_t k = 0; k < nom; k++ )
  {
    vec[id] = mtaueq[k];
    id += 1;
  }

  for ( idx_t k = 0; k < mode; k++ )
  {
    vec[id] = agepar[k];
    id += 1;
  }

  vec[id] = eqps;
  id += 1;

 /* for ( idx_t i = 0; i < strCount; i++ )
  {
    for ( idx_t j = 0; j < strCount; j++ )
    {
      vec[id]  = Stiff(i, j);
      id += 1;
    }
  }*/
  
//  System::out() << "vec: " << vec << "\n";
 }
}

  

