/*
 *  TU Delft 
 *
 *  Iuri Barcelos, August 2019
 *
 *  Material class that uses a nested neural network to compute
 *  stress and stiffness.
 *
 */

#include <jem/base/Error.h>
#include <jem/base/System.h>
#include <jem/base/Array.h>
#include <jem/util/Properties.h>
#include <jem/numeric/algebra/matmul.h>
#include <jem/numeric/algebra/utilities.h>
#include <jem/numeric/utilities.h>
#include <jem/io/FileReader.h>
#include <jive/model/Model.h>
#include <jive/model/StateVector.h>
#include <jive/app/ChainModule.h>
#include <jive/app/UserconfModule.h>
#include <jive/app/InfoModule.h>
#include <jive/fem/InitModule.h>
#include <jive/fem/ElementSet.h>
#include <jive/fem/NodeSet.h>
#include <jive/util/Globdat.h>
#include <jive/util/utilities.h>
#include <jive/util/DofSpace.h>
#include <jive/util/Assignable.h>

#include <jem/numeric/algebra/LUSolver.h>
#include <jem/numeric/algebra/EigenUtils.h>
#include <jem/numeric/algebra/utilities.h>


#include <jem/io/PrintWriter.h>

#include "ANNMaterial.h"

#include "utilities.h"
#include "LearningNames.h"

using namespace jem;

using jem::io::FileReader;

using jive::model::Model;
using jive::model::StateVector;
using jive::app::ChainModule;
using jive::app::UserconfModule;
using jive::app::InitModule;
using jive::app::InfoModule;
using jive::util::Globdat;
using jive::util::joinNames;
using jive::util::DofSpace;
using jive::util::Assignable;
using jive::fem::ElementSet;
using jive::fem::NodeSet;

using jem::numeric::matmul;
using jem::numeric::inverse;

using jem::io::PrintWriter;
//=======================================================================
//   class ANNMaterial
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------

const char* ANNMaterial::NETWORK    = "network";
const char* ANNMaterial::WEIGHTS    = "weights";
const char* ANNMaterial::NORMALIZER = "normalizer";
const char* ANNMaterial::RECURRENT  = "recurrent";

//-----------------------------------------------------------------------
//   constructor and destructor
//-----------------------------------------------------------------------

ANNMaterial::ANNMaterial

  ( const idx_t       rank,
    const Properties& globdat )

  : Material ( rank, globdat )

{
  System::out() << "ANNMaterial. rank " << rank << "\n";

  rank_ = rank;

  perm23_ = false; // Default
  first_ = true;

  JEM_PRECHECK ( rank_ >= 1 && rank_ <= 3 );

  data_. resize ( 0 );
  state_.resize ( 0 );

  // Check for permutation

  Assignable<ElementSet> eset;
  Assignable<NodeSet>    nset;
  eset = ElementSet::get ( globdat, "ANNMaterial" );
  nset = eset.getNodes ( );

  idx_t meshrank = nset.rank ( );

  System::out() << "ANNMaterial. rank_ " << rank_ << "\n";

  if ( meshrank < rank_ )
  {
    perm23_ = true;

    System::out() << "Performing plane strain analysis on 3D network\n";

    perm_.resize ( 3 );

    perm_[0] = 0;   
    perm_[1] = 1;   
    perm_[2] = 3;   
  }
}

ANNMaterial::~ANNMaterial ()
{}

//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------

void ANNMaterial::configure

  ( const Properties& props )

{
  using jem::System;
  System::out() << "ANNMaterial. configure.\n";
  String fname;

  Ref<ChainModule> chain;

  netData_ = Globdat::newInstance ( NETWORK );

  Globdat::getVariables ( netData_ );

  props.get ( fname, NORMALIZER );
  conf_.set ( NORMALIZER, fname );

  nizer_ = newNormalizer ( fname );

  if ( nizer_ != nullptr )
  {
    props.set ( "outNormalizer", nizer_ );
  }

  System::out() << "netData " << netData_ << "\n";
  
  chain = newInstance<ChainModule> ( joinNames ( NETWORK, "chain" ) );

  chain->pushBack ( newInstance<UserconfModule> ( joinNames ( NETWORK, "userinput"   ) ) );
  chain->pushBack ( newInstance<InitModule>     ( joinNames ( NETWORK, "init"        ) ) );
  chain->pushBack ( newInstance<InfoModule>     ( joinNames ( NETWORK, "info"        ) ) );

  chain->configure ( props,  netData_        );
  chain->getConfig ( conf_,  netData_        );
  chain->init      ( conf_,  props, netData_ );

  network_ = chain;

  props.get ( fname, WEIGHTS );
  conf_.set ( WEIGHTS, fname );

  initWeights_ ( fname );

  props.get ( recurrent_, RECURRENT  );
  conf_.set ( RECURRENT,  recurrent_ );

  if ( perm23_ )
  {
    System::out() << "Performing plane strain analysis on 3D network\n";

    perm_.resize ( 3 );

    perm_[0] = 0;   
    perm_[1] = 1;   
    perm_[2] = 3;   
  }
}

//-----------------------------------------------------------------------
//   getOutNormalizer
//-----------------------------------------------------------------------

inline Ref<Normalizer> ANNMaterial::getOutNormalizer () const
{
  return nizer_;
}

//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------

void ANNMaterial::getConfig

  ( const Properties& conf,
    const Properties& globdat ) const
    
{
  String fname;
  Properties netconf;

  conf_.get ( fname, WEIGHTS );
  conf. set ( WEIGHTS, fname );

  conf_.get ( fname, NORMALIZER );
  conf. set ( NORMALIZER, fname );

  conf_.get ( netconf, NETWORK );
  conf. set ( NETWORK, netconf );
}

//-----------------------------------------------------------------------
//  allocPoints 
//-----------------------------------------------------------------------

void ANNMaterial::allocPoints

  ( const idx_t       npoints )

{
  const idx_t STRAIN_COUNTS[4] = { 0, 1, 3, 9 };

  idx_t size = STRAIN_COUNTS[rank_];

  data_.resize ( npoints );

  for ( idx_t p = 0; p < npoints; ++p )
  {
    data_[p] = newInstance<NData> ( 1, size, size );
  }

  if ( recurrent_ )
  {
    System::out() << "Entered here \n";
    state_.resize ( npoints );

    for ( idx_t p = 0; p < npoints; ++p )
    {
      state_[p] = newInstance<NData> ( 1, size, size );
    }
  }
}

//-----------------------------------------------------------------------
//  update 
//-----------------------------------------------------------------------

void ANNMaterial::update

  ( const Vector&       stress,
    const Matrix&       stiff,
    const Matrix&       F,
    idx_t   ipoint )
{
  String context ( "ANNMaterial::update" );

  Properties params;

  idx_t ncomp = F.size(0);

  bool local = false; 
  bool rotate = true;

  Matrix stressTensor ( ncomp, ncomp );
  Matrix stressTensorR ( ncomp, ncomp );

  stiff = 0.0;

  Ref<Model> model = Model::get ( netData_, context );

  // Get R and U tensors from polar decomposition 

  Matrix R, U;

//  System::out() << "\nANNMaterial. Point " << ipoint << "\n"; 
//  System::out() << "Performing polar decomposition on F\n" << F << "\n";

  polarDecomposition ( R, U, F );

  Ref<PrintWriter> out = newInstance<PrintWriter> ( &System::out() );
  out->nformat.setFractionDigits( 10 );

 // *out << "Deformation tensor\n" << F << "\n";
//  *out << "Rotation tensor\n" << R << "\n";
//  *out << "Strech tensor\n" << U << "\n";

  if ( local )
  {
     // Transforming 9D vector in the frame x, y and z to local frame 1, 2 and 3
 
     Matrix auxF ( 4, 4 );
     auxF = 0.0;
     Matrix Fcopy = F.clone();

     System::out() << "F in x, y, z system: " << F << "\n";

     auxF ( slice(1, 4), slice(1, 4) ) = Fcopy;
     auxF ( 0, slice(1, 3) ) = Fcopy ( 2, slice(0, 2) );
     auxF ( slice (1, 3), 0 ) = Fcopy ( slice(0,2), 2 );

     auxF (0,0) = Fcopy (2, 2);  
     Matrix Flocal = auxF ( slice (0, 3), slice (0, 3) );

//     System::out() << "auxF " << auxF << "\n";
     System::out() << "F in local 1, 2, 3 system: " << Flocal << "\n"; 
     for ( idx_t i = 0; i < ncomp; i++ )
     {
       data_[ipoint]->inputs(slice(i*ncomp, (i+1)*ncomp),0) = Flocal(i, ALL); 
     }

  }
  else
  {
     if ( rotate )
     {
       for ( idx_t i = 0; i < ncomp; i++ )
       {
         data_[ipoint]->inputs(slice(i*ncomp, (i+1)*ncomp),0) = U(i, ALL);
       }
     }
     else
     {
       for ( idx_t i = 0; i < ncomp; i++ )
       {
         data_[ipoint]->inputs(slice(i*ncomp, (i+1)*ncomp),0) = F(i, ALL); 
       }
     }
  }

  params.set ( LearningParams::DATA, data_[ipoint] );
  params.set ( "dtglobal", dt_ );

  if ( recurrent_ && !first_ )
  {
    params.set ( LearningParams::STATE, state_[ipoint] );
  }

  params.set ( "offline", true );

  model->takeAction ( LearningActions::PROPAGATE,   params, netData_ );
  model->takeAction ( LearningActions::GETJACOBIAN, params, netData_ );
 
  Matrix jacnn ( data_[ipoint]->jacobian.clone() );

  if ( perm23_ )
  {
    stress = data_[ipoint]->outputs(ALL,0)[perm_];
  }
  else
  {

    if ( local )
    {
      // Condensing 9D vector in the local frame 1, 2 and 3 to 6D vector in the frame
      // x, y and z (where z is the fiber direction)
      // 0 -> z   1-> x   2-> y

      stress[0] = data_[ipoint]->outputs(4,0); 
      stress[1] = data_[ipoint]->outputs(8,0);
      stress[2] = data_[ipoint]->outputs(0,0);
      stress[3] = .5 * (data_[ipoint]->outputs(5, 0) + data_[ipoint]->outputs(7,0));
      stress[4] = .5 * (data_[ipoint]->outputs(6, 0) + data_[ipoint]->outputs(2,0));
      stress[5] = .5 * (data_[ipoint]->outputs(3, 0) + data_[ipoint]->outputs(1,0)); 
    }
    else
    { 
      // 9D vector is in the global frame x, y and z
      
      Vector stressnn =  nizer_->denormalize ( data_[ipoint]->outputs(ALL,0) ).clone();

      // Transforming 9D vector to 3x3 symmetric tensor

      to2ndOrderTensor ( stressTensor, stressnn );
      
      //System::out() << "ANNMaterial. Stresses in global frame:\n" << stressTensor << "\n";

       // Selecting relevant components 

       if ( rotate )
       {
          // Get stresses back to rotated system

          Matrix Rstress ( ncomp, ncomp );
	  Matrix RstressRt ( ncomp, ncomp );

	  matmul ( Rstress, R, stressTensor.clone() );
	  matmul ( RstressRt, RstressRt, R.transpose() );

	  stress[0] = RstressRt ( 0, 0 );
	  stress[1] = RstressRt ( 1, 1 );
	  stress[2] = RstressRt ( 2, 2 );
	
	  stress[3] = RstressRt ( 0, 1 );
	  stress[4] = RstressRt ( 1, 2 );
	  stress[5] = RstressRt ( 0, 2 );

	  stressTensorR = RstressRt.clone();
       }
       else
       {
	 stress[0] = stressTensor ( 0, 0 );
	 stress[1] = stressTensor ( 1, 1 );
	 stress[2] = stressTensor ( 2, 2 );
	
	 stress[3] = stressTensor ( 0, 1 );
	 stress[4] = stressTensor ( 1, 2 );
	 stress[5] = stressTensor ( 0, 2 );
       }
    }
  }

  // Get tangent stiffness 

  Matrix jacStretchRinv ( 6, 6 );
  Matrix stiffRinv ( 6, 6);
  Matrix stiffRinvUinv ( 6, 6);
  Matrix jacRU ( 6,6);
 
  if ( perm23_ )
  {
    stiff = jacnn(perm_,perm_); 
  }
  else
  {
    // Condensing stiffness matrix from 9x9 from network (fd format) to 6x6 matrix
    // in the same coordinate system fed to the network
  
    if ( rotate )
    {
      // Get tangent stiffness in the rotated frame
 
      // Related to stretches

      Matrix dstressRdstress, Rtexp;
      Matrix jacStretch ( 9, 9 );
      Matrix jacRotation ( 9, 9 );
      Matrix dstressRdU ( 9, 9 ); 

      kronProd ( dstressRdstress, R, R );
      matmul ( dstressRdU, dstressRdstress, jacnn );
      m2mm ( Rtexp, R.transpose() );

      // Related to rotation

      Matrix dsigmaRdR, dRdF, dUdF;
      evaldsigmaRdR ( dsigmaRdR, stressTensorR, R );
      
      evaldRUdF ( dRdF, dUdF, R, U );

/*      System::out() << "dsigmaRdR " << dsigmaRdR << "\n";
      System::out() << "dRdF " << dRdF << "\n";
      System::out() << "dUdF " << dUdF << "\n";
*/
      // First term ( dsigmaRdU x dUdF  )

      Matrix aux1 ( 9,9);
      Matrix aux2 (9,9);
      matmul ( aux1, Rtexp, dstressRdU );
      matmul ( jacStretch, dUdF, dstressRdU );

      // Second term ( dsigmaRdF x dRdF ) 

      Matrix Uinv;
      m2mm ( Uinv, inverse ( U ) );
      Matrix jacRotationUinv ( 9, 9);
      matmul ( jacRotation, dRdF, dsigmaRdR );
      matmul ( jacRotationUinv, Uinv, dsigmaRdR );

      // Adding both contributions 

      Matrix jactotal ( 9, 9);
      jactotal = jacStretch + jacRotation;
      aux2 = aux1 + jacRotation;
      Matrix aux3 ( 9,9);
      aux3 = aux1 + jacRotationUinv;

     // System::out() << "ANNMaterial. Point " << ipoint << " Stiffness ( bef sym ): \n" << jac << "\n";
   
       condenseSymMat ( jacRU, jactotal );
       stiff = jacRU; 

    /*   condenseSymMat ( jacStretchRinv, aux1 );
       condenseSymMat ( stiffRinv, aux2 );
       condenseSymMat ( stiffRinvUinv, aux3 );*/
    }
    else
    {
      // dsigmaRdF 

      stiff = jacnn;
    }
  }
   
  // Checking the tangent stiffness matrix using FD
  
  bool checkFD = false;

  Matrix stiffFD ( ncomp*ncomp, ncomp*ncomp );
  Matrix stiffFDcond;
  
  if ( checkFD )
  {
    double stepFD = 1e-6;
    Matrix Ffd = F.clone();
    Vector stressf ( ncomp*ncomp );
    Vector stressb ( ncomp*ncomp );
    Matrix stressTensorFD ( ncomp, ncomp );
    Vector stressfrot ( ncomp*ncomp );
    Vector stressbrot ( ncomp*ncomp );
    Matrix Rfd, Ufd;
  
    for ( idx_t i = 0; i < 3; i++ )
    {
      for ( idx_t j = 0; j < 3; j++ )
      {
        Ffd(i,j) +=  stepFD; 
	polarDecomposition ( Rfd, Ufd, Ffd );

	for ( idx_t r = 0; r < ncomp; r++ )
	{
	    data_[ipoint]->inputs(slice(r*ncomp, (r+1)*ncomp),0) = Ufd(r, ALL); 
	}     
	
	model->takeAction ( LearningActions::PROPAGATE,   params, netData_ );
	stressf = nizer_->denormalize ( data_[ipoint]->outputs(ALL,0) );
        to2ndOrderTensor ( stressTensorFD, stressf );

	Matrix stressRf = matmul ( Rfd, matmul ( stressTensorFD, Rfd.transpose() ) );

	for ( idx_t ii = 0; ii < 3; ii++ )
	{
	  for ( idx_t jj = 0; jj < 3; jj++ )
	  {
	    stressfrot[ii*3+jj] = stressRf(ii,jj);
	  }
	}

	//System::out() << "Stress rotated + " << stressRf << "\n";

	Ffd(i,j) -=  2*stepFD; 
	polarDecomposition ( Rfd, Ufd, Ffd );
	
	for ( idx_t r = 0; r < ncomp; r++ )
	{
	    data_[ipoint]->inputs(slice(r*ncomp, (r+1)*ncomp),0) = Ufd(r, ALL); 
	}     
	
	model->takeAction ( LearningActions::PROPAGATE,   params, netData_ );
	stressb = nizer_->denormalize ( data_[ipoint]->outputs(ALL,0) );
	to2ndOrderTensor ( stressTensorFD, stressb );

	Matrix stressRb = matmul ( Rfd, matmul (stressTensorFD, Rfd.transpose()  ));

	for ( idx_t ii = 0; ii < 3; ii++ )
	{
	  for ( idx_t jj = 0; jj < 3; jj++ )
	  {
	    stressbrot[ii*3+jj] = stressRb(ii, jj);
	  }
	}

	//System::out() << "Stress rotated -" << stressRb << "\n";

	Ffd(i,j) +=  stepFD;
	stiffFD(ALL, i*3+j) = ( stressfrot - stressbrot ) / 2.0 / stepFD;
    }
  }
  
    condenseSymMat ( stiffFDcond, stiffFD );
    stiff = stiffFDcond;

/*	for ( idx_t r = 0; r < ncomp; r++ )
	{
	    data_[ipoint]->inputs(slice(r*ncomp, (r+1)*ncomp),0) = U(r, ALL); 
}
       model->takeAction ( LearningActions::PROPAGATE,   params, netData_ );
*/
  //System::out() << "FD stiffness " << stiffFD << "\n";
  }
  
  if ( ipoint <= 100 )
  {
/*    TensorIndex i,j;
    Matrix errorRinv (6,6);
    Matrix errorRUinv (6,6);
    Matrix errorRU (6,6);
  

    errorRinv(i,j) = 100 * ( 1. - jacStretchRinv (i,j ) / stiffFDcond ( i,j ) );
    errorRUinv(i,j) = 100 * ( 1. - stiffRinv (i,j ) / stiffFDcond ( i,j ) );
    errorRU(i,j) = 100 * ( 1. - stiff (i,j ) / stiffFDcond ( i,j ) );
*/
      //stiff = jacRU;
  //  System::out() << "ANNMaterial. Point " << ipoint << " with Rinv: \n" << jacStretchRinv << "\n";
      System::out() << "ANNMaterial. Point " << ipoint << " FD: \n" << stiffFDcond << "\n";
      System::out() << "ANNMaterial. Point " << ipoint << " RU: \n" << stiff << "\n";
   /* System::out() << "ANNMaterial. Point " << ipoint << " RinvUinv: \n" << stiffRinvUinv << "\n";
    System::out() << "ANNMaterial. Point " << ipoint << " FD: \n" << stiffFDcond << "\n";*/
    System::out() << "ANNMaterial. Point " << ipoint << " Stress:\n" << stress << "\n";
  }
}

//-----------------------------------------------------------------------
//  condenseSymMat
//-----------------------------------------------------------------------

void ANNMaterial::condenseSymMat
  
    ( Matrix& Asym, const Matrix& A )
{
    Asym.resize ( 6, 6 );
    Asym = 0.0;

    Asym(0,0) = A(0,0);
    Asym(0,1) = .5 * ( A(0,4) + A(4,0) );
    Asym(0,2) = .5 * ( A(0,8) + A(8,0) );
    Asym(0,3) = .5 * ( A(0,1) + A(0,3) );
    Asym(0,4) = .5 * ( A(0,5) + A(0,7) );
    Asym(0,5) = .5 * ( A(0,2) + A(0,6) );

    Asym(1,0) = Asym(0,1);
    Asym(1,1) = A(4,4);
    Asym(1,2) = .5 * ( A(4,8) + A(8,4) );
    Asym(1,3) = .5 * ( A(4,1) + A(4,3) );
    Asym(1,4) = .5 * ( A(4,5) + A(4,7) );
    Asym(1,5) = .5 * ( A(4,2) + A(4,6) );

    Asym(2,0) = Asym(0,2);
    Asym(2,1) = Asym(1,2);
    Asym(2,2) = A(8,8);
    Asym(2,3) = .5 * ( A(8,1) + A(8,3) );
    Asym(2,4) = .5 * ( A(8,5) + A(8,7) );
    Asym(2,5) = .5 * ( A(8,2) + A(8,6) );

    Asym(3,0) = Asym(0,3);
    Asym(3,1) = Asym(1,3);
    Asym(3,2) = Asym(2,3);
    Asym(3,3) = .5 * ( A(1,1) + A(1,3) );
    Asym(3,4) = .5 * ( A(1,5) + A(1,7) );
    Asym(3,5) = .5 * ( A(1,2) + A(1,6) );

    Asym(4,0) = Asym(0,4);
    Asym(4,1) = Asym(1,4);
    Asym(4,2) = Asym(2,4);
    Asym(4,3) = Asym(3,4);
    Asym(4,4) = .5 * ( A(5,5) + A(5,7) );
    Asym(4,5) = .5 * ( A(5,2) + A(5,6) );

    Asym(5,0) = Asym(0,5);
    Asym(5,1) = Asym(1,5);
    Asym(5,2) = Asym(2,5);
    Asym(5,3) = Asym(3,5);
    Asym(5,4) = Asym(4,5);
    Asym(5,5) = .5 * ( A(6,2) + A(6,6) );   
}

//-----------------------------------------------------------------------
//  to2ndOrderTensor 
//-----------------------------------------------------------------------

void ANNMaterial::to2ndOrderTensor
  
    ( Matrix& stressTensor, const Vector& stress )
{
  // Condenses 9D vector into a 3x3 symmetric tensor

  stressTensor ( 0,0 ) = stress[0];
  stressTensor ( 1,1 ) = stress[4];
  stressTensor ( 2,2 ) = stress[8];

  stressTensor ( 0,1 ) = stressTensor ( 1,0 ) = .5 * ( stress[1] + stress[3] );
  stressTensor ( 1,2 ) = stressTensor ( 2,1 ) = .5 * ( stress[5] + stress[7] );
  stressTensor ( 0,2 ) = stressTensor ( 2,0 ) = .5 * ( stress[2] + stress[6] );
}

//-----------------------------------------------------------------------
//  evaldsigmaRdR
//-----------------------------------------------------------------------

void ANNMaterial::evaldsigmaRdR
  
    ( Matrix& dstressRdR_constU, const Matrix& stress, const Matrix& R )
{
  dstressRdR_constU.resize ( 9, 9 );
  dstressRdR_constU = 0.0;

  Matrix Eij (3,3);
  Matrix Eji (3,3);
  Matrix sum (9,9);

  Matrix U, Ubar, firstTerm, secondTerm;
  Matrix idMatrix (3,3);
  idMatrix = 0.;
  idMatrix (0,0) = idMatrix(1,1) = idMatrix(2,2) = 1.;

  for ( idx_t i = 0; i < 3; i++ )
  {
    for ( idx_t j = 0; j < 3; j++ )
    {
      Eij = 0.;
      Eij(i,j) = 1.;
      Matrix Eji = Eij.transpose();
      
      kronProd (U, Eij, Eji);      
      kronProd (Ubar, Eij, Eij);      
      kronProd (firstTerm, idMatrix, matmul (stress, R.transpose())) ;
      kronProd (secondTerm, idMatrix, matmul ( R, stress ) );
 
      sum = matmul ( Ubar, firstTerm ) + matmul ( secondTerm, U);

  //    System::out() << "i " << i << "j " << j << " sum " << sum << "\n";
      dstressRdR_constU(slice(0, 3), i*3+j ) = sum ( i*3, slice(j*3, (j+1)*3));
      dstressRdR_constU(slice(3, 6), i*3+j ) = sum ( i*3+1, slice(j*3, (j+1)*3));
      dstressRdR_constU(slice(6, 9), i*3+j ) = sum ( i*3+2, slice(j*3, (j+1)*3));
    } 
  }
}


//-----------------------------------------------------------------------
//  evaldRUdF
//-----------------------------------------------------------------------

void ANNMaterial::evaldRUdF
  

    ( Matrix& dRdF, Matrix& dUdF,
    		const Matrix& R, const Matrix& U )
{
  dRdF.resize ( 9, 9 );
  dUdF.resize ( 9, 9 );
  dRdF = 0.0;
  dUdF = 0.0;

  Matrix dRdFij ( 3,3 );
  Matrix dUdFij ( 3,3 );

  double trU = U(0,0) + U(1,1) + U(2,2);

  Matrix dUdFt ( 9, 9 );

  TensorIndex i,j;
  Matrix idMatrix (3,3);
  idMatrix(i,j) = where ( i == j, 1., 0. );

  Matrix trUMatrix (3,3);
  trUMatrix(i,j) = where ( i == j, trU, 0. );

  for ( idx_t i = 0; i < 3; i++ )
  {
    for ( idx_t j = 0; j < 3; j++ )
    {
      Matrix L (3,3);
      L = 0.0;
      L(i, j) = 1.;
     
      Matrix Lt = L.transpose();
  
      Matrix Y (3, 3);
      Y = trUMatrix - U;
      double detY = determinant ( Y );
      double detYinv = 1./detY;
  
      Matrix Rt = R.transpose();
      Matrix RY = matmul ( R, Y);
      Matrix YRt = matmul ( Y, Rt);
      Matrix RtL = matmul ( Rt, L);
      Matrix LtR = matmul ( Lt, R);
      Matrix RL = matmul ( R, L);
      Matrix LU = matmul ( L, U);
      Matrix ULt = matmul ( U, Lt);
      Matrix YU = matmul ( Y, U );
     
      // Second check
      Matrix RtLmLtR ( 3, 3 );
      RtLmLtR = RtL - LtR;

      Matrix dUdFterm = matmul ( Y, matmul ( RtLmLtR, YU ) );
      dUdFterm(i,j) = detYinv * dUdFterm (i,j); 

      Matrix dRdFterm = matmul ( R, matmul ( Y, matmul( RtLmLtR, Y) ) );
      dRdFterm(i,j) = detYinv * dRdFterm (i,j); 

      dUdFij = RtL - dUdFterm;
      dRdFij = dRdFterm;

      dRdF ( slice(0, 3), i*3+j ) = dRdFij ( 0, ALL ); 
      dRdF ( slice(3, 6), i*3+j ) = dRdFij ( 1, ALL ); 
      dRdF ( slice(6, 9), i*3+j ) = dRdFij ( 2, ALL ); 
      dUdF ( slice(0, 3), i*3+j ) = dUdFij ( 0, ALL ); 
      dUdF ( slice(3, 6), i*3+j ) = dUdFij ( 1, ALL ); 
      dUdF ( slice(6, 9), i*3+j ) = dUdFij ( 2, ALL ); 

    }
  }
}

//-----------------------------------------------------------------------
//  m2mm
//-----------------------------------------------------------------------

void ANNMaterial::m2mm
  
    ( Matrix& mm, const Matrix& m )
{
      mm.resize ( 9, 9 );
      mm = 0.0;

      mm(0,0) = m(0,0);
      mm(0,4) = m(0,1);
      mm(0,7) = m(0,2);
      mm(1,1) = m(1,1);
      mm(1,3) = m(1,0);
      mm(1,6) = m(1,2);
      mm(2,2) = m(2,2);
      mm(2,5) = m(2,1);
      mm(2,8) = m(2,0);
      mm(3,1) = m(0,1);
      mm(3,3) = m(0,0);
      mm(3,6) = m(0,2);
      mm(4,0) = m(1,0);
      mm(4,4) = m(1,1);
      mm(4,7) = m(1,2);
      mm(5,2) = m(1,2);
      mm(5,5) = m(1,1);
      mm(5,8) = m(1,0);
      mm(6,1) = m(2,1);
      mm(6,3) = m(2,0);
      mm(6,6) = m(2,2);
      mm(7,0) = m(2,0);
      mm(7,4) = m(2,1);
      mm(7,7) = m(2,2);
      mm(8,2) = m(0,2);
      mm(8,5) = m(0,1);
      mm(8,8) = m(0,0);
}


//-----------------------------------------------------------------------
//  mm2mmr
//-----------------------------------------------------------------------

void ANNMaterial::mm2mmr
  
    ( Matrix& mmr, const Matrix& mm )
{
  for (int i = 0; i < 9; i++)
  {
     mmr(3,i) = mm(4,i);
     mmr(4,i) = mm(3,i);
     mmr(5,i) = mm(6,i);
     mmr(6,i) = mm(5,i);
     mmr(7,i) = mm(8,i);
     mmr(8,i) = mm(7,i);
  }
}

//-----------------------------------------------------------------------
//  updateWriteTable
//-----------------------------------------------------------------------

void ANNMaterial::updateWriteTable

    ( const Vector&         stress,
      const Matrix&         stiff,
      const Matrix&         F,
      idx_t                 ipoint )

{
  String context ( "ANNMaterial::updateWriteTable" );

  Properties params;

  idx_t ncomp = F.size(0);

  bool local = false; 
  bool rotate = true;

  Matrix stressTensor ( ncomp, ncomp );
  stiff = 0.0;

  Ref<Model> model = Model::get ( netData_, context );

  // Get R and U tensors from polar decomposition 

  Matrix R, U;

//  System::out() << "\nANNMaterial. updateWriteTable. Point " << ipoint << "\n"; 
//  System::out() << "Performing polar decomposition on F\n" << F << "\n";

  polarDecomposition ( R, U, F );

  Ref<PrintWriter> out = newInstance<PrintWriter> ( &System::out() );
  out->nformat.setFractionDigits( 10 );

 /* *out << "Deformation tensor\n" << F << "\n";
  *out << "Rotation tensor\n" << R << "\n";
  *out << "Strech tensor\n" << U << "\n";
*/
  if ( local )
  {
     // Transforming 9D vector in the frame x, y and z to local frame 1, 2 and 3
 
     Matrix auxF ( 4, 4 );
     auxF = 0.0;
     Matrix Fcopy = F.clone();

     System::out() << "F in x, y, z system: " << F << "\n";

     auxF ( slice(1, 4), slice(1, 4) ) = Fcopy;
     auxF ( 0, slice(1, 3) ) = Fcopy ( 2, slice(0, 2) );
     auxF ( slice (1, 3), 0 ) = Fcopy ( slice(0,2), 2 );

     auxF (0,0) = Fcopy (2, 2);  
     Matrix Flocal = auxF ( slice (0, 3), slice (0, 3) );

//     System::out() << "auxF " << auxF << "\n";
     System::out() << "F in local 1, 2, 3 system: " << Flocal << "\n"; 
     for ( idx_t i = 0; i < ncomp; i++ )
     {
       data_[ipoint]->inputs(slice(i*ncomp, (i+1)*ncomp),0) = Flocal(i, ALL); 
     }

  }
  else
  {
     if ( rotate )
     {
       for ( idx_t i = 0; i < ncomp; i++ )
       {
         data_[ipoint]->inputs(slice(i*ncomp, (i+1)*ncomp),0) = U(i, ALL);
       }
     }
     else
     {
       for ( idx_t i = 0; i < ncomp; i++ )
       {
         data_[ipoint]->inputs(slice(i*ncomp, (i+1)*ncomp),0) = F(i, ALL); 
       }
     }
  }

  params.set ( LearningParams::DATA, data_[ipoint] );
  params.set ( "dtglobal", dt_ );

  if ( recurrent_ && !first_ )
  {
    params.set ( LearningParams::STATE, state_[ipoint] );
  }

  params.set ( "offline", true );

  model->takeAction ( LearningActions::PROPAGATE,   params, netData_ );
 
  if ( perm23_ )
  {
    stress = data_[ipoint]->outputs(ALL,0)[perm_];
  }
  else
  {

    if ( local )
    {
      // Condensing 9D vector in the local frame 1, 2 and 3 to 6D vector in the frame
      // x, y and z (where z is the fiber direction)
      // 0 -> z   1-> x   2-> y

      stress[0] = data_[ipoint]->outputs(4,0); 
      stress[1] = data_[ipoint]->outputs(8,0);
      stress[2] = data_[ipoint]->outputs(0,0);
      stress[3] = .5 * (data_[ipoint]->outputs(5, 0) + data_[ipoint]->outputs(7,0));
      stress[4] = .5 * (data_[ipoint]->outputs(6, 0) + data_[ipoint]->outputs(2,0));
      stress[5] = .5 * (data_[ipoint]->outputs(3, 0) + data_[ipoint]->outputs(1,0)); 
    }
    else
    { 

      // 9D vector is in the global frame x, y and z
      
      Vector stressnn =  nizer_->denormalize ( data_[ipoint]->outputs(ALL,0) );

      // Transforming 9D vector to 3x3 symmetric tensor


      to2ndOrderTensor ( stressTensor, stressnn );
      
      //System::out() << "ANNMaterial. Stresses in global frame:\n" << stressTensor << "\n";

       // Selecting relevant components 

       if ( rotate )
       {
          // Get stresses back to rotated system

          Matrix stressRt ( ncomp, ncomp );
	  Matrix RstressRt ( ncomp, ncomp );

	  matmul ( stressRt, stressTensor, R.transpose() );
	  matmul ( RstressRt, R, stressRt );

	  stress[0] = RstressRt ( 0, 0 );
	  stress[1] = RstressRt ( 1, 1 );
	  stress[2] = RstressRt ( 2, 2 );
	
	  stress[3] = RstressRt ( 0, 1 );
	  stress[4] = RstressRt ( 1, 2 );
	  stress[5] = RstressRt ( 0, 2 );
       }
       else
       {
	 stress[0] = stressTensor ( 0, 0 );
	 stress[1] = stressTensor ( 1, 1 );
	 stress[2] = stressTensor ( 2, 2 );
	
	 stress[3] = stressTensor ( 0, 1 );
	 stress[4] = stressTensor ( 1, 2 );
	 stress[5] = stressTensor ( 0, 2 );
       }
    }
  }
  
  // No need to calculate stiffness
}

//-----------------------------------------------------------------------
//  commit
//-----------------------------------------------------------------------

void ANNMaterial::commit ()

{
  if ( recurrent_ )
  {
    data_.swap ( state_ );
  }
  
  first_ = false;
}

//-----------------------------------------------------------------------
//  stressAtPoint 
//-----------------------------------------------------------------------

void ANNMaterial::stressAtPoint

  ( Vector&       stress,
    const Vector& strain,
    const idx_t   ipoint )
{
}

//-----------------------------------------------------------------------
//  clone 
//-----------------------------------------------------------------------

Ref<Material> ANNMaterial::clone ( ) const

{
  return newInstance<ANNMaterial> ( *this );
}

//-----------------------------------------------------------------------
//  addTableColumns 
//-----------------------------------------------------------------------

void ANNMaterial::addTableColumns

  ( IdxVector&     jcols,
    XTable&        table,
    const String&  name )

{
  if ( name == "nodalStress" || name == "ipStress" )
  {
    if ( rank_ == 2 )
    {
      jcols.resize ( 3 );

      jcols[0] = table.addColumn ( "s_xx" );
      jcols[1] = table.addColumn ( "s_yy" );
      jcols[2] = table.addColumn ( "s_xy" );
    }
    else
    {
      jcols.resize ( 6 );

      jcols[0] = table.addColumn ( "s_xx" );
      jcols[1] = table.addColumn ( "s_yy" );
      jcols[2] = table.addColumn ( "s_zz" );
      jcols[3] = table.addColumn ( "s_xy" );
      jcols[4] = table.addColumn ( "s_xz" );
      jcols[5] = table.addColumn ( "s_yz" );
    }
  }
}

//-----------------------------------------------------------------------
//  initWeights_
//-----------------------------------------------------------------------

void ANNMaterial::initWeights_

  ( const String&     fname )

{
  String context ( "ANNMaterial::initWeights_" );

  Ref<DofSpace> dofs  = DofSpace::get ( netData_, context );
  Ref<Model>    model = Model::   get ( netData_, context );

  Ref<FileReader> in    = newInstance<FileReader> ( fname );

  idx_t dc = dofs->dofCount();

  Vector wts ( dc );
  wts = 0.0;

  for ( idx_t i = 0; i < dc; ++i )
  {
    wts[i] = in->parseDouble();
  }

  System::out() << "Reading weights from file " << fname << "\n";

  StateVector::store ( wts, dofs, netData_ );

  model->takeAction ( LearningActions::UPDATE, Properties(), netData_ );
}

/* Purgatory

  //stiff  = data_[ipoint]->jacobian;

  //for ( idx_t row = 0; row < fac.size(); ++row )
  //{
  //  stiff(row,ALL) *= fac;
  //}

  // DBG: some tests

  //Vector tstrain ( 6 );

  //tstrain[0] = -0.032077;
  //tstrain[1] = 0.080439;
  //tstrain[2] = -0.019818;
  //tstrain[3] = 1.1599e-07;
  //tstrain[4] = 6.7652e-08;
  //tstrain[5] = 0.00038062;

  //data_[ipoint]->inputs(ALL,0) = nizer_->normalize ( tstrain );

  //model->takeAction ( NeuralActions::PROPAGATE,   params, netData_ );
  //model->takeAction ( NeuralActions::GETJACOBIAN, params, netData_ );

  //System::out() << "Test stress " << data_[ipoint]->outputs(ALL,0) << "\n";
  //System::out() << "Test jacobian " << data_[ipoint]->jacobian << "\n";


  // DEBUG: tangent check with finite diffs

  //idx_t icol = 5;

  //Vector strainp ( strain.clone() );

  //strainp[icol] += 1.e-10;

  //data_[ipoint]->inputs(ALL,0) = nizer_->normalize ( strainp );
  //model->takeAction ( NeuralActions::PROPAGATE, params, netData_ );

  //Vector stressp ( stress.shape() );
  //stressp = 0.;

  //stressp = data_[ipoint]->outputs(ALL,0);

  //Vector col ( 6 );
  //col = 0.0;

  //col = ( stressp - stress ) / 1.e-10;
  //System::out() << "Col " << col << "\n";
  //System::out() << "Jac col " << jac(ALL,icol) << "\n";

  //

  //stiff = 0.5 * ( jac + jac.transpose() );

  //if ( ipoint == 0 ) System::out() << "Jac shape " << data_[ipoint]->jacobian.shape() << "\n";
  //if ( ipoint == 0 ) System::out() << "Fac shape " << fac.shape() << "\n";
  //if ( ipoint == 0 ) System::out() << "Point " << ipoint << " fac " << fac << "\n";
  //if ( ipoint == 0 ) System::out() << "Point " << ipoint << " strain " << strain << "\n";
  //if ( ipoint == 0 ) System::out() << "Stiff " << stiff << "\n";
  //if ( ipoint == 0 ) System::out() << "Point " << ipoint << " stress " << stress << "\n";

*/
