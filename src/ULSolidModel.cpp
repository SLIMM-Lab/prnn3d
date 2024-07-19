/*
 * 
 *  Copyright (C) 2019 TU Delft. All rights reserved.
 *  
 *  This class implements a simple model for updated lagrangian 
 *  finite element analysis 
 *
 *  It passes only the incremental deformation gradient to the 
 *  constitutive law
 *
 *  Reference: Zienkiewicz, Taylor and Fox
 *  'The Finite Element Method for Solid and Structural Mechanics'
 *  7th edition, sections 5.3.3 and 6.5
 *
 *  Author:  F.P. van der Meer, F.P.vanderMeer@tudelft.nl
 *  Date:    March 2019
 *
 */

#include <jem/base/array/operators.h>
#include <jem/base/array/select.h>
#include <jem/base/Error.h>
#include <jem/base/IllegalInputException.h>
#include <jem/io/Writer.h>
#include <jem/numeric/algebra/MatmulChain.h>
#include <jem/numeric/algebra/matmul.h>
#include <jem/numeric/algebra/utilities.h>
#include <jem/util/StringUtils.h>
#include <jem/io/FileWriter.h>

#include <jive/geom/Geometries.h>
#include <jive/geom/IShapeFactory.h>
#include <jive/geom/Quad.h>
#include <jive/model/ModelFactory.h>
#include <jive/model/Actions.h>
#include <jive/model/StateVector.h>
#include <jive/util/utilities.h>
#include <jem/util/Timer.h>

#include "LargeDispUtilities.h"
#include "ULSolidModel.h"
#include "TbFiller.h"

#include <jem/numeric/algebra/LUSolver.h>
#include <jem/util/StringUtils.h>

using namespace jem;
using jem::io::endl;
using jem::io::Writer;
using jem::io::FileWriter;

using jem::util::Timer;

using jem::numeric::matmul;
using jem::numeric::MatmulChain;

using jive::geom::Geometries;
using jive::geom::Quad4;
using jive::model::StateVector;

using jem::numeric::norm2;
using jem::numeric::LUSolver;
using jem::numeric::invert;

using jem::util::StringUtils;

typedef MatmulChain<double,3>   MChain3;
typedef MatmulChain<double,2>   MChain2;
typedef MatmulChain<double,1>   MChain1;

//======================================================================
//   definition
//======================================================================

const char* ULSolidModel::DOF_NAMES[3]     = {"dx","dy","dz"};
const char* ULSolidModel::SHAPE_PROP       = "shape";
const char* ULSolidModel::MATERIAL_PROP    = "material";
const char* ULSolidModel::THICK_PROP       = "thickness";
const char* ULSolidModel::STATE_PROP       = "state";

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------

ULSolidModel::ULSolidModel

   ( const String&       name,
     const Properties&   conf,
     const Properties&   props,
     const Properties&   globdat ) : Super(name)
{
  using jive::util::joinNames;
  using jem::util::StringUtils;

  // create myTag_ (last part of myName_)
  
  StringVector names ( StringUtils::split( myName_, '.' ) );
  myTag_     = names [ names.size() - 1 ];

  Properties  myProps = props.getProps ( myName_ );
  Properties  myConf  = conf.makeProps ( myName_ );

  myProps.find( stateString_, STATE_PROP );
  myConf .set ( STATE_PROP, stateString_ );

  homogenize_ = false;
  myProps.find ( homogenize_, "homogenize" );
  myConf .set ( "homogenize", homogenize_ ); 

  const String context = getContext();

  egroup_ = ElemGroup::get ( myConf, myProps, globdat, context );

  numElem_   = egroup_.size();
  ielems_    . resize( numElem_ );
  ielems_    = egroup_.getIndices ();
  elems_     = egroup_.getElements ( );
  nodes_     = elems_.getNodes     ( );
  rank_      = nodes_.rank         ( );
  numNode_   = nodes_.size         ( );
  strCount_  = STRAIN_COUNTS[rank_];

  // Make sure that the number of spatial dimensions (the rank of the
  // mesh) is valid.

  System::out() << "ULSolid. rank " << rank_ << "\n";

  if ( rank_ < 1 || rank_ > 3 )
  {
    throw IllegalInputException (
      context,
      String::format (
        "invalid node rank: %d (should be 1, 2 or 3)", rank_    
      )
    );
  }

  String shapeProp = joinNames ( myName_, SHAPE_PROP );


  if ( stateString_ == "PLANE_STRAIN" )
  {
    shape_ = jive::geom::IShapeFactory::newInstance ( 
    joinNames ( myName_, SHAPE_PROP ), conf, props );

    strCount_ = 3;
  }

  else if ( stateString_ == "AXISYMMETRIC" )
  {
    shape_ = jive::geom::IShapeFactory::newInstance ( 
    joinNames ( myName_, SHAPE_PROP ), conf, props );

    strCount_ = 4;
  }

  else if ( stateString_ == "3D" )
  {
    shape_ = jive::geom::IShapeFactory::newInstance ( 
    joinNames ( myName_, SHAPE_PROP ), conf, props );
  }

  else
  {
    throw IllegalInputException (
      context,
      String::format (
        "unknown state problem %s",
        stateString_
      )
    );
  }

  nodeCount_  = shape_->nodeCount   ();
  ipCount_    = shape_->ipointCount ();
  dofCount_   = rank_ * nodeCount_;

  // Make sure that the rank of the shape matches the rank of the
  // mesh.

  if ( shape_->globalRank() != rank_ )
  {
    throw IllegalInputException (
      context,
      String::format (
        "shape has invalid rank: %d (should be %d)",
        shape_->globalRank (),
        rank_
      )
    );
  }

  // Make sure that each element has the same number of nodes as the
  // shape object.

  elems_.checkSomeElements (
    context,
    ielems_,
    shape_->nodeCount  ()
  );

  dofs_ = XDofSpace::get ( nodes_.getData(), globdat );
  
  dofTypes_.resize( rank_ );

  for( idx_t i = 0; i < rank_; i++)
  {
    dofTypes_[i] = dofs_->addType ( DOF_NAMES[i]);
  }

  dofs_->addDofs (
    elems_.getUniqueNodesOf ( ielems_ ),
    dofTypes_
  );

  // Compute the total number of integration points.

  idx_t  ipCount = shape_->ipointCount() * egroup_.size();

  // Create a material model object.

  material_ = newMaterial ( MATERIAL_PROP, myConf, myProps, globdat );

  Properties  matProps = myProps.findProps ( MATERIAL_PROP );

  // configure already here, so that allocPoints can use data from an input file

  material_->configure   ( matProps );
  material_->allocPoints ( ipCount );

  getShapeGrads_ = getShapeGradsFunc ( rank_ );

  thickness_ = 1.;

  // get name of material

  matProps.get ( matType_, "type" ); 

  // if ( rank_ == 2 )
  // {
  //   myProps.find( thickness_, THICK_PROP );
  //   myConf.set  ( THICK_PROP, thickness_ );
  // }
}

ULSolidModel::~ULSolidModel()
{}

//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------

void ULSolidModel::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  Properties  myProps  = props.findProps ( myName_ );
  Properties  matProps = myProps.findProps ( MATERIAL_PROP );

  material_->configure ( matProps );

  Properties  params;
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void ULSolidModel::getConfig 

  ( const Properties& conf,
    const Properties& globdat ) const

{
  Properties  myConf  = conf.makeProps ( myName_ );
  Properties  matConf = myConf.makeProps ( MATERIAL_PROP );

  material_->getConfig ( matConf );
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool ULSolidModel::takeAction

  ( const String&      action,
    const Properties&  params,
    const Properties&  globdat )

{
  using jive::model::Actions;
  using jive::model::ActionParams;

 // System::out () << "@ULSolidModel::takeAction(), action... " << action << "\n";

  if ( action == Actions::GET_MATRIX0 
    || action == Actions::GET_INT_VECTOR )
  {
    Ref<MatrixBuilder>  mbuilder;

    Vector  disp0;
    Vector  disp1;
    Vector  force;

    // Get the current displacements.

    StateVector::getOld ( disp0, dofs_, globdat );
    StateVector::get    ( disp1, dofs_, globdat );

    // Get the matrix builder and the internal force vector.

    params.find( mbuilder, ActionParams::MATRIX0 );
    params.get ( force,    ActionParams::INT_VECTOR );

    getMatrix_ ( mbuilder, force, disp0, disp1 );

    return true;
  }

  if ( action == "GET_DISSIPATION" )
  {
    getDissipation_ ( params );

    return false;
  }

  if ( action == Actions::COMMIT )
  {
    material_->commit ();

    return true;
  }
  
  if ( action == SolverNames::SET_STEP_SIZE )
  {
    double             dt;
    params.get       ( dt, SolverNames::STEP_SIZE );

 //   System::out() << "Setting " << SolverNames::STEP_SIZE << " in ULSolid model to " << dt << "\n";

    material_->setDT ( dt );
    return true;
  }

  if ( action == Actions::GET_TABLE )
  {
    return getTable_ ( params, globdat );
  }

  if ( action == "WRITE_XOUTPUT" )
  {
  //  if ( dw_.amFirstWriter ( this ) )
   //
      writeDisplacements_ ( params, globdat );
   // }
    // writeCohState_ ( globdat );

    return true;
  }

  return false;
}


//-----------------------------------------------------------------------
//   getMatrix_
//-----------------------------------------------------------------------


void ULSolidModel::getMatrix_

  ( Ref<MatrixBuilder>  mbuilder,
    const Vector&       force,
    const Vector&       disp0,
    const Vector&       disp1 ) const

{
  Writer&     dbgOut     ( System::debug ( "ul" ) );

  Cubix       grads0      ( rank_, nodeCount_, ipCount_ );
  Cubix       gradsuu     ( rank_, nodeCount_, ipCount_ );
  Cubix       grads     ( rank_, nodeCount_, ipCount_ );
  Matrix      stiff      ( strCount_, strCount_ );

  Matrix      coords0    ( rank_, nodeCount_ );  // reference 
  Matrix      coordsU    ( rank_, nodeCount_ );  // updated
  Matrix      coordsUU   ( rank_, nodeCount_ );


  Matrix      elemMat    ( dofCount_, dofCount_  );
  Vector      elemForce  ( dofCount_ );
  Vector      elemDisp0  ( dofCount_ );
  Vector      elemDisp1  ( dofCount_ );
  Vector      elemDispD  ( dofCount_ );

  Vector      stress     ( strCount_ );
  Matrix      df         ( 3, 3 );
  Matrix      F          ( 3, 3 );
  Matrix      b          ( strCount_, dofCount_  );
  Matrix      bt         = b.transpose ();

  IdxVector   inodes     ( nodeCount_ );
  IdxVector   idofs      ( dofCount_  );

  Vector      ipWeights  ( ipCount_   );
  Vector      dummy      ( ipCount_   );

  MChain1     mc1;
  MChain3     mc3;

  double      r = 1.; // needed for axisymmetric element

  idx_t       ipoint = 0;

  // Iterate over all elements assigned to this model.

//  System::out() << "getMatrix. numElem: " << numElem_ << "\n";

  for ( idx_t ie = 0; ie < numElem_; ie++ )
  {

    // Get the global element index.

    idx_t  ielem = ielems_[ie];

    // Get the element coordinates and DOFs.

    elems_.getElemNodes  ( inodes, ielem );
    nodes_.getSomeCoords ( coords0, inodes );
    dofs_->getDofIndices ( idofs,  inodes, dofTypes_ );

    // Get the displacements at the element nodes.

    elemDisp0 = select ( disp0, idofs );

    elemDisp1 = select ( disp1, idofs );

//    System::out() << "getMatrix. idofs " << idofs << "\n";

    // Get the displacement increment 

    elemDispD = elemDisp1 - elemDisp0;

    // Compute updated coordinates with previous converged disp

    updateCoords_ ( coordsUU, coords0, elemDisp0 );

    updateCoords_ ( coordsU, coords0, elemDisp1 ); 

    // Compute the spatial derivatives of the element shape
    // functions.

    shape_->getShapeGradients ( grads0, dummy, coords0 );       // NB: Undeformed
    shape_->getShapeGradients ( gradsuu, dummy, coordsUU );     // Reference ( start of load step )
    shape_->getShapeGradients ( grads, ipWeights, coordsU );    // Updated ( end of load step )

    // for 2D: multiply ipWeights with thickness

    ipWeights *= thickness_;

    // Assemble the element matrix.

    elemMat   = 0.0;
    elemForce = 0.0;
    
    Ref<PrintWriter> out = newInstance<PrintWriter> ( &System::out() );
    out->nformat.setFractionDigits( 8 );

    for ( idx_t ip = 0; ip < ipCount_; ip++ )
    {     
      
      // Compute the B-matrix for this integration point.
      // compute the deformation gradient increment f_{ij}^{(n)}


      if ( stateString_ == "AXISYMMETRIC" )
      {
        Matrix ipCoords   ( rank_, ipCount_ );

        Matrix sfuncs     = shape_->getShapeFunctions();

        shape_->getGlobalIntegrationPoints ( ipCoords, coordsU );

        r = ipCoords(1,ip);

        getBMatrix_ ( b, grads(ALL,ALL,ip), sfuncs(ALL,ip), r );

        axisymDeformationGradient ( df, elemDispD, grads(ALL,ALL,ip), sfuncs(ALL,ip), r );
      }

      else if ( stateString_ == "PLANE_STRAIN" )
      {
        getShapeGrads_ ( b, grads(ALL,ALL,ip) );

        planeStrainDeformationGradient ( df, elemDispD, grads(ALL,ALL,ip) );
      }

      else
      {
        getShapeGrads_ ( b, grads(ALL,ALL,ip) );

        Matrix ipCoords   ( rank_, ipCount_ );

   //     shape_->getGlobalIntegrationPoints ( ipCoords, coordsU );

        if ( matType_ == "Neural" )
	{
	  evalDeformationGradient ( df, elemDisp1, grads0(ALL,ALL,ip) );
	}
	else
	{
/*	  Matrix df0 ( 3,3) ; Matrix df1 ( 3,3 );
	  evalDeformationGradient ( df0, elemDisp0, grads0(ALL, ALL, ip) );
	  evalDeformationGradient ( df1, elemDisp1, grads0(ALL, ALL, ip) );
	  invert ( df0 );
	  Matrix dfundeformed ( 3, 3);
	  matmul ( dfundeformed, df1, df0 );
	  *out << "getMatrix Fprev inverted " << df0 << "\n";
	  *out << "getMatrix F current " << df1 << "\n";
	  *out << "getMatrix dF using Fcurrent * inv ( Fprev ) " << dfundeformed << "\n";*/
	  evalDeformationGradient ( df, elemDispD, gradsuu(ALL,ALL,ip) );
	}

	/**out << "getMatrix. elemDisp0 " << elemDisp0 << "\n";
	*out << "getMatrix. elemDisp1 " << elemDisp1 << "\n";
	*out << "getMatrix. elemD " << elemDispD << "\n";
	System::out() << "getMatrix. df " << df << "\n";*/
      }

      // Get the tangent stiffness matrix and the stress vector
      // from the Material given the deformation gradient increment

      material_->update ( stress, stiff, df, ipoint++ );

     //  *out << "getMatrix. elemDisp1 " << elemDisp1 << "\n";
     //  *out << "getMatrix. stiffMat " << stiff << "\n";

      // Compute the element force vector and the stiffness matrix

      if ( stateString_ == "AXISYMMETRIC" )
      {
        elemForce += r * ipWeights[ip] * mc1.matmul ( bt, stress );
        
        elemMat   += r * ipWeights[ip] * mc3.matmul ( bt, stiff, b );

        // Add element stiffness matrix large displacements

        addAxsymElemMatLargeD ( elemMat, stress, grads(ALL,ALL,ip), ipWeights[ip], r);
      }
      else
      {
        elemForce += ipWeights[ip] * mc1.matmul ( bt, stress );
        
        elemMat   += ipWeights[ip] * mc3.matmul ( bt, stiff, b );

        addElemMatLargeD ( elemMat, stress, grads(ALL,ALL,ip), ipWeights[ip]);
      }
    }

    // Add the element matrix to the global stiffness matrix.

    if ( mbuilder != nullptr )
    {
      mbuilder->addBlock ( idofs, idofs, elemMat );
    }

    // Add the element force vector to the global force vector.

    select ( force, idofs ) += elemForce;

    // System::out() << endl;
    // System::out() << "elem : " << ie << endl;
  }
}

//-----------------------------------------------------------------------
//   updateCoords_
//-----------------------------------------------------------------------

void ULSolidModel::updateCoords_ 

  ( const Matrix&  x, 
    const Matrix&  X, 
    const Vector&  U ) const

{
  idx_t k = 0;
  for ( idx_t j = 0; j < nodeCount_; ++j )
  {
    for ( idx_t i = 0; i < rank_; ++i )
    {
      x(i,j) = X(i,j) + U[k++];
    }
  }
}

//-----------------------------------------------------------------------
//   getDissipation_
//-----------------------------------------------------------------------

void ULSolidModel::getDissipation_

  ( const Properties&  params )

{
  const idx_t  nodeCount  = shape_->nodeCount   ();
  const idx_t  ipCount    = shape_->ipointCount ();
  const idx_t  ielemCount = ielems_.size        ();

  IdxVector    inodes    (        nodeCount );
  Matrix       coords    ( rank_, nodeCount );
  Vector       ipWeights (          ipCount );

  idx_t  ipoint      = 0;
  double dissipation = 0.;

  // bulk damage

  for ( idx_t ie = 0; ie < ielemCount; ++ie )
  {
    idx_t ielem = ielems_[ie];

    elems_.getElemNodes  ( inodes, ielem   );
    nodes_.getSomeCoords ( coords, inodes );

    // get the correct shape and then the number of idx_t points

    shape_->getIntegrationWeights ( ipWeights, coords );

    for ( idx_t ip = 0; ip < ipCount; ++ip )
    {
      dissipation += ipWeights[ip] * material_->giveDissipation ( ipoint++ );
    }
  }
  params.set ( myTag_, dissipation );
}

//-----------------------------------------------------------------------
//   getTable_
//-----------------------------------------------------------------------


bool ULSolidModel::getTable_

  ( const Properties&  params,
    const Properties&  globdat )

{
  using jive::model::Actions;
  using jive::model::ActionParams;
  using jive::model::StateVector;

  String       contents;
  Ref<XTable>  table;
  Vector       weights;
  String       name;

  Vector       disp0;
  Vector       disp1;

  StateVector::getOld ( disp0, dofs_, globdat );
  StateVector::get ( disp1, dofs_, globdat );

  // Get the table, the name of the table, and the table row weights
  // from the action parameters.

  params.get ( table,   ActionParams::TABLE );
  params.get ( name,    ActionParams::TABLE_NAME );
  params.get ( weights, ActionParams::TABLE_WEIGHTS );

  // Stress value are computed in the nodes.

//System::out() << "getTable ULsolid. ncolumns table " << table->columnCount() << "\n";

  if ( name == "stress" &&
       table->getRowItems() == nodes_.getData() )
  {
//    System::out() << "getStress!!\n";
    getStress_ ( *table, weights, disp0, disp1 );

    return true;
  }
  else if ( name == "xoutTable" )
  {
    globdat.get ( contents, TbFiller::TABLE_FILTER );

 // System::out() << "contents " << contents << "\n";
    
    getXOutTable_ ( table, weights, contents, disp0, disp1 );

    return true;
  }
  return false;
}

//-----------------------------------------------------------------------
//   getStress_
//-----------------------------------------------------------------------


void ULSolidModel::getStress_

  ( XTable&        table,
    const Vector&  weights,
    const Vector&  disp0,
    const Vector&  disp1 )

{
  IdxVector  ielems     = egroup_.getIndices  ();

  Cubix      grads0      ( rank_, nodeCount_, ipCount_ );
  Cubix      gradsuu     ( rank_, nodeCount_, ipCount_ );
  Cubix      grads      ( rank_, nodeCount_, ipCount_ );
  Matrix     ndStress   ( nodeCount_, strCount_ );  // nodal stress
  Vector     ndWeights  ( nodeCount_ );
  Matrix     stiff      ( strCount_,  strCount_ );

  Matrix     coords0    ( rank_,     nodeCount_ );
  Matrix     coordsU    ( rank_,     nodeCount_ );
  Matrix     coordsUU   ( rank_,     nodeCount_ );
  Matrix     b          ( strCount_, dofCount_  );

  Vector     stress     ( strCount_ );
  Matrix     df         ( 3, 3 );
  Vector     elemDisp0  ( dofCount_ );
  Vector     elemDisp1  ( dofCount_ );
  Vector     elemDispD  ( dofCount_ );

  IdxVector  inodes     ( nodeCount_ );
  IdxVector  idofs      ( dofCount_  );
  IdxVector  jcols      ( strCount_ + 1 );

  double     r = 1.; // needed for axisymmetric element

  // Add the columns for the stress components to the table.

  switch ( strCount_ )
  {
  case 1:

    jcols[0] = table.addColumn ( "xx" );

    break;

  case 3:

    jcols[0] = table.addColumn ( "xx" );
    jcols[1] = table.addColumn ( "yy" );
    jcols[2] = table.addColumn ( "xy" );

    break;

  case 4:

    // axisymmetric
    jcols[0] = table.addColumn ( "xx" );
    jcols[1] = table.addColumn ( "yy" );
    jcols[2] = table.addColumn ( "xy" );
    jcols[3] = table.addColumn ( "zz" );

    break;  

  case 6:

    jcols[0] = table.addColumn ( "xx" );
    jcols[1] = table.addColumn ( "yy" );
    jcols[2] = table.addColumn ( "zz" );
    jcols[3] = table.addColumn ( "xy" );
    jcols[4] = table.addColumn ( "yz" );
    jcols[5] = table.addColumn ( "xz" );
    jcols[6] = table.addColumn ( "vu" );

    break;

  default:

    throw Error (
      JEM_FUNC,
      "unexpected number of stress components: " +
      String ( strCount_ )
    );
  }

  idx_t         ipoint = 0;

  idx_t firstmodel = 0;

  if ( matType_ != "Neural" )
  {
    if ( numElem_ < 1000 ) firstmodel = 1657; // TODO: generalize this to something 
  }                                      // similar to FE2 material

//  System::out() << "nel " << numElem_ << " add " << firstmodel << "\n";

  Vector      ipWeights ( ipCount_ );
  Vector      dummy     ( ipCount_ );

  for ( idx_t ie = 0; ie < numElem_; ie++ )
  {
    idx_t  ielem = ielems[ie];

    ndStress   = 0.0;
    ndWeights  = 0.0;

    elems_.getElemNodes  ( inodes, ielem );
    dofs_->getDofIndices ( idofs,  inodes,  dofTypes_ );

    nodes_.getSomeCoords ( coords0, inodes );

    elemDisp0 = select ( disp0, idofs );
    elemDisp1 = select ( disp1, idofs );
    elemDispD = elemDisp1 - elemDisp0;

    updateCoords_ ( coordsU, coords0, elemDisp1 );
    updateCoords_ ( coordsUU, coords0, elemDisp0 );
    
/*    Vector auxl1 ( coords0[0]-coords0[1] );
    Vector auxl2 ( coords0[1]-coords0[2] );
    Vector auxl3 ( coords0[0]-coords0[2] );

    double l1 = norm2(auxl1);
    double l2 = norm2(auxl2);
    double l3 = norm2(auxl3);

    double p = (l1+l2+l3)/2.;

    double area = sqrt( p*(p-l1)*(p-l2)*(p-l3) );

    double volume = area*coords0(2, 3);
    double volumedef = area*coords0(2,3);
*/
    shape_->getShapeGradients ( grads0, dummy, coords0 );
    shape_->getShapeGradients ( grads, ipWeights, coordsU );
    shape_->getShapeGradients ( gradsuu, dummy, coordsUU );

    Matrix sfuncs = shape_->getShapeFunctions ();

    for ( idx_t ip = 0; ip < ipCount_; ip++ )
    {
      
      if ( stateString_ == "AXISYMMETRIC" )
      {
        Matrix ipCoords   ( rank_, ipCount_ );

        Matrix sfuncs     = shape_->getShapeFunctions();

        shape_->getGlobalIntegrationPoints ( ipCoords, coordsU );

        r = ipCoords(1,ip);

        getBMatrix_ ( b, grads(ALL,ALL,ip), sfuncs(ALL,ip), r );

        axisymDeformationGradient ( df, elemDispD, grads(ALL,ALL,ip), sfuncs(ALL,ip), r );

        material_->updateWriteTable ( stress, stiff, df, ipoint );
      }

      else if ( stateString_ == "PLANE_STRAIN" )
      {
        getShapeGrads_ ( b, grads(ALL,ALL,ip) );

        planeStrainDeformationGradient ( df, elemDispD, grads(ALL,ALL,ip) );

        Vector     stress1    ( 4 );

        material_->updateWriteTable ( stress1, stiff, df, ipoint );

        stress = stress1[slice(0,3)];
      }
      else
      {
        getShapeGrads_ ( b, grads(ALL,ALL,ip) );

        if ( matType_ == "Neural" )
        {
          evalDeformationGradient ( df, elemDisp1, grads0(ALL,ALL,ip) );
        }
        else
        {
          evalDeformationGradient ( df, elemDispD, gradsuu(ALL,ALL,ip) );
        }

        material_->updateWriteTable ( stress, stiff, df, ipoint );
      }
      
      if ( stateString_ == "AXISYMMETRIC" )
      {
        ndStress  += r * matmul ( sfuncs(ALL,ip), stress );

        ndWeights += r * sfuncs(ALL,ip);

        ++ipoint; 
      }
      else
      {
        ndStress  += matmul ( sfuncs(ALL,ip), stress );

        ndWeights += sfuncs(ALL,ip);

        ++ipoint; 
      }
    }

    select ( weights, inodes ) += ndWeights;

    // Add the stresses to the table.

    Matrix stressm(1, strCount_ +1 );
    stressm = -1.0;
    stressm(0,0 ) = stress[0];
    stressm(0,1) = stress[1];
    stressm(0,2) = stress[2];
    stressm(0,3) = stress[3];
    stressm(0,4) = stress[4];
    stressm(0,5) = stress[5];

    IdxVector ipointsvec ( 1 );
    ipointsvec[0] = ie + firstmodel;

    table.addBlock ( ipointsvec, jcols[slice(0,strCount_+1)],   stressm );
//    table.addBlock ( inodes, jcols[slice(0,strCount_)],   ndStress );
  }
}

//-----------------------------------------------------------------------
//   getXOutTable_
//-----------------------------------------------------------------------


void ULSolidModel::getXOutTable_

  ( Ref<XTable>        table,
    const Vector&      weights,
    const String&      contents,
    const Vector&      disp0,
    const Vector&      disp1 )

{
  Vector       ndWeights  ( nodeCount_ );
  StringVector hisNames   = material_->getHistoryNames ();

  Cubix        grads0      ( rank_, nodeCount_, ipCount_ );
  Cubix        gradsuu     ( rank_, nodeCount_, ipCount_ );
  Cubix        grads      ( rank_, nodeCount_, ipCount_ );
  Matrix       coords0    ( rank_, nodeCount_ );  // reference 
  Matrix       coordsU    ( rank_, nodeCount_ );  // updated
  Matrix       coordsUU   ( rank_, nodeCount_ );  
  Matrix       coords     ( rank_,     nodeCount_ );
  Matrix       stiff      ( strCount_, strCount_  );
  Matrix       b          ( strCount_, dofCount_  );

  Vector       elemDisp0  ( dofCount_ );
  Vector       elemDisp1  ( dofCount_ );
  Vector       elemDispD  ( dofCount_ );

  IdxVector    inodes     ( nodeCount_ );
  IdxVector    idofs      ( dofCount_  );

  const bool   tri6 = ( shape_->getGeometry() == Geometries::TRIANGLE
                      && nodeCount_ == 6 );

  const bool   wedge12 = ( shape_->getGeometry() == Geometries::WEDGE
                           && nodeCount_ == 12 );

  double       r = 1.; // for axisymmetric element

  // tell TbFiller which types are available to write

  TbFiller   tbFiller   ( rank_ );

  Slice      iistress   = tbFiller.announce ( "stress.tensor" );
  Slice      iihistory  = tbFiller.announce ( hisNames );

  Vector     ipValues   ( tbFiller.typeCount() );

  Matrix     df         ( 3, 3 );
  Matrix     F          ( 3, 3 );
  Vector     stress     ( strCount_ ); //ipValues[iistress]   );
  Vector     history    ( ipValues[iihistory]  );

  // Let TbFiller find out which columns of ndValues to write to 
  // which columns of the table (based on filter in input file)

  IdxVector  i2table;
  IdxVector  jcols;

  tbFiller . setFilter   ( contents );
  tbFiller . prepareTable( i2table, jcols, table );

  Matrix     ndValuesOut ( nodeCount_, i2table.size() );
  Vector     ipValuesOut ( i2table.size() );

  // fill table in loop over elements

  idx_t      ipCount;
  idx_t      ipoint = 0;

  Vector     ipWeights; Vector dummy; 
  
  Matrix homstress ( 1, strCount_+1 );
  homstress = 0.0;

  double voldef = 0.0; 

  // Add the columns for the stress components to the table.

  const idx_t nel = ielems_.size();
  
  idx_t firstmodel = 0;

  if ( matType_ != "Neural" )
  {
    if ( nel < 1000 && homogenize_ ) 
    {
      firstmodel = 1;
    }
    else
    {
      firstmodel = 1657; // TODO: generalize this to FE2?
    }
  }                                     

 // System::out() << "nel " << nel << " add " << firstmodel << "\n";
  Ref<PrintWriter> out = newInstance<PrintWriter> ( &System::out() );
  out->nformat.setFractionDigits( 16 );

  Matrix      sfuncs     = shape_->getShapeFunctions ();

  for ( idx_t ie = 0; ie < nel; ++ie )
  {
    idx_t ielem = ielems_[ie];

    ndValuesOut = 0.;
    ndWeights   = 0.;

    elems_.getElemNodes  ( inodes, ielem );
    dofs_->getDofIndices ( idofs,  inodes,  dofTypes_ );

    ipCount  = shape_->ipointCount ();

    ipWeights.resize ( ipCount   );
    dummy.resize ( ipCount );

    nodes_.getSomeCoords ( coords0, inodes );

    elemDisp0 = select ( disp0, idofs );

    elemDisp1 = select ( disp1, idofs );

    elemDispD = elemDisp1 - elemDisp0;

    updateCoords_ ( coordsU, coords0, elemDisp1 );
    updateCoords_ ( coordsUU, coords0, elemDisp0 );
  
    shape_->getShapeGradients ( grads0, dummy, coords0 );
    shape_->getShapeGradients ( grads, ipWeights, coordsU );
    shape_->getShapeGradients ( gradsuu, dummy, coordsUU );

    // Iterate over the integration points.
    // Gather all data, no matter which is asked, to keep code neat
    // The option to specify output is primarily for disk size, not CPU time

    for ( idx_t ip = 0; ip < ipCount; ip++, ++ipoint )
    {
      
      if ( stateString_ == "AXISYMMETRIC" )
      {
        Matrix ipCoords   ( rank_, ipCount_ );

        Matrix sfuncs     = shape_->getShapeFunctions();

        shape_->getGlobalIntegrationPoints ( ipCoords, coordsU );

        r = ipCoords(1,ip);

        getBMatrix_ ( b, grads(ALL,ALL,ip), sfuncs(ALL,ip), r );

        axisymDeformationGradient ( df, elemDispD, grads(ALL,ALL,ip), sfuncs(ALL,ip), r );
      }
      else if ( stateString_ == "PLANE_STRAIN" )
      {
        getShapeGrads_ ( b, grads(ALL,ALL,ip) );

        planeStrainDeformationGradient ( df, elemDispD, grads(ALL,ALL,ip) );
      }
      else
      {
        getShapeGrads_ ( b, grads(ALL,ALL,ip) );

        if ( matType_ == "Neural" )
        {
/*	  System::out() << "idofs " << idofs << "\n";
	  System::out() << "elemDisp0 " << elemDisp0 << "\n";
	  System::out() << "elemDisp1 " << elemDisp1 << "\n";*/
          evalDeformationGradient ( df, elemDisp1, grads0(ALL,ALL,ip) );
//	  System::out() << "getXOutput F " << df << "\n";
        }
        else
        {
          evalDeformationGradient ( df, elemDispD, gradsuu(ALL,ALL,ip) );
         // *out << "getXOutput df " << df << "\n";
        }
      }

      material_->updateWriteTable ( stress, stiff, df, ipoint );

//      material_-> getHistory ( history, ipoint );

      ipValuesOut  = ipValues[i2table];

      if ( stateString_ == "AXISYMMETRIC" )
      {
        ndValuesOut += r * matmul ( sfuncs(ALL,ip), ipValuesOut ); 
        ndWeights   += r * sfuncs(ALL,ip);
      }
      else
      {
        ndValuesOut += matmul ( sfuncs(ALL,ip), ipValuesOut ); 
        ndWeights   += sfuncs(ALL,ip);  
      }
      
      voldef += ipWeights[ip];

      if ( homogenize_ && matType_ != "Neural") 
      {
        for ( idx_t comp = 0; comp < strCount_; comp++ )
	{
	  homstress(0,comp) += stress[comp]*ipWeights[ip];        
        }                                                         
      }      
    }

    if ( tri6 ) TbFiller::permTri6 ( ndWeights, ndValuesOut );

    if ( wedge12 ) {TbFiller::permTri12 ( ndWeights, ndValuesOut );}

    select ( weights, inodes ) += ndWeights;

    // Add the values to the table.

    if ( matType_ == "Neural" ) 
    {
      Matrix stressm(3, strCount_ + 1);
      
      stressm(0,3) = stress[0];
      stressm(0,4) = stress[3];
      stressm(0,5) = stress[5];
      stressm(0,0) = df ( 0, 0 );
      stressm(0,1) = df ( 0, 1 );
      stressm(0,2) = df ( 0, 2 );

      stressm(1,3) = stress[3];
      stressm(1,4) = stress[1];
      stressm(1,5) = stress[4];
      stressm(1,0) = df ( 1, 0 );
      stressm(1,1) = df ( 1, 1 );
      stressm(1,2) = df ( 1, 2 );

      stressm(2,3) = stress[5];
      stressm(2,4) = stress[4];
      stressm(2,5) = stress[2];
      stressm(2,0) = df ( 2, 0 );
      stressm(2,1) = df ( 2, 1 );
      stressm(2,2) = df ( 2, 2 );
      stressm(0, 6) = voldef;

      IdxVector ipointsvec ( 3 );
      ipointsvec[0] = ie + firstmodel;
      ipointsvec[1] = ipointsvec[0] + 1;
      ipointsvec[2] = ipointsvec[0] + 2;

      table->addBlock ( ipointsvec, jcols,   stressm );
   }
   else
   {
      if ( homogenize_ )
      {
        if ( ie == numElem_ - 1)
	{
	  IdxVector ipointsvec ( 1 );
          ipointsvec[0] = firstmodel;

//	  *out << "voldef " << voldef << "\n";
          homstress(0, 6 ) = voldef;
          table->addBlock ( ipointsvec, jcols, homstress );
	}
      }
      else
      {
        Matrix stressm(1, strCount_ + 1);
        stressm(0,0) = stress[0];
        stressm(0,1) = stress[1];
        stressm(0,2) = stress[2];
        stressm(0,3) = stress[3];
        stressm(0,4) = stress[4];
        stressm(0,5) = stress[5];
	stressm(0,6) = voldef;

	IdxVector ipointsvec ( 1 );
        ipointsvec[0] = ie + firstmodel;

        table->addBlock ( ipointsvec, jcols, stressm );
      // table->addBlock ( inodes, jcols, ndValuesOut );
      }
   }
  }
}

//-----------------------------------------------------------------------
//    getBMatrix_
//-----------------------------------------------------------------------

//necessary for axisymmetric implementation
void ULSolidModel::getBMatrix_

  ( const Matrix&       b,
    const Matrix&       dN,
    const Vector&       N,
    const double        r ) const

{
  getShapeGrads_ ( b(slice(0,3),ALL), dN );

  b(3,slice(0,END,2)) = 0.;
  b(3,slice(1,END,2)) = N / r;
}

/*
//-----------------------------------------------------------------------
//    writeDisplacements_
//-----------------------------------------------------------------------

void ULSolidModel::writeDisplacements_

  ( const Properties&  globdat )

{
  Vector      dispi      ( rank_ );
  IdxVector   idofs      ( rank_ );
  Vector      disp;
  idx_t         it;

  if ( dispOut_ == nullptr )
  {
    dispOut_ = newInstance<PrintWriter>( newInstance<FileWriter> ( "all.disp" ) );
  }

  dispOut_->nformat.setFractionDigits ( 8 );

  globdat.get ( it, Globdat::TIME_STEP );

  *dispOut_ << "newXOutput " << it << '\n';

  Vector coords0 ( rank_ );   

  StateVector::get ( disp, dofs_, globdat );

  // regular nodes

  for ( idx_t inode = 0; inode < nodes_.size(); ++inode )
  {
    dofs_->getDofsForItem ( idofs,  dofTypes_,  inode );
    nodes_.getNodeCoords ( coords0, inode );

    dispi = disp[idofs] + coords0;

    *dispOut_ << inode << " ";

    for ( idx_t j = 0; j < rank_; ++j )
    {
      *dispOut_ << dispi[j] << " ";
    }
    *dispOut_ << '\n';
  }
  dispOut_->flush();
}
*/

//-----------------------------------------------------------------------
//    initWriter_
//-----------------------------------------------------------------------

Ref<PrintWriter>  ULSolidModel::initWriter_

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
    fileName[1] = myTag_;
    fileName[2] = name;
  }
  else
  {
    fileName.resize(2);

    fileName[0] = myTag_;
    fileName[1] = name;
  }

//  System::out() << "Filename for printing " << fileName << "\n";
  return newInstance<PrintWriter>( newInstance<FileWriter> (
         StringUtils::join( fileName, "." ) ) );
}

//-----------------------------------------------------------------------
//    writeDisplacements_
//-----------------------------------------------------------------------

void ULSolidModel::writeDisplacements_

  ( const Properties& params, 
    const Properties&  globdat )

{
  Vector      dispi      ( rank_ );
  IdxVector   idofs      ( rank_ );
  Vector      disp;
  idx_t         it;

  IdxVector ielems = egroup_.getIndices ( );
  IdxVector inodes    ( nodeCount_ );
  Matrix    coords    ( rank_, nodeCount_ );

  if ( dispOut_ == nullptr )
  {
    dispOut_ = initWriter_ ( params, "state" );
  }

  dispOut_->nformat.setFractionDigits ( 12 );

  globdat.get ( it, Globdat::TIME_STEP );

  *dispOut_ << "newXOutput " << it << '\n';

  Vector coords0 ( rank_ );

  StateVector::get ( disp, dofs_, globdat );

  for ( idx_t i = 0; i < numElem_; ++i )
  {
    idx_t ielem = ielems[i];

    elems_.getElemNodes  ( inodes, ielem            );
    nodes_.getSomeCoords ( coords, inodes           );

    for ( idx_t j = 0; j < nodeCount_; ++j )
    {
      dofs_->getDofIndices ( idofs, inodes[j], dofTypes_ );

      *dispOut_ << inodes[j] << " ";

      for ( idx_t r = 0; r < rank_; ++r )
      {
        *dispOut_ << coords(r,j) + disp[idofs[r]] << " ";
      }

      *dispOut_ << '\n';
    }
  }

  dispOut_->flush();
}


//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   newULSolidModel
//-----------------------------------------------------------------------


static Ref<Model>     newULSolidModel

  ( const String&       name,
    const Properties&   conf,
    const Properties&   props,
    const Properties&   globdat )

{
  return newInstance<ULSolidModel> ( name, conf, props, globdat );
}


//-----------------------------------------------------------------------
//   declareULSolidModel
//-----------------------------------------------------------------------


void declareULSolidModel ()
{
  using jive::model::ModelFactory;

  ModelFactory::declare ( "ULSolid", & newULSolidModel );
}

