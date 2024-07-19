/*
 * 
 *  Copyright (C) 2019 TU Delft. All rights reserved.
 *  
 *  This class implements a virtual class for a constitutive model
 *
 *  Author:  F.P. van der Meer, F.P.vanderMeer@tudelft.nl
 *  Date:    March 2019
 *
 */

#include "Material.h"
#include "EGPMaterial.h"
#include "NeoHookeMaterial.h"
#include "OrthotropicULMaterial.h"
#include "BonetMaterial.h"
#include "ANNMaterial.h"
#include "ObserverMaterial.h"

using namespace jem;


//=======================================================================
//   class Material
//=======================================================================

Material::Material 

  ( const idx_t        rank,
    const Properties&  globdat )

  : rank_(rank)

{
}


Material::~Material()
{}

void   Material::commit()
{}

void Material::checkCommit

  ( const Properties& params )

{}

void   Material::cancel()
{}

void   Material::configure

  ( const Properties& props )

{}

void   Material::getConfig 

  ( const Properties& props ) const

{}

//--------------------------------------------------------------------
//   allocPoints
//--------------------------------------------------------------------

void Material::allocPoints ( const idx_t count )
{}

void Material::allocPoints 

  ( const idx_t      count,
    const Matrix&    transfer,
    const IdxVector& oldPoints )

{
  // default implementation: ignore history input

  allocPoints ( count );
}

//--------------------------------------------------------------------
//   allocPoints
//--------------------------------------------------------------------

void Material::deallocPoints ( const idx_t count )
{}

//-----------------------------------------------------------------------
//   getHistoryNames
//-----------------------------------------------------------------------

StringVector Material::getHistoryNames 

  () const
{
  return historyNames_;
}

//-----------------------------------------------------------------------
//   getHistory
//-----------------------------------------------------------------------

void Material::getHistory
 
  ( const Vector&  history,
    const idx_t    ipoint ) const

{
}


//-----------------------------------------------------------------------
//   getInitHistory
//-----------------------------------------------------------------------

void Material::getInitHistory
 
  ( const Vector&  history,
    const idx_t    ipoint ) const

{
}

//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   newMaterial
//-----------------------------------------------------------------------


Ref<Material>         newMaterial

  ( const String&       name,
    const Properties&   conf,
    const Properties&   props,
    const Properties&   globdat )

{
  Properties     matProps = props.getProps ( name );
  Properties     matConf  = conf.makeProps ( name );

  Ref<Material>  mat;
  String         type;
  idx_t          dim;

  matProps.get ( type, "type" );
  matConf .set ( "type", type );

  matProps.get ( dim, "dim"   );
  matConf .set ( "dim", dim   );
  
  if ( type == "EGP" )
  {
    mat = newInstance<EGPMaterial> ( dim, globdat );
  }
  else if ( type == "NeoHooke" )
  {
    mat = newInstance<NeoHookeMaterial> ( dim, globdat );
  }
  else if ( type == "OrthotropicBonet" )
  {
    mat = newInstance<BonetMaterial> ( dim, globdat );
  }
  else if ( type == "Neural" )
  {
    mat = newInstance<ANNMaterial> ( dim, globdat );
  }
  else if ( type == "Observer" )
  {
    mat = newInstance<ObserverMaterial> ( matProps, matConf, dim, globdat );
  }
  else
  {
    matProps.propertyError (
      name,
      "invalid material type: " + type
    );
  }

  return mat;
}

//-----------------------------------------------------------------------
//   setHistory
//-----------------------------------------------------------------------

void Material::setHistory

  ( const Vector&    hvals,
    const idx_t      mpoint )

{
  // Default implementation
}


//-----------------------------------------------------------------------
//   updateYieldFunc
//-----------------------------------------------------------------------

void Material::updateYieldFunc

  ( const Vector&    learnProp )

{
  // Default implementation
}

