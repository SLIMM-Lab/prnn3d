/*
 * Copyright (C) 2019 TU Delft. All rights reserved.
 *
 * ItemSet-like entity for artificial axons. X version
 *
 * Author: Iuri Barcelos, i.barceloscarneiromrocha@tudelft.nl
 * Date:   May 2019
 * 
 */


#include <jem/base/System.h>
#include <jem/base/assert.h>
#include <jem/base/Class.h>
#include <jem/base/array/utilities.h>
#include <jem/base/IllegalInputException.h>
#include <jive/util/error.h>
#include <jive/fem/error.h>
#include <jive/fem/Globdat.h>

#include "XAxonSet.h"

using jem::newInstance;
using jive::Vector;

//=======================================================================
//   class XAxonSet
//=======================================================================

//-----------------------------------------------------------------------
//   addAxon
//-----------------------------------------------------------------------

idx_t XAxonSet::addAxon ()

{
  Vector nulcoords ( 0 );

  return nodes_.addNode ( nulcoords );
}

//-----------------------------------------------------------------------
//   addAxons
//-----------------------------------------------------------------------

IdxVector XAxonSet::addAxons

  ( const idx_t count )

{
  IdxVector ins ( count );
  ins = -1;

  for ( idx_t i = 0; i < count; ++i )
  {
    ins[i] = addAxon();
  }

  return ins;
}

//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------

Ref<XAxonSet> XAxonSet::get

  ( const Properties&  globdat,
    const String&      context )

{
  Ref<XAxonSet> axons = jem::dynamicCast<XAxonSet> 
    ( globdat.get ( AxonSet::GLOBNAME ) );

  if ( axons == nullptr )
  {
    throw jem::IllegalInputException (
      context,
      "the standard axon set is not editable (not an XAxonSet)"
    );
  }

  return axons;
}
