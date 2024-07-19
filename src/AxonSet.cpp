/*
 * Copyright (C) 2019 TU Delft. All rights reserved.
 *
 * ItemSet skin for artificial axons.
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
#include <jive/fem/XNodeSet.h>
#include <jive/util/Assignable.h>
#include <jive/util/StdGroupSet.h>

#include "AxonSet.h"

using jem::newInstance;
using jive::util::Assignable;
using jive::util::StdGroupSet;

//=======================================================================
//   class AxonSet
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------

const char* AxonSet::GLOBNAME = "AxonSet";

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------

AxonSet::AxonSet ()

{
  using jive::fem::newXNodeSet;

  nodes_ = newXNodeSet ( "axons" );
}

//-----------------------------------------------------------------------
//   store
//-----------------------------------------------------------------------

void AxonSet::store

  ( const Properties&  globdat ) const

{
  //JEM_ASSERT ( *this != jem::NIL );

  globdat.set ( GLOBNAME, const_cast<AxonSet*> ( this ) );

  nodes_.store ( globdat );
}

//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------

Ref<AxonSet> AxonSet::get

  ( const Properties&  globdat,
    const String&      context )

{
  Ref<AxonSet> neurons;

  globdat.find ( neurons, GLOBNAME );

  if ( !neurons )
  {
    throw jem::IllegalInputException (
      context,
      "no neurons have been defined"
    );
  }

  return neurons;
}
