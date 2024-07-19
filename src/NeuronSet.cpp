/*
 * Copyright (C) 2019 TU Delft. All rights reserved.
 *
 * ItemSet skin for artificial neurons.
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
#include <jive/fem/XElementSet.h>
#include <jive/util/Assignable.h>
#include <jive/util/StdGroupSet.h>

#include "NeuronSet.h"

using jem::newInstance;
using jive::util::Assignable;
using jive::util::StdGroupSet;
using jive::fem::XElementSet;

//=======================================================================
//   class NeuronSet
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------

const char* NeuronSet::GLOBNAME = "NeuronSet";

//-----------------------------------------------------------------------
//   constructor
//-----------------------------------------------------------------------

NeuronSet::NeuronSet  

  ( const Ref<AxonSet> axons )

{
  using jive::fem::newXElementSet;

  elemsI_ = newXElementSet ( NodeSet ( axons->getData() ) );
  elemsO_ = newXElementSet ( NodeSet ( axons->getData() ) );

  pairsI_ = newInstance<StdGroupSet>
    ( "inputNeurons", "neurons", elemsI_.getData() );

  pairsO_ = newInstance<StdGroupSet>
    ( "inputNeurons", "neurons", elemsO_.getData() );
}

//-----------------------------------------------------------------------
//   getInput
//-----------------------------------------------------------------------

void NeuronSet::getInput

  (       IdxVector& iaxons,
          IdxVector& ineurons,
    const idx_t      ineuron  ) const

{
  iaxons. resize       ( elemsI_.getElemNodeCount ( ineuron ) );
  elemsI_.getElemNodes ( iaxons, ineuron );

  ineurons.resize          ( pairsI_->getGroupSize ( ineuron ) );
  pairsI_->getGroupMembers ( ineurons, ineuron );
}

//-----------------------------------------------------------------------
//   getOutput
//-----------------------------------------------------------------------

void NeuronSet::getOutput

  (       IdxVector& iaxons,
          IdxVector& ineurons,
    const idx_t      ineuron  ) const

{
  iaxons. resize       ( elemsO_.getElemNodeCount ( ineuron ) );
  elemsO_.getElemNodes ( iaxons, ineuron );

  ineurons.resize          ( pairsO_->getGroupSize ( ineuron ) );
  pairsO_->getGroupMembers ( ineurons, ineuron );
}


//-----------------------------------------------------------------------
//   store
//-----------------------------------------------------------------------

void NeuronSet::store

  ( const Properties&  globdat ) const

{
  //JEM_ASSERT ( *this != jem::NIL );

  globdat.set ( GLOBNAME, const_cast<NeuronSet*> ( this ) );

  elemsI_.store ( globdat );
}

//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------

Ref<NeuronSet> NeuronSet::get

  ( const Properties&  globdat,
    const String&      context )

{
  Ref<NeuronSet> neurons;

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
