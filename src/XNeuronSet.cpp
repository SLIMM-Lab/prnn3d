/*
 * Copyright (C) 2019 TU Delft. All rights reserved.
 *
 * ItemSet skin for artificial neurons. X version.
 *
 * Author: Iuri Barcelos, i.barceloscarneiromrocha@tudelft.nl
 * Date:   May 2019
 * 
 */

#include <jem/base/Slice.h>
#include <jem/base/assert.h>
#include <jem/base/Error.h>
#include <jem/base/IllegalInputException.h>
#include <jive/util/XItemMap.h>
#include <jive/util/StdGroupSet.h>
#include <jive/fem/Globdat.h>

#include "XNeuronSet.h"

using jem::newInstance;
using jem::Error;
using jem::slice;
using jem::END;

//=======================================================================
//   class XElementSet
//=======================================================================

//-----------------------------------------------------------------------
//   addNeuron
//-----------------------------------------------------------------------

idx_t XNeuronSet::addNeuron ()
{
  IdxVector inodes (0);
  IdxVector ielems (0);

  idx_t i = -1;

  i  = elemsI_.addElement ( inodes );

  if ( elemsO_.addElement ( inodes ) != i )
  {
    throw Error ( JEM_FUNC, "XNeuronSet: Invalid neuron index" );
  }

  if ( pairsI_->addGroup ( inodes ) != i )
  {
    throw Error ( JEM_FUNC, "XNeuronSet: Invalid neuron index" );
  }

  if ( pairsO_->addGroup ( inodes ) != i )
  {
    throw Error ( JEM_FUNC, "XNeuronSet: Invalid neuron index" );
  }

  return i;
}

//-----------------------------------------------------------------------
//   setInput
//-----------------------------------------------------------------------

void XNeuronSet::setInput

  ( const idx_t      ineuron,
    const IdxVector& iaxons,
    const IdxVector& ineurons )

{
  elemsI_ .setElemNodes    ( ineuron, iaxons   );
  pairsI_->setGroupMembers ( ineuron, ineurons );
}

//-----------------------------------------------------------------------
//   addOutput
//-----------------------------------------------------------------------

void XNeuronSet::addOutput

  ( const idx_t      ineuron,
    const IdxVector& iaxons,
    const IdxVector& ineurons )

{
  for ( idx_t ia = 0; ia < iaxons.size(); ++ia )
  {
    idx_t iaxon = iaxons[ia];
    idx_t ineur = ineurons[ia];
    idx_t size  = elemsO_.getElemNodeCount ( ineur );

    IdxVector members ( size );

    elemsO_.getElemNodes ( members, ineur );

    members.reshape ( size + 1 );
    members[size] = iaxon;

    elemsO_.setElemNodes ( ineur, members );

    size = pairsO_->getGroupSize ( ineur );

    members.resize ( size );

    pairsO_->getGroupMembers ( members, ineur );

    members.reshape ( size + 1 );
    members[size] = ineuron;

    pairsO_->setGroupMembers ( ineur, members );
  }
}

//-----------------------------------------------------------------------
//   get
//-----------------------------------------------------------------------

Ref<XNeuronSet> XNeuronSet::get

  ( const Properties&  globdat,
    const String&      context )

{
  Ref<XNeuronSet> neurons = jem::dynamicCast<XNeuronSet> 
    ( globdat.get ( NeuronSet::GLOBNAME ) );

  if ( !neurons )
  {
    throw jem::IllegalInputException (
      context,
      "the standard neuron set is not editable (not an XNeuronSet)"
    );
  }

  return neurons;
}

