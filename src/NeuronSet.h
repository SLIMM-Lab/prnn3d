/*
 * Copyright (C) 2019 TU Delft. All rights reserved.
 *
 * ItemSet-like entity for artificial neurons.
 *
 * Author: Iuri Barcelos, i.barceloscarneiromrocha@tudelft.nl
 * Date:   May 2019
 * 
 */

#ifndef NEURONSET_H
#define NEURONSET_H

#include <jem/base/Object.h>
#include <jem/base/Ref.h>
#include <jem/base/Array.h>
#include <jem/util/Properties.h>
#include <jive/util/XGroupSet.h>
#include <jive/fem/XElementSet.h>
#include <jive/fem/NodeSet.h>
#include <jive/Array.h>
#include <jive/util/Assignable.h>

#include "AxonSet.h"

using jem::Object;
using jem::Ref;
using jem::idx_t;
using jem::String;
using jem::util::Properties;

using jive::util::XGroupSet;
using jive::util::Assignable;
using jive::fem::XElementSet;
using jive::fem::NodeSet;
using jive::IdxVector;

//-----------------------------------------------------------------------
//   class NeuronSet
//-----------------------------------------------------------------------

class NeuronSet : public Object
{
 public:

  static const char*          GLOBNAME;

                              NeuronSet        

    ( const Ref<AxonSet>        axons );

          void                getInput

    (       IdxVector&          iaxons,
            IdxVector&          ineurons,
      const idx_t               ineuron )            const;

          void                getOutput

    (       IdxVector&          iaxons,
            IdxVector&          ineurons,
      const idx_t               ineuron )            const;
       
  inline  idx_t               getOutSize

    ( const idx_t               ineuron )            const;

  inline  idx_t               size     ()            const;

  void                        store

    ( const Properties&         globdat )            const;

  static Ref<NeuronSet>       get

    ( const Properties&         globdat,
      const String&             context );

 protected:

  Assignable<XElementSet>     elemsI_;
  Assignable<XElementSet>     elemsO_;

  Ref<XGroupSet>              pairsI_;
  Ref<XGroupSet>              pairsO_;
};

//-----------------------------------------------------------------------
//   getOutSize
//-----------------------------------------------------------------------

inline idx_t NeuronSet::getOutSize

  ( const idx_t ineuron ) const
{
  return elemsO_.getElemNodeCount ( ineuron );
}

//-----------------------------------------------------------------------
//   size
//-----------------------------------------------------------------------

inline idx_t NeuronSet::size () const
{
  return elemsO_.size();
}

#endif
