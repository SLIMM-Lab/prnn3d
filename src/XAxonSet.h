/*
 * Copyright (C) 2019 TU Delft. All rights reserved.
 *
 * ItemSet-like entity for artificial axons. X version
 *
 * Author: Iuri Barcelos, i.barceloscarneiromrocha@tudelft.nl
 * Date:   May 2019
 * 
 */

#ifndef XAXONSET_H
#define XAXONSET_H

#include <jem/base/Object.h>
#include <jem/base/Ref.h>
#include <jem/base/Array.h>
#include <jem/util/Properties.h>
#include <jive/fem/XNodeSet.h>
#include <jive/Array.h>
#include <jive/util/Assignable.h>

#include "AxonSet.h"

using jem::Object;
using jem::Ref;
using jem::idx_t;
using jem::String;
using jem::util::Properties;

using jive::util::PointSet;
using jive::util::Assignable;
using jive::fem::NodeSet;
using jive::fem::XNodeSet;
using jive::IdxVector;

//-----------------------------------------------------------------------
//   class XAxonSet
//-----------------------------------------------------------------------

class XAxonSet : public AxonSet
{
 public:
   typedef AxonSet            Super;

  inline                      XAxonSet    ();    

  inline void                 clear       ();

  inline void                 reserve

    ( idx_t                     count );      

  inline void                 trimToSize  ();

  idx_t                       addAxon     ();

  IdxVector                   addAxons

    ( const idx_t               count );

  static Ref<XAxonSet>         get

    ( const Properties&         globdat,
      const String&             context );
};

//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   XAxonSet
//-----------------------------------------------------------------------

XAxonSet::XAxonSet () : Super ()
{}

//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------

inline void XAxonSet::clear ()
{
  nodes_.clear();
}

//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------

inline void XAxonSet::reserve ( idx_t n )
{
  nodes_.reserve ( n );
}

//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------

inline void XAxonSet::trimToSize ()
{
  nodes_.trimToSize ();
}

#endif
