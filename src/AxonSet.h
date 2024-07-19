/*
 * Copyright (C) 2019 TU Delft. All rights reserved.
 *
 * ItemSet-like entity for artificial axons.
 *
 * Author: Iuri Barcelos, i.barceloscarneiromrocha@tudelft.nl
 * Date:   May 2019
 * 
 */

#ifndef AXONSET_H
#define AXONSET_H

#include <jem/base/Object.h>
#include <jem/base/Ref.h>
#include <jem/base/Array.h>
#include <jem/util/Properties.h>
#include <jive/fem/XNodeSet.h>
#include <jive/Array.h>
#include <jive/util/Assignable.h>

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
//   class AxonSet
//-----------------------------------------------------------------------

class AxonSet : public Object
{
 public:

  static const char*          GLOBNAME;

                              AxonSet    ();    

  inline PointSet*            getData    () const;

  void                        store

    ( const Properties&         globdat   ) const;

  static Ref<AxonSet>         get

    ( const Properties&         globdat,
      const String&             context );

 protected:

  Assignable<XNodeSet>        nodes_;
};

//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   getNodes
//-----------------------------------------------------------------------

inline PointSet* AxonSet::getData () const
{
  return nodes_.getData();
}

#endif
