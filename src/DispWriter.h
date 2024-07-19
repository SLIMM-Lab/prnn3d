/*
 * 
 *  Copyright (C) 2013 TU Delft. All rights reserved.
 *  
 *  This class makes sure displacements are written only once.
 *  Any model that has displacements shoud be have a DispWriter
 *  
 *  Author: F.P. van der Meer, f.p.vandermeer@tudelft.nl
 *  Date: April 2013
 *
 */

#ifndef DISP_WRITER_H
#define DISP_WRITER_H

#include <jem/base/Object.h>
#include <jive/model/Model.h>

using jem::Object;
using jem::Ref;
using jive::model::Model;

class DispWriter : public Object
{
 public:

  bool                    amFirstWriter

    ( Ref<Model>            model );


 protected:

  static Ref<Model>       writer_;
};

inline bool DispWriter::amFirstWriter

  ( Ref<Model>  model )

{
  if ( writer_ == nullptr ) 
  {
    writer_ = model;

    return true;
  }
  else
  {
    return ( model == writer_ );
  }
}

#endif
