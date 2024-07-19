/*
 * Copyright (C) 2019 TU Delft. All rights reserved.
 *
 * Base class for covariance (distance) kernels (measures)
 * For machine learning purposes
 *
 * Rasmussen, C. E. and Williams, C. K. I. Gaussian Processes
 * for Machine Learning. MIT Press, 2016. <www.gaussianprocess.org>
 *
 * Author: Iuri Barcelos, i.barceloscarneiromrocha@tudelft.nl
 * Date:   Oct 2019
 *
 */

#ifndef KERNEL_H
#define KERNEL_H

#include <jem/base/System.h>
#include <jem/util/Properties.h>
#include <jem/base/Object.h>
#include <jive/Array.h>
#include <jive/util/XTable.h>

using jem::System;
using jem::idx_t;
using jem::Ref;
using jem::Object;
using jem::String;
using jem::util::Properties;
using jive::Vector;
using jive::Matrix;
using jive::Cubix;
using jive::IdxVector;
using jive::BoolVector;
using jive::util::XTable;


//-----------------------------------------------------------------------
//   class Kernel
//-----------------------------------------------------------------------

class Kernel : public Object
{
 public:
                         Kernel

    ( const String&        name,
      const Properties&    conf,
      const Properties&    props,
      const Properties&    globdat );

  virtual void           configure
  
    ( const Properties&    props,
      const Properties&    globdat );

  virtual void           getConfig

    ( const Properties&    conf,
      const Properties&    globdat ) const;

  virtual void           update  
  
    ( const Properties&    globdat )       = 0;

  virtual double         eval

    ( const Vector&        xp      ) const = 0;

  virtual double         eval

    ( const Vector&        xp,
      const Vector&        xq      ) const = 0;

  virtual Vector         eval

    ( const Vector&        xp, 
      const Matrix&        xq      ) const = 0;

  virtual Matrix         eval

    ( const Matrix&        xp      ) const = 0;

  virtual Matrix         eval

    ( const Matrix&        xp,
      const Matrix&        xq      ) const = 0;

  virtual Matrix         evalDerivs

    ( const Vector&        xp,
      const Matrix&        xq      ) const = 0;

  virtual void           gradients

    (       Cubix&         G,
      const Matrix&        xp      )       = 0;

  virtual idx_t          varCount () const = 0;

  virtual Vector         getVars  () const = 0;

  virtual double         getNoise () const = 0;

  virtual Ref<Kernel>    clone    () const = 0;

 protected:

                       ~Kernel ();
};

//-----------------------------------------------------------------------
//   newKernel
//-----------------------------------------------------------------------

Ref<Kernel>  newKernel

    ( const String&       name,
      const Properties&   conf,
      const Properties&   props,
      const Properties&   globdat );

#endif
