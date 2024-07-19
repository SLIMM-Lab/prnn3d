/*
 * Copyright (C) 2019 TU Delft. All rights reserved.
 *
 * Squared exponential (Radial Basis Functions) kernel
 * Isotropic version with noise. For machine learning purposes
 *
 * Rasmussen, C. E. and Williams, C. K. I. Gaussian Processes
 * for Machine Learning. MIT Press, 2016. <www.gaussianprocess.org>
 *
 * Author: Iuri Barcelos, i.barceloscarneiromrocha@tudelft.nl
 * Date:   Oct 2019
 *
 */

#ifndef ISORBF_KERNEL_H
#define ISORBF_KERNEL_H

#include <jem/base/Array.h>
#include <jem/base/array/operators.h>
#include <jem/base/System.h>
#include <jem/util/Properties.h>
#include <jem/base/Object.h>
#include <jive/Array.h>
#include <jive/util/XTable.h>

#include "Kernel.h"

using jem::System;
using jem::idx_t;
using jem::Ref;
using jem::Object;
using jem::String;
using jem::util::Properties;
using jive::Vector;
using jive::Matrix;
using jive::IdxVector;
using jive::BoolVector;
using jive::util::XTable;


//-----------------------------------------------------------------------
//   class IsoRBFKernel
//-----------------------------------------------------------------------

class IsoRBFKernel : public Kernel

{
 public:

  static const char*     INITVARIANCE;
  static const char*     INITLENSCALE;
  static const char*     INITNOISE;
  static const char*     VARIANCEBOUNDS;
  static const char*     LENSCALEBOUNDS;
  static const char*     NOISEBOUNDS;

                         IsoRBFKernel

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
  
    ( const Properties&    globdat );

  virtual double         eval

    ( const Vector&        xp      ) const;
      
  virtual double         eval

    ( const Vector&        xp,
      const Vector&        xq      ) const;

  virtual Vector         eval

    ( const Vector&        xp, 
      const Matrix&        xq      ) const;

  virtual Matrix         eval

    ( const Matrix&        xp      ) const;

  virtual Matrix         eval

    ( const Matrix&        xp,
      const Matrix&        xq      ) const;

  virtual Matrix         evalDerivs

    ( const Vector&        xp,
      const Matrix&        xq      ) const;

  virtual void           gradients

    (       Cubix&         G,
      const Matrix&        xp      );

  virtual Matrix         geEval

    ( const Vector&        xp      ) const;

  virtual Matrix         geEval

    ( const Vector&        xp,
      const Vector&        xq      ) const;

  virtual Matrix         geEval

    ( const Vector&        xp, 
      const Matrix&        xq      ) const;     

  virtual Matrix         geEval

    ( const Matrix&        xp      ) const;

  virtual Matrix         geEval

    ( const Matrix&        xp,
      const Matrix&        xq      ) const;

  virtual void           geGradients

    (       Cubix&         G,
      const Matrix&        xp      );

  virtual idx_t          varCount () const;

  virtual Vector         getVars  () const;

  virtual double         getNoise () const;

  virtual Ref<Kernel>    clone    () const;

 protected:

                       ~IsoRBFKernel ();
		       
	  Cubix         grads_

  ( const Vector&         xp       ) const;

          Cubix         grads_

  ( const Vector&         xp,
    const Vector&         xq       ) const;

          void          load_
	
  ( const String& fname            );

 private:

  double                l_;
  double                l2_;

  double                sigF2_;
  double                sigN2_;

  IdxVector             iDofs_;

  Vector                ubnd_;
  Vector                lbnd_;

};

#endif
