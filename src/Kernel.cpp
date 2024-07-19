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

#include <jem/base/Error.h>
#include <jem/util/Properties.h>

#include "Kernel.h"
#include "IsoRBFKernel.h"

using namespace jem;

//-----------------------------------------------------------------------
//   newInstance
//-----------------------------------------------------------------------

Ref<Kernel>  newKernel

  ( const String&     name,
    const Properties& conf,
    const Properties& props,
    const Properties& globdat )

{
  Properties    kernelProps = props.getProps ( name );
  Properties    kernelConf  = conf.makeProps ( name );

  Ref<Kernel>   kernel;
  String        type;

  kernelProps.get ( type, "type" );
  kernelConf.set  ( "type", type );

  if      ( type == "IsoRBF" )
    kernel = newInstance<IsoRBFKernel> ( name, kernelConf, kernelProps, globdat );
  else
    kernelProps.propertyError ( name, "Invalid kernel: " + type );

  return kernel;
}

//=======================================================================
//   class Kernel
//=======================================================================

//-----------------------------------------------------------------------
//   constructors and destructor
//-----------------------------------------------------------------------

Kernel::Kernel

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{}

Kernel::~Kernel()
{}

//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------

void Kernel::configure

  ( const Properties& props,
    const Properties& globdat )
{}

//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------

void Kernel::getConfig

  ( const Properties& props,
    const Properties& globdat ) const
{}
