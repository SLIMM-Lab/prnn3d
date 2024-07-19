/*
 * Copyright (C) 2019 TU Delft. All rights reserved.
 *
 * Utility functions for artificial neural networks.
 *
 * Author: Iuri Barcelos, i.barceloscarneiromrocha@tudelft.nl
 * Date:   May 2019
 * 
 */

#ifndef NEURALUTILS_H
#define NEURALUTILS_H

#include <jem/base/System.h>
#include <jem/base/Array.h>
#include <jem/base/Float.h>
#include <jem/base/array/tensor.h>
#include <jem/util/Properties.h>
#include <jem/numeric/algebra/utilities.h>
#include <jive/Array.h>
#include <jem/base/Tuple.h>

extern "C"
{
  #include  <math.h>
}

using namespace jem;

using jem::Array;
using jem::String;
using jem::util::Properties;

using jive::Vector;
using jive::Matrix;
using jive::IntVector;
using jive::IdxVector;
using jem::Tuple;
using jem::idx_t;

//-----------------------------------------------------------------------
//   typedefs
//-----------------------------------------------------------------------


namespace NeuralUtils
{
  typedef void (*ActivationFunc)

    ( Matrix& x );
    
  typedef void (*ActivationGrad)

    ( Matrix& x );

  typedef void (*ActivationHess)

    ( Matrix& x );

  typedef double (*LossFunc)

    ( const Matrix& pred,
      const Matrix& real );

  typedef void (*LossGrad)
   
    (       Matrix& pred,
      const Matrix& real );

  typedef void (*InitFunc)

    (       Matrix&     w, 
      const Properties& globdat );

  ActivationFunc getActivationFunc

    ( const String&    name );

  ActivationGrad getActivationGrad

    ( const String&    name );

  ActivationHess getActivationHess

    ( const String&    name );

  LossFunc       getLossFunc

    ( const String&    name );

  LossGrad       getLossGrad

    ( const String&    name );

  InitFunc       getInitFunc

    ( const String&    name );

  namespace Activations
  {
    inline void evalIdentityFunc

      ( Matrix& x )

    {}

    inline void evalSigmoidFunc

      ( Matrix& x )

    {
      x = 1. / ( 1. + exp(-x) );
    }

    inline void evalTanhFunc

      ( Matrix& x )

    {
      //x = 2. / ( 1. + exp(-2.*x) ) - 1.;
      x = tanh(x);
    }

    inline void evalReLUFunc

      ( Matrix& x )

    {
      TensorIndex i, j;
      x(i,j) = where ( x(i,j) > 0., x(i,j), 0. );
    }

    inline void evalAbsFunc

      ( Matrix& x )

    {
      TensorIndex i, j;
      x(i,j) = where ( x(i,j) >= 0., x(i,j), abs(x(i,j)) ); 
    }

    inline void evalSoftPlusFunc

      ( Matrix& x )

    {
        x = log ( 1. + exp ( x) );  
    }
    
    inline void evalSoftPlusDiagFunc

      ( Matrix& x )

    {
      if ( x.size(0) == x.size(1) )
      {
        TensorIndex i, j;
        x(i,j) = where ( i == j, log ( 1. + exp( x(i,j) ) ), x(i,j) );
      }
      else if ( x.size(1) == 6 )
      {
        idx_t nBlocks = x.size(0);
        for ( idx_t nb = 0; nb < nBlocks; nb++ )
        {
           x(nb,0) = log ( 1. + exp ( x(nb,0) ));
           x(nb,3) = log ( 1. + exp ( x(nb,3) ));
           x(nb,5) = log ( 1. + exp ( x(nb,5) ));
        }
      }
    }
    
    inline void evalLeakyReLUDiagFunc

      ( Matrix& x )

    {
      if ( x.size(0) == x.size(1) )
      {
        TensorIndex i, j;
        x(i,j) = where ( i == j, abs ( x(i,j) ), x(i,j) );
      }
      else if ( x.size(1) == 6 )
      {
        idx_t nBlocks = x.size(0);
        for ( idx_t nb = 0; nb < nBlocks; nb++ )
        {
           x(nb,0) = abs ( x(nb,0) );
           x(nb,3) = abs ( x(nb,3) );
           x(nb,5) = abs ( x(nb,5) );
        }
      }
    }

    inline void evalMinusFunc

      ( Matrix& x )

    {
      TensorIndex i, j;
      x(i,j) = where ( i == j, x(i,j) - 1.0, x(i,j) );
    }

    inline void evalSinFunc

      ( Matrix& x )

    { 
      x = sin ( 100. * x );
    }

    inline void evalIdentityGrad

      ( Matrix& x )

    {
      x = 1.;
    }

    inline void evalMinusGrad

      ( Matrix& x )

    {
      x = 1.;     
    }

    inline void evalSigmoidGrad

      ( Matrix& x )

    {
      Matrix temp = x.clone();
      evalSigmoidFunc(temp);

      x = temp * ( 1. - temp );
    }

    inline void evalTanhGrad

      ( Matrix& x )

    {
      Matrix temp = x.clone();
      evalTanhFunc ( temp );

      x = 1. - temp * temp;
    }

    inline void evalReLUGrad

      ( Matrix& x )

    {
      TensorIndex i, j;

      x(i,j) = where ( x(i,j) > 0., 1., 0. );
    }

    inline void evalAbsGrad

      ( Matrix& x )

    {
      TensorIndex i, j;
      x(i,j) = where ( x(i,j) >= 0., 1., -1. );
    }

    inline void evalSoftPlusGrad

      ( Matrix& x )

    {
       x = 1. / ( 1. + exp ( -x ) );
      
    }
    
    inline void evalSoftPlusDiagGrad

      ( Matrix& x )

    {
      if ( x.size(1) == 6 )
      { 
        Matrix temp = x.clone();
	temp = 1.0;

        for ( idx_t i = 0; i < x.size(0); i++ )
        {
	  temp(i,0) =  1. / ( 1. + exp(-x(i,0)) );
          temp(i,3) = 1. / ( 1. + exp(-x(i,3)) );
	  temp(i,5) = 1. / ( 1. + exp(-x(i,5)) ); 
        }

	x = temp;
      }
      else if ( x.size(0) == x.size(1) ) 
      {
        Matrix temp = x.clone();
	temp = 1.0;

        TensorIndex i, j;
        temp(i,j) = where ( i == j, 1. / ( 1. + exp ( -x(i,j)) ), 1.0 ); 

	 x = temp;
      }
    } 
    
    inline void evalLeakyReLUDiagGrad

      ( Matrix& x )

    {
      if ( x.size(1) == 6 )
      { 
        Matrix temp = x.clone();
	temp = 1.0;

        for ( idx_t i = 0; i < x.size(0); i++ )
        {
	  if ( x(i,0) >= 0 ) 
	  {
   	    temp(i,0) = 1.;
	  }
	  else
	  {
	    temp(i,0) = -1.;
	  }

	  if ( x(i,3) >= 0 ) 
	  {
   	    temp(i,3) = 1.;
	  }
	  else
	  {
	    temp(i,3) = -1.;
	  }

	  if ( x(i,5) >= 0 ) 
	  {
   	    temp(i,5) = 1.;
	  }
	  else
	  {
	    temp(i,5) = -1.;
	  }
        }

	x = temp;
      }
      else if ( x.size(0) == x.size(1) ) 
      {
        Matrix temp = x.clone();
	temp = 1.0;

        for ( idx_t ii = 0; ii < x.size(0); ii++ )
        {  
          if ( x(ii,ii) >= 0) 
          {
            temp (ii, ii ) = 1.; 
          }
          else
          {
            temp (ii, ii ) = -1.; 
          }
	}
	 x = temp;
      }
    }        

    inline void evalSinGrad

      ( Matrix& x )

    {
      Matrix temp = x.clone();
      x = 100. * cos ( 100. * temp );
    }

    inline void evalIdentityHess

      ( Matrix& x )

    {
      x = 0.0;
    }

    inline void evalSigmoidHess

      ( Matrix& x )

    {
      Matrix temp = x.clone();
      evalSigmoidFunc(temp);

      x = temp * ( 1. - temp ) * ( 1. - 2.*temp );
    }

    inline void evalTanhHess

      ( Matrix& x )

    {
      x = -2.0 * tanh(x)/cosh(x)/cosh(x);
    }

    inline void evalReLUHess

      ( Matrix& x )

    {
    }

    inline void evalAbsHess

      ( Matrix& x )

    {
    }

    inline void evalSoftPlusHess

      ( Matrix& x )

    {
    }
    
     inline void evalSoftPlusDiagHess

      ( Matrix& x )

    {
    }

    inline void evalSinHess

      ( Matrix& x )

    {
    }
  }

  namespace Losses
  {
    inline double evalSquaredErrorFunc

      ( const Matrix& pred,
	const Matrix& real )

    {
      return 0.5 * dot ( real - pred );
    }

    inline double evalRelativeSquaredErrorFunc

      ( const Matrix& pred,
	const Matrix& real )

    {

      return 0.5 * dot ( real - pred ) / dot ( real ) ;
    }

    inline void evalSquaredErrorGrad

      (       Matrix& pred,
	const Matrix& real )

    {
      pred -= real;
    }

    inline void evalRelativeSquaredErrorGrad

      (       Matrix& pred,
	const Matrix& real )

    {
      pred -= real;
      pred /= dot ( real );
    }

  }

  namespace Initializations
  {
	   void zeroInit

      (       Matrix&     w,
        const Properties& globdat );

           void glorotInit
 
      (       Matrix&     w,
        const Properties& globdat );

           void martensInit
 
      (       Matrix&     w,
        const Properties& globdat );

           void sutskeverInit

      (       Matrix&     w,
        const Properties& globdat );

           void orthoInit

      (       Matrix&     w,
        const Properties& globdat );

	   void heInit

      (       Matrix&    w,
        const Properties& globdat );

	   void heNormalInit

      (       Matrix&    w,
        const Properties& globdat );

	   void glorotExpInit

      (       Matrix&    w,
        const Properties& globdat );

  }

  void           shuffle

    (       IdxVector&  vec,
      const Properties& globdat );
}

#endif

