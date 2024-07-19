/*
 * Copyright (C) 2019 TU Delft. All rights reserved.
 *
 * Utility functions for artificial neural networks.
 *
 * Author: Iuri Barcelos, i.barceloscarneiromrocha@tudelft.nl
 * Date:   May 2019
 * 
 */

#include <cstdlib>

#include <jem/base/System.h>
#include <jem/base/array/select.h>
#include <jem/base/array/utilities.h>
#include <jem/base/array/operators.h>
#include <jem/base/PrecheckException.h>
#include <jem/base/Error.h>
#include <jem/base/Float.h>
#include <jem/numeric/utilities.h>
#include <jem/numeric/algebra/matmul.h>
#include <jem/numeric/algebra/utilities.h>

#include <jive/util/Random.h>

#include "NeuralUtils.h"
#include "utilities.h"

using namespace jem;
using namespace NeuralUtils;

using jem::numeric::matmul;
using jem::numeric::norm2;
using jem::ALL;

using jive::util::Random;

//-----------------------------------------------------------------------
//   getActivationFunc
//-----------------------------------------------------------------------

ActivationFunc NeuralUtils::getActivationFunc

  ( const String& name )

{
  if      ( name.equalsIgnoreCase ( "identity" ) ||
            name.equalsIgnoreCase ( "linear"   )    )
  {
    return & NeuralUtils::Activations::evalIdentityFunc;
  }
  else if ( name.equalsIgnoreCase ( "sigmoid" ) )
  {
    return & NeuralUtils::Activations::evalSigmoidFunc;
  }
  else if ( name.equalsIgnoreCase ( "tanh"    ) )
  {
    return & NeuralUtils::Activations::evalTanhFunc;
  }
  else if ( name.equalsIgnoreCase ( "relu" ) )
  {
    return & NeuralUtils::Activations::evalReLUFunc;
  }
  else if ( name.equalsIgnoreCase ( "abs" ) )
  {
    return & NeuralUtils::Activations::evalAbsFunc;
  }
  else if ( name.equalsIgnoreCase ( "leakyreludiag" ) )
  {
    return & NeuralUtils::Activations::evalLeakyReLUDiagFunc;
  }
  else if ( name.equalsIgnoreCase ( "softplus" ) )
  {
    return & NeuralUtils::Activations::evalSoftPlusFunc;
  }
  else if ( name.equalsIgnoreCase ( "softplusdiag" ) )
  {
    return & NeuralUtils::Activations::evalSoftPlusDiagFunc;
  }  
  else if ( name.equalsIgnoreCase ( "sin" ) )
  {
    return & NeuralUtils::Activations::evalSinFunc;
  }
  else if ( name.equalsIgnoreCase ( "minus" ) )
  {
    return & NeuralUtils::Activations::evalMinusFunc;
  }
  else
  {
    throw Error ( JEM_FUNC, "Unknown activation function." );
  }

  return 0;
}

//-----------------------------------------------------------------------
//   getActivationGrad
//-----------------------------------------------------------------------

ActivationGrad NeuralUtils::getActivationGrad

  ( const String& name )

{
  if      ( name.equalsIgnoreCase ( "identity" ) ||
            name.equalsIgnoreCase ( "linear"   )    )
  {
    return & NeuralUtils::Activations::evalIdentityGrad;
  }
  else if ( name.equalsIgnoreCase ( "sigmoid" ) )
  {
    return & NeuralUtils::Activations::evalSigmoidGrad;
  }
  else if ( name.equalsIgnoreCase ( "tanh"    ) )
  {
    return & NeuralUtils::Activations::evalTanhGrad;
  }
  else if ( name.equalsIgnoreCase ( "relu" ) )
  {
    return & NeuralUtils::Activations::evalReLUGrad;
  }
  else if ( name.equalsIgnoreCase ( "abs" ) )
  {
    return & NeuralUtils::Activations::evalAbsGrad;
  }
  else if ( name.equalsIgnoreCase ( "leakyreludiag" ) )
  {
    return & NeuralUtils::Activations::evalLeakyReLUDiagGrad;
  } 
  else if ( name.equalsIgnoreCase ( "softplus" ) )
  {
    return & NeuralUtils::Activations::evalSoftPlusGrad;
  }
  else if ( name.equalsIgnoreCase ( "softplusdiag" ) )
  {
    return & NeuralUtils::Activations::evalSoftPlusDiagGrad;
  }  
  else if ( name.equalsIgnoreCase ( "sin" ) )
  {
    return & NeuralUtils::Activations::evalSinGrad;
  }
  else if ( name.equalsIgnoreCase ( "minus" ) )
  {
    return & NeuralUtils::Activations::evalMinusGrad;
  }
  else
  {
    throw Error ( JEM_FUNC, "Unknown activation function." );
  }

  return 0;
}

//-----------------------------------------------------------------------
//   getActivationHess
//-----------------------------------------------------------------------

ActivationGrad NeuralUtils::getActivationHess

  ( const String& name )

{
  if      ( name.equalsIgnoreCase ( "identity" ) ||
            name.equalsIgnoreCase ( "linear"   )    )
  {
    return & NeuralUtils::Activations::evalIdentityHess;
  }
  else if ( name.equalsIgnoreCase ( "sigmoid" ) )
  {
    return & NeuralUtils::Activations::evalSigmoidHess;
  }
  else if ( name.equalsIgnoreCase ( "tanh"    ) )
  {
    return & NeuralUtils::Activations::evalTanhHess;
  }
  else if ( name.equalsIgnoreCase ( "relu" ) )
  {
    return & NeuralUtils::Activations::evalReLUHess;
  }
  else if ( name.equalsIgnoreCase ( "abs" ) )
  {
    return & NeuralUtils::Activations::evalAbsHess;
  }
  else if ( name.equalsIgnoreCase ( "softplus" ) )
  {
    return & NeuralUtils::Activations::evalSoftPlusHess;
  }
  else if ( name.equalsIgnoreCase ( "softplusdiag" ) )
  {
    return & NeuralUtils::Activations::evalSoftPlusDiagHess;
  }  
  else if ( name.equalsIgnoreCase ( "sin" ) )
  {
    return & NeuralUtils::Activations::evalSinHess;
  }
  else if ( name.equalsIgnoreCase ( "minus" ) )
  {
    return & NeuralUtils::Activations::evalIdentityHess;
  }
  else
  {
    throw Error ( JEM_FUNC, "Unknown activation function." );
  }

  return 0;
}

//-----------------------------------------------------------------------
//   getLossFunc
//-----------------------------------------------------------------------

LossFunc NeuralUtils::getLossFunc

  ( const String& name )

{
  if ( name.equalsIgnoreCase ( "squarederror" ) )
  {
    return & NeuralUtils::Losses::evalSquaredErrorFunc;
  }
  else if ( name.equalsIgnoreCase ( "relativesquarederror" ) )
  {
    return & NeuralUtils::Losses::evalRelativeSquaredErrorFunc;
  }
  else
  {
    throw Error ( JEM_FUNC, "Unknown loss function." );
  }

  return 0;
}

//-----------------------------------------------------------------------
//   getLossGrad
//-----------------------------------------------------------------------

LossGrad NeuralUtils::getLossGrad

  ( const String& name )

{
  if ( name.equalsIgnoreCase ( "squarederror" ) )
  {
    return & NeuralUtils::Losses::evalSquaredErrorGrad;
  }
  else if ( name.equalsIgnoreCase ( "relativesquarederror" ) )
  {
    return & NeuralUtils::Losses::evalRelativeSquaredErrorGrad;
  }
  else
  {
    throw Error ( JEM_FUNC, "Unknown loss function." );
  }

  return 0;
}

//-----------------------------------------------------------------------
//   getInitFunc
//-----------------------------------------------------------------------

InitFunc NeuralUtils::getInitFunc

  ( const String& name )

{
  if      ( name.equalsIgnoreCase  ( "zeros" ) )
  {
    return & NeuralUtils::Initializations::zeroInit;
  }
  else if ( name.equalsIgnoreCase ( "glorot" ) )
  {
    return & NeuralUtils::Initializations::glorotInit;
  }
  else if ( name.equalsIgnoreCase ( "martens" ) )
  {
    return & NeuralUtils::Initializations::martensInit;
  }
  else if ( name.equalsIgnoreCase ( "sutskever" ) )
  {
    return & NeuralUtils::Initializations::sutskeverInit;
  }
  else if ( name.equalsIgnoreCase ( "orthogonal" ) )
  {
    return & NeuralUtils::Initializations::orthoInit;
  }
  else if ( name.equalsIgnoreCase ( "he" ) )
  {
    return & NeuralUtils::Initializations::heInit;
  }
  else if ( name.equalsIgnoreCase ( "henormal" ) )
  {
    return & NeuralUtils::Initializations::heNormalInit;
  }
  else if ( name.equalsIgnoreCase ( "glorotexp" ) )
  {
    return & NeuralUtils::Initializations::glorotExpInit;
  }
  else
  {
    throw Error ( JEM_FUNC, "Unknown initialization function." );
  }

  return 0;
}

//-----------------------------------------------------------------------
//   zeroInit
//-----------------------------------------------------------------------

void NeuralUtils::Initializations::zeroInit

  (       Matrix&     w,
    const Properties& globdat )

{
  w = 0.0;
}

//-----------------------------------------------------------------------
//   glorotInit
//-----------------------------------------------------------------------

void NeuralUtils::Initializations::glorotInit

  (       Matrix&     w,
    const Properties& globdat )

{
  // Variance-preserving weight initialization
  // Glorot, X. and Bengio, Y. (2010). Understanding the difficulty of
  // training deep feedforward neural networks. In: Proceedings of 
  // AISTATS 2010, 9:249-256.

  Ref<Random> generator = Random::get ( globdat );

  idx_t nrow = w.size(0);
  idx_t ncol = w.size(1);

  double fac = sqrt ( 6. / ( nrow + ncol ) );

  for ( idx_t i = 0; i < nrow; ++i )
  {
    for ( idx_t j = 0; j < ncol; ++j )
    {
      double rand = 2.0*generator->next() - 1.0;
      if ( ncol == 6 )
      {
          if ( j == 0 || j == 3 || j == 5 )
          {
            w(i,j) =  0.5 + 0.01 * ( rand * fac);
          }
          else
          {
            w(i,j) = 0.01 * (rand * fac);
          } 
      }
      else
      {
        w(i,j) = rand*fac;
      }
    }
  }
}


//-----------------------------------------------------------------------
//   heInit
//-----------------------------------------------------------------------

void NeuralUtils::Initializations::heInit

  (       Matrix&     w,
    const Properties& globdat )

{
  // Variance-preserving weight initialization
  // Glorot, X. and Bengio, Y. (2010). Understanding the difficulty of
  // training deep feedforward neural networks. In: Proceedings of 
  // AISTATS 2010, 9:249-256.

  Ref<Random> generator = Random::get ( globdat );


  idx_t nrow = w.size(0);
  idx_t ncol = w.size(1);

  double fac = sqrt( 2.) * sqrt ( 6. / ( nrow + ncol ) );

  for ( idx_t i = 0; i < nrow; ++i )
  {
    for ( idx_t j = 0; j < ncol; ++j )
    {
      double rand = 2.0*generator->next() - 1.0;
      w(i,j) = rand * fac;
    }
  }
}

//-----------------------------------------------------------------------
//   heNormalInit
//-----------------------------------------------------------------------

void NeuralUtils::Initializations::heNormalInit

  (       Matrix&     w,
    const Properties& globdat )

{
  // Variance-preserving weight initialization
  // Glorot, X. and Bengio, Y. (2010). Understanding the difficulty of
  // training deep feedforward neural networks. In: Proceedings of 
  // AISTATS 2010, 9:249-256.

  Ref<Random> generator = Random::get ( globdat );


  idx_t nrow = w.size(0);
  idx_t ncol = w.size(1);

  double fac = sqrt( 2.) * sqrt ( 2. / ( nrow + ncol ) );

  for ( idx_t i = 0; i < nrow; ++i )
  {
    for ( idx_t j = 0; j < ncol; ++j )
    {
      double rand = generator->nextGaussian();
      w(i,j) = rand * fac;
    }
  }
}


//-----------------------------------------------------------------------
//   glorotExpInit
//-----------------------------------------------------------------------

void NeuralUtils::Initializations::glorotExpInit

  (       Matrix&     w,
    const Properties& globdat )

{
  // Variance-preserving weight initialization
  // Glorot, X. and Bengio, Y. (2010). Understanding the difficulty of
  // training deep feedforward neural networks. In: Proceedings of 
  // AISTATS 2010, 9:249-256.

  Ref<Random> generator = Random::get ( globdat );


  idx_t nrow = w.size(0);
  idx_t ncol = w.size(1);

  double fac = sqrt ( 6. / ( nrow + ncol ) );

  for ( idx_t i = 0; i < nrow; ++i )
  {
    for ( idx_t j = 0; j < ncol; ++j )
    {
      double rand = 2.0*generator->next() - 1.0;
      w(i,j) = exp ( rand * fac );
    }
  }
}

//-----------------------------------------------------------------------
//   martensInit
//-----------------------------------------------------------------------

void NeuralUtils::Initializations::martensInit

  (       Matrix&     w, 
    const Properties& globdat )

{
  // Sparse weight initialization
  // Martens, J. (2010). Deep learning via hessian-free optimization.
  // In: Proceedings of ICML-27, pp. 735-742

  Ref<Random> generator = Random::get ( globdat );

  idx_t nrow = w.size(0);
  idx_t ncol = w.size(1);

  IdxVector cols ( iarray ( ncol ) );

  for ( idx_t i = 0; i < nrow; ++i )
  {
    if ( ncol <= 15 )
    {
      for ( idx_t j = 0; j < ncol; ++j )
      {
	w(i,j) = generator->nextGaussian(); 
      }
    }
    else
    {
      shuffle ( cols, globdat );

      for ( idx_t j = 0; j < 15; ++j )
      {
        w(i,cols[j]) = generator->nextGaussian();
      }
    }
  }
}


//-----------------------------------------------------------------------
//   sutskeverInit
//-----------------------------------------------------------------------

void NeuralUtils::Initializations::sutskeverInit

  (       Matrix&     w, 
    const Properties& globdat )

{
  // Sparse initialization followed by eigenvalue scaling
  // Sutskever, I (2013). Training recurrent neural networks. PhD thesis,
  // University of Toronto

  martensInit ( w, globdat );

  // Throw an error if w is not square. 
  // NB: the recurrent weight matrix should always be square

  if ( w.size(0) != w.size(1) )
  {
    throw Error ( JEM_FUNC,
      "Eigenvalue scaling only available for square weight matrices" );
  }

  double fac = powerMethod ( w ) / 1.2;

  w = w / fac;
}

//-----------------------------------------------------------------------
//   orthoInit
//-----------------------------------------------------------------------

void NeuralUtils::Initializations::orthoInit

  (       Matrix&     w, 
    const Properties& globdat )

{
  // Orthogonal recurrent weight matrix initialization
  // Saxe, A. M., McClelland, J. L. and Ganguli, S. (2014). Exact 
  // solutions to the nonlinear dynamics of learning in deep linear
  // neural networks. arXiv:1312.6120v3

  Ref<Random> generator = Random::get ( globdat );

  idx_t nrow = w.size(0);
  idx_t ncol = w.size(1);

  for ( idx_t i = 0; i < nrow; ++i )
  {
    for ( idx_t j = 0; j < ncol; ++j )
    {
      double rand = 2.0*generator->next() - 1.0;
      w(i,j) = rand; 
    }
  }

  // Throw an error if w is not square. 
  // NB: the recurrent weight matrix should always be square

  if ( w.size(0) != w.size(1) )
  {
    throw Error ( JEM_FUNC, 
      "Orthogonalization only available for square weight matrices" );
  }

  Matrix Q, R;

  QR ( Q, R, w );

  w = Q;
}

//-----------------------------------------------------------------------
//   shuffle
//-----------------------------------------------------------------------

void NeuralUtils::shuffle

  ( IdxVector&        vec,
    const Properties& globdat )

{
  // Fisher-Yates shuffling algorithm

  Ref<Random> generator = Random::get ( globdat );

  int  temp;
  jem::lint rand;  

  int size = vec.size();

  while ( size > 1 )
  {
    rand = generator->next ( size - 1 );
    size--;

    temp = vec[size];
    vec[size] = vec[rand];
    vec[rand] = temp;
  }
}
