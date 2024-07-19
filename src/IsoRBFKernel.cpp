/*
 * Copyright (C) 2019 TU Delft. All rights reserved.
 *
 * Base class for covariance (distance) kernels (measures)
 * Parameters to be optimized are log-scaled versions of the
 * actual kernel hyperparameters. 
 *
 * Rasmussen, C. E. and Williams, C. K. I. Gaussian Processes
 * for Machine Learning. MIT Press, 2016. <www.gaussianprocess.org>
 *
 * Author: Iuri Barcelos, i.barceloscarneiromrocha@tudelft.nl
 * Date:   Oct 2019
 *
 */

#include <jem/base/Error.h>
#include <jem/base/IllegalInputException.h>
#include <jem/util/Properties.h>
#include <jive/util/XDofSpace.h>
#include <jive/util/DummyItemSet.h>
#include <jive/model/StateVector.h>
#include <jem/numeric/algebra/utilities.h>

#include "IsoRBFKernel.h"
#include "LearningNames.h"

using namespace jem;

using jive::util::DofSpace;
using jive::util::XDofSpace;
using jive::util::ItemSet;
using jive::util::DummyItemSet;
using jive::model::StateVector;

//=======================================================================
//   class IsoRBFKernel
//=======================================================================

const char* IsoRBFKernel::INITVARIANCE   = "variance";
const char* IsoRBFKernel::INITLENSCALE   = "lengthScale";
const char* IsoRBFKernel::INITNOISE      = "noise";
const char* IsoRBFKernel::VARIANCEBOUNDS = "sBounds";
const char* IsoRBFKernel::LENSCALEBOUNDS = "lBounds";
const char* IsoRBFKernel::NOISEBOUNDS    = "nBounds";

//-----------------------------------------------------------------------
//   constructors and destructor
//-----------------------------------------------------------------------

IsoRBFKernel::IsoRBFKernel

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

  : Kernel ( name, props, conf, globdat ) 

{
  using jem::IllegalInputException;

  String context = "IsoRBF kernel initialization";

  Ref<XDofSpace>    dofs  = XDofSpace::get ( globdat, context );
  Ref<DummyItemSet> items = dynamicCast<DummyItemSet> (
			    ItemSet::get ( LearningNames::HYPERSET, globdat, context ) );
  
  idx_t doftype = dofs->findType ( LearningNames::HYPERDOF );

  iDofs_.resize ( varCount() );
  iDofs_ = -1;

  for ( idx_t v = 0; v < varCount(); ++v )
  {
    iDofs_[v] = dofs->addDof ( items->addItem(), doftype );
  }

  lbnd_.resize ( varCount() ); 
  ubnd_.resize ( varCount() ); 

  lbnd_ = log(1.e-5);
  ubnd_ = log(1.e+5);

  System::out() << "Created IsoRBFKernel.\n";
}

IsoRBFKernel::~IsoRBFKernel()
{}

//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------

void IsoRBFKernel::configure

  ( const Properties& props,
    const Properties& globdat )
{
  String context = "IsoRBFKernel::configure";

  System::out() << "Configure RBF Kernel.\n";
  
  System::out() << "Props passed to IsoRBFKernel " << props << "\n";

  Ref<XDofSpace> dofs  = XDofSpace::get ( globdat, context );

  Vector bnds;

  if ( props.find ( bnds, VARIANCEBOUNDS ) )
  {
    lbnd_[0] = log ( min ( bnds ) );
    ubnd_[0] = log ( max ( bnds ) );
  }

  if ( props.find ( bnds, LENSCALEBOUNDS ) )
  {
    lbnd_[1] = log ( min ( bnds ) );
    ubnd_[1] = log ( max ( bnds ) );
  }

  if ( props.find ( bnds, NOISEBOUNDS ) )
  {
    lbnd_[2] = log ( min ( bnds ) );
    ubnd_[2] = log ( max ( bnds ) );
  }

  Vector init ( varCount() );
  System::out() << "Upper bounds at the moment " << ubnd_ << "\n";
  init = exp ( 0.5 * ( ubnd_ - lbnd_ ) + lbnd_ );

  props.find ( init[0], INITVARIANCE );
  props.find ( init[1], INITLENSCALE );
  props.find ( init[2], INITNOISE    );

  Vector vars;
  StateVector::get ( vars, dofs, globdat );

  for ( idx_t v = 0; v < varCount(); ++v )
  {
    if ( log(init[v]) < lbnd_[v] || log(init[v]) > ubnd_[v] )
    {
      throw IllegalInputException ( JEM_FUNC,
        "IsoRBF kernel: out-of-bounds initial value" );
    }

    vars[iDofs_[v]] = ( log(init[v]) - lbnd_[v] ) / ( ubnd_[v] - lbnd_[v] );
  }

}

//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------

void IsoRBFKernel::getConfig

  ( const Properties& props,
    const Properties& globdat ) const
{}

//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------

void IsoRBFKernel::update 

  ( const Properties& globdat )

{
  Ref<DofSpace> dofs = DofSpace::get ( globdat, "IsoRBF kernel update" );

  Vector vars;
  StateVector::get ( vars, dofs, globdat );

  sigF2_ = exp ( vars[iDofs_[0]] * ( ubnd_[0] - lbnd_[0] ) + lbnd_[0] );
  l_     = exp ( vars[iDofs_[1]] * ( ubnd_[1] - lbnd_[1] ) + lbnd_[1] );
  sigN2_ = exp ( vars[iDofs_[2]] * ( ubnd_[2] - lbnd_[2] ) + lbnd_[2] );

  l2_    = l_ * l_;

 /* System::out() << "vars " << vars << "\n";
  System::out() << "ubnd " << ubnd_ << " lbnd " << lbnd_ << "\n";
  System::out() << "sigf2 " << sigF2_ << "\n";
  System::out() << "l " << l_ << "\n";
  System::out() << "sign2 " << sigN2_ << "\n";*/
}

//-----------------------------------------------------------------------
//   eval
//-----------------------------------------------------------------------

double IsoRBFKernel::eval

  ( const Vector& xp ) const

{
  return sigF2_ + sigN2_;
}

//-----------------------------------------------------------------------
//   eval
//-----------------------------------------------------------------------

double IsoRBFKernel::eval

  ( const Vector& xp,
    const Vector& xq  ) const

{
//  System::out() << "l2_ " << l2_ << "\n";
  return sigF2_ * exp ( - dot ( xp - xq ) / 2. / l2_ );
}

//-----------------------------------------------------------------------
//   eval
//-----------------------------------------------------------------------

Vector IsoRBFKernel::eval
 
  ( const Vector& xp,
    const Matrix& xq  ) const

{
  idx_t nq = xq.size ( 1 );

  Vector k ( nq );
  k = 0.0;

  for ( idx_t q = 0; q < nq; ++q )
  {
    k[q] = eval ( xp, xq(ALL,q) );
  }

  return k;
}

//-----------------------------------------------------------------------
//   eval
//-----------------------------------------------------------------------

Matrix IsoRBFKernel::eval

  ( const Matrix& xp ) const

{
  idx_t n = xp.size(1);

  Matrix K ( n, n );
  K = 0.0;

  for ( idx_t p = 0; p < n; ++p )
  {
    for ( idx_t q = 0; q < n; ++q )
    {
      if ( p == q )
      {
	K(p,q) = sigF2_ + sigN2_;
      }
      else
      {
	K(p,q) = eval ( xp(ALL,p), xp(ALL,q) );
      }
    }
  }

  return K;
}

//-----------------------------------------------------------------------
//   eval
//-----------------------------------------------------------------------

Matrix IsoRBFKernel::eval

  ( const Matrix& xp,
    const Matrix& xq  ) const

{
  idx_t np = xp.size(1);
  idx_t nq = xq.size(1);

  Matrix K ( np, nq );
  K = 0.0;

  for ( idx_t p = 0; p < np; ++p )
  {
    for ( idx_t q = 0; q < nq; ++q )
    {
      K(p,q) = eval ( xp(ALL,p), xq(ALL,q) );
    }
  }

  return K;
}

//-----------------------------------------------------------------------
//   evalDerivs
//-----------------------------------------------------------------------

Matrix IsoRBFKernel::evalDerivs

  ( const Vector& xp,
    const Matrix& xq ) const

{
  idx_t s = xp.size( );
  idx_t n = xq.size(1);

  Matrix K ( s, n );
  K = 0.0;

  for ( idx_t i = 0; i < n; ++i )
  {
    double k = eval ( xp, xq(ALL,i) );

    //K(ALL,i) = k / l2_ * ( xp - xq(ALL,i) );
    K(ALL,i) = k / l2_ * ( xq(ALL,i) - xp );
  }

  return K;
}

//-----------------------------------------------------------------------
//   gradients
//-----------------------------------------------------------------------

void IsoRBFKernel::gradients

  (       Cubix&  G,
    const Matrix& xp )

{
  idx_t n = xp.size(1);

  G.resize ( varCount(), n, n );
  G = 0.0;

  Matrix diff ( n, n );
  diff = 0.0;

  for ( idx_t p = 0; p < n; ++p )
  {
    for ( idx_t q = 0; q < n; ++q )
    {
      diff(p,q) = dot ( xp(ALL,p) - xp(ALL,q ) );
    }
  }

  // NB: These derivatives do not appear correct, but the additional
  // terms sigF2_, l_ (cancelled with the l3 in the denominator)
  // and sigN2_ appear because of the log-scaling of trainable
  // parameters. The ubnd, lbnd terms appear because of the [0,1]
  // scaling on top of the log-scaling

  G(0,ALL,ALL) = sigF2_ * exp ( - diff / 2. / l2_ ) * ( ubnd_[0] - lbnd_[0] );

  G(1,ALL,ALL) = sigF2_ / l2_ * diff * 
                 exp ( - diff / 2. / l2_ ) * ( ubnd_[1] - lbnd_[1] );

  for ( idx_t p = 0; p < n; ++p )
  {
    G(2,p,p) = sigN2_ * ( ubnd_[2] - lbnd_[2] );
  }
}

//-----------------------------------------------------------------------
//  geEval
//-----------------------------------------------------------------------

Matrix IsoRBFKernel::geEval

  ( const Vector&        xp      ) const

{
  idx_t D = xp.size();

  Matrix k ( D+1, D+1 );
  k = 0.0;

  k(0,0) = eval ( xp );

  for ( idx_t d = 1; d < D+1; ++d )
  {
    k(d,d) = eval ( xp ) / l2_;
  }

  return k;
}

//-----------------------------------------------------------------------
//  geEval
//-----------------------------------------------------------------------

Matrix IsoRBFKernel::geEval

  ( const Vector&        xp,
    const Vector&        xq      ) const

{
  idx_t D = xp.size();

  Matrix k ( D+1, D+1 );
  k = 0.0;

  k = eval ( xp, xq );

  k(0,slice(1,END)) *= ( xq - xp ) / l2_;
  k(slice(1,END),0) *= ( xq - xp ) / l2_;

  for ( idx_t d = 1; d < D+1; ++d )
  {
    k(d,slice(1,END)) *= ( xq - xp ) * ( xp[d-1] - xq[d-1] ) / l2_ / l2_;
    k(d,d) += eval ( xp, xq ) / l2_;
  }

  return k;
}

//-----------------------------------------------------------------------
//  geEval
//-----------------------------------------------------------------------

Matrix IsoRBFKernel::geEval

  ( const Vector&        xp, 
    const Matrix&        xq      ) const     

{
  idx_t D = xp.size();
  idx_t n = xq.size(1);

  Matrix K ( n*(D+1), (D+1) );
  K = 0.0;

  Matrix k ( D+1, D+1 );

  for ( idx_t q = 0; q < n; ++q )
  {
    k = geEval ( xq(ALL,q), xp );

    // Copy blocks to the correct places

    idx_t qs = n + q*D;

    K(q,0) = k(0,0);
    K(q,slice(1,END))   = -k(0,slice(1,END));
    K(slice(qs,qs+D),0) =  k(0,slice(1,END));

    K(slice(qs,qs+D),slice(1,END)) = k(slice(1,END),slice(1,END));
  }

  return K;
}

//-----------------------------------------------------------------------
//  geEval
//-----------------------------------------------------------------------

Matrix IsoRBFKernel::geEval

  ( const Matrix&        xp      ) const

{
  idx_t D = xp.size(0);
  idx_t m = xp.size(1);

  Matrix K ( m*(D+1), m*(D+1) );
  K = 0.0;

  Matrix k ( D+1, D+1 );

  for ( idx_t p = 0; p < m; ++p )
  {
    for ( idx_t q = 0; q < m; ++q )
    {
      if ( p == q )
      {
        k = geEval ( xp(ALL,p) );
      }
      else
      {
        k = geEval ( xp(ALL,q), xp(ALL,p) );
      }

      // Copy blocks to the correct places

      idx_t ps = m + p*D;
      idx_t qs = m + q*D;

      K(p,q) = k(0,0);
      K(p,slice(qs,qs+D)) = K(slice(qs,qs+D),p) = k(0,slice(1,END));
      K(slice(ps,ps+D),slice(qs,qs+D)) = k(slice(1,END),slice(1,END));
    }
  }

  return K;
}

//-----------------------------------------------------------------------
//  geEval
//-----------------------------------------------------------------------

Matrix IsoRBFKernel::geEval

  ( const Matrix&        xp,
    const Matrix&        xq      ) const

{
  idx_t D = xp.size(0);
  idx_t m = xp.size(1);
  idx_t n = xq.size(1);

  Matrix K ( m*(D+1), n*(D+1) );
  K = 0.0;

  // TODO
  throw Error ( JEM_FUNC, "geEval(matrix,matrix) not implemented yet" );

  return K;
}

//-----------------------------------------------------------------------
//   gradients
//-----------------------------------------------------------------------

void IsoRBFKernel::geGradients

  (       Cubix&  G,
    const Matrix& xp )

{
  idx_t D = xp.size(0);
  idx_t m = xp.size(1);

  G.resize ( varCount(), m*(D+1), m*(D+1) );
  G = 0.0;

  Cubix g ( varCount(), D+1, D+1 );

  for ( idx_t p = 0; p < m; ++p )
  {
    for ( idx_t q = 0; q < m; ++q )
    {
      if ( p == q )
      {
        g = grads_ ( xp(ALL,p) );
      }
      else
      {
        g = grads_ ( xp(ALL,q), xp(ALL,p) );
      }

      // Copy blocks to the correct places

      idx_t ps = m + p*D;
      idx_t qs = m + q*D;

      G(ALL,p,q) = g(ALL,0,0);
      G(ALL,p,slice(qs,qs+D)) = G(ALL,slice(qs,qs+D),p) = g(ALL,0,slice(1,END));
      G(ALL,slice(ps,ps+D),slice(qs,qs+D)) = g(ALL,slice(1,END),slice(1,END));
    }
  }
}

//-----------------------------------------------------------------------
//  varCount
//-----------------------------------------------------------------------

idx_t IsoRBFKernel::varCount () const

{
  return 3;
}

//-----------------------------------------------------------------------
//  getVars
//-----------------------------------------------------------------------

Vector IsoRBFKernel::getVars () const

{
  Vector vars ( 3 );

  vars[0] = sigF2_;
  vars[1] = l_;
  vars[2] = sigN2_;

  return vars;
}

//-----------------------------------------------------------------------
//  getNoise
//-----------------------------------------------------------------------

double IsoRBFKernel::getNoise () const

{
  return sqrt(sigN2_);
}

//-----------------------------------------------------------------------
//  clone 
//-----------------------------------------------------------------------

Ref<Kernel> IsoRBFKernel::clone ( ) const

{
  return newInstance<IsoRBFKernel> ( *this );
}

//-----------------------------------------------------------------------
//  grads_
//-----------------------------------------------------------------------

Cubix IsoRBFKernel::grads_

  ( const Vector& xp ) const

{
  idx_t D = xp.size();

  Cubix g ( varCount(), D+1, D+1 );
  g = 0.0;

  g(0,0,0) = sigF2_ * ( ubnd_[0] - lbnd_[0] );
  g(2,0,0) = sigN2_ * ( ubnd_[2] - lbnd_[2] );

  for ( idx_t d = 1; d < D+1; ++d )
  {
    g(0,d,d) = sigF2_ / l2_ * ( ubnd_[0] - lbnd_[0] );
    g(1,d,d) = -2.0 * sigF2_ / l2_ * ( ubnd_[1] - lbnd_[1] );
  }

  return g;
}

//-----------------------------------------------------------------------
//  grads_
//-----------------------------------------------------------------------

Cubix IsoRBFKernel::grads_

  ( const Vector& xp,
    const Vector& xq ) const

{
  idx_t D = xp.size();

  Cubix g ( varCount(), D+1, D+1 );
  g = 0.0;

  double sqnrm = dot ( xp - xq );
  double eterm = exp( -sqnrm / 2. / l2_ );

  double sbnd = ( ubnd_[0] - lbnd_[0] );
  double lbnd = ( ubnd_[1] - lbnd_[1] );

  // (Kgg)'

  g(0,0,0) = sigF2_ * eterm * sbnd;
  g(1,0,0) = sqnrm * sigF2_ * eterm * lbnd / l2_;

  // (Kgd)' = (Kdg)'

  Vector diff ( xp - xq );

  g(0,0,slice(1,END)) = g(0,slice(1,END),0) = -sigF2_ * eterm * sbnd / l2_ * diff;
  g(1,0,slice(1,END)) = g(1,slice(1,END),0) = -sigF2_ * eterm * lbnd / 
                                               l2_ / l2_ * diff * ( -2.0 * l2_ + sqnrm );

  // (Kdd)'

  for ( idx_t d = 1; d < D+1; ++d )
  {
    double diff2 = xp[d-1] - xq[d-1];

    g(0,d,slice(1,END)) = - diff * diff2 * sigF2_ * sbnd * eterm / l2_ / l2_;
    g(0,d,d)            = sigF2_ * sbnd * eterm * ( l2_ - diff2*diff2 ) / l2_ / l2_;

    g(1,d,slice(1,END)) = - diff * diff2 * sigF2_ * lbnd * eterm / 
                            l2_ / l2_ / l2_ * ( sqnrm - 4.0*l2_ );
    g(1,d,d)            = sigF2_ * lbnd * eterm / l2_ / l2_ / l2_ *
                          ( 4.0 * l2_ * diff2*diff2 - 2.0*l2_*l2_ + sqnrm*l2_ - sqnrm*diff2*diff2 );
  }

  return g;
}
