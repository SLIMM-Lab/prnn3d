#include <jem/numeric/utilities.h>
#include <jem/base/String.h>
#include <jem/base/Error.h>

#include "utilities.h"
#include "voigtUtilities.h"

using jem::String;
using jem::ALL;
using jem::Error;

// -----------------------------------------------------------------------
//   getPrincipalDirection2D
// -----------------------------------------------------------------------

double voigtUtilities::getPrincipalDirection2D

  ( const Vector&       stress )
{
  double d = stress[0] - stress[1];

  double theta;

  if ( jem::numeric::abs ( d ) < 1e-6 )
  {
    theta = 0.78539816339744830962;
  }
  else
  {
    theta = 0.5 * ::atan2 ( stress[2], 0.5 * d );
  }
  
  return theta;
}

//-----------------------------------------------------------------------
//    tensor2VoigtStress3D
//-----------------------------------------------------------------------

void voigtUtilities::tensor2VoigtStress3D

  ( Tuple<double,6> vec,
    const Matrix&   tens )

{
  vec[0] = tens(0,0);
  vec[1] = tens(1,1);
  vec[2] = tens(2,2);
  vec[3] = ( tens(0,1) + tens(1,0) ) * 0.5;
  vec[4] = ( tens(1,2) + tens(2,1) ) * 0.5;
  vec[5] = ( tens(2,0) + tens(0,2) ) * 0.5;
} 

//-----------------------------------------------------------------------
//    tensor2VoigtStress
//-----------------------------------------------------------------------

void voigtUtilities::tensor2VoigtStress

  ( const Vector& vec,
    const Matrix& tens )

{
  // no checks on sizes!
 
  const idx_t rank = tens.size ( 0 );

  if ( rank == 2 )
  {
    vec[0] = tens(0,0);
    vec[1] = tens(1,1);
    vec[2] = ( tens(0,1) + tens(1,0) ) / 2.;
  }
  else if ( rank == 3 )
  {
    vec[0] = tens(0,0);
    vec[1] = tens(1,1);
    vec[2] = tens(2,2);
    vec[3] = ( tens(0,1) + tens(1,0) ) / 2.;
    vec[4] = ( tens(1,2) + tens(2,1) ) / 2.;
    vec[5] = ( tens(2,0) + tens(0,2) ) / 2.;
  }
  else if ( rank == 1 )
  {
    vec[0] = tens(0,0);
  }
} 

//-----------------------------------------------------------------------
//    tensor2VoigtStrain
//-----------------------------------------------------------------------

void voigtUtilities::tensor2VoigtStrain

  ( const Vector& vec,
    const Matrix& tens )

{
  // no checks on sizes!
 
  const idx_t rank = tens.size ( 0 );

  if ( rank == 2 )
  {
    vec[0] = tens(0,0);
    vec[1] = tens(1,1);
    vec[2] = tens(0,1) + tens(1,0);
  }
  else if ( rank == 3 )
  {
    vec[0] = tens(0,0);
    vec[1] = tens(1,1);
    vec[2] = tens(2,2);
    vec[3] = tens(0,1) + tens(1,0);
    vec[4] = tens(1,2) + tens(2,1);
    vec[5] = tens(2,0) + tens(0,2);
  }
  else if ( rank == 1 )
  {
    vec[0] = tens(0,0);
  }
} 

//-----------------------------------------------------------------------
//    voigt2TensorStress
//-----------------------------------------------------------------------

void voigtUtilities::voigt2TensorStress

  ( const Matrix& tens,
    const Vector& vec  )

{
  const idx_t rank = tens.size ( 0 );

  if ( rank == 2 )
  {
    tens(0,0) = vec[0];
    tens(1,1) = vec[1];
    tens(0,1) = tens(1,0) = vec[2];
  }
  else if ( rank == 3 )
  {
    tens(0,0) = vec[0];
    tens(1,1) = vec[1];
    tens(2,2) = vec[2];
    tens(0,1) = tens(1,0) = vec[3];
    tens(1,2) = tens(2,1) = vec[4];
    tens(2,0) = tens(0,2) = vec[5];
  }
  else if ( rank == 1)
  {
    tens(0,0) = vec[0];
  }
}

//-----------------------------------------------------------------------
//    voigt2TensorStrain
//-----------------------------------------------------------------------

void voigtUtilities::voigt2TensorStrain

  ( const Matrix& tens,
    const Vector& vec  )

{
  const idx_t rank = tens.size ( 0 );

  if ( rank == 2 )
  {
    tens(0,0) = vec[0];
    tens(1,1) = vec[1];
    tens(0,1) = tens(1,0) = .5 * vec[2];
  }
  else if ( rank == 3 )
  {
    tens(0,0) = vec[0];
    tens(1,1) = vec[1];
    tens(2,2) = vec[2];
    tens(0,1) = tens(1,0) = .5 * vec[3];
    tens(1,2) = tens(2,1) = .5 * vec[4];
    tens(2,0) = tens(0,2) = .5 * vec[5];
  }
  else if ( rank == 1)
  {
    tens(0,0) = vec[0];
  }
}

//-----------------------------------------------------------------------
//    multiplyVoigtNormal
//-----------------------------------------------------------------------

void voigtUtilities::multiplyVoigtNormal

  ( const Vector& sign,
    const Vector& sig,
    const Vector& n )

{
  idx_t rank     =   n.size();

  JEM_ASSERT ( sign.size() == rank );

  if ( rank == 2 )
  {
    JEM_ASSERT( sig.size() == 3 );

    sign[0] = sig[0]*n[0] + sig[2]*n[1];
    sign[1] = sig[2]*n[0] + sig[1]*n[1];
  }
  else
  {
    JEM_ASSERT( sig.size() == 6 );

    sign[0] = sig[0]*n[0] + sig[3]*n[1] + sig[5]*n[2];
    sign[1] = sig[3]*n[0] + sig[1]*n[1] + sig[4]*n[2];
    sign[2] = sig[5]*n[0] + sig[4]*n[1] + sig[2]*n[2];
  }
}

//-----------------------------------------------------------------------
//    multiplyVoigtNormal (overloaded matrix version)
//-----------------------------------------------------------------------

void voigtUtilities::multiplyVoigtNormal

  ( const Matrix& dbn,
    const Matrix& db,
    const Vector& n )

{
  idx_t dofCount = db.size(1);

  JEM_ASSERT ( dbn.size(0) == n.size() );
  JEM_ASSERT ( dbn.size(1) == dofCount );

  for ( idx_t i = 0; i < dofCount; ++i )
  {
    multiplyVoigtNormal( dbn(ALL,i), db(ALL,i), n );
  }
}


//-----------------------------------------------------------------------
//   getVectorRotationMat
//-----------------------------------------------------------------------

void voigtUtilities::getVectorRotationMat

  ( const Matrix&       q,
    const Vector&       normal )

{
  // get matrix to rotate vectors to frame aligned with unit normal vector
  // only implemented for 2D

  // normal = [ cos(theta); sin(theta) ]
  // [ v_n; v_s ] = q * [ v_x; v_y ]

  JEM_ASSERT ( normal.size() == 2 );
  JEM_ASSERT ( q.size(0)     == 2 );
  JEM_ASSERT ( q.size(1)     == 2 );

  q(0,0) =  normal[0];   q(0,1) = normal[1];
  q(1,0) = -normal[1];   q(1,1) = normal[0];
}

//-----------------------------------------------------------------------
//   getStrainRotationMat
//-----------------------------------------------------------------------

void voigtUtilities::getStrainRotationMat

  ( const Matrix&       q,
    const Vector&       normal )

{
  // get matrix to rotate strain to frame aligned with unit normal vector
  // only implemented for 2D

  // normal = [ cos(theta); sin(theta) ]
  // [ e_nn; e_ee; e_ne ] = q * [ e_xx; e_yy; e_xy ]

  JEM_ASSERT ( normal.size() == 2 );
  JEM_ASSERT ( q.size(0)     == 3 );
  JEM_ASSERT ( q.size(1)     == 3 );

  double c2 = normal[0]*normal[0];
  double s2 = normal[1]*normal[1];
  double sc = normal[1]*normal[0];

  q(0,0) = c2;     q(0,1) =  s2;     q(0,2) = -sc;
  q(1,0) = s2;     q(1,1) =  c2;     q(1,2) =  sc;
  q(2,0) = 2.*sc;  q(2,1) = -2.*sc;  q(2,2) = c2-s2;
}

//-----------------------------------------------------------------------
//   getStressRotationMat
//-----------------------------------------------------------------------

void voigtUtilities::getStressRotationMat

  ( const Matrix&       q,
    const Vector&       normal )
{
  // get matrix to rotate stress to frame aligned with unit normal vector
  // only implemented for 2D

  // normal = [ cos(theta); sin(theta) ]
  // [ s_nn; s_ss; s_ns ] = q * [ s_xx; s_yy; s_xy ]

  JEM_ASSERT ( normal.size() == 2 );
  JEM_ASSERT ( q.size(0)     == 3 );
  JEM_ASSERT ( q.size(1)     == 3 );

  double c2 = normal[0]*normal[0];
  double s2 = normal[1]*normal[1];
  double sc = normal[1]*normal[0];

  q(0,0) = c2;   q(0,1) =  s2;   q(0,2) = -2.*sc;
  q(1,0) = s2;   q(1,1) =  c2;   q(1,2) =  2.*sc;
  q(2,0) = sc;   q(2,1) = -sc;   q(2,2) = c2-s2;
}

//-----------------------------------------------------------------------
//   transformStress
//-----------------------------------------------------------------------

void voigtUtilities::transformStress2D

  ( const Vector&      sign, 
    const double       co,  
    const double       si,
    const Vector&      sigx )

{
  double c2 = co*co;
  double s2 = si*si;
  double cs = co*si;

  sign[0] =   c2 * sigx[0] + s2 * sigx[1] +    2 * cs   * sigx[2];
  sign[1] =   s2 * sigx[0] + c2 * sigx[1] -    2 * cs   * sigx[2];
  sign[2] = - cs * sigx[0] + cs * sigx[1] + ( c2 - s2 ) * sigx[2];
}

//-----------------------------------------------------------------------
//   transformStrain2D
//-----------------------------------------------------------------------

void voigtUtilities::transformStrain2D

  ( const Vector&      epsn, 
    const double       co,  
    const double       si,
    const Vector&      epsx )

{
  double c2 = co*co;
  double s2 = si*si;
  double cs = co*si;

  epsn[0] =       c2 * epsx[0] +     s2 * epsx[1] +        cs   * epsx[2];
  epsn[1] =       s2 * epsx[0] +     c2 * epsx[1] -        cs   * epsx[2];
  epsn[2] = - 2 * cs * epsx[0] + 2 * cs * epsx[1] + ( c2 - s2 ) * epsx[2];
}

//-----------------------------------------------------------------------
//   voigtIndex
//-----------------------------------------------------------------------

idx_t voigtUtilities::voigtIndex

  ( const idx_t     i,
    const idx_t     j,
    const idx_t     rank )

{
  if ( i == j ) return i;    // 00, 11, 22

  if ( i+j == 1 ) return rank;  // 01, 10
  if ( i+j == 3 ) return 4;     // 12, 21
  if ( i+j == 2 ) return 5;     // 20, 02
  return -1;
}

//-----------------------------------------------------------------------
//   tensorIndex
//-----------------------------------------------------------------------

void  voigtUtilities::tensorIndex

  (       idx_t&    i,
          idx_t&    j,
    const idx_t     iv,
    const idx_t     rank )

{
  // returns tensor idices from voigt index

  JEM_ASSERT ( rank <= 3 );
  JEM_ASSERT ( iv >= 0 );

  i = j = -1;

  if ( iv < rank )
  {
    i = j = iv;
  }
  else
  {
    if ( iv == rank )
    {
      JEM_ASSERT ( rank > 1 );
      i = 0;
      j = 1;
    }
    else if ( iv == rank+1 )
    {
      JEM_ASSERT ( rank == 3 );
      i = 1; 
      j = 2;
    }
    else if ( iv == rank+2 )
    {
      JEM_ASSERT ( rank == 3 );
      i = 0;
      j = 2;
    }
    else
    {
      throw Error ( JEM_FUNC, "Voigt index too high: " + String(iv) );
    }
  }
}

//-----------------------------------------------------------------------
//   tensorIndex
//-----------------------------------------------------------------------

Tuple<idx_t,2> voigtUtilities::tensorIndex

  ( const idx_t     iv,
    const idx_t     rank )

{
  idx_t i;
  idx_t j;

  tensorIndex ( i, j, iv, rank );

  return Tuple<idx_t,2> ( i, j );
}


//-----------------------------------------------------------------------
//   engng2tensorial (on copy)
//-----------------------------------------------------------------------

void  voigtUtilities::engng2tensorial

  ( const Vector& tens,
    const Vector& eng )

{
  tens = eng;
  engng2tensorial ( tens );
}

//-----------------------------------------------------------------------
//   engng2tensorial (on original)
//-----------------------------------------------------------------------

void  voigtUtilities::engng2tensorial

  ( const Vector&         eng2tens )

{
  const idx_t strCount = eng2tens.size();

  if ( strCount == 3 )
  {
    eng2tens[2] *= 0.5;
  }
  else if ( strCount == 6 )
  {
    eng2tens[3] *= 0.5;
    eng2tens[4] *= 0.5;
    eng2tens[5] *= 0.5;
  }
}

