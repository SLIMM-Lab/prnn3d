#include <jem/numeric/utilities.h>
#include <jem/base/String.h>
#include <jem/base/Error.h>

#include "utilities.h"
#include "mcmlib.h"

using jem::String;
using jem::ALL;
using jem::Error;


void mcmlib::inimI ( M33& m )
{
  m = 0.0;

  m(0,0) = 1.0; m(1,1) = 1.0; m(2,2) = 1.0;

  // for (int i = 0; i < 3; i++)
  // {
  //   m(i,i) = 1.0;
  // }
}


void mcmlib::inimmI4 ( M99& mI )
{

  mI = 0.0;

  mI (0,0) = 1.0;
  mI (1,1) = 1.0;
  mI (2,2) = 1.0;
  mI (3,4) = 1.0;
  mI (4,3) = 1.0;
  mI (5,6) = 1.0;
  mI (6,5) = 1.0;
  mI (7,8) = 1.0;
  mI (8,7) = 1.0;
}

void mcmlib::m2cc (Tuple<double,9,1>& c, const M33& m, int s)
{
  c = 0.0;

  // column vector represented as Sx1 matrix to enable transpose 
  // in jem/jive it is not possible to transpose Vector class

  switch ( s )
  {
    case 9:

      c(0,0) = m(0,0);
      c(1,0) = m(1,1);
      c(2,0) = m(2,2);
      c(3,0) = m(0,1);
      c(4,0) = m(1,0);
      c(5,0) = m(1,2);
      c(6,0) = m(2,1);
      c(7,0) = m(2,0);
      c(8,0) = m(0,2);
      break;

    default:
      throw Error ( JEM_FUNC, "Int value s not valid " );
  }
}


void mcmlib::cc2m (Matrix& m, const Matrix& c, int s)
{
  m.resize( 3, 3 ); 

  m = 0.0;

  // column vector represented as Sx1 matrix to enable transpose 
  // in jem/jive it is not possible to transpose Vector class

  switch ( s )
  {
    case 9:

      m(0,0) = c(0,0);
      m(1,1) = c(1,0);
      m(2,2) = c(2,0);
      m(0,1) = c(3,0);
      m(1,0) = c(4,0);
      m(1,2) = c(5,0);
      m(2,1) = c(6,0);
      m(2,0) = c(7,0);
      m(0,2) = c(8,0);
      break;
  
    case 6:
  
      m(0,0) = c(0,0);
      m(1,1) = c(1,0);
      m(2,2) = c(2,0);
      m(0,1) = c(3,0);
      m(1,0) = c(3,0);
      m(1,2) = c(4,0);
      m(2,1) = c(4,0);
      m(2,0) = c(5,0);
      m(0,2) = c(5,0);
      break;

    default:
      throw Error ( JEM_FUNC, "Int value s not valid " );
  }
}


void mcmlib::m2mm (M99& mm, const M33& m, int s)
{  
  switch ( s )
  {
    case 9:

      mm = 0.0;

      mm(0,0) = m(0,0);
      mm(0,4) = m(0,1);
      mm(0,7) = m(0,2);
      mm(1,1) = m(1,1);
      mm(1,3) = m(1,0);
      mm(1,6) = m(1,2);
      mm(2,2) = m(2,2);
      mm(2,5) = m(2,1);
      mm(2,8) = m(2,0);
      mm(3,1) = m(0,1);
      mm(3,3) = m(0,0);
      mm(3,6) = m(0,2);
      mm(4,0) = m(1,0);
      mm(4,4) = m(1,1);
      mm(4,7) = m(1,2);
      mm(5,2) = m(1,2);
      mm(5,5) = m(1,1);
      mm(5,8) = m(1,0);
      mm(6,1) = m(2,1);
      mm(6,3) = m(2,0);
      mm(6,6) = m(2,2);
      mm(7,0) = m(2,0);
      mm(7,4) = m(2,1);
      mm(7,7) = m(2,2);
      mm(8,2) = m(0,2);
      mm(8,5) = m(0,1);
      mm(8,8) = m(0,0);
      break;
  
    // case 5:
  
    //   mm(0,0) = m(0,0);
    //   mm(0,4) = m(0,1);
    //   mm(1,1) = m(1,1);
    //   mm(1,3) = m(1,0);
    //   mm(2,2) = m(2,2);
    //   mm(3,1) = m(0,1);
    //   mm(3,3) = m(0,0);
    //   mm(4,0) = m(1,0);
    //   mm(4,4) = m(1,1);

    //   break;

    default:
      throw Error ( JEM_FUNC, "Int value s not valid " );
  }

}


void mcmlib::mm2mmc (M99& mmc, const M99& mm, int s)
{
  mmc = mm;

  switch ( s )
  {
    case 9:
  
      for (int i = 0; i < s; i++)
      {
        mmc(i,3) = mm(i,4);
        mmc(i,4) = mm(i,3);
        mmc(i,5) = mm(i,6);
        mmc(i,6) = mm(i,5);
        mmc(i,7) = mm(i,8);
        mmc(i,8) = mm(i,7);
      }
      break;

    default:
      throw Error ( JEM_FUNC, "Int value s not valid " );
  }

}


void mcmlib::mm2mmr (M99& mmr, const M99& mm, int s)
{

  mmr = mm;

  switch ( s )
  {
    // case 5:

    //   for (int i = 0; i < s; i++)
    //   {
    //     mmr(3,i) = mm(4,i);
    //     mmr(4,i) = mm(3,i);
    //   }
    //   break;
  
    case 9:
  
      for (int i = 0; i < s; i++)
      {
        mmr(3,i) = mm(4,i);
        mmr(4,i) = mm(3,i);
        mmr(5,i) = mm(6,i);
        mmr(6,i) = mm(5,i);
        mmr(7,i) = mm(8,i);
        mmr(8,i) = mm(7,i);
      }
      break;

    default:
      throw Error ( JEM_FUNC, "Int value s not valid " );
  }

}

void mcmlib::cc2cct (Tuple<double,9,1>& cct, const Tuple<double,9,1>& cc, int s)
{
  cct = cc;

  // column vector represented as Sx1 matrix to enable transpose 
  // in jem/jive it is not possible to transpose Vector class

  if (s == 9)
  {
    cct(3,0) = cc(4,0);
    cct(4,0) = cc(3,0);
    cct(5,0) = cc(6,0);
    cct(6,0) = cc(5,0);
    cct(7,0) = cc(8,0);
    cct(8,0) = cc(7,0);
  }

}

