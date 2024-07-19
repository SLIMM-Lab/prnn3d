//-----------------------------------------------------------------------
// 
// 
// inimI   - initialize unit tensor
// inimmI4 - initialize fourth order unit tensor
// m2cc    - transform matrix into column
// cc2m    - transform column into matrix
// m2mm    - transform matrix into expanded matrix
// mm2mmc  - switch columns of expanded matrix
// mm2mmr  - switch rows of expanded matrix
// cc2cct  - switch the elements that correspond to
//           shear terms of a column array of size (s)
//----------------------------------------------------------------------- 

#ifndef MCMLIB_H
#define MCMLIB_H

#include <jem/base/Tuple.h>
#include <jive/Array.h>


using jem::idx_t;
using jem::Tuple;
using jive::Vector;
using jive::Matrix;

typedef Tuple <double,3,3> M33;
typedef Tuple <double,9,9> M99;

namespace mcmlib
{   
  void inimI   ( Tuple<double,3,3>& m );

  void inimmI4 ( Tuple<double,9,9>& mI ); 

  void m2cc    (Tuple<double,9,1>& c, const M33& m, int s);

  void cc2m    (Matrix& m, const Matrix& c, int s);

  void m2mm    (M99& mm, const M33& m, int s);

  void mm2mmc  (M99& mmc, const M99& mm, int s);

  void mm2mmr  (M99& mmr, const M99& mm, int s);

  void cc2cct  (Tuple<double,9,1>& cct, const Tuple<double,9,1>& cc, int s);

};

#endif
