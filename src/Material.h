/*
 * 
 *  Copyright (C) 2019 TU Delft. All rights reserved.
 *  
 *  This class implements a virtual class for a constitutive model
 *
 *  Author:  F.P. van der Meer, F.P.vanderMeer@tudelft.nl
 *  Date:    March 2019
 *
 */

#ifndef MATERIAL_H
#define MATERIAL_H

#include <jem/util/Properties.h>
#include <jem/base/array/utilities.h>
#include <jive/Array.h>
#include <jem/base/Object.h>

using jem::idx_t;
using jem::Ref;
using jem::String;
using jem::Object;
using jem::util::Properties;
using jive::Vector;
using jive::Matrix;
using jive::IdxVector;
using jive::StringVector;


//-----------------------------------------------------------------------
//   class Material
//-----------------------------------------------------------------------

class Material : public Object
{
 public:

  explicit                Material

    ( const idx_t           rank,
      const Properties&     globdat );

  virtual void            configure

    ( const Properties&     props );

  virtual void            getConfig

    ( const Properties&     conf )            const;

  virtual void            update

    ( const Vector&         stress,
      const Matrix&         stiff,
      const Matrix&         df,
      const idx_t           ip ) = 0;

  virtual void            updateWriteTable

    ( const Vector&         stress,
      const Matrix&         stiff,
      const Matrix&         df,
      const idx_t           ip ) = 0;

  virtual void            commit ();

  virtual void            checkCommit

    ( const Properties&     params  );

  virtual void            cancel ();

  virtual void            allocPoints

    ( const idx_t           count );

  virtual void            allocPoints

    ( const idx_t           count,
      const Matrix&         transfer,
      const IdxVector&      oldPoints );

  virtual void            deallocPoints 
    
    ( const idx_t           count );

  StringVector            getHistoryNames   () const;

  virtual void            getHistory

    ( const Vector&         hvals,
      const idx_t           mpoint ) const;


  virtual void            getInitHistory

    ( const Vector&         hvals,
      const idx_t           mpoint ) const;

  virtual void           setHistory

    ( const Vector&        hvals,
      const idx_t          mpoint );

  inline virtual idx_t    getHistoryCount() const;

  virtual void           updateYieldFunc

    ( const Vector&      learnProp );

  inline virtual idx_t    pointCount ()     const;

  inline virtual double   giveDissipation   ( const idx_t point  ) const;

  void                    setDT

    ( const double          dt );

   double                 getDT ();

    

 protected:

  virtual                ~Material      ();

 protected:

  idx_t                  rank_;
  double                 dt_;
  StringVector           historyNames_;
};

//-----------------------------------------------------------------------
//   implementation
//-----------------------------------------------------------------------


inline double Material::giveDissipation ( const idx_t ipoint ) const
{   
  // default implementation: no dissipation
  return 0.;
}

inline idx_t Material::pointCount () const
{
  return 0;
}

inline void Material::setDT ( const double dt ) 
{
  dt_ = dt;
}

inline double Material::getDT ( )
{
  return dt_;
}

inline idx_t Material::getHistoryCount () const
{
  return 0;
}

//-----------------------------------------------------------------------
//   related functions
//-----------------------------------------------------------------------


Ref<Material>             newMaterial

  ( const String&           name,
    const Properties&       conf,
    const Properties&       props,
    const Properties&       globdat );

#endif 
