/*
 * Copyright (C) 2019 TU Delft. All rights reserved.
 *
 * ItemSet skin for artificial neurons. X version.
 *
 * Author: Iuri Barcelos, i.barceloscarneiromrocha@tudelft.nl
 * Date:   May 2019
 * 
 */

#ifndef XNEURONSET_H
#define XNEURONSET_H

#include "NeuronSet.h"

//-----------------------------------------------------------------------
//   class XNeuronSet
//-----------------------------------------------------------------------

class XNeuronSet : public NeuronSet
{
 public:
  
  typedef NeuronSet         Super;

  inline                    XNeuronSet 
  
    ( const Ref<AxonSet>      axons );

  inline void               clear       ();

  inline void               reserve

    ( idx_t                   count );      

  inline void               trimToSize  ();

         idx_t              addNeuron   ();

	 void               setInput
 
    ( const idx_t             ineuron,
      const IdxVector&        iaxons,
      const IdxVector&        ineurons   );

         void               addOutput

    ( const idx_t             ineuron,
      const IdxVector&        iaxons,
      const IdxVector&        ineurons   );

  static Ref<XNeuronSet>    get

    ( const Properties&       globdat,
      const String&           context );
};

//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class XNeuronSet
//=======================================================================

//-----------------------------------------------------------------------
//   constructor 
//-----------------------------------------------------------------------

inline XNeuronSet::XNeuronSet 

  ( const Ref<AxonSet> axons ) : Super ( axons )
{
}

//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------

inline void XNeuronSet::clear ()
{
  elemsI_.clear ();
  elemsO_.clear ();

  pairsI_->clear ();
  pairsO_->clear ();
}

//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------

inline void XNeuronSet::reserve ( idx_t n )
{
  elemsI_.reserve ( n );
  elemsO_.reserve ( n );

  pairsI_->reserve ( n );
  pairsO_->reserve ( n );
}

//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------

inline void XNeuronSet::trimToSize ()
{
  elemsI_.trimToSize ();
  elemsO_.trimToSize ();

  pairsI_->trimToSize ();
  pairsO_->trimToSize ();
}

#endif
