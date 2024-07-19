/*
 *  Copyright (C) 2012 TU Delft. All rights reserved.
 *
 *  Object that gathers data in a table for output purposes
 *
 *  Author: F.P. van der Meer
 *  Date: November 2012
 *
 */

#ifndef TB_FILLER_H
#define TB_FILLER_H

#include <jem/base/Slice.h>
#include <jem/base/System.h>
#include <jem/io/OutputStream.h>
#include <jem/io/Writer.h>
#include <jem/util/ArrayBuffer.h>

#include <jive/Array.h>
#include <jive/util/XTable.h>

using namespace jem;

using jem::util::ArrayBuffer;
using jem::io::OutputStream;
using jive::util::XTable;
using jive::BoolVector;
using jive::IdxVector;
using jive::StringVector;
using jive::Vector;
using jive::Matrix;
using jive::Cubix;

//=====================================================================
//   class TbFiller
//=====================================================================

class TbFiller : public Object {
  public:

    static const char*        TABLE_FILTER;

    class NamedSlice {
      public:
        NamedSlice ();
        NamedSlice ( const Slice slice, const String& name );
        Slice   slice() const;
        String  name() const;
      private:
        Slice   slice_;
        String  name_;
    };

    typedef ArrayBuffer<NamedSlice> NamedSliceBuffer;
    typedef TbFiller                Self;

    TbFiller

    ( const idx_t         rank,
      const bool          axisym = false );

    ~TbFiller ();

  protected:

  public:

    Slice                announce

    ( const String&       name );

    Slice                announce

    ( const StringVector& names );

    void                 setFilter

    ( const String&       filter );

    void                 prepareTable

    (       IdxVector&    i2table,
            IdxVector&    jcols,
            const Ref<XTable>   table ) const;

    inline idx_t         typeCount () const;

    static void          permTri6

    ( const Vector&       weights,
      const Matrix&       values );

    static void          permTri12

    ( const Vector&       weights,
      const Matrix&       values );

  protected:

    void                 announceTensor_

    ( const String&       name );

    void                 announceVector_

    ( const String&       name );

    void                 announceDiag_

    ( const String&       name );

  protected:

    idx_t                rank_;
    idx_t                strCount_;
    idx_t                ntype_;
    NamedSliceBuffer     superNames_;
    ArrayBuffer<String>  typeNames_;
    BoolVector           write_;
    bool                 axisym_;

    static IdxVector     permi_;
    static IdxVector     permj1_;
    static IdxVector     permj2_;
};

//-----------------------------------------------------------------------
//   typeCount
//-----------------------------------------------------------------------

idx_t TbFiller::typeCount () const
{
  return ntype_;
}

//=======================================================================
//   related functions
//=======================================================================

void print

(      jem::io::Writer&      out,
       const TbFiller::NamedSlice& ns );

#endif
