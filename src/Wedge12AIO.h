/*
 *
 *  This file contains all necessary classes for Wedge12 shape.
 *
 *
 *  Lu Ke, 04-05-2020
 *
 *
 */


#ifndef JIVE_GEOM_WEDGE12AIO_H
#define JIVE_GEOM_WEDGE12AIO_H

#include <jive/geom/shfuncs/PrismFuncs.h>
#include <jive/geom/shfuncs/LineFuncs.h>
#include <jive/geom/shfuncs/TriangFuncs.h>
#include <jive/geom/StdWedge.h>
#include <jive/geom/Wedge.h>


JIVE_BEGIN_PACKAGE ( geom )


//-----------------------------------------------------------------------
//   class StdWedge12
//-----------------------------------------------------------------------


class StdWedge12 : public StdWedge, public Serializable {
  public:

    JEM_DECLARE_CLASS     ( StdWedge12, StdWedge );

    static const int        VERTEX_COUNT     = 12;
    static const int        SHAPE_FUNC_COUNT = 12;


    StdWedge12       ();

    virtual void            readFrom

    ( ObjectInput&          in );

    virtual void            writeTo

    ( ObjectOutput&         out )             const;

    virtual idx_t           shapeFuncCount   () const;
    virtual idx_t           vertexCount      () const;
    static  Matrix          vertexCoords     ();
    virtual Matrix          getVertexCoords  () const;

    virtual void            evalShapeFunctions

    ( const Vector&         f,
      const Vector&         u )               const;

    virtual void            evalShapeGradients

    ( const Vector&         f,
      const Matrix&         g,
      const Vector&         u )               const;

    virtual void            evalShapeGradGrads

    ( const Vector&         f,
      const Matrix&         g,
      const Matrix&         h,
      const Vector&         u )               const;

    static Ref<SShape>      makeNew

    ( const String&         geom,
      const Properties&     conf,
      const Properties&     props );

    static void             declare          ();


  protected:

    virtual                ~StdWedge12       ();

};

//-----------------------------------------------------------------------
//   class Wedge12
//-----------------------------------------------------------------------

class Wedge12 : public Wedge {
  public:

    static const char*      TYPE_NAME;

    static const int        NODE_COUNT      = 12;

    static const char*      ISCHEME;
    static const char*      TSCHEME;
    static const char*      QSCHEME;


    static Ref<IShape>      getShape

    ( const String&         name    = "wedge12",
      const Ref<SShape>&    sfuncs  = nullptr );

    static Ref<IShape>      getShape

    ( const String&         name,
      String&               ischeme,
      String&               tscheme,
      String&               qscheme,
      const Ref<SShape>&    sfuncs  = nullptr);

    static Ref<IShape>      getShape

    ( const String&         name,
      const MatrixVector&   ischeme,
      const Ref<SShape>&    sfuncs  = nullptr );

    static Ref<IShape>      getShape

    ( const String&         name,
      const Properties&     conf,
      const Properties&     props );

    static util::Topology   getBoundaryTopology ();

    static void             declare             ();

  private:

    class                   myUtils_;

};

//-----------------------------------------------------------------------
//   class Wedge12Funcs
//-----------------------------------------------------------------------


class Wedge12Funcs : public ShapeFuncs<3, 12> {
  public:

    static inline void  evalFuncs

    ( FuncVector&       f,
      const Point&      u );

    static inline void  evalGrads

    ( FuncVector&       f,
      GradMatrix&       g,
      const Point&      u );

    static inline void  evalGrads2

    ( FuncVector&       f,
      GradMatrix&       g,
      Grad2Matrix&      h,
      const Point&      u );

};


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_WEDGE12_FUNCS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_WEDGE12_FUNCS
#define JIVE_EVAL_WEDGE12_FUNCS( f, u )                   \
                                                          \
  JIVE_EVAL_PRISM_FUNCS( f, u,                            \
                         jive::geom::Triang6Funcs,        \
                         jive::geom::Line2Funcs )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_WEDGE12_GRADS
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_WEDGE12_GRADS
#define JIVE_EVAL_WEDGE12_GRADS( f, g, u )                \
                                                          \
  JIVE_EVAL_PRISM_GRADS( f, g, u,                         \
                         jive::geom::Triang6Funcs,        \
                         jive::geom::Line2Funcs )


//-----------------------------------------------------------------------
//   macro JIVE_EVAL_WEDGE12_GRADS2
//-----------------------------------------------------------------------


#undef  JIVE_EVAL_WEDGE12_GRADS2
#define JIVE_EVAL_WEDGE12_GRADS2( f, g, h, u )            \
                                                          \
  JIVE_EVAL_PRISM_GRADS2( f, g, h, u,                     \
                          jive::geom::Triang6Funcs,       \
                          jive::geom::Line2Funcs )



//#######################################################################
//   Implementation
//#######################################################################

//=======================================================================
//   class Wedge12Funcs
//=======================================================================


inline void Wedge12Funcs::evalFuncs

( FuncVector&   f,
  const Point&  u )

{
  JIVE_EVAL_PRISM_FUNCS ( f, u, Triang6Funcs, Line2Funcs );
}


inline void Wedge12Funcs::evalGrads

( FuncVector&   f,
  GradMatrix&   g,
  const Point&  u )

{
  JIVE_EVAL_PRISM_GRADS ( f, g, u, Triang6Funcs, Line2Funcs );
}


inline void Wedge12Funcs::evalGrads2

( FuncVector&   f,
  GradMatrix&   g,
  Grad2Matrix&  h,
  const Point&  u )

{
  JIVE_EVAL_PRISM_GRADS2 ( f, g, h, u, Triang6Funcs, Line2Funcs );
}


//-----------------------------------------------------------------------
//   compatibility typedefs
//-----------------------------------------------------------------------

typedef StdWedge12        StdQuadraticLinearWedge;
typedef Wedge12           QuadraticLinearWedge;


JIVE_END_PACKAGE ( geom )

#endif
