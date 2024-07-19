#ifndef AXISYM_MODEL_H
#define AXISYM_MODEL_H

#include <jem/io/PrintWriter.h>
#include <jem/numeric/algebra/matmul.h>
#include <jem/util/Properties.h>
#include <jem/util/SparseArray.h>
#include <jem/util/StringUtils.h>

#include <jive/util/utilities.h>
#include <jive/util/XDofSpace.h>
#include <jive/util/XTable.h>
#include <jive/util/Assignable.h>
#include <jive/geom/InternalShape.h>
#include <jive/algebra/MatrixBuilder.h>
#include <jive/model/Model.h>
#include <jive/model/Actions.h>
#include <jive/model/StateVector.h>
#include <jive/fem/ElementGroup.h>
#include <jive/fem/Globdat.h>

#include "models.h"
#include "utilities.h"
// #include "Softening.h"
// #include "DispWriter.h"
#include "Material.h"
// #include "Plasticity.h"
// #include "ThermNames.h"
#include "SolverNames.h"

using namespace jem;

using jem::io::PrintWriter;
using jem::util::Properties;
using jem::util::SparseArray;
using jem::util::StringUtils;
using jive::Vector;
using jive::IdxVector;
using jive::StringVector;
using jive::Cubix;
using jive::util::XDofSpace;
using jive::util::XTable;
using jive::util::Assignable;
using jive::algebra::MatrixBuilder;
using jive::model::Model;
using jive::model::StateVector;
using jive::geom::IShape;
using jive::fem::NodeSet;
using jive::fem::ElementSet;
using jive::fem::ElementGroup;
using jive::fem::Globdat;

// some typedef to avoid typing

typedef ElementSet              ElemSet;
typedef ElementGroup            ElemGroup;

class AxisymModel : public Model
{
 public:

  typedef AxisymModel        Self;
  typedef Model              Super;

  static const char*         DOF_NAMES[3];
  static const char*         SHAPE_PROP;
  static const char*         MATERIAL_PROP;
  static const char*         THICK_PROP;
  static const char*	       LARGE_DISP_PROP;

                       AxisymModel
			 
    ( const String&       name,
      const Properties&   conf,
      const Properties&   props,
      const Properties&   globdat );

  virtual void         configure

    ( const Properties&   props,
      const Properties&   globdat );

  virtual void         getConfig

    ( const Properties&   conf,
      const Properties&   globdat )      const;

  virtual bool         takeAction

    ( const String&       action,
      const Properties&   params,
      const Properties&   globdat );

 protected:

  virtual              ~AxisymModel ();

  virtual void         getMatrix_

    ( Ref<MatrixBuilder>  mbuilder,
      const Vector&       force,
      const Vector&       disp )       const;

  // virtual void         getThermForce_

  //   ( const Vector&       fth0,
  //     const Vector&       disp )   const;

  // void                 getDissForce_

  //   ( const Ref<Plasticity>  p,
  //     const Vector&       fstar,
  //     const Vector&       disp )   const;
    
  // void                 initCharLength_ ();

  void                 getDissipation_

    ( const Properties&   params );

  bool                 getTable_

    ( const Properties&   params,
      const Properties&   globdat );
 
  virtual void         getXOutTable_

    ( Ref<XTable>         table,
      const Vector&       weights,
      const String&       contents,
      const Vector&       disp );

  void                 getStress_

    ( XTable&             table,
      const Vector&       weights,
      const Vector&       disp );

  void                 getBMatrix_

    ( const Matrix&       b,
      const Matrix&       dN,
      const Vector&       N,
      const double        r ) const;

  Ref<PrintWriter>     initWriter_

    ( const Properties&   params,
      const String        name ) const;

  void                 writeDisplacements_

    ( const Properties&   params,
      const Properties&   globdat );

 protected:

  Assignable<ElemGroup>   egroup_;
  Assignable<ElemSet>     elems_;
  Assignable<NodeSet>     nodes_;

  IdxVector               ielems_;

  idx_t                   rank_;
  idx_t                   nodeCount_;
  idx_t                   numElem_;
  idx_t                   numNode_;
  idx_t                   strCount_;
  idx_t                   dofCount_;
  idx_t                   ipCount_;
  bool			              largeDisp_;

  Ref<IShape>             shape_;

  Ref<XDofSpace>          dofs_;
  IdxVector               dofTypes_;

  Ref<Material>           material_;
  // Ref<Softening>          softening_;
  bool                    crackBandMethod_;
  
  Vector                  charLength_;
    
  Ref<PrintWriter>        dispOut_;
  Ref<PrintWriter>        elemOut_;
  Ref<PrintWriter>        damFOut_;
  Ref<PrintWriter>        damMOut_;

  ShapeGradsFunc          getShapeGrads_;
  ShapeFuncsFunc          getShapeFuncs_;

  String                  myTag_;

  static Ref<PrintWriter> nodeOut_;
  static idx_t            nodesWritten_;

  // DispWriter              dw_;
};

#endif
