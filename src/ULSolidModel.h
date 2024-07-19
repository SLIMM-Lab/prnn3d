/*
 * 
 *  Copyright (C) 2019 TU Delft. All rights reserved.
 *  
 *  This class implements a simple model for updated lagrangian 
 *  finite element analysis 
 *
 *  It passes only the incremental deformation gradient to the 
 *  constitutive law
 *
 *  Reference: Zienkiewicz, Taylor and Fox
 *  'The Finite Element Method for Solid and Structural Mechanics'
 *  7th edition, sections 5.3.3 and 6.5
 *
 *  Author:  F.P. van der Meer, F.P.vanderMeer@tudelft.nl
 *  Date:    March 2019
 *
 */

#ifndef UL_SOLID_MODEL_H
#define UL_SOLID_MODEL_H

#include <jem/util/Properties.h>
#include <jem/io/PrintWriter.h>

#include <jive/util/XDofSpace.h>
#include <jive/util/XTable.h>
#include <jive/util/Assignable.h>
#include <jive/algebra/MatrixBuilder.h>
#include <jive/model/Model.h>
#include <jive/fem/ElementGroup.h>
#include <jive/geom/InternalShape.h>
#include <jive/fem/Globdat.h>

#include "models.h"
#include "utilities.h"
#include "Material.h"
#include "SolverNames.h"
#include "DispWriter.h"

using namespace jem;

using jem::util::Properties;
using jem::io::PrintWriter;
using jive::Vector;
using jive::IdxVector;
using jive::StringVector;
using jive::Cubix;
using jive::util::XDofSpace;
using jive::util::XTable;
using jive::util::Assignable;
using jive::algebra::MatrixBuilder;
using jive::model::Model;
using jive::fem::NodeSet;
using jive::fem::ElementSet;
using jive::fem::ElementGroup;
using jive::geom::IShape;
using jive::fem::Globdat;

// some typedef to avoid typing

typedef ElementSet              ElemSet;
typedef ElementGroup            ElemGroup;

class ULSolidModel : public Model
{
 public:

  typedef ULSolidModel       Self;
  typedef Model              Super;

  static const char*         DOF_NAMES[3];
  static const char*         SHAPE_PROP;
  static const char*         MATERIAL_PROP;
  static const char*         THICK_PROP;
  static const char*         STATE_PROP;

                       ULSolidModel
       
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

  virtual              ~ULSolidModel ();

  virtual void         getMatrix_

    ( Ref<MatrixBuilder>  mbuilder,
      const Vector&       force,
      const Vector&       disp0,
      const Vector&       disp1 )       const;

  void                 updateCoords_ 

    ( const Matrix&       x, 
      const Matrix&       X, 
      const Vector&       U ) const;

  void                 getDissipation_

    ( const Properties&   params );

  Ref<PrintWriter>          initWriter_

    ( const Properties&       params,
      const String            name ) const;

  bool                 getTable_

    ( const Properties&   params,
      const Properties&   globdat );
 
  virtual void         getXOutTable_

    ( Ref<XTable>         table,
      const Vector&       weights,
      const String&       contents,
      const Vector&       disp0,
      const Vector&       disp1 );

  void                 getStress_

    ( XTable&             table,
      const Vector&       weights,
      const Vector&       disp0,
      const Vector&       disp1 );

  void                 getBMatrix_

  ( const Matrix&       b,
    const Matrix&       dN,
    const Vector&       N,
    const double        r ) const;

  void         writeDisplacements_

    ( 
     const Properties&   params,
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
  double                  thickness_;
  String                  stateString_;
  bool                    homogenize_;

  Ref<IShape>             shape_;

  Ref<XDofSpace>          dofs_;
  IdxVector               dofTypes_;

  Ref<Material>           material_;
  String                  matType_;

  ShapeGradsFunc          getShapeGrads_;
  ShapeFuncsFunc          getShapeFuncs_;
  String                  myTag_;

  DispWriter              dw_;
  Ref<PrintWriter>        dispOut_;
};

#endif
