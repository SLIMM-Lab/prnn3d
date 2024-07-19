/*
 *  Copyright (C) 2008 TU Delft. All rights reserved.
 *  
 *  Frans van der Meer, August 2008 
 *  
 *  This module generates a quasi-3D mesh for laminate analysis
 *  from an existing 2D mesh.
 *
 *  - nodes are multiplied (1 layer of nodes per ply)
 *  - element groups with solid (plane) elements are defined for each ply
 *  - element groups with interface elements are defined between plies
 *  - all nodegroups that are defined in the 2D mesh are extended with 
 *    the nodes with the same in-plane coordinates
 *  - NB: constraints are not copied to new plies
 *
 */

#ifndef LAMINATE_MESH_MODULE_H
#define LAMINATE_MESH_MODULE_H

#include <jive/app/Module.h>
#include <jive/fem/ElementGroup.h>
#include <jive/fem/XElementSet.h>
#include <jive/fem/XNodeSet.h>
#include <jive/util/Constraints.h>
#include <jive/util/XDofSpace.h>
#include <jive/util/Assignable.h>

#include "GroupInputModule.h"

using namespace jem;

using jem::util::Properties;
using jive::BoolVector;
using jive::IdxVector;
using jive::IdxMatrix;
using jive::StringVector;
using jive::app::Module;
using jive::fem::ElementGroup;
using jive::fem::XElementSet;
using jive::fem::XNodeSet;
using jive::util::Constraints;
using jive::util::XDofSpace;
using jive::util::Assignable;

typedef ElementGroup        ElemGroup;
typedef XElementSet         XElemSet;

//-----------------------------------------------------------------------
//   class LaminateMeshModule
//-----------------------------------------------------------------------


class LaminateMeshModule : public Module
{
 public:

  typedef LaminateMeshModule Self;
  typedef Module             Super;

  static const char*        DOF_NAMES[3];
  static const char*        TYPE_NAME;
  static const char*        N_LAYERS;
  static const char*        DIM;
  static const char*        THICKNESS;
  static const char*        INTERFACES;
  static const char*        LAYER_NAMES;
  static const char*        INTERFACE_NAMES;
  static const char*        NODE_GROUPS;
  static const char*        TIE_GROUPS;
  static const char*        SKIP_NGROUPS;
  static const char*        SKIP_EGROUPS;
  static const char*        ELAS_EGROUPS;
  static const char*        NONE;
  static const char*        NORMAL;
  static const char*        PERIODIC;
  static const char*        TIE_ALL;
  static const char*        IELEMS_N;
  static const char*        IELEMS_E;
  static const char*        TRANSITION;
  static const char*        THICK_DIRECTION;
  static const char*        WRITE_MESH;

  explicit                  LaminateMeshModule

    ( const String&           name   = "laminateMesh" );

  virtual Status            init

    ( const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

  virtual Status            run

    ( const Properties&       globdat );

  virtual void              shutdown

    ( const Properties&       globdat );

  static Ref<Module>        makeNew

    ( const String&           name,
      const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

 protected:

  virtual                  ~LaminateMeshModule  ();

 private:

  idx_t                    nLayers_;
  idx_t                    nInterf_;
  idx_t                    nElGroups_;
  idx_t                    rank_;
  idx_t                    rankFac_;
  idx_t                    nng_;

  idx_t                    numNodes_;
  idx_t                    numElems_;

  double                   dz_;

  String                   interfaces_;
  bool                     tieAll_;
  bool                     active_;
  bool                     doTieGroups_;
  String                   ofilename_;

  StringVector             layerNames_;
  StringVector             interfaceNames_;
  StringVector             groupNames_;
  StringVector             tieGroups_;
  StringVector             skipNGroups_;

  Assignable<ElemGroup>    egroup_;
  Assignable<XElemSet>     elems_;
  Assignable<XNodeSet>     nodes_;

  Ref<Constraints>         cons_;
  Ref<GroupInputModule>    groupInput_;

  // some quantities for when element groups are skipped

  StringVector             skipEGroups_;
  StringVector             elasticEGroups_;

  IdxVector                ielemsY_;  // Yes: will be duplicated
  IdxVector                ielemsN_;  // No: will not be duplicated
  BoolVector               doElem_;   // same information, stored differently
  BoolVector               doNode_;  

  bool                     doPermutate_;
  idx_t                    thickDir_;
  IdxVector                iperm_;

  IdxVector                inodesY_;  // nodes from ielemsY_
  IdxVector                inodesN_;  // nodes from ielemsN_

  IdxMatrix                nMatrix_;
  IdxMatrix                eMatrix_;

  idx_t                    numElemY_;
  idx_t                    numNodeY_;

  // functions

  void                     configure_ 

    ( const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

  void                     prepareMesh_

    ( const Properties&   globdat );

  void                     prepareSkip_

    ( const Properties&   globdat );

  void                     prepareElastic_

    ( const Properties&   globdat );

  void                     createMesh_ 

    ( const Properties&   globdat );

  void                     createMesh3_ 

    ( const Properties&   globdat );

  void                     permutate3_

    ( const Properties&   globdat );

  void                     updateConstraints_

    ( const Properties&   globdat );

  void                     updateNodeGroups_

    ( const Properties&   globdat );

  bool                     tieGroupsVert_

    ( const Properties&   globdat );

  void                     writeToFile_

    ( const String&       filename,
      const Properties&   globdat ) const;
};


#endif
