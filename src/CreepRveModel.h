/*
 *  TU Delft; Dragan Kovacevic 
 *
 *  2022
 *
 *  finite deformation model to apply off-axis creep stress on RVE
 * 
 *  model derived from strain-rate based arclength model
 *  SRArclenModel.cpp 
 *
 */

#ifndef CREEPRVE_MODEL_H 
#define CREEPRVE_MODEL_H

#include <jem/numeric/algebra/matmul.h>
#include <jive/Array.h>
#include <jive/model/Model.h>
#include <jive/fem/NodeGroup.h>
#include <jive/model/Actions.h>
#include <jive/model/ModelFactory.h>
#include <jive/util/Assignable.h>
#include <jive/util/XDofSpace.h>
#include <jem/io/PrintWriter.h>

using namespace jem;

using jem::numeric::matmul;
using jem::util::Properties;
using jive::Vector;
using jive::IdxVector;
using jive::IdxVector;
using jive::IdxMatrix;
using jive::StringVector;
using jive::model::Model;
using jive::fem::NodeGroup;
using jive::fem::NodeSet;
using jive::util::Assignable;
using jive::util::XDofSpace;
using jive::util::DofSpace;
using jive::util::Constraints;

using jem::io::PrintWriter;

typedef Tuple<double,2,2> M22;
typedef Tuple<double,3,3> M33;
typedef Tuple<double,2>   Vec2;

class CreepRveModel : public Model
{
 public:

  static const char* NODEGROUPS_PROP;
  static const char* LOADGROUPS_PROP;
  static const char* DOFS_PROP;
  static const char* LOADDOFS_PROP;
  static const char* STRESSRATE_PROP;
  static const char* OFFANGLE_PROP;
  static const char* MAXSTRESS_PROP;
  static const char* MAXSTRAIN_PROP;

                     CreepRveModel

    ( const String&     name,
      const Properties& conf,
      const Properties& props,
      const Properties& globdat );

  virtual void       configure

    ( const Properties& props,
      const Properties& globdat );

  virtual void       getConfig

    ( const Properties& conf,
      const Properties& globdat )          const;

  virtual bool       takeAction

    ( const String&     action,
      const Properties& params,
      const Properties& globdat );

  static Ref<Model>   makeNew

    ( const String&     name,
      const Properties& conf,
      const Properties& props,
      const Properties& globdat );

 protected:
  
  virtual           ~CreepRveModel ();

 private:

  void               applyDisps_

    ( const Properties& params,
      const Properties& globdat );

  void               connect_ ();

  void               dofsChanged_ ();

  void               getExtVector_

    ( const Properties& params,
      const Properties& globdat );

  void               updateTransform_

    ( );

  void               initMasters_

    ( const Properties& globdat );

  void               initDirection_

    (  );

  void               updateDefGrad_

    ( const Properties& globdat );

  void               updateStrain_

    ( );

 private:

  Ref<Constraints>    cons_;
  Ref<DofSpace>       dofs_;
  Assignable<NodeSet> nodes_;

  idx_t               numgroups_;
  idx_t               numloadgroups_;

  StringVector        nodeGroups_;
  StringVector        loadGroups_;

  StringVector        dofTypes_;
  StringVector        ldofTypes_;

  // Time control

  double              time_;
  double              time0_;
  double              dt_;

  IdxVector           masters_;

  double              maxStrain_;
  double              strain_;
  double              strain0_;

  // Geometry

  double              l1_;
  double              l2_;
  double              l3_;

  double              offAngle_;
  double              theta0_;
  double              c0_;
  double              s0_;
  double              phi_;

  double              stressRate_;
  double              stress_;
  double              stress0_;
  double              maxStress_;

  Vector              transform0_;
  Vector              transformForce_;

  M33                 F1loc_;

  M33                 F0loc_;
  M33                 Sbar1_;

  idx_t               idofY_;

  idx_t               ndofTypes_;

  Ref<PrintWriter>          strainOut_;

  Ref<PrintWriter>          initWriter_

    ( const Properties&       params,
      const String            name ) const;


    void                      printStrains_

    ( const Properties&       params,
      const Properties&       globdat );
  
  
};

#endif
