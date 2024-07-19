/*
 *  TU Delft 
 *
 *  2021
 *
 *  Arclength model with constraint equation based 
 *  on the global strain rate in the loading direction
 * 
 *  Loading direction assumed along global y-axis 
 *  model partially derived from Iuri's BCModel 
 *
 */

#ifndef SRARCLEN_MODEL_H 
#define SRARCLEN_MODEL_H

#include <jem/numeric/func/Function.h>
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

using jem::numeric::Function;
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

class SRArclenModel : public Model
{
 public:

  static const char* NODEGROUPS_PROP;
  static const char* LOADGROUPS_PROP;
  static const char* DOFS_PROP;
  static const char* LOADDOFS_PROP;
  static const char* STRAINRATE_PROP;
  static const char* OFFANGLE_PROP;
  static const char* MAXSTRAIN_PROP;

                     SRArclenModel

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
  
  virtual           ~SRArclenModel ();

 private:

  void               applyDisps_

    ( const Properties& params,
      const Properties& globdat );

  void               connect_ ();

  void               dofsChanged_ ();

  void               initUnitLoad_ 
  
    ( const Properties& globdat );

  void               getUnitLoad_

    ( const Properties& params,
      const Properties& globdat );

  void               getArcFunc_

    ( const Properties& params,
      const Properties& globdat );

  void               initTransform_

    ( const Properties& globdat );

  void               initMasters_

    ( const Properties& globdat );

  void               initDirection_

    (  );

  void               updateDefGrad_

    (       M33&        F,
      const Properties& globdat );

  void               updateTransform_

    ( const double&     theta1 );

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

  // Arclen control

  Vector              unitLoad_;

  IdxVector           masters_;

  double              maxStrain_;
  double              strain0_;
  double              strain_;

  // Geometry

  double              l1_;
  double              l2_;
  double              l3_;

  double              offAngle_;
  double              theta0_;
  double              c0_;
  double              s0_;
  double              phi_;
  double              strainRate_;

  Vector              transform0_;
  Vector              transformDefGrad_;
  Vector              transformForce_;


  M33                 F0loc_;
  M33                 Sbar1_;

  idx_t               idofY_;

  idx_t               ndofTypes_;

  // Printing
 
   Ref<PrintWriter>   strainOut_;

   Ref<PrintWriter>   stressOut_;
  

  Ref<PrintWriter>          initWriter_

    ( const Properties&       params,
      const String            name ) const;


    void                      printStrains_

    ( const Properties&       params,
      const Properties&       globdat );

    void                      printStresses_

    ( const Properties&       params,
      const Properties&       globdat );

};

#endif
