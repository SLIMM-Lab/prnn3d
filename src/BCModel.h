/*
 *  TU Delft 
 *
 *  Iuri Barcelos, August 2018
 *
 *  BC model adapted from VEVPLoadModel. 
 *
 */

#ifndef BC_MODEL_H 
#define BC_MODEL_H

#include <jem/numeric/func/Function.h>
#include <jive/Array.h>
#include <jive/model/Model.h>
#include <jive/fem/NodeGroup.h>
#include <jive/model/Actions.h>
#include <jive/model/ModelFactory.h>
#include <jive/util/Assignable.h>
#include <jive/util/XDofSpace.h>
#include <jive/model/PointLoadModel.h>

#include "TrainingData.h"
#include "Normalizer.h"
#include <jem/io/PrintWriter.h>

using namespace jem;

using jem::numeric::Function;
using jem::util::Properties;
using jive::Vector;
using jive::IntVector;
using jive::IdxVector;
using jive::IdxMatrix;
using jive::StringVector;
using jive::model::Model;
using jive::model::PointLoadModel;
using jive::fem::NodeGroup;
using jive::fem::NodeSet;
using jive::util::Assignable;
using jive::util::XDofSpace;
using jive::util::DofSpace;
using jive::util::Constraints;

using jive::app::Module;
using jem::io::PrintWriter;

typedef Array< Properties,        1 > Globdats;
typedef Array< Ref<TrainingData>, 1 > Datas;
typedef Array< Ref<Module>,       1 > Chains;

class BCModel : public Model
{
 public:

  static const char* MODE_PROP;
  static const char* NODEGROUPS_PROP;
  static const char* DOFS_PROP;
  static const char* UNITVEC_PROP;
  static const char* SHAPE_PROP;
  static const char* SHAPE_GP;
  static const char* INIT_HYPERS;
  static const char* STEP_PROP;
  static const char* ALENGROUP_PROP;
  static const char* LOAD_PROP;
  static const char* MAXNORM_PROP;

  enum               Mode      { LOAD, DISP, ALEN };

                     BCModel

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

  static Ref<Model>  makeNew

    ( const String&     name,
      const Properties& conf,
      const Properties& props,
      const Properties& globdat );

 protected:
  
  virtual           ~BCModel ();

 private:

  void               applyLoads_

    ( const Vector&     fext,
      const Properties& globdat );

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
      
  void               updateDefGrad_

    (       Matrix&        F,
      const Properties& globdat );
      
 private:

  Ref<Constraints>    cons_;
  Ref<DofSpace>       dofs_;
  Assignable<NodeSet> nodes_;

  Ref<Function>       shape_;

  idx_t               numgroups_;
  idx_t               numGP_;
  IntVector           idofs_;

  Vector              unitVec_;
  Vector              initVals_;

  StringVector        nodeGroups_;
  StringVector        dofTypes_;

  Mode                mode_;

  // Sample control
  
  bool                useShapeGP_;
  bool                updateGP_;
  StringVector        hpFiles_;
  Chains              gp_; 
  Globdats            gpGlobdat_;
  Datas               gpData_;
  int                 rseed_;
  Vector              meanprev_;

  // Time control

  double              time_;
  double              time0_;
  double              dt_;
  double              stepF_;

  // Arclen control

  bool                ulUpd_;
  Vector              unitLoad_;
  idx_t               master_;
  idx_t               alenGroup_;

  double              maxNorm_;

  IdxVector           masters_;
  Vector              signs_;

  Ref<PointLoadModel> pLoad_;
  
  // Geometry
  
  double             l1_;
  double             l2_;
  double             l3_;
 
  idx_t              ndofTypes_;

  idx_t              idofXX_, idofXY_, idofXZ_;
  idx_t              idofYX_, idofYY_, idofYZ_;
  idx_t              idofZX_, idofZY_, idofZZ_;

  bool               symmetric_;
  Vector             symFactors_;

  // Printing
 
  Ref<PrintWriter>   strainOut_;
  Ref<PrintWriter>   rotationOut_;
  Ref<PrintWriter>   stretchOut_;

  Ref<PrintWriter>          initWriter_

    ( const Properties&       params,
      const String            name ) const;


    void                      printStrains_

    ( const Properties&       params,
      const Properties&       globdat );

};

#endif
