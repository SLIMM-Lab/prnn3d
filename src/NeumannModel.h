/*
 * 
 *  Copyright (C) 2010 TU Delft. All rights reserved.
 *  
 *  This class implements a model for neumann boundary conditions.
 *  As opposed to the PointLoadModel, this can apply loads on changing
 *  NodeGroups
 *
 *  Author:  F.P. van der Meer, F.P.vanderMeer@tudelft.nl
 *  Date:    April 2011
 *
 */


#ifndef NEUMANN_MODEL_H
#define NEUMANN_MODEL_H

#include <jem/io/Writer.h>
#include <jive/Array.h>
#include <jive/model/Model.h>
#include <jive/fem/NodeGroup.h>
#include <jive/model/ModelFactory.h>
#include <jive/util/Assignable.h>
#include <jive/util/XDofSpace.h>

using namespace jem;

using jem::util::Properties;
using jem::io::Writer;
using jive::Vector;
using jive::IdxVector;
using jive::model::Model;
using jive::StringVector;
using jive::fem::NodeGroup;
using jive::fem::NodeSet;
using jive::model::Model;
using jive::util::Assignable;
using jive::util::XDofSpace;
using jive::util::DofSpace;
using jive::util::Constraints;

//-----------------------------------------------------------------------
//   class NeumannModel
//-----------------------------------------------------------------------


class NeumannModel : public Model
{
 public:

  typedef NeumannModel    Self;
  typedef Model             Super;

  static const char*        TYPE_NAME;

  static const char*        LOAD_INCR_PROP;
  static const char*        INIT_LOAD_PROP;
  static const char*        MIN_LOAD_PROP;
  static const char*        MAX_LOAD_PROP;
  static const char*        REDUCTION_PROP;
  static const char*        NODES_PROP;
  static const char*        DOF_PROP;
  static const char*        FACTORS_PROP;

  explicit                  NeumannModel

    ( const String&           name );

  virtual void              configure

    ( const Properties&       props,
      const Properties&       globdat );

  virtual void              getConfig

    ( const Properties&       conf,
      const Properties&       globdat )      const;

  virtual bool              takeAction

    ( const String&           action,
      const Properties&       params,
      const Properties&       globdat );

  void                      setLoadIncr

    ( double                  incr );

  inline double             getLoadIncr     () const;

  static Ref<Model>         makeNew

    ( const String&           name,
      const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );


 protected:

  virtual                  ~NeumannModel  ();


 private:

  void                      init_

    ( const Properties&       globdat );

  void                      getExtVector_

    ( const Vector&           fext,
      const Properties&       globdat ) const;

  void                      advance_

    ( const Properties&       globdat );

  void                      commit_

    ( const Properties&       params,
      const Properties&       globdat );

  void                      reduceStep_

    ( const Properties&       params,
      const Properties&       globdat );

  void                      increaseStep_

    ( const Properties&       params,
      const Properties&       globdat );

 private:

  Ref<DofSpace>             dofs_;
  Assignable<NodeSet>       nodes_;

  idx_t                     ngroups_;
  IdxVector                 idofs_;

  double                    loadScale0_;
  double                    loadScale_;
  double                    loadIncr_;

  /*
   * the following members are input for specifying boundary conditions
   *
   */

  StringVector              nodeGroups_;
  StringVector              dofTypes_;
  Vector                    factors_;

  /* the following members are constant input variables 
   *
   * reduction:   factor for reduction of increments
   * loadIncr:    initial load increment
   * minLoadIncr: minimum (absolute) load increment
   * maxLoadVal:  maximum (absolute) load value (in disp.control)
   *
   */

  double                    reduction_;
  double                    loadIncr0_; 
  double                    minLoadIncr_; 
  double                    maxLoadVal_;
  double                    initLoad_; 
};

#endif
