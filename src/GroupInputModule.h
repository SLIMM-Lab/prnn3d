/*
 *  Copyright (C) 2014 TU Delft. All rights reserved.
 *  
 *  Frans van der Meer, September 2014
 *  
 *  Module to generate NodeGroups and ElementGroups from input file
 *
 */

#ifndef GROUP_INPUT_MODULE_H
#define GROUP_INPUT_MODULE_H

#include <jive/app/Module.h>
#include <jive/fem/ElementGroup.h>
#include <jive/fem/NodeSet.h>
#include <jive/fem/ElementSet.h>
#include <jive/util/Constraints.h>
#include <jive/util/XDofSpace.h>
#include <jive/util/Assignable.h>

using namespace jem;

using jem::util::Properties;
using jive::StringVector;
using jive::app::Module;
using jive::fem::ElementGroup;
using jive::fem::NodeSet;
using jive::fem::ElementSet;
using jive::util::Constraints;
using jive::util::XDofSpace;
using jive::util::Assignable;
using jive::Vector;
using jive::Matrix;
using jive::IdxVector;

//-----------------------------------------------------------------------
//   class GroupInputModule
//-----------------------------------------------------------------------


class GroupInputModule : public Module
{
 public:

  typedef GroupInputModule   Self;
  typedef Module             Super;

  static const char*        TYPE_NAME;
  static const char*        NODE_GROUPS;
  static const char*        ELEM_GROUPS;

  explicit                  GroupInputModule

    ( const String&           name   = "groupInput" );

  virtual Status            init

    ( const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

  static Ref<Module>        makeNew

    ( const String&           name,
      const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

 protected:

  virtual                  ~GroupInputModule  ();

 private:

  idx_t                    numNodes_;

  // functions

  void                     configure_ 

    ( const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

  void                     makeNodeGroups_

    ( const Properties&   globdat );

  //=======================================================================
  //   class NGroup_
  //=======================================================================

  /*
   *  Class that generates NodeGroups from input in .pro-file
   *
   *  Use xtype/ytype/ztype = "min"/"max",
   *   or xval/yval/zval = value,
   *   to specify NodeGroup location
   *  Specify 1, 2 or 3 coordinates for plane, line or point. 
   *  Use restrictToGroups to use only nodes from specific ElemGroups
   *      specify restrictToPartial to use only some nodes from those
   *      elements (e.g. top layer of interface elements)
   */

  class NGroup_ : public Object
  {
   public:

    explicit                NGroup_  ();

    ~NGroup_ ();

    void                    configure

      ( const Properties&     conf,
        const Properties&     props,
        const String&         name );

    void                    makeGroup

      ( const Properties&     globdat );

    static const char*      X_NAMES[3];
    static const double     PI;

    StringVector            knownTypes;

    enum    Check   { NONE, VALUE, BOUNDS };   

   private:

    void                    findCandidates_

      ( const ElementSet&     elems,
        const Properties&     globdat );

    void                    findNodes_

      ( const NodeSet&        nodes );

    void                    store_

      ( const NodeSet&        nodes,
        const Properties&     globdat );

    idx_t            rank_;

    bool             all_;
    double           eps_;
    double           angle_;
    double           radius_;

    // every entry of doX_ is a switch that determines how a 
    // node is checked in that dimension (x,y,z)
    // if ( all(doX_==0) ) a line is specified

    Array<Check>     doX_;  

    Tuple<Vector,3>  xvals_;
    Tuple<String,3>  xtype_;
    Matrix           xbounds_;

    IdxVector        inodes_;
    String           myName_;
    StringVector     restrictTo_;
    IdxVector        restrictPartial_;
  };

  //=======================================================================
  //   class EGroup_
  //=======================================================================

  /*
   *  Class that generates ElementGroups from input in .pro-file
   *
   *  Use xtype/ytype/ztype = "min"/"max",
   *   or xval/yval/zval = value,
   *   to specify ElementGroup location
   *  Specify 1, 2 or 3 coordinates for plane, line or point. 
   */

  class EGroup_ : public Object
  {
   public:

    explicit                EGroup_  ();

    ~EGroup_ ();

    void                    configure

      ( const Properties&     conf,
        const Properties&     props,
        const String&         name );

    void                    makeGroup

      ( const Properties&     globdat );

    static const char*      X_NAMES[3];
    static const double     PI;

    StringVector            knownTypes;

    enum    Check   { NONE, VALUE, BOUNDS };   

   private:

    void                    findElems_

      ( const ElementSet&     nodes,
        const Properties&     globdat );

    void                    store_

      ( const ElementSet&     nodes,
        const Properties&     globdat );

    idx_t            rank_;

    bool             all_;
    double           eps_;

    // every entry of doX_ is a switch that determines how an 
    // element is checked in that dimension (x,y,z)

    Array<Check>     doX_;  

    Tuple<Vector,3>  xvals_;
    Tuple<String,3>  xtype_;
    Tuple<bool,3>    completely_;
    Matrix           xbounds_;

    IdxVector        ielems_;
    String           myName_;
    String           parent_;
  };

};



#endif
