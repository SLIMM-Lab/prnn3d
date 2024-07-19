/*
 * Copyright (C) 2019 TU Delft. All rights reserved.
 *
 * Class that implements a sequential neural network. Similar
 * to jive::model::MultiModel, but adapted to loop over layers
 * in reverse order during backpropagation.
 *
 * Author: Iuri Barcelos, i.barceloscarneiromrocha@tudelft.nl
 * Date:   May 2019
 * 
 */

#ifndef JIVE_MODEL_ANNMODEL_H
#define JIVE_MODEL_ANNMODEL_H

#include <jem/io/Serializable.h>
#include <jem/util/ObjFlex.h>
#include <jive/model/import.h>
#include <jive/model/Model.h>

JIVE_BEGIN_PACKAGE( model )

//-----------------------------------------------------------------------
//   class ANNModel
//-----------------------------------------------------------------------

class ANNModel : public Model,
                 public Serializable
{
 public:

  JEM_DECLARE_CLASS       ( ANNModel, Model );

  static const char*        TYPE_NAME;
  static const char*        LAYERS;
  static const char*        SEED;
  static const char*        SYMMETRIC;
  static const char*        WARM_START;
  static const char*        OFFLINE;

  explicit                  ANNModel

    ( const String&           name = "" );

  virtual void              readFrom

    ( ObjectInput&            in );

  virtual void              writeTo

    ( ObjectOutput&           out )        const;

  virtual Model*            findModel

    ( const String&           name )       const;

  virtual void              configure

    ( const Properties&       props,
      const Properties&       globdat );

  virtual void              getConfig

    ( const Properties&       props,
      const Properties&       globdat )    const;

  virtual bool              takeAction

    ( const String&           action,
      const Properties&       params,
      const Properties&       globdat );

  void                      clear       ();

  void                      reserve

    ( idx_t                   modCount );

  void                      trimToSize  ();

  void                      addLayer

    ( const Ref<Model>&       layer );

  static Ref<Model>         makeNew

    ( const String&           name,
      const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

  static void               initWeights_

    ( const String&           fname,
      const Properties&       globdat);

  static void               declare     ();

 protected:

  virtual                  ~ANNModel  ();

 private:

  jem::util::ObjFlex        layers_;
  String                    warmFile_;
  bool                      offline_;

};

JIVE_END_PACKAGE( model )

#endif
