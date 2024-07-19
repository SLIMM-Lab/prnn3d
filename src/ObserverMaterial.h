/*
 * Copyright (C) 2021 TU Delft. All rights reserved.
 *
 *  Material shell that takes stress/strain/history observations
 *  from anchor models and prints them to a file.
 *
 * Author: Iuri Rocha, i.rocha@tudelft.nl
 * Date:   Apr 2021
 *
 */

#ifndef OBSERVER_MATERIAL_H
#define OBSERVER_MATERIAL_H

#include <jem/base/Tuple.h>
#include <jem/base/Array.h>
#include <jem/util/Flex.h>
#include <jem/util/ArrayBuffer.h>
#include <jive/util/XTable.h>
#include <jive/app/Module.h>
#include <jive/app/ChainModule.h>
#include <jive/model/Model.h>
#include <jem/io/PrintWriter.h>
#include <jem/io/Logger.h>
#include <jem/util/Timer.h>
#include <jive/algebra/MatrixBuilder.h>

#include "Material.h"
#include "TrainingData.h"

using jem::Array;
using jem::util::Flex;
using jem::util::ArrayBuffer;
using jive::util::XTable;
using jive::app::Module;
using jive::app::ChainModule;
using jem::io::Logger;
using jem::io::PrintWriter;
using jem::io::Writer;
using jem::util::Timer;
using jive::BoolMatrix;
using jive::StringVector;
using jive::algebra::MatrixBuilder;

typedef Ref<Material>              MatRef;

//-------------------------------------------------------------------------------------------------
//   class ObserverMaterial
//-------------------------------------------------------------------------------------------------

class ObserverMaterial : public Material
{
 public:

  static const char*     MATERIAL;
 // static const char*     FEATUREEXTRACTOR;
  static const char*     OUTFILE;
  static const char*     RSEED;
  static const char*     SHUFFLE;

  explicit               ObserverMaterial

    ( const Properties&    props,
      const Properties&    conf,
      const idx_t          rank,
      const Properties&    globdat );

  virtual void           configure

    ( const Properties&    props );

  virtual void           getConfig

    ( const Properties&    conf ) const;

  virtual void           update

    ( const Vector&        stress,
      const Matrix&        stiff,
      const Matrix&       df,
            idx_t          ipoint );

  virtual void           commit  ();

  virtual void           checkCommit

    ( const Properties& params    );

  virtual void           cancel  ();

  virtual void            updateWriteTable

  ( const Vector&         stress,
    const Matrix&         stiff,
    const Matrix&         df,
    idx_t                 ipoint );

  virtual void           getHistory

    ( Vector&              hvals,
      const idx_t          mpoint );

  virtual void           allocPoints

    ( const idx_t          npoints );

  virtual Ref<Material>  clone ( ) const;

 protected:

  virtual               ~ObserverMaterial();

 protected:

  class                  Hist_
  {
   public:
    
                           Hist_ ();
                          ~Hist_ ();

    Flex<Matrix>           strains;
    Flex<Vector>           stresses;
    Flex<Vector>           history;
  //  Flex<Vector>           features;

    MatRef                 extractor;
  };

 protected:

  MatRef                 child_;

  Flex<Matrix>           strains_;
  Flex<Vector>           stresses_;
  Flex<Vector>           history_;
  Flex<Hist_>            hist_;

  idx_t                  rank_;
  idx_t                  strCount_;

  Properties             globdat_;
  Properties             props_;
  Properties             conf_;

  String                 file_;

  Ref<PrintWriter>       out_;

  int                    rseed_;
  bool                   shuffle_;
  bool                   writeHistory_;

};

#endif
