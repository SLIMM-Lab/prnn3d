/*
 *  TU Delft 
 *
 *  Iuri Barcelos, August 2019
 *
 *  Material class that uses a nested neural network to compute
 *  stress and stiffness.
 *
 */

#ifndef ANN_MATERIAL_H
#define ANN_MATERIAL_H

#include <jive/app/Module.h>
#include <jive/model/Model.h>
#include <jive/util/XTable.h>

#include "Material.h"
#include "NData.h"
#include "Normalizer.h"

using jive::app::Module;
using jive::util::XTable;

//-----------------------------------------------------------------------
//   class ANNMaterial
//-----------------------------------------------------------------------

class ANNMaterial : public Material
{
 public:

  typedef Array< Ref<NData>, 1 > Batch;

  static const char*     NETWORK;
  static const char*     WEIGHTS;
  static const char*     NORMALIZER;
  static const char*     RECURRENT;

  explicit               ANNMaterial

    ( const idx_t          rank,
      const Properties&    globdat );

  virtual void           configure

    ( const Properties&    props );

  virtual void           getConfig

    ( const Properties&    conf,
      const Properties&    globdat )   const;

  virtual void           update

    ( const Vector&        stress,
      const Matrix&        stiff,
      const Matrix&        F,
            idx_t          ipoint );

  virtual void            updateWriteTable

  ( const Vector&         stress,
    const Matrix&         stiff,
    const Matrix&         df,
    idx_t                 ipoint );

  virtual void           commit  ();

  virtual void           stressAtPoint

    ( Vector&              stress,
      const Vector&        strain,
      const idx_t          ipoint );

  virtual void           addTableColumns

    ( IdxVector&           jcols,
      XTable&              table,
      const String&        name );
  
  virtual void           allocPoints

    ( const idx_t           npoints );

  inline Ref<Normalizer> getOutNormalizer () const;

  virtual Ref<Material>  clone ( ) const;

 protected:

  virtual               ~ANNMaterial();

 protected:

  idx_t                   rank_;
  bool                    recurrent_;
  bool                    first_;

  Ref<Module>             network_;
  Properties              netData_;
  Ref<Normalizer>         nizer_;

  Batch                   data_;
  Batch                   state_;

  Properties              conf_;

  IdxVector               perm_;
  bool                    perm23_;

 protected:

  void                    initWeights_

    ( const String&         fname        );

  void                     m2mm 

    ( Matrix& mm, const Matrix& m );

   void                    mm2mmr 

    ( Matrix& mmr, const Matrix& mm );

   void                    evaldsigmaRdR

    ( Matrix& dsigmaRdR_constU, const Matrix& stress,

                                const Matrix& R );

   void                    evaldRUdF

    ( Matrix& dRdF, Matrix& dUdF, 

               const Matrix& R, const Matrix& U );

   void                    condenseSymMat

    ( Matrix& Asym, const Matrix& A ); 

   void                    to2ndOrderTensor

    ( Matrix& stressTensor, const Vector& stress ); 

};

#endif
