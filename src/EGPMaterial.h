/*
 * 
 *  Copyright (C) 2019 TU Delft. All rights reserved.
 *  
 *  This class implements the Eindhoven Glassy Polymer
 *  material model.
 *
 *  reference: "Extending the EGP constitutive model for polymer 
 *  glasses to multiple relaxation times" Van Breemen et al. 2011
 * 
 *
 */

#ifndef EGPMATERIAL_H
#define EGPMATERIAL_H

#include <jem/base/String.h>
#include <jem/util/Flex.h>
#include <jem/base/Array.h>
#include <jem/util/Timer.h>
#include <jem/io/PrintWriter.h>

#include "Material.h"


using jem::String;
using jem::Tuple;
using jem::Array;
using jem::util::Flex;
using jem::util::Timer;
using jem::io::PrintWriter;

using namespace jem;

typedef Tuple <double,3,3> M33;
typedef Tuple <double,9,9> M99;
typedef Tuple <double,9,1> M91;


// =======================================================
//  class EGP
// =======================================================

// This class implements Eindhoven Glassy Polymer material

class EGPMaterial : public Material
{
 public:

  static const char*      STATE_PROP;

  enum ProblemType {
    PlaneStrain,
    PlaneStress,
    AxiSymmetric
  };

  explicit                EGPMaterial

    ( idx_t                 rank,
      const Properties&     globdat );

  virtual void            configure

    ( const Properties&     props );

  virtual void            getConfig

    ( const Properties&     conf )         const;

  virtual void            update

    ( const Vector&         stress,
      const Matrix&         stiff,
      const Matrix&         df,
      idx_t                 ipoint );

  virtual void            updateWriteTable

  ( const Vector&         stress,
    const Matrix&         stiff,
    const Matrix&         df,
    idx_t                 ipoint );

  virtual void            commit ();

  virtual void            allocPoints

    ( const idx_t           count );

  // Tuple<double,6>         fill3DStress

  // ( const Vector&         v3 )             const;

  virtual void            getHistory

  ( const Vector&         hvals,
    const idx_t           mpoint ) const;

  virtual void            getInitHistory

  ( const Vector&         hvals,
    const idx_t           mpoint ) const;

  virtual idx_t                getHistoryCount ()    const override;

  virtual void                setHistory

    ( const Vector&             hvals,
      const idx_t               mpoint );

 protected:

  virtual                ~EGPMaterial   ();

  double t0a_;
  double t0b_;
  double Gr_;
  double T_;      // absolute temperature [K]
  double dH0a_;   // activation enthalpy for alpha process [J / mol]
  double dH0b_;   // activation enthalpy for beta process [J / mol]
  double SHa_;    //init value of aging dependent part of the activ. enthalpy
  double SHb_;    //init value of aging dependent part of the activ. enthalpy
  double evNskT_; // edwards-vilgis >> Ns * k * T
  double eva_;    // edwards-vilgis >> alpha parameter
  double evh_;    // edwards-vilgis >> eta parameter
  double Cr1a_;   // 1st viscous strain hardening contribution
  double Cr2a_;   // 2nd viscous strain hardening contribution
  double Cr1b_;   // 1st viscous strain hardening contribution
  double Cr2b_;   // 2nd viscous strain hardening contribution
  double SSa_;    // initial value of state parameter Ss
  double SSb_;    // initial value of state parameter Ss
  double r0a_;    // parameter of the softening function
  double r1a_;
  double r2a_;
  double r0b_;
  double r1b_;
  double r2b_;
  double Vacta_;  // stress activation volume for the alpha process
  double Vactb_;
  double ma_;    // pressure dependence of the viscosity for the alpha process
  double mb_;
  double k_;      // bulk modulus
  int    mode_;   // 1 or 2 (alpha / alpha and beta)
  int    nom_;    // total number of modes
  int    nam_;    // number of modes associated with the alpha process
  Vector G_;      // shear moduli of all modes i 
  Vector h0_;     // initial (rejuvenated) viscosities of all modes i
  double R_;      // gas constant
  double kb_;     // Boltzman constant

  String                  stateString_;
  String                  strHardening_;
  ProblemType             state_;
  idx_t                   strCount_;

  Ref<PrintWriter>        lsrOut_;

  bool                    nn_;
  bool                    first_;

protected:

  void            newtonrap_

    (const double dt, double& eqps, const double& eqpsBi, const double& p,
                  
    const Array<M33>& mCpnB, const M33& mtCn, const M33& mtUn,

    const M33& mtB, Array<M33>& mCpn, M33& mbSs, Array<M33>& mbtBen,

    Vector& lambda, Vector& h, Vector& teq, Vector& mteq,

    Vector& S, Vector& gamma, const idx_t noe, const double Jn );


  void            updateEqStress_

    (const Array<M33>& mCpnB, const Array<M33>& mCpn, const M33& mtCn, 

      const M33& mtUn, const Array<M33>& mbtBen, const Vector& lambda, 

      const Vector& teq, const Vector& mteq, M33& mbsda, M33& mbsdb);

  void            calcAgingPar_

    (const Vector& Rg, const Vector& S);

  void            calcSofteningFn_

    (const double& eqps, const Vector& Rg);

  void            calcDSDeqps_

    (double& dSdea, double& dSdeb, const double& eqps);

  void            updateViscosity_

    (const Vector& h, const Vector& Rg, const double& htB, const Vector& S,

      const double& p, const double& t0a, const double& t0b, const Vector& teq);

  void            checkConvergence_

    (idx_t& convpar, const double& Reta, const double& R0);

  
  void            jacobimat_

          (Matrix& mat, 

          const Vector& gamma, const Vector& lambda, const Vector& mteq,

          const Vector& teq, const idx_t& noe, 

          const double& deqpsdt, const double& dt,
           
          const double& dSdea, const double& dSdeb,

          const double& t0a, const double& t0b);


  void            edvilStress_ 

    (M33& mSr, const M33& mI, double J, const M33& mtB);


  void            deviatoricStress_ 

    (M33& mSs, const M33 DF, M33& mtB, Array<M33>& mtBe,

        Array<M33>& mtBeB, Vector& lambda, Vector& h, Vector& teq, 
        
        Vector& mteq, Vector& S, Vector& gamma, 

        double& eqps, double JB, double& eqpsB);


  void            stiffness_ 

    (const bool& nn, const Matrix& StiffFull, const Matrix& Stiff, 

    const M33& mF, const M33& mFB, M33& mtB, 

    const M33& mStress, const Array<M33>& mtBe, const Array<M33>& mtBeB,

    const Vector& h, const Vector& mteq, const Vector& lambda,

    const Vector& teq, const Vector& gamma, double J, double JB );


  
  void            edvilStiff_ 

    (M99& mmStiffrc, M33& mtB, M33& mB, const M33 mF, double J);


  
  // history variables

  class                   Hist_
  {
    public:

                            Hist_

      ( const idx_t nom, const idx_t mode, const idx_t strCount );

                            Hist_

      ( const Hist_&          h );

      void   toVector ( const Vector& vec ) const;

      M33  F;

      Array<M33> mtBe;

      Vector  lmbd, visc, taueq, mtaueq, agepar;

      double  eqps;

      Matrix Stiff;

  };

  Flex<Hist_>             initHist_; 
  Flex<Hist_>             preHist_;    // history of previous load step
  Flex<Hist_>             newHist_;    // history of current iteration
  Flex<Hist_>*            latestHist_; // points to latest history
};

#endif 
