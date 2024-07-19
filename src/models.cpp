
#include "models.h"
#include "ANNModel.h"
#include <jive/model/ModelFactory.h>


//-----------------------------------------------------------------------
//   declareModels
//-----------------------------------------------------------------------


void declareModels ()
{
  declareDirichletModel();
  declareLoadDispModel();
  declareNeumannModel();
  declareULSolidModel();
  declareAxisymModel();
  declarePeriodicBCModel ();
  declareSRArclenModel   ();
  declareCreepRveModel   (); 
  declareBCModel         ();
  jive::model::ANNModel::declare();
  declareDenseLayer                         ();
  declareSparseLayer                        ();
  declareMatRecLayer                        ();
  declareBlockLayer                         ();
  declareBlockDecLayer                      ();
  declareModDenseLayer                      ();
  declareGPModel                            ();
}
