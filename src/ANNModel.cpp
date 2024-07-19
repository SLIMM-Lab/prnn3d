/*
 * Copyright (C) 2019 TU Delft. All rights reserved.
 *
 * Class that implements a sequential artificial neural network.
 *
 * Similar to jive::model::ANNModel, but adapted to loop over 
 * layers in reverse order during backpropagation.
 *
 * Author: Iuri Barcelos, i.barceloscarneiromrocha@tudelft.nl
 * Date:   May 2019
 * 
 */

#include <jem/base/System.h>
#include <jem/base/assert.h>
#include <jem/base/Thread.h>
#include <jem/base/Error.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/CancelledException.h>
#include <jem/base/IllegalArgumentException.h>
#include <jem/io/ObjectInput.h>
#include <jem/io/ObjectOutput.h>
#include <jem/util/Properties.h>
#include <jive/util/utilities.h>
#include <jive/util/Assignable.h>
#include <jive/util/XDofSpace.h>
#include <jive/util/Random.h>
#include <jive/model/Names.h>
#include <jive/model/ModelFactory.h>
#include <jive/model/Actions.h>
#include <jive/model/StateVector.h>
#include <jive/fem/XNodeSet.h>
#include <jem/io/FileReader.h>

#include "ANNModel.h"
#include "LearningNames.h"
#include "XAxonSet.h"
#include "XNeuronSet.h"

JEM_DEFINE_SERIABLE_CLASS( jive::model::ANNModel );

JIVE_BEGIN_PACKAGE( model )

using jem::newInstance;
using jem::Error;
using jem::io::FileReader;

//=======================================================================
//   class ANNModel
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------

const char*  ANNModel::TYPE_NAME   = "Neural";
const char*  ANNModel::LAYERS      = "layers";
const char*  ANNModel::SEED        = "rseed";
const char*  ANNModel::SYMMETRIC   = "symmetric";
const char*  ANNModel::WARM_START  = "warmStart";
const char*  ANNModel::OFFLINE     = "offline";

//-----------------------------------------------------------------------
//   constructors & destructor
//-----------------------------------------------------------------------

ANNModel::ANNModel ( const String& name ) :

  Super ( name )

{
}


ANNModel::~ANNModel ()
{}

//-----------------------------------------------------------------------
//   readFrom
//-----------------------------------------------------------------------

void ANNModel::readFrom ( ObjectInput& in )
{
  decode ( in, myName_, layers_ );
}

//-----------------------------------------------------------------------
//   writeTo
//-----------------------------------------------------------------------

void ANNModel::writeTo ( ObjectOutput& out ) const
{
  encode ( out, myName_, layers_ );
}

//-----------------------------------------------------------------------
//   findModel
//-----------------------------------------------------------------------

Model* ANNModel::findModel ( const String& name ) const
{
  if ( name == myName_ )
  {
    return const_cast<Self*> ( this );
  }
  else
  {
    const idx_t  n   = layers_.size ();

    Model*       mod = 0;

    for ( idx_t i = 0; i < n; i++ )
    {
      mod = layers_.getAs<Model>(i)->findModel ( name );

      if ( mod )
      {
        break;
      }
    }

    return mod;
  }
}

//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------

void ANNModel::configure

  ( const Properties&  props,
    const Properties&  globdat )

{

  Properties params;
  
  using jem::System;

  System::out() << "ANNModel. configure. \n";
  
  for ( idx_t i = 0; i < layers_.size(); i++ )
  {
    Ref<Model>  mod = layers_.getAs<Model> ( i );

    mod->configure ( props, globdat );
  }
  
  // Check if warm start is to be used and read weight file if so
  // and check if offline mode is on

  warmFile_ = ""; //mnn.net";
  offline_ = false;

  Properties       myProps = props.findProps ( "model" );
  myProps.find ( warmFile_, WARM_START );
  myProps.find ( offline_, OFFLINE );

  if ( offline_ && warmFile_ == "")
  {
    throw Error ( JEM_FUNC,
      "No file was provided for reading weights and biases. Please run again." );
  }

  if ( warmFile_ != "" )
  {  
     initWeights_( warmFile_, globdat );
     
     for ( idx_t i = 0; i < layers_.size(); i++ )
     {
       Ref<Model>  mod = layers_.getAs<Model> ( i );

       mod->takeAction ( LearningActions::UPDATE, params, globdat );
     }
  }
}

//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------

void ANNModel::getConfig

  ( const Properties&  props,
    const Properties&  globdat ) const

{
  for ( idx_t i = 0; i < layers_.size(); i++ )
  {
    Ref<Model>  mod = layers_.getAs<Model> ( i );

    mod->getConfig ( props, globdat );
  }
}

//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------

bool ANNModel::takeAction

  ( const String&      action,
    const Properties&  params,
    const Properties&  globdat )

{
  using jive::model::Actions;

  using jem::System;
  //System::out() << "@ANNModel::takeAction. Action: " << action << ".\n";
 
  if ( action == Actions::SHUTDOWN )
  {
    for ( idx_t i = 0; i < layers_.size(); ++i )
    {
      Ref<Model> layer = layers_.getAs<Model> ( i );

      layer->takeAction ( action, params, globdat );
    }

    return true;
  }

  if ( action == LearningActions::PROPAGATE  ||
       action == LearningActions::FORWARDJAC ||
       action == LearningActions::RECALL     ||
       action == LearningActions::UPDATE       )
  {
    // Move in the direction input->output 
       
    if ( offline_ == true && action == LearningActions::UPDATE )
    {
       return true;  
    }      
    else
    {
      for ( idx_t i = 0; i < layers_.size(); ++i )
      {
        if ( offline_ ) params.set ( "offline", true );
        Ref<Model> layer = layers_.getAs<Model> ( i );
//	System::out() << "Eval layer " << i << "\n";
        layer->takeAction ( action, params, globdat );
//	System::out() << "Next\n";
      }
    }

    return true;
  }

  if ( action == LearningActions::GETDETERMINANT  )
  {
    // Move in the direction input->matreclayer 

   for ( idx_t i = 0; i <= 2; ++i )
   {
     Ref<Model> layer = layers_.getAs<Model> ( i );
     layer->takeAction ( action, params, globdat );
   }
  }

  if ( action == LearningActions::GETMIDJACOBIAN )
  {
    // Move backwards but stop halfway through

    idx_t size = layers_.size();

    // NB: The core layer is the symmetry line, so
    // the total number of layers should be odd

    JEM_ASSERT ( size % 2 == 1 );

    for ( idx_t i = size - 1; i > size/2; --i )
    {
      Ref<Model> layer = layers_.getAs<Model> ( i );

      layer->takeAction ( LearningActions::GETJACOBIAN, params, globdat );
    }

    return true;
  }

  if ( action == LearningActions::BACKPROPAGATE ||
       action == LearningActions::BACKWARDJAC   ||
       action == LearningActions::GETJACOBIAN      )
  {
    // Move in the direction output->input

    for ( idx_t i = layers_.size()-1; i >= 0; --i )
    {
      Ref<Model> layer = layers_.getAs<Model> ( i );

      layer->takeAction ( action, params, globdat );
    }

    return true;
  }

  return false;
}

//-----------------------------------------------------------------------
//   clear
//-----------------------------------------------------------------------

void ANNModel::clear ()
{
  layers_.clear ();
}

//-----------------------------------------------------------------------
//   reserve
//-----------------------------------------------------------------------

void ANNModel::reserve ( idx_t n )
{
  layers_.reserve ( n );
}

//-----------------------------------------------------------------------
//   trimToSize
//-----------------------------------------------------------------------

void ANNModel::trimToSize ()
{
  layers_.trimToSize ();
}

//-----------------------------------------------------------------------
//   addLayer
//-----------------------------------------------------------------------

void ANNModel::addLayer 

  ( const Ref<Model>&  layer )

{
  JEM_PRECHECK ( layer != nullptr );

  layers_.pushBack ( layer );
}

//=======================================================================
//   related functions
//=======================================================================

//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------

Ref<Model> ANNModel::makeNew

  ( const String&      myName,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  using jive::util::joinNames;
  using jive::util::Assignable;
  using jive::util::XDofSpace;
  using jive::util::Random;
  using jive::fem::XNodeSet;
  using jive::fem::newXNodeSet;

  Properties       myConf  = conf .makeProps ( myName );
  Properties       myProps = props.findProps ( myName );

  Ref<ANNModel>    network;
  StringVector     layers;
  String           name;
  Ref<Model>       layer;
  idx_t            n, seed;

  using jem::System;
  System::out() << "Initialize NN. ANNModel::makeNew. myName" << myName << " \n";

  // Initialize the RNG

  myProps.get ( seed, SEED );
  myConf.set  ( SEED, seed );

  Ref<Random> generator = Random::get ( globdat );
  
  generator->restart ( seed );

  // Create sets for neurons and axons

  Ref<XAxonSet>   axons   = newInstance<XAxonSet>   ();
  Ref<XNeuronSet> neurons = newInstance<XNeuronSet> ( axons );

  axons  ->store ( globdat );
  neurons->store ( globdat );

  // Initialize DofSpace

  Ref<XDofSpace> dofs = XDofSpace::get ( axons->getData(), globdat );

  dofs->addType ( LearningNames::WEIGHTDOF );
  dofs->addType ( LearningNames::BIASDOF   );

  // Create layers

  myProps.get ( layers, LAYERS );
  myConf .set ( LAYERS, layers );

  network = newInstance<ANNModel> ( myName );
  n       = layers.size ();
  
  // Get info on weight sharing
  
  IdxVector weightSharing_ ( n ); 
  weightSharing_ = 0; // Default is no weight sharing is considered

  myProps.find ( weightSharing_, "weightSharing" );

  bool symmetric = false;

  myProps.find ( symmetric, SYMMETRIC );

  for ( idx_t i = 0; i < n; i++ )
  {
    name = layers[i];
    
    String compname ( joinNames ( myName, name ) );

    if ( name.size() == 0 )
    {
      myProps.propertyError ( LAYERS, "empty layer name" );
    }

    if ( i == 0 )
    {
      props.set   ( joinNames ( compname, LearningNames::FIRSTLAYER ), true );
      props.erase ( joinNames ( compname, LearningNames::LASTLAYER  )       );
    }
    else if ( i == n - 1 && !symmetric )
    {
      props.erase ( joinNames ( compname, LearningNames::FIRSTLAYER )       );
      props.set   ( joinNames ( compname, LearningNames::LASTLAYER  ), true );
    }
    else
    {
      props.erase ( joinNames ( compname, LearningNames::FIRSTLAYER )       );
      props.erase ( joinNames ( compname, LearningNames::LASTLAYER  )       );
    }

    if ( weightSharing_[i] > 0 )
    {
    	String imname = joinNames ( myName, layers[i-1] );
    	String name   = imname + "T";

    	Ref<Model> image = network->findModel ( joinNames ( myName, layers[i-weightSharing_[i]] ) );
      
    	System::out() << "i " << i << " n " << n << " " << joinNames ( myName, layers[i-weightSharing_[i]] ) << "\n";

	if ( image == nullptr )
       {
         throw Error ( JEM_FUNC, "Error constructing symmetric network" );
       }

       Properties propsco = props.clone();
       Properties imProps = propsco.getProps ( imname );
       imProps.set ( LearningParams::IMAGE, image );
      // props.set ( "model.first.imageLayer", image );
       if ( i == n - 1 )
       {
          imProps.set ( LearningNames::LASTLAYER, true       );
        }
       props.set ( compname, imProps );
     //  props.set ( "model.lower.type", "Block" );
     //  props.set ( "model.lower.osize", 9 );

       System::out() << "Name of layer with weight sharing " << compname << "\n";
       System::out() << "Props of layer with weight sharing " << props << "\n";

       layer = explicitNewModel ( compname,
                                 conf,
				 props,
				 globdat );
      
       network->addLayer ( layer );       
    }
    else
    {
      System::out() << "Layer name: " << compname << "\n";
      System::out() << "props non sym " << props << "\n";
      layer = explicitNewModel ( compname,
                               conf,
                               props,
                               globdat );

      network->addLayer ( layer );
    }
  }

  // Create symmetric layers

  if ( symmetric )
  {
    for ( idx_t i = 0; i < n-1; ++i )
    {
      String imname = joinNames ( myName, layers[n-2-i] );
      String name   = imname + "T";

      Ref<Model> image = network->findModel ( joinNames ( myName, layers[n-1-i] ) );
      
      System::out() << "i " << i << " n " << n << " " << joinNames ( myName, layers[n-1-i] ) << "\n";

      if ( image == nullptr )
      {
        throw Error ( JEM_FUNC, "Error constructing symmetric network" );
      }

      Properties imProps = props.getProps ( imname );

      imProps.set ( LearningParams::IMAGE, image );

      if ( i == n - 2 )
      {
        imProps.set   ( LearningNames::LASTLAYER, true );
	imProps.erase ( LearningNames::FIRSTLAYER );
      }

      props.set ( name, imProps );

      System::out() << "name " << name << "\npropsss " << props << "\n";

      layer = explicitNewModel ( name,
                                 conf,
				 props,
				 globdat );
      
      network->addLayer ( layer );
    }
  }

  network->trimToSize ();
  
  return network;
  
}

//-----------------------------------------------------------------------
//  initWeights_
//-----------------------------------------------------------------------

void ANNModel::initWeights_

  ( const String&     fname,
    const Properties&  globdat)

{
  using jem::System;
  
  String context ( "ANNModel::initWeights_" );
 
  Ref<DofSpace> dofs  = DofSpace::get ( globdat, context );

  System::out() << "Read file\n";
  
  Ref<FileReader> in    = newInstance<FileReader> ( fname );

  idx_t dc = dofs->dofCount();

  Vector wts ( dc );
  wts = 0.0;

  System::out() << "dc " << dc << "\n";

  for ( idx_t i = 0; i < dc; ++i )
  {
    wts[i] = in->parseDouble();
  }

 System::out() << "Setting state: " << wts << "\n";
 
 Vector state;
 StateVector::get ( state, dofs, globdat );
 state = wts;
}

//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------

void ANNModel::declare ()
{
  ModelFactory::declare ( TYPE_NAME,  & makeNew );
  ModelFactory::declare ( CLASS_NAME, & makeNew );
}

JIVE_END_PACKAGE( model )
