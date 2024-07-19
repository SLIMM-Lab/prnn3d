
/*
 *
 *  Copyright (C) 2011 Habanera. All rights reserved.
 *
 *  This file is part of Jive, an object oriented toolkit for
 *  solving partial differential equations.
 *
 *  Please read the file LICENSE included in this distribution for
 *  usage and distribution terms.
 *
 *  For more information contact Habanera at http://www.habanera.nl.
 *
 *  Jive version: 2.0
 *  Date:         Mon Oct 24 11:01:03 CEST 2011
 *
 */


#include <jem/base/Thread.h>
#include <jem/base/Exception.h>
#include <jem/base/CancelledException.h>
#include <jem/base/ClassTemplate.h>
#include <jem/base/System.h>
#include <jem/util/None.h>
#include <jem/util/Properties.h>
#include <jive/util/utilities.h>
#include <jive/app/Names.h>
#include <jive/app/Actions.h>
#include <jive/app/ModuleFactory.h>
#include <jive/app/UserconfModule.h>


JEM_DEFINE_CLASS( jive::app::UserconfModule );


JIVE_BEGIN_PACKAGE( app )


using jem::System;
using jem::staticCast;
using jem::newInstance;
using jem::util::ObjFlex;


//=======================================================================
//   class UserconfModule
//=======================================================================

//-----------------------------------------------------------------------
//   static data
//-----------------------------------------------------------------------


const char*  UserconfModule::TYPE_NAME = "Userconf";


//-----------------------------------------------------------------------
//   constructor & destructor
//-----------------------------------------------------------------------


UserconfModule::UserconfModule ( const String& name ) :

  Super ( name )

{
  running_ = false;
}


UserconfModule::~UserconfModule ()
{}


//-----------------------------------------------------------------------
//   init
//-----------------------------------------------------------------------


UserconfModule::Status UserconfModule::init

  ( const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  System::out() << "userconf " << myName_ << " init \n";

  Ref<Module>  mod;
  Status       stat;


  runList_  .clear   ();
  runList_  .reserve ( modules_.size() );
  aliveList_.clear   ();
  aliveList_.reserve ( modules_.size() );

  running_ = false;

  try
  {
    for ( idx_t i = 0; i < modules_.size(); i++ )
    {
      mod  = modules_.getAs<Module> ( i );
      stat = mod->init ( conf, props, globdat );

      aliveList_.pushBack ( mod );

      if      ( stat == EXIT )
      {
        return EXIT;
      }
      else if ( stat != DONE )
      {
        runList_.pushBack ( mod );
      }
    }
  }
  catch ( const jem::Exception& )
  {
    for ( idx_t i = 0; i < aliveList_.size(); i++ )
    {
      mod = aliveList_.getAs<Module> ( i );
      mod->shutdown ( globdat );
    }

    runList_  .clear ();
    aliveList_.clear ();

    throw;
  }

  running_ = true;

  return OK;
}


//-----------------------------------------------------------------------
//   run
//-----------------------------------------------------------------------


UserconfModule::Status UserconfModule::run

  ( const Properties&  globdat )

{
  Ref<Module>  mod;
  Status       stat;


  if ( jem::Thread::cancelled() )
  {
    throw jem::CancelledException (
      getContext (),
      "program execution cancelled"
    );
  }

  if ( ! running_ )
  {
    return EXIT;
  }

  if ( ! runList_.size() )
  {
    return OK;
  }

  for ( idx_t i = 0; i < runList_.size(); i++ )
  {
    mod  = runList_.getAs<Module> ( i );
    stat = mod->run ( globdat );

    if      ( stat == EXIT )
    {
      running_ = false;

      return EXIT;
    }
    else if ( stat == DONE )
    {
      runList_.erase ( i-- );
    }
  }

  return OK;
}


//-----------------------------------------------------------------------
//   shutdown
//-----------------------------------------------------------------------


void UserconfModule::shutdown ( const Properties& globdat )
{
  ObjFlex  list;

  list.swap ( aliveList_ );

  runList_.clear ();

  running_ = false;

  for ( idx_t i = 0; i < list.size(); i++ )
  {
    Ref<Module>  mod = modules_.getAs<Module> ( i );

    mod->shutdown ( globdat );
  }
}


//-----------------------------------------------------------------------
//   configure
//-----------------------------------------------------------------------


void UserconfModule::configure

  ( const Properties&  props,
    const Properties&  globdat )

{
  System::out() << "userconf " << myName_ << " configure \n";
  using jem::util::isNone;
  using jive::util::joinNames;

  Properties    myProps = props.findProps ( myName_ );

  StringVector  names;
  StringVector  names2;

  Ref<Module>   mod;
  Status        stat;

  idx_t         i, n;


  // Erase all or selected modules.

  if ( myProps.find( names, PropNames::MODULES ) )
  {
    for ( i = 0; i < aliveList_.size(); i++ )
    {
      mod = aliveList_.getAs<Module> ( i );
      mod->shutdown ( globdat );
    }

    modules_  .clear ();
    runList_  .clear ();
    aliveList_.clear ();
  }
  else
  {
    Ref<Object>  obj;

    for ( i = 0; i < modules_.size(); i++ )
    {
      mod = modules_.getAs<Module> ( i );

      if ( ! props.find( obj, mod->getName() ) )
      {
        continue;
      }

      if ( isNone( obj ) )
      {
        modules_.erase ( i );
        runList_.erase ( mod );

        if ( aliveList_.erase( mod ) )
        {
          mod->shutdown ( globdat );
        }

        i--;
      }
    }
  }

  // Configure the current modules.

  for ( i = 0; i < modules_.size(); i++ )
  {
    mod = modules_.getAs<Module> ( i );
    mod->configure ( props, globdat );
  }

  if ( myProps.find( names2, PropNames::NEW_MODULES ) )
  {
    n = names.size ();

    if ( n )
    {
      names.reshape ( n + names2.size() );
      names[slice(n,END)] = names2;
    }
    else
    {
      names.ref ( names2 );
    }
  }

  // Add new modules.

  n = names.size ();

  if ( n )
  {
    Properties  conf;

    modules_.reserve ( modules_.size() + n );

    for ( i = 0; i < n; i++ )
    {
      mod = newModule ( joinNames( myName_, names[i] ),
                        conf, props, globdat );

      mod->configure ( props, globdat );

      modules_.pushBack ( mod );

      if ( running_ )
      {
        stat = mod->init ( conf, props, globdat );

        aliveList_.pushBack ( mod );

        if      ( stat == EXIT )
        {
          running_ = false;
          return;
        }
        else if ( stat != DONE )
        {
          runList_.pushBack ( mod );
        }
      }
    }
  }
}


//-----------------------------------------------------------------------
//   getConfig
//-----------------------------------------------------------------------


void UserconfModule::getConfig

  ( const Properties&  conf,
    const Properties&  globdat ) const

{
  for ( idx_t i = 0; i < modules_.size(); i++ )
  {
    Ref<Module>  mod = modules_.getAs<Module> ( i );

    mod->getConfig ( conf, globdat );
  }
}


//-----------------------------------------------------------------------
//   takeAction
//-----------------------------------------------------------------------


bool UserconfModule::takeAction

  ( const String&      action,
    const Properties&  params,
    const Properties&  globdat )

{
  bool  result = false;

  for ( idx_t i = 0; i < modules_.size(); i++ )
  {
    Ref<Module>  mod = modules_.getAs<Module> ( i );

    if ( mod->takeAction( action, params, globdat ) )
    {
      result = true;
    }
  }

  return result;
}


//-----------------------------------------------------------------------
//   makeNew
//-----------------------------------------------------------------------


Ref<Module> UserconfModule::makeNew

  ( const String&      name,
    const Properties&  conf,
    const Properties&  props,
    const Properties&  globdat )

{
  return newInstance<Self> ( name );
}


//-----------------------------------------------------------------------
//   declare
//-----------------------------------------------------------------------


void UserconfModule::declare ()
{
  ModuleFactory::declare ( TYPE_NAME,  & makeNew );
  ModuleFactory::declare ( CLASS_NAME, & makeNew );
}


JIVE_END_PACKAGE( app )
