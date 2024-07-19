#ifndef XOUTPUT_MODULE_H
#define XOUTPUT_MODULE_H

#include <jem/io/PrintWriter.h>
#include <jive/app/Module.h>
#include <jive/util/Table.h>

using jem::Ref;
using jem::idx_t;
using jem::String;
using jem::io::Writer;
using jem::io::PrintWriter;
using jem::util::Properties;
using jive::app::Module;
using jive::util::Table;


//-----------------------------------------------------------------------
//   class XOutputModule
//-----------------------------------------------------------------------


class XOutputModule : public Module {
  public:

    typedef XOutputModule      Self;
    typedef Module             Super;

    static const char*        TYPE_NAME;
    static const char*        TABLE_NAME;
    static const char*        TABLE_CONTENTS;

    // public member functions

    explicit                  XOutputModule

    ( const String&           name    = "xoutput" );

    virtual Status            init

    ( const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

    virtual Status            run

    ( const Properties&       globdat );

    virtual void              shutdown

    ( const Properties&       globdat );

    virtual void              configure

    ( const Properties&       props,
      const Properties&       globdat );

    virtual void              getConfig

    ( const Properties&       props,
      const Properties&       globdat )    const;

    static Ref<Module>        makeNew

    ( const String&           name,
      const Properties&       conf,
      const Properties&       props,
      const Properties&       globdat );

  protected:

    virtual                  ~XOutputModule  ();


  private:

    // private member functions

    Ref<PrintWriter>         initWriter_

    ( const String           name,
      const Properties&      myProps ) const;

    void                    shutdownWriter_

    ( const Ref<PrintWriter> outName ) const;

    void                    writeDissipation_

    ( const Properties&      params,
      const Properties&      globdat );

    void                    writeTable_

    ( const Ref<Table>       table,
      const Properties&      globdat );

    // private member data

    bool                     write_;
    bool                     writeCr_;
    bool                     writeCohInfo_;

    idx_t                    interval_;
    bool                     dissipation_;
    String                   tableFilter_;

    Ref<Writer>              out_;

    Ref<PrintWriter>         tracOut_;
    Ref<PrintWriter>         dissOut_;
    Ref<PrintWriter>         tableOut_;;
};


#endif
