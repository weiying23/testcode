//****************************************************************************\
//*                    National Numerical Windtunnel                          *
//*         FlowStar -- Flow Simulation Tools for Aerospace Research          *
//*                  Computational Aerodynamics Institute                     *
//*              China Aerodynamics Research&Development Center               *
//*                       Mianyang, Sichuan, China                            *
//****************************************************************************/
///
/// \file   io_log.h
/// \brief  A wrapper class for std::cout with more controlling for MPI parallel
///         and overlap grid.
/// \author tangj
/// \date   2020-02-18
/// \copyright  C.All rights reserved. 2020-2020, CAI/CARDC
/// 
/// \par Usage
/// <pre>        
///   1. Include io_log.h in which .cpp file you want to use the log
///   2. Use mflog::log to replace std::cout
///   3. If user wants to output information by only one processor in
///      MPI prallel environment, call mflog::log.set_one_processor_out()
///      before passing in information by operator "<<".
///   4. If user wants to output info. by each grid for overlap in MPI
///      parallel environment, call mflog::log.set_each_grid_out() before
///      passing in information by operator "<<".
///   5. Call mflog::log.set_all_processors_out() to make all processor
///      output information.
/// </pre>
///
/// \par    Update records:
/// <pre>
/// Date        Author     Description
/// 
/// </pre>

#ifndef MFL_IO_LOG_OUT_H
#define MFL_IO_LOG_OUT_H

// C++ includes
#include <iostream>
#include <string>

#ifdef MPICH
#include <mpi.h>
#endif

namespace mflog
{

// Forward Declarations

// ------------------------------------------------------------------
// OStreamProxy class definition
//
template <typename charT=char, typename traits=std::char_traits<charT> >
class BasicOStreamProxy
 {
 public:
    /**
    * This class is going to be used to proxy for ostream, but other
    * character and traits types are possible
    */
    typedef std::basic_ostream<charT,traits> streamT;

    /**
    * This class is going to be used to proxy for ostream, but other
    * character and traits types are possible
    */
    typedef std::basic_streambuf<charT,traits> streambufT;

    /**
    * Default constructor.  Takes a reference to the 'target' ostream
    * to which we pass output.  The user is responsible for ensuring
    * that this target exists for as long as the proxy does.
    */
    explicit BasicOStreamProxy (streamT& target) : _target(&target) {}

    /**
    * Shallow copy constructor.  Output in the new object is passed to
    * the same target ostream as in the old object.  The user is
    * responsible for ensuring that this target exists for as long as
    * the proxies do.
    */
    BasicOStreamProxy (BasicOStreamProxy& old) : _target(old._target) {}

    // Reset the internal target to a new \p target output stream.
    BasicOStreamProxy& operator= (streamT& target)
    {
        _target = &target;
        return *this;
    }

    // Reset the target to the same output stream as in old
    BasicOStreamProxy& operator= (const BasicOStreamProxy& old)
    {
        _target = old._target;
        return *this;
    }

    // Default destructor.
    ~BasicOStreamProxy () {}

    // Conversion to ostream&, for when we get passed to a function requesting one.
    operator streamT&() { return *_target; }

    // Conversion to const ostream&, for when we get passed to a function requesting one.
    operator const streamT&() const { return *_target; }

    // Redirect any output to the target.
    template<typename T>
    BasicOStreamProxy& operator<< (const T& in) 
    {
        (*_target) << in; return *this;
    }

    // Redirect any ostream manipulators to the target.
    BasicOStreamProxy& operator<< (const streamT& (*in)(streamT&)) 
    {
        (*_target) << in; return *this;
    }

    // Redirect any ios manipulators to the target.
    BasicOStreamProxy& operator<< (const std::basic_ios<charT,traits>& (*in)(std::basic_ios<charT,traits>&)) 
    {
        (*_target) << in; return *this;
    }

    // Redirect any ios_base manipulators to the target.
    BasicOStreamProxy& operator<< (const std::ios_base& (*in)(std::ios_base&)) 
    {
        (*_target) << in; return *this;
    }


    // Return the writable ostream reference.
    streamT* get() { return _target; }

    // Return the const ostream reference.
    const streamT* get() const { return _target; }

 private:

    // The pointer to the "real" ostream we send everything to.
    streamT* _target;

 };

typedef BasicOStreamProxy<> OStreamProxy;


// ------------------------------------------------------------------
// Class to print log onto screen with more controlling parameters
//
class Logout : public OStreamProxy
{
public:  

    typedef std::basic_ios<char, std::char_traits<char> > os_ios_type;

    // Default constructor with a parameter of a reference to the 'target' ostream    
    explicit Logout (streamT& target);

    ~Logout(){};

    // Output object of type T to the target.
    template<typename T>
    Logout& operator<< (const T& in) 
    {
        if (need_out_)
        {
            (*OStreamProxy::get()) << in;
        }        
        return *this;
    }

    // Output object of type ostream to the target.
    Logout& operator<< (streamT& (*in)(streamT&));

    // Redirect any ios manipulators to the target.
    Logout& operator<< (os_ios_type& (*in)(os_ios_type&));

    // Redirect any ios_base manipulators to the target.
    Logout& operator<< (std::ios_base& (*in)(std::ios_base&));

    // Make all processors output message
    void set_all_processors_out();

    // Make only one processor output message
    void set_one_processor_out();

    // Make each grid output message
    void set_each_grid_out();

    // return parallel rank id when use MPI
    int rank_id();

#ifdef MPICH
    // initialize for MPI parallel 
    void mpi_init(const MPI_Comm global_comm_world);
    void mpi_init(const unsigned int grid_id, const MPI_Comm global_comm_world);
#endif

private:

    // 
    bool need_out_;

    // indicate the current rank will output message if
    bool root_rank_out_;

    bool each_grid_out_;

    // processor rank
    int rank_id_;

    // grid id
    int grid_id_;
};

// Make all processors output message
inline void Logout::set_all_processors_out()
{
    need_out_ = true;
}

// Make only one processor output message
inline void Logout::set_one_processor_out()
{
    need_out_ = root_rank_out_;
}

// Make each grid output message
inline void Logout::set_each_grid_out()
{
    need_out_ = each_grid_out_;
}

// return parallel rank id
inline int Logout::rank_id()
{
    return rank_id_;
}

extern Logout log;

} // ~namespace mflog

#endif // ~MFL_IO_LOG_OUT_H
