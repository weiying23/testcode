
#ifndef _PARAMETERS_H
#define _PARAMETERS_H

// you need #include "yourfile.h" to include define of real
#include "number_type.h"
typedef mflow::RealFlow real;
const int MAX_STRING = mflow::MAX_STRING;

// C++ includes
#include <string>
#include <vector>
#include <map>
#include <set>
#include <cstddef>
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <stdexcept>
#include <cstring>

//#include <complex> //have real??
#include <typeinfo> // std::bad_cast

#ifndef _LINUX_
#include <io.h>  // access()
#else
#include <unistd.h> // access()
#include <cxxabi.h> //need for abi::__cxa_demangle
#endif

#define _HAVE_RTTI

// Use the built-in nullptr keyword
//#define _nullptr nullptr

// No nullptr and workaround doesn't compile. This should not be common, but in this case, 
// We have to fall back on the C++03 definition of NULL.
#define _nullptr NULL

//#include <cassert>

#define ASSERT(x) \
    if(!(x))  \
{             \
    std::cout << "ERROR: Assert " << #x << std::endl; \
    std::cout << "On line " << __LINE__ << std::endl; \
    std::cout << "On file " << __FILE__ << std::endl; \
    std::abort();                                     \
}

//#define Throw(e) do { throw e; } while (0)
#define Throw(e) do { std::abort(); } while (0)

#define _error_msg(msg)                                             \
    do {                                                             \
    std::cout << msg << std::endl;                                   \
    std::cout << __FILE__ << " line " << __LINE__ << " " << __DATE__ << " " << __TIME__ << std::endl; \
    Throw("");                               \
    } while(0)


#ifdef _HAS_CPP0X
using std::to_string;
#else
template<typename T>
std::string to_string( const T & i )
{
    std::stringstream s;
    s << i;
    return s.str();
}
#endif

#define Error(msg)                                                                  \
  do                                                                                \
  {                                                                                 \
    std::ostringstream _error_oss_;                                                 \
    _error_oss_ << "\n\n"                                                           \
                << "\n\n*** ERROR ***\n"                                            \
                << msg                                                              \
                << "\n\n";                                                          \
    _error_msg( _error_oss_.str() );                                                    \
    throw std::runtime_error(_error_oss_.str());                                  \
  } while (0)


// We use two different function names to avoid an odd overloading
// ambiguity bug with icc 10.1.008
template <typename Tnew, typename Told>
inline Tnew cast_ptr( Told * oldvar )
{
#if !defined(NDEBUG) && defined(_HAVE_RTTI)
    Tnew newvar = dynamic_cast<Tnew>(oldvar);
    if (!newvar)
    {
        std::ostringstream oss;

        oss << "Failed to convert " << typeid(Told).name()
            << " pointer to " << typeid(Tnew).name()
            << std::endl;
        oss << "The " << typeid(Told).name()
            << " appears to be a "
            << typeid(*oldvar).name() << std::endl;
        Error(oss.str());
    }
    return newvar;
#else
    return(static_cast<Tnew>(oldvar));
#endif
}


#if defined(__GLIBC__) || defined(__GLIBCXX__) //gcc g++
#define _HAVE_GCC_ABI_DEMANGLE
#endif

// demangle() is used by the parameters class for demangling typeid's.
// If configure determined that your compiler does not support demangling, it simply returns the input string.
#if defined(_HAVE_GCC_ABI_DEMANGLE)
inline
std::string demangle( const char * name )
{
    int status = 0;
    std::string ret = name;

    // Actually do the demangling
    char * demangled_name = abi::__cxa_demangle( name, 0, 0, & status );

    // If demangling returns non-NULL, save the result in a string.
    if( demangled_name ) ret = demangled_name;

    // According to cxxabi.h docs, the caller is responsible for deallocating memory.
    std::free( demangled_name );

    return ret;
}
#else
inline
std::string demangle( const char * name ) { return std::string(name); }
#endif

namespace parameter_space
{

/**
 * This class provides the ability to map between arbitrary, user-defined strings and several data types. 
 * This can be used to provide arbitrary user-specified options.
 *
 * \author 
 * \date 
 */
class parameters_base
{
public:
    parameters_base() {} /* Default constructor. Does nothing. */

    parameters_base( const parameters_base & ); /* Copy constructor. */

    virtual ~parameters_base(); /* Destructor. Clears any allocated memory. */

    parameters_base & operator = ( const parameters_base & rhs ); /* Assignment operator. Removes all parameters in \p this and inserts copies of all parameters from \p rhs */

    /* Addition/Assignment operator. Inserts copies of all parameters from \p rhs.
       Any parameters of the same name already in \p this are replaced. // if it is with different type? replace it also!*/
    parameters_base & operator += ( const parameters_base & rhs );

    template<typename T> /* @returns a constant reference to the specified parameter value. Requires, of course, that the parameter exists. */
    const T & get( const std::string & name ) const;

    /* Inserts a new parameter into the object but does not return a writable reference.
       The value of the newly inserted parameter may not be valid. */
    template<typename T>
    void insert( const std::string & name );

    /* @returns a writeable reference to the specified parameter.
       This method will create the parameter if it does not exist, so it can be used to define parameters
       which will later be accessed with the \p get() member. */
    template<typename T>
    T & set( const std::string & name );

    virtual void remove( const std::string & ); /* Removes the specified parameter from the list, if it exists. */

    std::size_t n_parameters() const { return _values.size(); } /* @returns the total number of parameters. */

#ifdef _HAVE_RTTI
    template<typename T>
    std::size_t n_parameters() const; /* @returns the number of parameters of the requested type. */
#endif // _HAVE_RTTI

    virtual void clear(); /* Clears internal data structures & frees any allocated memory. */

    void print( std::ostream & os = std::cout ) const; /* Prints the contents, by default to xxx::out. */

private:
    class value /* Abstract definition of a parameter value. */
    {
    public:
        virtual ~value() {} /* Destructor. */

#ifdef _HAVE_RTTI
        virtual std::string type() const = 0; /* String identifying the type of parameter stored. Must be reimplemented in derived classes. */
#endif // _HAVE_RTTI

        virtual void print( std::ostream & ) const = 0; /* Prints the parameter value to the specified stream. Must be reimplemented in derived classes. */

        virtual value * clone() const = 0; /* Clone this value.  Useful in copy-construction.  Must be reimplemented in derived classes. */
    };

public:
    template<typename T>
    class parameter : public value /* Concrete definition of a parameter value for a specified type. */
    {
    public:
        parameter() {}
        ~parameter() {}

    public:
        const T & get() const { return _value; } /* @returns a read-only reference to the parameter value. */

        T & set() { return _value; } /* @returns a writeable reference to the parameter value. */

#ifdef _HAVE_RTTI
        virtual std::string type() const; /* String identifying the type of parameter stored. */
#endif // _HAVE_RTTI

        virtual void print( std::ostream & ) const; /* Prints the parameter value to the specified stream. */

        virtual value * clone() const; /* Clone this value. Useful in copy-construction. */

    private:
        T _value; /* Stored parameter value. */
    };

public:
    typedef std::map<std::string, value *>::iterator iterator; /* Parameter map iterator. */

    typedef std::map<std::string, value *>::const_iterator const_iterator; /* Constant parameter map iterator. */

    typedef std::map<std::string, value *>::size_type size_type;

    /**
    * @returns \p true if a parameter of type \p T with a specified name exists, \p false otherwise.
    *
    * If RTTI has been disabled then we return \p true if a parameter of specified name exists regardless of its type.
    */
    template <typename T>
    bool have_parameter( const std::string & ) const;
    template <typename T>
    bool have_parameter( parameters_base::const_iterator & it ) const;

    iterator begin();               /* Iterator pointing to the beginning of the set of parameters. */
    const_iterator begin() const;   /* Iterator pointing to the beginning of the set of parameters. */
    iterator end();                 /* Iterator pointing to the end of the set of parameters */
    const_iterator end() const;     /* Iterator pointing to the end of the set of parameters */

protected:
    std::map<std::string, value *> _values; /* Data structure to map names with values. */
    //std::unordered_map<std::string, value *> _values;

public:
    //typedef value value_type;
};

// ------------------------------------------------------------
// parameters_base::parameter<> class inline methods

// This only works with Run-Time Type Information, even though typeid(T) *should* be determinable at compile time regardless...
#ifdef _HAVE_RTTI
template<typename T>
inline
std::string
parameters_base::parameter<T>::type() const
{
    return demangle( typeid(T).name() );
}
#endif

/* Helper functions for printing scalar, vector and vector<vector> types. Called from parameters_base::parameter<T>::print(...). */
template<typename T>
void print_helper( std::ostream & os, const T * param );

template<typename T>
void print_helper( std::ostream & os, const std::vector<T> * param );

template<typename T>
void print_helper( std::ostream & os, const std::vector<std::vector<T> > * param );

template<typename T>
inline
void
parameters_base::parameter<T>::print( std::ostream & os ) const
{
    print_helper( os, static_cast<const T *>(&_value) ); // Call helper function overloaded for basic scalar and vector types
}

template<typename T>
inline
parameters_base::value * 
parameters_base::parameter<T>::clone() const
{
    parameter<T> * copy = new parameter<T>;
    
    ASSERT(copy);

    copy->_value = _value;

    return copy;
}

// ---------------------------------------------------------------------------------
// parameters_base class inline methods
inline
void parameters_base::clear() // since this is inline we must define it before its first use (for some compilers)
{
    while( !_values.empty() )
    {
        parameters_base::iterator it = _values.begin();

        delete it->second;
        it->second = _nullptr;

        _values.erase(it);
    }
}

inline
parameters_base::parameters_base( const parameters_base & p )
{
    * this = p;
}

inline
parameters_base::~parameters_base()
{
    this->clear();
}

inline
parameters_base & parameters_base::operator= ( const parameters_base & rhs )
{
    if( this == & rhs )
        ;
    else
    {
        this->clear();
        *this += rhs;
    }

    return *this;
}

inline
parameters_base & parameters_base::operator+= ( const parameters_base & rhs )
{
    if( this == & rhs )
        ;
    else
    for( parameters_base::const_iterator it = rhs._values.begin(); it != rhs._values.end(); ++it )
    {
        if( _values.find(it->first) != _values.end() ) delete _values[it->first];
        _values[it->first] = it->second->clone();
    }

    return * this;
}

inline
void parameters_base::print( std::ostream & os ) const
{
    parameters_base::const_iterator it = _values.begin();

    os  << "\n Name\t\t\t Type\t\t\t Value\t\t\t #Size = " << _values.size() << "\n"
        << "{----------------------------------------------------------------\n";

    while( it != _values.end() )
    {
        os << " " << it->first
           << "\t\t\t "
#ifdef _HAVE_RTTI
           << it->second->type()
#endif // _HAVE_RTTI
           << "\t\t\t ";   it->second->print(os);
        os << '\n';

        ++it;
    }

    os << "----------------------------------------------------------------}\n";
}

// Declare this now that parameters_base::print() is defined.
// By declaring this early we can use it in subsequent methods. Required for gcc-4.0.2
inline
std::ostream & operator << ( std::ostream & os, const parameters_base & p )
{
    p.print(os);
    return os;
}

// name exist && same type return true
template<typename T>
inline
bool parameters_base::have_parameter( parameters_base::const_iterator & it ) const
{
    //parameters_base::const_iterator it = _values.find(name);

    if( it != _values.end() ) //found it
#ifdef _HAVE_RTTI
        if( dynamic_cast<const parameter<T> *>(it->second) != _nullptr ) //same type
#else // _HAVE_RTTI
        if( cast_ptr<const parameter<T> *>(it->second) != _nullptr )
#endif // _HAVE_RTTI
            return true;

    return false;
}

// name exist && same type
template<typename T>
inline
bool parameters_base::have_parameter( const std::string & name ) const
{
    parameters_base::const_iterator it = _values.find(name);

    return have_parameter<T>( it );
}

template<typename T>
inline
const T & parameters_base::get( const std::string & name ) const
{
    parameters_base::const_iterator it = _values.find(name);

    if( ! this->have_parameter<T>(it) )
    {
        std::ostringstream oss;

        oss << "ERROR: no <";
#ifdef _HAVE_RTTI
        oss << ' ' << demangle(typeid(T).name()) << ' ';
#endif
        oss << "> parameter named \""
            << name << "\" found.\n\n"
            << "Known parameters:\n"
            << *this;

        _error_msg(oss.str());
    }

    ///ASSERT( it != _values.end() ); //_assert
    ///ASSERT( it->second );

    return cast_ptr<parameter<T> *>(it->second)->get();
}


//does not exist new
//exist but different type delete & renew
//exist and same type donothing, and this is fit for group
template<typename T>
inline
void parameters_base::insert( const std::string & name )
{
    parameters_base::iterator it = _values.find(name);

    if( it != _values.end() ) //found it
    {
#ifdef _HAVE_RTTI
        if( dynamic_cast<const parameter<T> *>(it->second) == _nullptr ) //different type
#else // _HAVE_RTTI
        if( cast_ptr<const parameter<T> *>(it->second) == _nullptr )
#endif // _HAVE_RTTI
        {
            delete it->second;
            it->second = new parameter<T>;
        }
    }
    else
    {
        _values[name] = new parameter<T>;
    }
}


template<typename T>
inline
T & parameters_base::set( const std::string & name )
{
    this->insert<T>( name );

    return cast_ptr<parameter<T> *>(_values[name])->set();
}


inline
void parameters_base::remove( const std::string & name )
{
    parameters_base::iterator it = _values.find(name);

    if( it != _values.end() )
    {
        delete it->second;
        it->second = _nullptr;

        _values.erase(it);
    }
}


#ifdef _HAVE_RTTI
template<typename T>
inline
std::size_t parameters_base::n_parameters() const
{
    std::size_t count = 0;

    parameters_base::const_iterator it = _values.begin();
    const parameters_base::const_iterator vals_end = _values.end();

    for( ; it != vals_end; ++it )
        if( dynamic_cast<parameter<T> *>(it->second) != _nullptr )
            ++count;

    return count;
}
#endif

inline
parameters_base::iterator parameters_base::begin()
{
    return _values.begin();
}

inline
parameters_base::const_iterator parameters_base::begin() const
{
    return _values.begin();
}

inline
parameters_base::iterator parameters_base::end()
{
    return _values.end();
}

inline
parameters_base::const_iterator parameters_base::end() const
{
    return _values.end();
}

//non-member scalar print function
template<typename P>
void print_helper( std::ostream & os, const P * param )
{
    os << *param;
}

//non-member vector print function
template<typename P>
void print_helper( std::ostream & os, const std::vector<P> * param )
{
    for(typename std::vector<P>::size_type i=0; i!=param->size(); ++i)
        os << (*param)[i] << " ";
}

//non-member vector<vector> print function
template<typename P>
void print_helper( std::ostream & os, const std::vector<std::vector<P> > * param )
{
    for(typename std::vector<P>::size_type i=0; i!=param->size(); ++i)
        for(typename std::vector<P>::size_type j=0; j!=(*param)[i].size(); ++j)
            os << (*param)[i][j] << " ";
}


/* The main class responsible for handling user-defined parameters. */
class parameters : public parameters_base
{
public:
    parameters();
    parameters( const parameters & rhs );
    parameters( const parameters_base & rhs );

    virtual ~parameters();

    virtual void remove( const std::string & name );
    virtual void clear();
        
    template<typename T> T & set( const std::string & name );

    using parameters_base::insert;
    //template<typename T> void insert( const std::string & name );

    using parameters_base::get;

    /* These methods add an parameter */
    template<typename T, typename S>
    void add( const std::string & name, const S & val_, bool is_const = false );

    ///void add( const std::string & name, parameters::value_type * val_, bool is_const = false );

    //nested_group_names
    void insert_group( const std::string & name ) { insert<group_of_params_type>( name ); }

private:
    /* This method checks to make sure that we aren't adding a parameter with the same name but a different type. It throws a Error if an inconsistent type */
    template <typename T>
    void check_consistent_type( const std::string & name, const parameters::const_iterator it ) const;

    template <typename T>
    void check_consistent_type( const std::string & name ) const;

public:
    bool is_const( const std::string & name ) const; /* Returns a boolean indicating whether the specified parameter is constant or not */

    std::string type( const std::string & name );/* Return the type of the requested parameter by name */

    /* Copy and Copy/Add operators */
    //using parameters_base::operator=;
    //using parameters_base::operator+=;
    parameters & operator=  ( const parameters & rhs );
    parameters & operator+= ( const parameters & rhs );
    
private:
    /// This method is called when adding a parameter with a default value, can be specialized for non-matching types
    template<typename T, typename S>
    void set_param_helper(const std::string & name, T & l_value, const S & r_value) const; 

    /// If a parameters value was constant, it will appear in this list
    std::set<std::string> _const_params;

public: //copy and modified from global function, be careful when use it
    typedef parameters * group_of_params_type_ptr;
    typedef parameters   group_of_params_type;

    void add_sub_group( const std::string & name, const group_of_params_type & val_, bool is_const = false ) 
    { add<group_of_params_type,group_of_params_type>( name, val_, is_const ); }

    group_of_params_type & get_sub_group( const std::string & name ) { return set<group_of_params_type>( name ); }

    std::size_t n_sub_groups() const { return n_parameters<group_of_params_type>(); } /* @returns the total number of sub-groups. */

    /// return true if the sub-group 'name' is exist
    bool have_sub_group( const std::string &name ) const { return have_parameter<group_of_params_type>(name); }

private:
    //search group top-to-down
    //nested_group_names "" means this::, a b means this::a::b::
    group_of_params_type & get_nested_group( const std::vector<std::string> & nested_group_names )
    {
        group_of_params_type_ptr p_actual_group = (this);

        size_t i = 0;
        size_t size = nested_group_names.size();
        while( i < size ) p_actual_group = const_cast<group_of_params_type_ptr>( & (p_actual_group->get<group_of_params_type>(nested_group_names[i++])) );

        return * p_actual_group;
    }

public:
    //add var name to actual group, set value and is_const
    //nested_group_names(name_of_space) a b means a::b::
    template<typename T>
    void add( const std::string & name, const T & value_, const std::vector<std::string> & nested_group_names, bool is_const = false )
    {
        get_nested_group( nested_group_names ).add<T>( name, value_, is_const );
    }

private:
    //add vector name[size] to actual group, set value and is_const
    template<typename T>
    void add( const std::string & name, size_t size, const std::vector<T> & value_, group_of_params_type & actual_group, bool is_const )
    {
        if( size == 0 ) //type name[] = ...; name[0] or name[neg] processed
        {
            if( value_.size() == 0 )
            {
                Error( "Array parameter \"" << name << "\" has blank content and not assign its size." );
            }
            else
            {
                actual_group.add< std::vector<T> >( name, value_, is_const );
            }
        }
        else
        {
            if( value_.size() > size )
            {
                Error( "Array parameter \"" << name << "\" has too many initial contents." );
            }
            else if( value_.size() == size )
            {
                actual_group.add< std::vector<T> >( name, value_, is_const );
            }
            else
            {
                Error( "Array parameter \"" << name << "\" has too little initial contents." );
                std::vector<T> r_value(value_);
                r_value.resize(size);
                actual_group.add< std::vector<T> >( name, r_value, is_const );
            }
        }
    }

public:
    //add vector name[size] to actual group, set value and is_const
    //nested_group_names(name_of_space) a b means a::b::
    template<typename T>
    void add( const std::string & name, size_t size, const std::vector<T> & value_, const std::vector<std::string> & nested_group_names, bool is_const = false )
    {
        add( name, size, value_, get_nested_group( nested_group_names ), is_const );
    }

private:
    //add vector vector name[size0][size1])to actual group, set value and is_const
    template<typename T>
    void add( const std::string & name, size_t size0, size_t size1, const std::vector< std::vector<T> > & value_, group_of_params_type & actual_group, bool is_const )
    {
        size_t v_size = value_.size();

        if( size0 == 0 ) //type name[][x] = ...; name[0][x] or name[neg][x] processed
        {
            if( v_size == 0 )
            {
                Error( "Array2D parameter \"" << name << "\" has blank content." );
            }
            else
            {
                if( size1 == 0 )
                {
                    for( size_t i = 0; i != v_size; ++i )
                    {
                        if( value_[i].empty() ) Error( "Array2D parameter \"" << name << "\"[ " << i << " ] has blank content." );
                    }
                }
                else
                {
                    for( size_t i = 0; i != v_size; ++i )
                    {
                        if( value_[i].size() != size1 ) Error( "Array2D parameter \"" << name << "\"[ " << i << " ] size mismatching." );
                    }
                }

                actual_group.add< std::vector< std::vector<T> > >( name, value_, is_const );
            }
        }
        else //size0 != 0
        {
            if( v_size == size0 )
            {
                if( size1 == 0 )
                {
                    for( size_t i = 0; i != v_size; ++i )
                    {
                        if( value_[i].empty() ) Error( "Array2D parameter \"" << name << "\"[ " << i << " ] has blank content." );
                    }
                }
                else
                {
                    for( size_t i = 0; i != v_size; ++i )
                    {
                        if( value_[i].size() != size1 ) Error( "Array2D parameter \"" << name << "\"[ " << i << " ] size mismatching." );
                    }
                }

                actual_group.add< std::vector< std::vector<T> > >( name, value_, is_const );
            }
            else if( v_size > size0 )
            {
                Error( "Array2D parameter \"" << name << "\" has too many initial contents." );
            }
            else 
            {
                Error( "Array2D parameter \"" << name << "\" has too little initial contents." );
                //
                //std::vector< vector<T> > r_value(value_);
                //r_value.resize(size0);
                //actual_group.add_param<vector< vector<T> >>( name, r_value, is_const );
            }
        }
    }

public:
    //add vector vector name[size0][size1])to actual group, set value and is_const
    //nested_group_names(name_of_space) a b means a::b::
    template<typename T>
    void add( const std::string & name, size_t size0, size_t size1, const std::vector< std::vector<T> > & value_, const std::vector<std::string> & nested_group_names, bool is_const = false )
    {
        add( name, size0, size1, value_, get_nested_group( nested_group_names ), is_const );
    }

private:
    //all group ptr from top to bottom in a Array, the caller be responsible for deleting the Array
    group_of_params_type_ptr * get_array_of_nested_groups_ptr( const std::vector<std::string> & nested_group_names )
    {
        size_t size = nested_group_names.size();

        group_of_params_type_ptr * array_of_nested_groups_ptr = new group_of_params_type_ptr[size + 1];

        array_of_nested_groups_ptr[0] = (this); //means top group

        size_t i = 0;
        while( i < size )
        {
            array_of_nested_groups_ptr[i+1] = const_cast<group_of_params_type_ptr>( & (array_of_nested_groups_ptr[i]->get<group_of_params_type>(nested_group_names[i])) );
            ++i;
        }

        return array_of_nested_groups_ptr;
    }

public:
    //find name in nested_group bottom-to-up;
    //size of nested_group_names >= 0; a b means ::a::b:: name, first in b, then in a,finally in the top group
    template <typename T>
    const T & get( const std::string & name, const std::vector<std::string> & nested_group_names )
    {
        size_t size = nested_group_names.size();

        if( size == 0 ) return get<T>( name );

        group_of_params_type_ptr * array_of_nested_groups_ptr = get_array_of_nested_groups_ptr( nested_group_names );

        for( int i = size; i >= 0; --i )
        {
            if( array_of_nested_groups_ptr[i]->have_parameter<T>( name ) )
            {
                const T & ret = array_of_nested_groups_ptr[i]->get<T>( name );

                delete [] array_of_nested_groups_ptr;

                return ret;
            }
        }

        delete [] array_of_nested_groups_ptr;

        Error( "Parameter \"" << name << "\" not found in group \"" << nested_group_names[size-1] << "\" and the upper group." );
    }

    //find scoped_name[0] bottom-to-up in nested_group, the find scoped_name[1], till last var name
    //size of nested_group_names >= 0; size of scoped_name >= 1; //scoped_name a var means a::var
    template <typename T>
    const T & get( const std::vector<std::string> & scoped_name, const std::vector<std::string> & nested_group_names = std::vector<std::string>() )
    {
        if( scoped_name.size() == 1 ) return get<T>( scoped_name[0], nested_group_names ); //only var in scoped_name
        else //i.e. group0::var;
        {
            group_of_params_type_ptr p_actual_group = const_cast<group_of_params_type_ptr>( & (get<group_of_params_type>(scoped_name[0], nested_group_names)) );

            for( size_t j = 1; j < scoped_name.size() - 1; ++j )
                p_actual_group = const_cast<group_of_params_type_ptr>( & (p_actual_group->get<group_of_params_type>(scoped_name[j])) );

            return p_actual_group->get<T>( scoped_name.back() );

            ///end another method

            size_t size = nested_group_names.size();

            group_of_params_type_ptr * array_of_nested_groups_ptr = get_array_of_nested_groups_ptr( nested_group_names );

            for( int i = size; i >= 0; --i )
            {
                if( array_of_nested_groups_ptr[i]->have_parameter<group_of_params_type>(scoped_name[0]) )
                {
                    group_of_params_type_ptr p_actual_group = const_cast<group_of_params_type_ptr>( & (array_of_nested_groups_ptr[i]->get<group_of_params_type>(scoped_name[0])) );

                    for( size_t j = 1; j < scoped_name.size() - 1; ++j )
                        p_actual_group = const_cast<group_of_params_type_ptr>( & (p_actual_group->get<group_of_params_type>(scoped_name[j])) );

                    const T & ret = p_actual_group->get<T>( scoped_name.back() );

                    delete [] array_of_nested_groups_ptr;

                    return ret;
                }
            }

            delete [] array_of_nested_groups_ptr;

            Error( "Scope \"" << scoped_name[0] << "\" of parameter \"" << scoped_name.back() << "\" not found." );
        }
    }

    template <typename T>
    bool is_const( const std::string & name, const std::vector<std::string> & nested_group_names )
    {
        std::vector<std::string>::size_type size = nested_group_names.size();

        if( size == 0 ) return is_const( name );

        group_of_params_type_ptr * array_of_nested_groups_ptr = get_array_of_nested_groups_ptr( nested_group_names );

        for( int i = size; i >= 0; --i )
        {
            if( array_of_nested_groups_ptr[i]->have_parameter<T>( name ) )
            {
                bool ret = array_of_nested_groups_ptr[i]->is_const( name );

                delete [] array_of_nested_groups_ptr;

                return ret;
            }
        }

        delete [] array_of_nested_groups_ptr;

        Error( "Parameter \"" << name << "\" not found in scope \"" << nested_group_names[size-1] << "\" and the upper scope." );
    }

    //var, A::var, A::B::var...
    template <typename T>
    bool is_const( const std::vector<std::string> & scoped_name, const std::vector<std::string> & nested_group_names = std::vector<std::string>() )
    {
        if( scoped_name.size() == 1 ) //just var
            return is_const<T>( scoped_name[0], nested_group_names );
        else //i.e. A::var;
        {
            group_of_params_type_ptr p_actual_group = const_cast<group_of_params_type_ptr>( & (get<group_of_params_type>(scoped_name[0], nested_group_names)) );

            for( size_t j = 1; j < scoped_name.size() - 1; ++j )
                p_actual_group = const_cast<group_of_params_type_ptr>( & (p_actual_group->get<group_of_params_type>(scoped_name[j])) );

            return p_actual_group->is_const( scoped_name.back() );

            ///end another method

            size_t size = nested_group_names.size();

            group_of_params_type_ptr * array_of_nested_groups_ptr = get_array_of_nested_groups_ptr( nested_group_names );

            for( int i = size; i >= 0; --i )
            {
                if( array_of_nested_groups_ptr[i]->have_parameter<group_of_params_type>(scoped_name[0]) )
                {
                    group_of_params_type_ptr p_actual_group = const_cast<group_of_params_type_ptr>( & (array_of_nested_groups_ptr[i]->get<group_of_params_type>(scoped_name[0])) );

                    for( size_t j = 1; j < scoped_name.size() - 1; ++j )
                        p_actual_group = const_cast<group_of_params_type_ptr>( & (p_actual_group->get<group_of_params_type>(scoped_name[j])) );

                    bool ret = p_actual_group->is_const( scoped_name.back() );

                    delete [] array_of_nested_groups_ptr;

                    return ret;
                }
            }

            delete [] array_of_nested_groups_ptr;

            Error( "Scope \"" << scoped_name[0] << "\" of parameter \"" << scoped_name.back() << "\" not found." );
        }
    }

    //nested_group_names "" means top group; a::b:: means ::a::b::
    void insert_group( const std::string & name, const std::vector<std::string> & nested_group_names )
    {
        get_nested_group( nested_group_names ).insert<group_of_params_type>( name );
    }

    //nested_group_names "" means this group; a::b:: means this::a::b::
    //no name, 
    std::string insert_group( const std::vector<std::string> & nested_group_names )
    {
        group_of_params_type & actual_group = get_nested_group( nested_group_names );

        size_t size = nested_group_names.size();
        std::string group_name = nested_group_names[size-1];
        size_t i = 0;
        while( actual_group.have_parameter<group_of_params_type>( group_name + "_" + to_string(i) ) ) ++i;

        group_name += "_" + to_string(i);

        actual_group.insert<group_of_params_type>( group_name );

        return group_name;
    }

    bool data_to_flowstar( const_iterator it, std::string & name, mflow::IntType & type, mflow::IntType & size, void * & value_ )
    {
        //copy form flowstar\Common\include\data_pool.h
        const mflow::IntType fs_INT    = 1;
        const mflow::IntType fs_FLOAT  = 2;
        const mflow::IntType fs_DOUBLE = 3;
        const mflow::IntType fs_STRING = 4;
        const mflow::IntType fs_CHAR   = 5;
        const mflow::IntType fs_LONG   = 6;

#ifdef _HAVE_RTTI
        if( dynamic_cast<const parameter<int> *>(it->second) != _nullptr ) //same type
#else // _HAVE_RTTI
        if( cast_ptr<const parameter<int> *>(it->second) != _nullptr )
#endif // _HAVE_RTTI
        {
            name = it->first; type = fs_INT; size = 1;
            int param_value = cast_ptr<parameter<int>*>(it->second)->get();
            value_ = new mflow::IntType[size];
            (static_cast<mflow::IntType*>(value_))[0] = param_value;   
            return true;
        }
        else
#ifdef _HAVE_RTTI
        if( dynamic_cast<const parameter<float> *>(it->second) != _nullptr ) //same type
#else // _HAVE_RTTI
        if( cast_ptr<const parameter<float> *>(it->second) != _nullptr )
#endif // _HAVE_RTTI
        {
            name = it->first; type = fs_FLOAT; size = 1;
            value_ = new float[size];
            memcpy( value_, &(cast_ptr<parameter<float>*>(it->second)->get()), sizeof(float) * size );

            return true;
        }
        else
#ifdef _HAVE_RTTI
        if( dynamic_cast<const parameter<double> *>(it->second) != _nullptr ) //same type
#else // _HAVE_RTTI
        if( cast_ptr<const parameter<double> *>(it->second) != _nullptr )
#endif // _HAVE_RTTI
        {
            name = it->first; type = fs_DOUBLE; size = 1;
            value_ = new double[size];
            memcpy( value_, &(cast_ptr<parameter<double>*>(it->second)->get()), sizeof(double) * size );

            return true;
        }
        else
#ifdef _HAVE_RTTI
        if( dynamic_cast<const parameter< std::vector<int> > *>(it->second) != _nullptr ) //same type
#else // _HAVE_RTTI
        if( cast_ptr<const parameter< std::vector<int> > *>(it->second) != _nullptr )
#endif // _HAVE_RTTI
        {
            name = it->first; type = fs_INT; size = cast_ptr<parameter< std::vector<int> >*>(it->second)->get().size();

            const std::vector<int> &param_value = cast_ptr<parameter< std::vector<int> >*>(it->second)->get();

            value_ = new mflow::IntType[size];

            for (int iparam = 0; iparam < size; ++iparam)
            {
                (static_cast<mflow::IntType*>(value_))[iparam] = param_value[iparam];
            }
            return true;
        }
        else
#ifdef _HAVE_RTTI
        if( dynamic_cast<const parameter< std::vector<float> > *>(it->second) != _nullptr ) //same type
#else // _HAVE_RTTI
        if( cast_ptr<const parameter< std::vector<float> > *>(it->second) != _nullptr )
#endif // _HAVE_RTTI
        {
            name = it->first; type = fs_FLOAT; size = cast_ptr<parameter< std::vector<float> >*>(it->second)->get().size();
            value_ = new float[size];
            memcpy( value_, &(cast_ptr<parameter< std::vector<float> >*>(it->second)->get()[0]), sizeof(float) * size );

            return true;
        }
        else
#ifdef _HAVE_RTTI
        if( dynamic_cast<const parameter< std::vector<double> > *>(it->second) != _nullptr ) //same type
#else // _HAVE_RTTI
        if( cast_ptr<const parameter< std::vector<double> > *>(it->second) != _nullptr )
#endif // _HAVE_RTTI
        {
            name = it->first; type = fs_DOUBLE; size = cast_ptr<parameter< std::vector<double> >*>(it->second)->get().size();
            value_ = new double[size];
            memcpy( value_, &(cast_ptr<parameter< std::vector<double> >*>(it->second)->get()[0]), sizeof(double) * size );

            return true;
        }
        else
#ifdef _HAVE_RTTI
        if( dynamic_cast<const parameter< std::string > *>(it->second) != _nullptr ) //same type
#else // _HAVE_RTTI
        if( cast_ptr<const parameter< std::string > *>(it->second) != _nullptr )
#endif // _HAVE_RTTI
        {
            name = it->first; type = fs_STRING; size = 1;
            int buf_size = cast_ptr<parameter< std::string >*>(it->second)->get().size() + 1;
            value_ = new char[buf_size]; char * t_value = (char*)value_;
            strcpy( (char*)value_, cast_ptr<parameter< std::string >*>(it->second)->get().c_str() );
            //t_value[buf_size-1] = '\0';
            //memcpy( value_, &(cast_ptr<parameter< std::string >*>(it->second)->get()[0]), buf_size - 1 );

            return true;
        }
        else
#ifdef _HAVE_RTTI
        if( dynamic_cast<const parameter< std::vector< std::string > > *>(it->second) != _nullptr ) //same type
#else // _HAVE_RTTI
        if( cast_ptr<const parameter< std::vector< std::string > > *>(it->second) != _nullptr )
#endif // _HAVE_RTTI
        {
            name = it->first; type = fs_STRING; size = cast_ptr<parameter< std::vector< std::string > >*>(it->second)->get().size();

            int buf_size = size * MAX_STRING; //

            value_ = new char[buf_size]; char * t_value = (char*)value_;

            int p = 0;
            for( int i = 0; i < size; ++i )
            {
                strcpy( t_value + p, cast_ptr<parameter< std::vector< std::string > >*>(it->second)->get()[i].c_str() );
                //memcpy( ((char*)value_) + p, &(cast_ptr<parameter< std::vector< std::string > >*>(it->second)->get()[i][0]), xsize - 1 );
                p += MAX_STRING;
            }

            return true;
        }

        return false;
    }
};


// ---------------------------------------------------------------------------------
// parameters class inline methods
inline
parameters::parameters() : parameters_base()
{
}

inline
parameters::parameters( const parameters & rhs ) : parameters_base()
{
    * this = rhs;
}

inline
parameters::parameters( const parameters_base & rhs )
{
    parameters_base::operator=(rhs);
}

inline
parameters::~parameters()
{
    //std::cout << "~input_parameters" << std::endl;
}

inline
void parameters::remove( const std::string & name )
{
    parameters_base::remove( name );
    _const_params.erase( name );
}

inline
void
parameters::clear()
{
    parameters_base::clear();
    _const_params.clear();
}

// Template and inline function implementations

template<typename T>
inline
T & 
parameters::set( const std::string & name )
{
    parameters_base::const_iterator it = _values.find(name);

    if( this->have_parameter<T>(it) ) // found and same type
    {
        if( is_const( name ) )
            Error("Attempting to set parameter \"" << name << "\" with type (" << demangle(typeid(T).name()) 
            << ")\nbut the parameter is constant, you can remove it");
        else
            return cast_ptr<parameter<T>*>(it->second)->set();
    }
    else
    {
        if( it == _values.end() ) //not found
            Error("Attempting to set parameter \"" << name << "\" with type (" << demangle(typeid(T).name()) 
            << ")\nbut the parameter does not exists, you can add it");
        else
            Error("Attempting to set parameter \"" << name << "\" with type (" << demangle(typeid(T).name()) 
                << ")\nbut the parameter already exists as type (" << it->second->type() << "), you can remove it");
    }
}

template<typename T, typename S>
inline
void
parameters::set_param_helper( const std::string & /*name*/, T & l_value, const S & r_value ) const
{
    l_value = r_value;
}

template<typename T, typename S>
inline
void
parameters::add( const std::string & name, const S & val_, bool is_const )
{
    parameters::const_iterator it = _values.find(name);

    if( it != _values.end() ) //found
        Error("Attempting to add parameter \"" << name << "\" with type (" << demangle(typeid(T).name())
        << ")\nbut the parameter already exists, its type is (" << it->second->type() << "), you can remove it, or set a new value");

    _values[name] = new parameter<T>;
    T & l_value = cast_ptr<parameter<T>*>(_values[name])->set();
    set_param_helper(name, l_value, val_);

    if( is_const ) _const_params.insert(name);
}

//exist && different type --> error
// Do we have a paremeter with the same name but a different type?
template <typename T>
inline
void
parameters::check_consistent_type( const std::string & name, const parameters::const_iterator it ) const
{
    //parameters::const_iterator it = _values.find(name);
    if( it != _values.end() && dynamic_cast<const parameter<T>*>(it->second) == _nullptr )
        ///Error("Attempting to set parameter \"" << name << "\" with type (" << demangle(typeid(T).name()) << ")\nbut the parameter already exists as other type" );
        Error("Attempting to set parameter \"" << name << "\" with type (" << demangle(typeid(T).name()) 
        << ")\nbut the parameter already exists as type (" << it->second->type() << ")");
}

//exist && different type --> error
// Do we have a paremeter with the same name but a different type?
template<typename T>
inline
void
parameters::check_consistent_type( const std::string & name ) const
{
    parameters::parameters_base::const_iterator it = _values.find(name);

    if( it != _values.end() && dynamic_cast<const parameter<T>*>(it->second) == _nullptr )
        ///Error("Attempting to set parameter \"" << name << "\" with type (" << demangle(typeid(T).name()) << ")\nbut the parameter already exists as other type" );
            Error("Attempting to set parameter \"" << name << "\" with type (" << demangle(typeid(T).name()) 
            << ")\nbut the parameter already exists as type (" << it->second->type() << ")");
}

inline
parameters &
parameters::operator= ( const parameters & rhs )
{
    if( this == & rhs )
        ;
    else
    {
        parameters_base::operator=(rhs);
        _const_params = rhs._const_params;
    }

    return *this;
}

inline
parameters &
parameters::operator+= ( const parameters & rhs )
{
    if( this == & rhs )
        ;
    else
    {
        parameters_base::operator+=(rhs);
        _const_params.insert( rhs._const_params.begin(), rhs._const_params.end() );
    }

    return *this;
}

inline
bool
parameters::is_const( const std::string & name ) const
{
    return _const_params.find( name ) != _const_params.end();
}

inline
std::string
parameters::type( const std::string & name )
{
    return _values[name]->type();
}




inline
void trim_all( std::string & str, const std::string & delimiter = " " )
{
    if( str.empty() ) return;

    std::string::size_type pos = 0;

    while( ( pos = str.find_first_of(delimiter, pos) ) != std::string::npos ) str.erase( pos, 1 );
}


} //namespace parameter_space

extern int read_parameters( const std::string & file_name, parameter_space::parameters & params );

#endif // _PARAMETERS_H
