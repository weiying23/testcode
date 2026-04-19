//****************************************************************************\
//*                    National Numerical Windtunnel                          *
//*         FlowStar -- Flow Simulation Tools for Aerospace Research          *
//*                  Computational Aerodynamics Institute                     *
//*              China Aerodynamics Research&Development Center               *
//*                       Mianyang, Sichuan, China                            *
//****************************************************************************/
///
/// \file   parameter_reader.cpp
/// \brief  A class to read mflow parameters
/// \author tangj
/// \date   2020-06-10
/// \copyright  C.All rights reserved. 2020, CAI/CARDC
///
/// \par    Update records:
/// <pre>
/// Date        Author     Description
/// 2020-07-06  tangj      Normalize notation according coding guideline of FlowStar
/// </pre>

// direct head file
#include "parameter_reader.h"

// C/C++ build-in head files
#include <cassert>   // assert()
#include <string.h>
#include <cctype>    // isspace()

// other mflow head files
#include "algm.h"
#include "grid_patch_type.h"
#include "io_log.h"
#include "memory_util.h"
#include "system_base_functions.h"
#include "parameters.h"


namespace mflow
{
#ifdef CPP_FILD_ID
#undef CPP_FILD_ID
#endif
#define CPP_FILD_ID 11610  // define file id

#ifdef MPICH
    extern int myZone;
    extern int numprocs;
    extern MPI_Comm GridComm;  //for each grid, tangj
#endif


// constructor
ParameterReader::ParameterReader(void) :
    steady_                  (1),
    dynamic_                 (0),
    n_grids_                 (1),
    simu_parameters_         (NULL),
    zones_common_parameters_ (NULL),
    grid_index_              (1),
    is_overset_              (false)
{
#ifdef MPICH
    n_cores_for_grid_.push_back(numprocs);
#else
    n_cores_for_grid_.push_back(1);
#endif
}

// destructor
ParameterReader::~ParameterReader()
{
    mfmem::sdel_object(simu_parameters_);
    mfmem::sdel_object(zones_common_parameters_);

    for (IntType igrid = 0; igrid < n_grids_; ++igrid)
    {
        mfmem::sdel_object(zones_parameters_[igrid]);
        mfmem::sdel_object(zones_bc_records_[igrid]);
    }
}

// Read parameter files
// app_case 1->preprocessor, 2->solver, 3->postprocessor
void ParameterReader::read_parameter(const IntType app_case, const bool is_overset)
{
    mflog::log.set_one_processor_out();
    mflog::log << "Start to read parameter files" << std::endl;

    this->is_overset_ = is_overset;

    // guess which type of parameter file exist.
    if (CheckFileReadable("input.para"))
    {
        this->read_parameter_v2(app_case);
    }
    else if (CheckFileReadable("input.par") || CheckFileReadable("input1.par"))
    {
        this->read_parameter_v1(app_case);
    }
    else
    {
        std::cerr << "No parameter file exists, please check and retry" << std::endl;
        mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
    }
}


// ProcessGeneric() needs a function UpdateData().
// So we set an auxiliary class to support a that function.
class UpdateParamAuxiliary
{
public:

    UpdateParamAuxiliary(DataSafe * data) : data_(data) {};

    // Update one data in data pool binded with this patch
    void UpdateData(void *data, IntType typein, IntType size, const ShortString name)
    {
        data_->UpdateDataSafe(data, typein, size, name);
    }

    // Get one data in data pool binded with this patch
    void GetData(void *data, IntType typein, IntType size, const ShortString name)
    {
        data_->GetDataByName(data, typein, size, name);
    }

    ~UpdateParamAuxiliary(){};

private:
    DataSafe * data_;
};


// read parameter files according to applications
// 
void ParameterReader::read_parameter_v1(const IntType app_case)
{
    std::string filename;
    filename = "input.par";

    mflog::log.set_one_processor_out();
    mflog::log << "Start to read input.par file" << std::endl;

    // firstly, try to read input.par or input1.par
    if(CheckFileReadable(filename))
    {
        mfmem::snew_object(simu_parameters_, dmrfl);
        mfmem::snew_object(zones_common_parameters_, dmrfl);

        DataSafe *zone_params = NULL;
        mfmem::snew_object(zone_params, dmrfl);
        zones_parameters_.push_back(zone_params);

        BCond * zone_bc_record = NULL;
        mfmem::snew_object(zone_bc_record, dmrfl);
        zones_bc_records_.push_back(zone_bc_record);

        this->read_file_input_v1(filename, 0);
    }
    else
    {
        std::cerr << "file "<< filename << " is not exist" << std::endl;
        mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
    }

    // add common parameters of zone to all zones
    for (IntType igrid = 0; igrid < n_grids_; ++igrid)
    {
        zones_parameters_[igrid]->CopyDataFrom(zones_common_parameters_);
    }
}


// read input.par of old format
void ParameterReader::read_file_input_v1(const std::string & file, const IntType zone_id)
{
    // check file
    FILE *fpin = NULL;
    if( (fpin = fopen(file.c_str(), "r")) == NULL)
    {
        std::cerr << "Could not open file " << file << std::endl;
        mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
    }

    // Auxiliary objects to support function ProcessGeneric().
    UpdateParamAuxiliary
        aux_params_simu(simu_parameters_),
        aux_params_all_zone(zones_common_parameters_),
        aux_params_current_zone(zones_parameters_[zone_id]);

    // 
    char line1[MAXLINE+1], line2[MAXLINE+1];
    ShortString word1;
    String word2;
    IntType p1=0,p2=0,i,j,n,nzones,type,n_patch;
    char *error;
    IntType   zn=0; // zn=0: simulation; >0 the zone; <0 all zone;
    BCRecord *bcr;
    ShortString *bcName;

    for(;;) 
    {
        do
        {
            error = fgets(line1, MAXLINE, fpin);
            if(!error) break;
        } while (line1[0] == '#'); // filter notation lines
        if(error == NULL) break;

        do
        {
            error = fgets(line2, MAXLINE, fpin);
            if(!error) break;
        } while (line2[0] == '#'); // filter notation lines
        if(error == NULL) break;

        p1 = 0;
        GetNextWord(word1, line1, &p1);   

        // check for key words
        if(strcmp(word1, "Zone") == 0)
        {
            p2 = 0;
            GetNextWord(word2, line2, &p2);
            if(strcmp(word2, "ALL") == 0 || strcmp(word2, "All") == 0 || strcmp(word2, "all") == 0)
            {
                    zn = -1;
            }
            else
            {
                zn = atoi(word2);
                assert(zn > 0 && zn <= nzones);
            }
            continue;
        }
        // boundary condition block
        else if (strcmp(word1, "BC") == 0) 
        {
            if(zn > 0)
            {
                p2 = 0;
                GetNextWord(word2, line2, &p2);
                n_patch = atoi(word2);

                for(j = 0; j < n_patch; ++j)
                {
                    bcr = NULL;
                    mfmem::snew_object(bcr, dmrfl);

                    // use key as type for now
                    do
                    {
                        error = fgets(line1, MAXLINE, fpin);
                        if(!error) break;
                    } while (line1[0] == '#');
                    if(error == NULL) break;

                    do
                    {
                        error = fgets(line2, MAXLINE, fpin);
                        if(!error) break;
                    } while (line2[0] == '#');
                    if(error == NULL) break;

                    n = NumberOfWords(line1);

                    p1 = 0; p2 = 0;

                    for(i = 0; i < n; ++i)
                    {
                        GetNextWord(word1, line1, &p1);
                        if(strcmp(word1, "type") == 0)
                        {
                            GetNextWord(word2, line2, &p2);
                            type = atoi(word2);
                            bcr->SetType(type);
                            bcName = fromTypeToName(type);
                            bcr->SetTypeSymbol(*bcName);
                            mfmem::sdel_array_1D(bcName);
                        }
                        else if(strcmp(word1, "patch") == 0)
                        {
                            GetNextWord(word2, line2, &p2);
                            type = atoi(word2);
                            bcr->SetPatchID(type);
                        }
                        else if(word1[0] == '$')
                        {
                            ProcessGeneric(bcr, word1, line2, &p2);
                        }
                    }

                    // insert into BCond object of this zone
                    zones_bc_records_[zone_id]->AddBCRecord(bcr);
                }
            } 
            else
            {
                mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
            }

            // detect whether BC is enough, viz. patch numbers is equal to n_patch
            if(j != n_patch)
            {
                std::cerr << std::endl << "Error! Patch numbers is not full! Please check and modify." << std::endl;
                mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
            }

            continue;
        }

        // other key word found    
        if(strcmp(word1, "title") == 0)
        {
            titile_ = line2;
        } 
        else 
        {
            n = NumberOfWords(line1);
            p1 = 0; p2 = 0;
            for(i = 0; i < n; ++i)
            {
                GetNextWord(word1, line1, &p1);
                if(strcmp(word1, "nZones") == 0)
                {
                    GetNextWord(word2, line2, &p2);
                    nzones = atoi(word2);

                    //for(j=0; j<nzones; j++)
                    //{
                    //    zone = NULL;
                    //    mfmem::snew_object(zone,dmrfl);
                    //    zone->SetZone(j);
                    //    AddZone(zone);
                    //}
                }
                else if(strcmp(word1, "steady") == 0) 
                {
                    GetNextWord(word2, line2, &p2);
                    steady_ = atoi(word2);
                }
                else if(strcmp(word1, "dynamic") == 0) 
                {
                    GetNextWord(word2, line2, &p2);
                    dynamic_ = atoi(word2);
                }
                else // no key word found
                {
                    if(word1[0] != '$')
                    {
                        std::cerr << "Line error: " << line1 << std::endl;
                        mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
                    }
                    if(zn == 0)  // parameters belong to simulation
                    {
                        ProcessGeneric(&aux_params_simu, word1, line2, &p2);
                    } 
                    else if(zn > 0) // parameters belong to one zone
                    {
                        ProcessGeneric(&aux_params_current_zone, word1, line2, &p2);
                    } 
                    else // parameters belong to all zones
                    {
                        ProcessGeneric(&aux_params_all_zone, word1, line2, &p2);
                    }
                }
            }
        }
    }

    fclose(fpin);
}


/// \brief Read parameter files of old version, such as input.para and chmrinput.para
/// \param[in] app_case 1->preprocessor, 2->solver, 3->postprocessor
void ParameterReader::read_parameter_v2(const IntType app_case)
{
    std::string input_name("input.para");
    this->read_file_input_v2(input_name);    

    // read merged input.par and forceget.par for post-processing
    if (app_case == 3)
    {
        // first read the merged input.par
        string forceget_name = "forceget.para";        
        this->read_file_forceget_v2(forceget_name);
    }
}


// Get parameters and pass them to this object
typedef parameter_space::parameters::group_of_params_type group_type;
typedef parameter_space::parameters::parameters_base::const_iterator iter_type;

void PostComponentsHelper(group_type &componenets_group, UpdateParamAuxiliary &zone_params)
{
    IntType n_components = componenets_group.get<int>("n_components");

    IntType *n_patches_component = NULL, *global_half_component = NULL;
    String *name_component = NULL;

    mfmem::snew_array_1D(n_patches_component, n_components, dmrfl);
    mfmem::snew_array_1D(global_half_component, n_components, dmrfl);
    mfmem::snew_array_1D(name_component, n_components, dmrfl);

    std::vector<IntType> patches_component;

    for (int icomp = 0; icomp < n_components; ++icomp)
    {
        std::string sub_component_name = "components_" + int2str(icomp);
        group_type &zone_component = componenets_group.get_sub_group(sub_component_name);

        const std::string &name = zone_component.get<std::string> ("name");
        strcpy(name_component[icomp], name.c_str());

        global_half_component[icomp] = zone_component.get<int>("global_half");

        const std::vector<int> &bc_ptaches = zone_component.get<std::vector<int> >("bc_patches");

        n_patches_component[icomp] = static_cast<IntType>(bc_ptaches.size());

        for (int ipatch = 0; ipatch < bc_ptaches.size(); ++ipatch)
        {
            patches_component.push_back(bc_ptaches[ipatch]);
        }
    }

    zone_params.UpdateData(&n_components, INT, 1, "nAssem");
    zone_params.UpdateData(n_patches_component, INT, n_components, "NAssem");
    zone_params.UpdateData(global_half_component, INT, n_components, "AssemGorH");
    zone_params.UpdateData(name_component, STRING, n_components, "NameAssem");
    zone_params.UpdateData(&(patches_component[0]), INT, static_cast<IntType>(patches_component.size()), "NumAssem");

    mfmem::sdel_array_1D(n_patches_component);
    mfmem::sdel_array_1D(global_half_component);
    mfmem::sdel_array_1D(name_component);
}

void BoundaryConditionHelper(group_type &bc_group, int group_id, BCond * bc_records)
{
    std::map<int, BCRecord *> patch_id_to_record;

    // variables to get parameters
    std::string name; IntType type; IntType size; void * value_;

    int n_patch_groups = bc_group.get<int>("n_patch_groups");
    for (int igroup = 0; igroup < n_patch_groups; ++igroup)
    {
        std::string patch_group_name = "BC_" + int2str(igroup);
        group_type & bc_patch_group = bc_group.get_sub_group(patch_group_name);

        int bc_type = bc_patch_group.get<int>("type");
        std::vector<int> ids = bc_patch_group.get<std::vector<int> > ("ids");

        DataSafe *data = NULL;
        mfmem::snew_object(data, dmrfl);        
        UpdateParamAuxiliary bc_data(data);

        // Delete type and ids from parameters. Here we assume that the remaining parameters
        // belong to extra data of BCRecord.
        bc_patch_group.remove("type");
        bc_patch_group.remove("ids");
        for (iter_type iter = bc_patch_group.begin(); iter != bc_patch_group.end(); ++iter)
        {
            if( bc_patch_group.data_to_flowstar( iter, name, type, size, value_ ) )
            {
                bc_data.UpdateData(value_, type, size, name.c_str());

                // user must delete the memory
                delete [] static_cast<char *> (value_);
            }
        }
        
        ShortString *bc_type_symbol = fromTypeToName(bc_type);    

        for (int patch_id = 0; patch_id < ids.size(); ++patch_id)
        {
            BCRecord *bc_record = NULL;
            mfmem::snew_object(bc_record, dmrfl);
            bc_record->SetType(bc_type);
            bc_record->SetTypeSymbol(*bc_type_symbol);
            bc_record->SetPatchID(ids[patch_id]);
            bc_record->GetDataReference().CopyDataFrom(data);

            // save the BCRecord
            patch_id_to_record.insert(std::make_pair(ids[patch_id], bc_record));
        }

        mfmem::sdel_array_1D(bc_type_symbol);
        mfmem::sdel_object(data);
    }

    // Check and add BC record.
    // Patch id should start from 1 and numbering contiguously
    // However, here I don't know how many patches should appear.
    typedef std::map<int, BCRecord *>::iterator iter_type;
    int correct_patch_id = 1;
    for (iter_type iter = patch_id_to_record.begin(); iter != patch_id_to_record.end(); ++iter)
    {
        if (iter->first != correct_patch_id)
        {
            std::cerr << "Some mistakes exist in BC section of Zone " << group_id << std::endl;
            std::cerr << "Patch id should start from 1 and numbering contiguously!" << std::endl;
            mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
        }

        // add to BCond
        bc_records->AddBCRecord(iter->second);

        ++correct_patch_id;
    }
}

// Read parameter files of new version, such as input.para
void ParameterReader::read_file_input_v2(const std::string & file)
{
    // check file
    if(!CheckFileReadable(file))
    {
        std::cerr << "Could not open file " << file << std::endl;
        mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
    }

    // read parameter from file
    parameter_space::parameters params_root;
    read_parameters(file, params_root);

    //params.print();    

    // parameters in section Simulation
    group_type &param_simu = params_root.get_sub_group("Simulation");


    mfmem::snew_object(simu_parameters_, dmrfl);
    UpdateParamAuxiliary aux_params_simu(simu_parameters_);

    // variables to get parameters
    std::string name; IntType type; IntType size; void * value_;

    // pass into simulation parameters
    for (iter_type iter = param_simu.begin(); iter != param_simu.end(); ++iter)
    {
        if( param_simu.data_to_flowstar( iter, name, type, size, value_ ) )
        {
            aux_params_simu.UpdateData(value_, type, size, name.c_str());

            // user must delete the memory
            delete [] static_cast<char *> (value_);
        }
    }

    // get the value of steady and dynamic
    this->steady_ = param_simu.get<int>("steady");
    this->dynamic_ = param_simu.get<int>("dynamic");

    // parameters in section Zone
    group_type &param_zone = params_root.get_sub_group("Zone"); 

    // pass into common Zone parameters
    mfmem::snew_object(zones_common_parameters_, dmrfl);
    UpdateParamAuxiliary aux_params_all_zones(zones_common_parameters_);
    
    for(iter_type iter = param_zone.begin() ; iter != param_zone.end(); ++ iter)
    {
        if( param_zone.data_to_flowstar( iter, name, type, size, value_ ))
        {
            aux_params_all_zones.UpdateData(value_, type, size, name.c_str());
            delete[] static_cast<char *> (value_);
        }
    }

    // sub-groups in Zone
    size_t n_sub_zones = param_zone.n_sub_groups();
    IntType n_grids_in_file = 0;
    for (int izone = 0; izone < n_sub_zones; ++izone)
    {
        // default name for sub-group: "group-name" + "_" + "i" 
        std::string zone_name = "Zone_" + int2str(izone);

        if (param_zone.have_sub_group(zone_name))
        {
            group_type &param_sub_zone = param_zone.get_sub_group(zone_name);

            DataSafe * zone_parameters = NULL;
            mfmem::snew_object(zone_parameters, dmrfl);
            zones_parameters_.push_back(zone_parameters);

            // Copy common parameters
            zone_parameters->CopyDataFrom(zones_common_parameters_);

            // Add special parameter
            UpdateParamAuxiliary aux_params_zone(zone_parameters);
            for(iter_type iter = param_sub_zone.begin() ; iter != param_sub_zone.end(); ++ iter)
            {
                if( param_zone.data_to_flowstar( iter, name, type, size, value_ ))
                {
                    aux_params_zone.UpdateData(value_, type, size, name.c_str());
                    delete[] static_cast<char *> (value_);
                }
            }

            // deal with boundary conditions
            group_type & bc_groups = param_sub_zone.get_sub_group("BC");
            BCond * bc_record = NULL;
            mfmem::snew_object(bc_record, dmrfl);
            zones_bc_records_.push_back(bc_record);
            BoundaryConditionHelper(bc_groups, izone, bc_record);

            ++n_grids_in_file;
        }
    }

}


// read forceget.par of new format
void ParameterReader::read_file_forceget_v2(const std::string & file)
{
    // check file
    if(!CheckFileReadable(file))
    {
        std::cerr << "Could not open file " << file << std::endl;
        mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
    }

    // read parameter from file
    parameter_space::parameters params_root;
    read_parameters(file, params_root);

    //params.print();    

    // parameters in section Simulation
    group_type &param_simu = params_root.get_sub_group("Simulation");

    assert(simu_parameters_ != NULL);
    UpdateParamAuxiliary aux_params_simu(simu_parameters_);

    // variables to get parameters
    std::string name; IntType type; IntType size; void * value_;

    // pass into simulation parameters
    for (iter_type iter = param_simu.begin(); iter != param_simu.end(); ++iter)
    {
        if( param_simu.data_to_flowstar( iter, name, type, size, value_ ) )
        {
            aux_params_simu.UpdateData(value_, type, size, name.c_str());

            // user must delete the memory
            delete [] static_cast<char *> (value_);
        }
    }

    //
    // Get the parameters for post-process
    const std::vector<int> &post_zones = param_simu.get<std::vector<int> > ("post_zones");
    for (std::vector<int>::const_iterator iter = post_zones.begin(); iter != post_zones.end(); ++iter)
    {
        this->post_zones_.push_back(*iter);
    }
    // delete post_zones from parameters list
    this->simu_parameters_->DeleteDataByName("post_zones");

    group_type &param_post_dirs = param_simu.get_sub_group("Dirname");
    const std::vector<std::string> &post_dirs = param_post_dirs.get<std::vector<std::string> >("dir_list");
    for (std::vector<std::string>::const_iterator iter = post_dirs.begin(); iter != post_dirs.end(); ++iter)
    {
        this->post_directories_.push_back(*iter);
    }

    // parameters in section Zone
    group_type &param_zone = params_root.get_sub_group("Zone"); 

    // pass into common Zone parameters
    DataSafe *zones_common_parameters_forceget = NULL;
    mfmem::snew_object(zones_common_parameters_forceget, dmrfl);
    UpdateParamAuxiliary aux_params_all_zones(zones_common_parameters_forceget);

    for(iter_type iter = param_zone.begin() ; iter != param_zone.end(); ++ iter)
    {
        if( param_zone.data_to_flowstar( iter, name, type, size, value_ ))
        {
            aux_params_all_zones.UpdateData(value_, type, size, name.c_str());
            delete[] static_cast<char *> (value_);
        }
    }

    // deal with force/moment integration for components
    if (param_zone.have_sub_group("components"))
    {
        group_type &param_zone_components = param_zone.get_sub_group("components");
        PostComponentsHelper(param_zone_components, aux_params_all_zones);   
    }

    // Copy common parameters
    for (int izone = 0; izone < zones_parameters_.size(); ++izone)
    {
        DataSafe * zone_parameters = zones_parameters_[izone];        
        zone_parameters->CopyDataFrom(zones_common_parameters_forceget);
    }

    mfmem::sdel_object(zones_common_parameters_forceget);

    // sub-groups in Zone
    size_t n_sub_zones = param_zone.n_sub_groups();
    for (int izone = 0; izone < n_sub_zones; ++izone)
    {
        // default name for sub-group: "group-name" + "_" + "i" 
        std::string zone_name = "Zone_" + int2str(izone);

        if (param_zone.have_sub_group(zone_name))
        {
            group_type &param_sub_zone = param_zone.get_sub_group(zone_name);

            IntType object_zone = this->post_zones_[izone-1];

            assert(object_zone < n_grids_);
            DataSafe * zone_parameters = zones_parameters_[object_zone];

            // Add special parameter
            UpdateParamAuxiliary aux_params_zone(zone_parameters);
            for(iter_type iter = param_sub_zone.begin() ; iter != param_sub_zone.end(); ++ iter)
            {
                if( param_zone.data_to_flowstar( iter, name, type, size, value_ ))
                {
                    aux_params_zone.UpdateData(value_, type, size, name.c_str());
                    delete[] static_cast<char *> (value_);
                }
            }

            // deal with force/moment integration for components
            if (param_sub_zone.have_sub_group("components"))
            {
                group_type &param_zone_components = param_sub_zone.get_sub_group("components");
                PostComponentsHelper(param_zone_components, aux_params_zone);
            }
        }
    }
}


/// \brief Modify the value of parameter in the input file of old format
/// \param[in] file patch of file
/// \param[in] keys key of parameters
/// \param[in] values value of parameters
// Modify the value of parameter in the input file of old format
// Waring: this is only used to modify restart.
void modify_parameter_file_v1(const std::string & file, std::map<std::string, std::string> & params)
{   
    FILE *fp;

    // check file
    if( (fp = fopen(file.c_str(), "r")) == NULL)
    {
        std::cerr << "Could not open " << file << " when read it" << endl;
        mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
    }

    // use dynamic array instead of static array allocated on stack, because
    // the array is too large, which may cause stack overflow. tangj20191113
    const IntType N_MAX_LINES = 2500;
    char **line1 = NULL, *tem = NULL;
    mfmem::snew_array_2D(line1, N_MAX_LINES, MAXLINE+1, dmrfl, true);
    mfmem::snew_array_1D(tem, MAXLINE+1, dmrfl);

    IntType i, j, n;
    char c, *error;

    // read file
    n = 0;
    do
    {
        error = fgets(line1[n], MAXLINE, fp);
        ++n;
    }
    while (error != NULL);

    fclose(fp);
    --n;

    // modify parameters
    for(i=0; i<n; i++)
    {
        j=0;
        for(;;)
        {
            c = line1[i][j];
            if(c == '\n' || c == '\0' || c == ' ' || c == '\r') break;
            tem[j] = c;
            j++;
        }
        tem[j] = '\0';
        if(!strcmp(tem,"$I-restart") || !strcmp(tem,"$I-turbRst"))
        {
            strcpy(line1[i+1],"1\n");
        }
    }

    // write file
    if( (fp = fopen(file.c_str(), "w")) == NULL)
    {
        std::cerr << "Could not open " << file << " when write it" << endl;

        // delete dynamic arrays before exit. tangj
        // if not exit, DO NOT delete array too!
        mfmem::sdel_array_2D(line1);
        mfmem::sdel_array_1D(tem);
        mflow_exit(mflow_error_flag(CPP_FILD_ID, CPP_LINE));
    }
    for(i=0; i<n; ++i)
    {
        fputs(line1[i],fp);
    }
    fclose(fp);

    // delete dynamic arrays
    mfmem::sdel_array_2D(line1);
    mfmem::sdel_array_1D(tem);
}

/// \brief Modify the value of parameter in the input file of new format
/// \param[in] file patch of file
/// \param[in] keys key of parameters
/// \param[in] values value of parameters
/// \attention if the parameter is array, the multi-values are separated by ',', such as "-3.5, 25.0";
/// 
/// \par    Update records:
/// <pre>
/// Date        Author     Description
/// 2020-07-08  tangj      bug fix for parameter of array type
/// </pre>
void modify_parameter_file_v2(const std::string & file, std::map<std::string, std::string> & params)
{
    // check file
    std::ifstream fin(file.c_str(), std::ifstream::in);
    if (fin.fail())
    {
        std::cerr << "Could not open " << file << " when read it" << std::endl;
    }

    // stream for saving content of new file
    std::ostringstream file_content;

    std::string line;
    while (getline(fin, line))
    {
        std::string::size_type line_size = line.size();

        // figure out how many blank char at the beginning of the line
        std::string::size_type n_blank = 0;
        while (n_blank < line_size)
        {
            if (std::isspace(line[n_blank])) ++n_blank;
            else break;
        }

        if(n_blank == line_size)  // blank line
        {
            file_content << line << std::endl;
        }
        else if(line[n_blank] == '/' && line[n_blank+1] == '/')  // notation line
        {
            file_content << line << std::endl;
        } 
        else // parameter line
        {
            bool param_is_array = false;
            std::string::size_type index = 0;
            std::map<std::string, std::string>::iterator param_iter = params.begin();
            for ( ; param_iter != params.end(); ++param_iter)
            {
                index = line.find(param_iter->first);
                if (index != std::string::npos)
                {
                    // all the char matches
                    std::string::size_type index_before = index - 1;
                    std::string::size_type index_after = index + param_iter->first.size();
                    if (std::isspace(line[index_before]) && // the before one char is blank and 
                        (std::isspace(line[index_after]) || // the next char is blank or '='
                        (line[index_after] == '=') ||
                        (line[index_after] == '[') ))       // or '[' 
                    {
                        // get rid of notation words
                        std::string::size_type end = line.find("//");
                        if (end != std::string::npos)
                        {
                            line.resize(end);
                        }
                        // if char '[' exists, the parameter must be an array
                        param_is_array = !(std::string::npos == line.find("["));

                        break;  // exist this parameter
                    }    
                }
            }

            if (param_iter == params.end()) // do not match any one of the object parameter
            {
                file_content << line << std::endl;
            } 
            else // object parameter line
            {
                std::string new_line(line, 0, index);
                new_line += param_iter->first;  // key
                if(param_is_array) new_line += "[]";
                new_line += " = ";
                if(param_is_array) new_line += "{";
                new_line += param_iter->second; // value
                if(param_is_array) new_line += "}";
                new_line += ';';
                file_content << new_line << std::endl;
            }
        }
    }
    fin.close();

    // write to file
    // check file
    std::ofstream fout(file.c_str(), std::ofstream::out);
    if (fout.fail())
    {
        std::cerr << "Could not open " << file << " when write it" << std::endl;
    }
    fout << file_content.str();
    fout.close();
}

// 
void modify_parameter_file(const std::string & file, std::map<std::string, std::string> & params)
{
    // figure out the format according the postfix of file
    if (file.rfind(".para") != string::npos)
    {
        modify_parameter_file_v2(file, params);
    } 
    else
    {
        modify_parameter_file_v1(file, params);
    }
}


// ------------------------------------------------------------------
// Auxiliary functions declaration for interpreting parameter file
// ------------------------------------------------------------------

// Count the number of words in a line
IntType NumberOfWords(char *line)
{
    IntType p=0, n=0;
    char c;

    for(;;)
    {    
        while(isspace(c = line[p]))
        {
            p++;
        }

        if(c == '\n' || c == '\0') return n;
        n++;

        while(!isspace(c = line[p]))
        {
            p++;
        }
    }
}


// Get the next word from line, starting at position p
IntType GetNextWord(char *word, char *line, IntType *p)
{
    char c;

    while(isspace(c = line[(*p)++]))
    {
        ;
    }

    if(c == '\n' || c == '\0')
    {
        *word = '\0';
        return 0;
    }
    else
    {
        *word++ = c;  
    }

    while(!isspace(c = line[*p]))
    {
        (*p)++;
        if(c == '\n' || c == '\0')
        {
            *word = '\0';
            return 0; 
        }
        else
        {
            *word++ = c;
        }
    }

    *word = '\0';
    return 0;  
}


#undef CPP_FILD_ID  // clear out file id
} // ~namespace mflow

