
#include "para_json.h"

#include "module_base/global_variable.h"

#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>
#ifdef __RAPIDJSON
#include "json_output/abacusjson.h"
#include "json_output/general_info.h"
#include "json_output/init_info.h"
#include "json_output/readin_info.h"

#endif // __RAPIDJSON

namespace Json
{

// void create_Json(ModuleSymmetry::Symmetry *symm,Atom *atoms,Input *input){
// #ifdef __RAPIDJSON
//     gen_general_info(input);
//     gen_init(symm,atoms);
// #endif
// }

void json_output()
{
#ifdef __RAPIDJSON
#ifdef __MPI
    if (GlobalV::MY_RANK == 0)
        AbacusJson::write_to_json("abacus.json");
#else
    AbacusJson::write_to_json("abacus.json");
#endif
#endif // __RAPIDJSON
}

void create_Json(UnitCell* ucell, const Parameter& param)
{
#ifdef __RAPIDJSON
    gen_general_info(param);
    gen_init(ucell);
    // gen_stru(ucell);
#endif
    json_output();
}

void gen_stru_wrapper(UnitCell* ucell)
{
#ifdef __RAPIDJSON
#ifdef __MPI
    if (GlobalV::MY_RANK == 0)
        gen_stru(ucell);
#else
    gen_stru(ucell);
#endif
#endif
}

void convert_time(std::time_t time_now, std::string& time_str)
{
    std::tm* tm = std::localtime(&time_now);
    std::ostringstream oss;
    oss << std::put_time(tm, "%Y-%m-%d %H:%M:%S");
    time_str = oss.str();
}

} // namespace Json
