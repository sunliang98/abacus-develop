#include "module_elecstate/elecstate_getters.h"

#include "module_cell/unitcell.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"
#include "module_parameter/parameter.h"
#include "module_hamilt_general/module_xc/xc_functional.h"

namespace elecstate
{

double get_ucell_omega()
{
    return GlobalC::ucell.omega;
}

double get_ucell_tpiba()
{
    return GlobalC::ucell.tpiba;
}

int get_xc_func_type()
{
    return XC_Functional::get_func_type();
}

std::string get_input_vdw_method()
{
    return PARAM.inp.vdw_method;
}

double get_ucell_tot_magnetization()
{
    return GlobalC::ucell.magnet.tot_magnetization;
}

double get_ucell_abs_magnetization()
{
    return GlobalC::ucell.magnet.abs_magnetization;
}

double get_ucell_tot_magnetization_nc_x()
{
    return GlobalC::ucell.magnet.tot_magnetization_nc[0];
}

double get_ucell_tot_magnetization_nc_y()
{
    return GlobalC::ucell.magnet.tot_magnetization_nc[1];
}

double get_ucell_tot_magnetization_nc_z()
{
    return GlobalC::ucell.magnet.tot_magnetization_nc[2];
}

std::string get_ks_solver_type()
{
    return PARAM.inp.ks_solver;
}

} // namespace elecstate
