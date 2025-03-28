#include "pot_local_paw.h"

#include "module_base/timer.h"
#include "module_base/tool_title.h"

#include <complex>

#ifdef USE_PAW
#include "module_cell/module_paw/paw_cell.h"
#endif

namespace elecstate
{

//==========================================================
// This routine computes the local potential in real space
//==========================================================
void PotLocal_PAW::cal_fixed_v(double *vl_pseudo // store the local pseudopotential
)
{
    ModuleBase::TITLE("PotLocal_PAW", "cal_fixed_v");
    ModuleBase::timer::tick("PotLocal_PAW", "cal_fixed_v");

#ifdef USE_PAW
    int nrxx = GlobalC::paw_cell.get_nrxx();
    double* tmp = new double[nrxx];
    double* vl_hartree = new double[nrxx];
    GlobalC::paw_cell.get_vloc_ncoret(vl_hartree,tmp);

    for(int ir = 0; ir < nrxx; ir ++)
    {
        vl_pseudo[ir] = vl_hartree[ir] * 2;
    }
    delete[] tmp;
    delete[] vl_hartree;
#endif

    // GlobalV::ofs_running <<" set local pseudopotential done." << std::endl;
    ModuleBase::timer::tick("PotLocal_PAW", "cal_fixed_v");
    return;
}

} // namespace elecstate