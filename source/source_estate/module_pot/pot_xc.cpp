#include "pot_xc.h"

#include "source_base/timer.h"
#include "source_pw/hamilt_pwdft/global.h"

#ifdef USE_LIBXC
#include "source_hamilt/module_xc/xc_functional_libxc.h"
#endif

namespace elecstate
{

void PotXC::cal_v_eff(const Charge*const chg, const UnitCell*const ucell, ModuleBase::matrix& v_eff)
{
    ModuleBase::TITLE("PotXC", "cal_veff");
    ModuleBase::timer::tick("PotXC", "cal_veff");
    const int nrxx_current = chg->nrxx;
    
    //----------------------------------------------------------
    //  calculate the exchange-correlation potential
    //----------------------------------------------------------

    if (XC_Functional::get_ked_flag())
    {
#ifdef USE_LIBXC
        const std::tuple<double, double, ModuleBase::matrix, ModuleBase::matrix> etxc_vtxc_v
            = XC_Functional_Libxc::v_xc_meta(XC_Functional::get_func_id(), nrxx_current, ucell->omega, ucell->tpiba, chg);
        *(this->etxc_) = std::get<0>(etxc_vtxc_v);
        *(this->vtxc_) = std::get<1>(etxc_vtxc_v);
        v_eff += std::get<2>(etxc_vtxc_v);
        *(this->vofk) = std::get<3>(etxc_vtxc_v);
#else
        ModuleBase::WARNING_QUIT("v_of_rho", "to use mGGA, compile with LIBXC");
#endif
    }
    else
    {
        const std::tuple<double, double, ModuleBase::matrix> etxc_vtxc_v
            = XC_Functional::v_xc(nrxx_current, chg, ucell);
        *(this->etxc_) = std::get<0>(etxc_vtxc_v);
        *(this->vtxc_) = std::get<1>(etxc_vtxc_v);
        v_eff += std::get<2>(etxc_vtxc_v);
    }
    ModuleBase::timer::tick("PotXC", "cal_veff");
}

} // namespace elecstate
