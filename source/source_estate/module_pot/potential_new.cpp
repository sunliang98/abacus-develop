#include "potential_new.h"

#include "source_base/global_function.h"
#include "source_base/global_variable.h"
#include "source_base/memory.h"
#include "source_base/timer.h"
#include "source_base/tool_quit.h"
#include "source_base/tool_title.h"
#include "source_hamilt/module_xc/xc_functional.h"
#include "module_parameter/parameter.h"

#include <map>

namespace elecstate
{

Potential::Potential(const ModulePW::PW_Basis* rho_basis_in,
                     const ModulePW::PW_Basis* rho_basis_smooth_in,
                     const UnitCell* ucell_in,
                     const ModuleBase::matrix* vloc_in,
                     Structure_Factor* structure_factors_in,
                     surchem* solvent_in,
                     double* etxc_in,
                     double* vtxc_in)
    : ucell_(ucell_in), vloc_(vloc_in), structure_factors_(structure_factors_in), solvent_(solvent_in), etxc_(etxc_in),
      vtxc_(vtxc_in)
{
    this->rho_basis_ = rho_basis_in;
    this->rho_basis_smooth_ = rho_basis_smooth_in;
    this->fixed_mode = true;
    this->dynamic_mode = true;
    this->use_gpu_ = (PARAM.inp.basis_type == "pw" && PARAM.inp.device == "gpu");

    // allocate memory for Potential.
    this->allocate();
}

Potential::~Potential()
{
    if (this->components.size() > 0)
    {
        for (auto comp: this->components)
        {
            delete comp;
        }
        this->components.clear();
    }
    if (use_gpu_)
    {
        delmem_sd_op()(s_veff_smooth);
        delmem_sd_op()(s_vofk_smooth);
        delmem_dd_op()(d_veff_smooth);
        delmem_dd_op()(d_vofk_smooth);
    }
    else
    {
        delmem_sh_op()(s_veff_smooth);
        delmem_sh_op()(s_vofk_smooth);
    }
}

void Potential::pot_register(const std::vector<std::string>& components_list)
{
    ModuleBase::TITLE("Potential", "pot_register");
    // delete old components first.
    if (this->components.size() > 0)
    {
        for (auto comp: this->components)
        {
            delete comp;
        }
        this->components.clear();
    }

    // register components
    //---------------------------
    // mapping for register
    //---------------------------
    for (auto comp: components_list)
    {
        PotBase* tmp = this->get_pot_type(comp);
        this->components.push_back(tmp);
        //        GlobalV::ofs_running << "Successful completion of Potential's registration : " << comp << std::endl;
    }

    // after register, reset fixed_done to false
    this->fixed_done = false;

    return;
}

void Potential::allocate()
{
    ModuleBase::TITLE("Potential", "allocate");
    int nrxx = this->rho_basis_->nrxx;
    int nrxx_smooth = this->rho_basis_smooth_->nrxx;
    if (nrxx == 0) 
	{
		return;
	}
	if (nrxx_smooth == 0) 
	{
		return;
	}

    this->v_effective_fixed.resize(nrxx);
    ModuleBase::Memory::record("Pot::veff_fix", sizeof(double) * nrxx);

    this->v_effective.create(PARAM.inp.nspin, nrxx);
    ModuleBase::Memory::record("Pot::veff", sizeof(double) * PARAM.inp.nspin * nrxx);

    this->veff_smooth.create(PARAM.inp.nspin, nrxx_smooth);
    ModuleBase::Memory::record("Pot::veff_smooth", sizeof(double) * PARAM.inp.nspin * nrxx_smooth);

    if (XC_Functional::get_ked_flag())
    {
        this->vofk_effective.create(PARAM.inp.nspin, nrxx);
        ModuleBase::Memory::record("Pot::vofk", sizeof(double) * PARAM.inp.nspin * nrxx);

        this->vofk_smooth.create(PARAM.inp.nspin, nrxx_smooth);
        ModuleBase::Memory::record("Pot::vofk_smooth", sizeof(double) * PARAM.inp.nspin * nrxx_smooth);
    }
    if (use_gpu_)
    {
        if (PARAM.globalv.has_float_data)
        {
            resmem_sd_op()(s_veff_smooth, PARAM.inp.nspin * nrxx_smooth);
            resmem_sd_op()(s_vofk_smooth, PARAM.inp.nspin * nrxx_smooth);
        }
        if (PARAM.globalv.has_double_data)
        {
            resmem_dd_op()(d_veff_smooth, PARAM.inp.nspin * nrxx_smooth);
            resmem_dd_op()(d_vofk_smooth, PARAM.inp.nspin * nrxx_smooth);
        }
    }
    else
    {
        if (PARAM.globalv.has_float_data)
        {
            resmem_sh_op()(s_veff_smooth, PARAM.inp.nspin * nrxx_smooth, "POT::sveff_smooth");
            resmem_sh_op()(s_vofk_smooth, PARAM.inp.nspin * nrxx_smooth, "POT::svofk_smooth");
        }
        if (PARAM.globalv.has_double_data)
        {
            this->d_veff_smooth = this->veff_smooth.c;
            this->d_vofk_smooth = this->vofk_smooth.c;
        }
        // There's no need to allocate memory for double precision pointers while in a CPU environment
    }
}

void Potential::update_from_charge(const Charge*const chg, const UnitCell*const ucell)
{
    ModuleBase::TITLE("Potential", "update_from_charge");
    //ModuleBase::timer::tick("Potential", "update_from_charge");

    if (!this->fixed_done)
    {
        this->cal_fixed_v(this->v_effective_fixed.data());
        this->fixed_done = true;
    }

    this->cal_v_eff(chg, ucell, this->v_effective);

    // interpolate potential on the smooth mesh if necessary
    this->interpolate_vrs();

    if (this->use_gpu_)
    {
        if (PARAM.globalv.has_float_data)
        {
            castmem_d2s_h2d_op()(s_veff_smooth, this->veff_smooth.c, this->veff_smooth.nr * this->veff_smooth.nc);
            castmem_d2s_h2d_op()(s_vofk_smooth, this->vofk_smooth.c, this->vofk_smooth.nr * this->vofk_smooth.nc);
        }
        if (PARAM.globalv.has_double_data)
        {
            syncmem_d2d_h2d_op()(d_veff_smooth, this->veff_smooth.c, this->veff_smooth.nr * this->veff_smooth.nc);
            syncmem_d2d_h2d_op()(d_vofk_smooth, this->vofk_smooth.c, this->vofk_smooth.nr * this->vofk_smooth.nc);
        }
    }
    else
    {
        if (PARAM.globalv.has_float_data)
        {
            castmem_d2s_h2h_op()(s_veff_smooth, this->veff_smooth.c, this->veff_smooth.nr * this->veff_smooth.nc);
            castmem_d2s_h2h_op()(s_vofk_smooth, this->vofk_smooth.c, this->vofk_smooth.nr * this->vofk_smooth.nc);
        }
        // There's no need to synchronize memory for double precision pointers while in a CPU environment
    }

    //ModuleBase::timer::tick("Potential", "update_from_charge");
}

void Potential::cal_fixed_v(double* vl_pseudo)
{
    ModuleBase::TITLE("Potential", "cal_fixed_v");
    ModuleBase::timer::tick("Potential", "cal_fixed_v");

    this->v_effective_fixed.assign(this->v_effective_fixed.size(), 0.0);
    for (size_t i = 0; i < this->components.size(); i++)
    {
        if (this->components[i]->fixed_mode)
        {
            this->components[i]->cal_fixed_v(vl_pseudo);
        }
    }

    ModuleBase::timer::tick("Potential", "cal_fixed_v");
}

void Potential::cal_v_eff(const Charge*const chg, const UnitCell*const ucell, ModuleBase::matrix& v_eff)
{
    ModuleBase::TITLE("Potential", "cal_veff");
    ModuleBase::timer::tick("Potential", "cal_veff");

    const int nspin_current = this->v_effective.nr;
    const int nrxx = this->v_effective.nc;
    // first of all, set v_effective to zero.
    this->v_effective.zero_out();

    // add fixed potential components
    // nspin = 2, add fixed components for all
    // nspin = 4, add fixed components on first colomn
    for (int i = 0; i < nspin_current; i++)
    {
        if (i == 0 || nspin_current == 2)
        {
            ModuleBase::GlobalFunc::COPYARRAY(this->v_effective_fixed.data(), this->get_effective_v(i), nrxx);
        }
    }

    // cal effective by every components
    for (size_t i = 0; i < this->components.size(); i++)
    {
        if (this->components[i]->dynamic_mode)
        {
            this->components[i]->cal_v_eff(chg, ucell, v_eff);
        }
    }

    ModuleBase::timer::tick("Potential", "cal_veff");
}

void Potential::init_pot(int istep, const Charge*const chg)
{
    ModuleBase::TITLE("Potential", "init_pot");
    ModuleBase::timer::tick("Potential", "init_pot");

    assert(istep >= 0);
    // fixed components only calculated in the beginning of SCF
    this->fixed_done = false;

    this->update_from_charge(chg, this->ucell_);

    ModuleBase::timer::tick("Potential", "init_pot");
    return;
}

void Potential::get_vnew(const Charge* chg, ModuleBase::matrix& vnew)
{
    ModuleBase::TITLE("Potential", "get_vnew");
    vnew.create(this->v_effective.nr, this->v_effective.nc);
    vnew = this->v_effective;

    this->update_from_charge(chg, this->ucell_);
    //(used later for scf correction to the forces )
    for (int iter = 0; iter < vnew.nr * vnew.nc; ++iter)
    {
        vnew.c[iter] = this->v_effective.c[iter] - vnew.c[iter];
    }

    return;
}

void Potential::interpolate_vrs()
{
    ModuleBase::TITLE("Potential", "interpolate_vrs");
    ModuleBase::timer::tick("Potential", "interpolate_vrs");

    if ( PARAM.globalv.double_grid)
    {
        if (rho_basis_->gamma_only != rho_basis_smooth_->gamma_only)
        {
            ModuleBase::WARNING_QUIT("Potential::interpolate_vrs", "gamma_only is not consistent");
        }

        ModuleBase::ComplexMatrix vrs(PARAM.inp.nspin, rho_basis_->npw);
        for (int is = 0; is < PARAM.inp.nspin; is++)
        {
            rho_basis_->real2recip(&v_effective(is, 0), &vrs(is, 0));
            rho_basis_smooth_->recip2real(&vrs(is, 0), &veff_smooth(is, 0));
        }

        if (XC_Functional::get_ked_flag())
        {
            ModuleBase::ComplexMatrix vrs_ofk(PARAM.inp.nspin, rho_basis_->npw);
            for (int is = 0; is < PARAM.inp.nspin; is++)
            {
                rho_basis_->real2recip(&vofk_effective(is, 0), &vrs_ofk(is, 0));
                rho_basis_smooth_->recip2real(&vrs_ofk(is, 0), &vofk_smooth(is, 0));
            }
        }
    }
    else
    {
        this->veff_smooth = this->v_effective;
        this->vofk_smooth = this->vofk_effective;
    }

    ModuleBase::timer::tick("Potential", "interpolate_vrs");
}

template <>
float* Potential::get_veff_smooth_data()
{
    return this->veff_smooth.nc > 0 ? this->s_veff_smooth : nullptr;
}

template <>
double* Potential::get_veff_smooth_data()
{
    return this->veff_smooth.nc > 0 ? this->d_veff_smooth : nullptr;
}

template <>
float* Potential::get_vofk_smooth_data()
{
    return this->vofk_smooth.nc > 0 ? this->s_vofk_smooth : nullptr;
}

template <>
double* Potential::get_vofk_smooth_data()
{
    return this->vofk_smooth.nc > 0 ? this->d_vofk_smooth : nullptr;
}

} // namespace elecstate
