#include "source_base/global_function.h"
#include "gint_fvl.h"
#include "gint_common.h"
#include "phi_operator.h"

namespace ModuleGint
{

void Gint_fvl::cal_gint()
{
    ModuleBase::TITLE("Gint", "cal_gint_fvl");
    ModuleBase::timer::tick("Gint", "cal_gint_fvl");
    init_dm_gint_();
    transfer_dm_2d_to_gint(*gint_info_, dm_vec_, dm_gint_vec_);
    cal_fvl_svl_();
    ModuleBase::timer::tick("Gint", "cal_gint_fvl");
}

void Gint_fvl::init_dm_gint_()
{
    dm_gint_vec_.resize(nspin_);
    for (int is = 0; is < nspin_; is++)
    {
        dm_gint_vec_[is] = gint_info_->get_hr<double>();
    }
}

void Gint_fvl::cal_fvl_svl_()
{
#pragma omp parallel
    {
        PhiOperator phi_op;
        std::vector<double> phi;
        std::vector<double> phi_vldr3;
        std::vector<double> phi_vldr3_dm;
        std::vector<double> dphi_x;
        std::vector<double> dphi_y;
        std::vector<double> dphi_z;
        ModuleBase::matrix* fvl_thread = nullptr;
        ModuleBase::matrix* svl_thread = nullptr;
        if(isforce_)
        {
            fvl_thread = new ModuleBase::matrix(*fvl_);
            fvl_thread->zero_out();
        }
        if(isstress_)
        {
            svl_thread = new ModuleBase::matrix(*svl_);
            svl_thread->zero_out();
        }
#pragma omp for schedule(dynamic)
        for(const auto& biggrid: gint_info_->get_biggrids())
        {
            if(biggrid->get_atoms().size() == 0)
            {
                continue;
            }
            phi_op.set_bgrid(biggrid);
            const int phi_len = phi_op.get_rows() * phi_op.get_cols();
            phi.resize(phi_len);
            phi_vldr3.resize(phi_len);
            phi_vldr3_dm.resize(phi_len);
            dphi_x.resize(phi_len);
            dphi_y.resize(phi_len);
            dphi_z.resize(phi_len);
            phi_op.set_phi_dphi(phi.data(), dphi_x.data(), dphi_y.data(), dphi_z.data());
            for (int is = 0; is < nspin_; is++)
            {
                phi_op.phi_mul_vldr3(vr_eff_[is], dr3_, phi.data(), phi_vldr3.data());
                phi_op.phi_mul_dm(phi_vldr3.data(), dm_gint_vec_[is], false, phi_vldr3_dm.data());
                if(isforce_)
                {
                    phi_op.phi_dot_dphi(phi_vldr3_dm.data(), dphi_x.data(), dphi_y.data(), dphi_z.data(), fvl_thread);
                }
                if(isstress_)
                {
                    phi_op.phi_dot_dphi_r(phi_vldr3_dm.data(), dphi_x.data(), dphi_y.data(), dphi_z.data(), svl_thread);
                }
            }
        }
#pragma omp critical
        {
            if(isforce_)
            {
                fvl_[0] += fvl_thread[0];
                delete fvl_thread;
            }
            if(isstress_)
            {
                svl_[0] += svl_thread[0];
                delete svl_thread;
            }
        }
    }
}


}