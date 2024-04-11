#include "FORCE_gamma.h"
#include "module_base/memory.h"
#include "module_base/parallel_reduce.h"
#include "module_base/timer.h"
#include "module_cell/module_neighbor/sltk_grid_driver.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"
#ifdef __DEEPKS
#include "module_hamilt_lcao/module_deepks/LCAO_deepks.h" //caoyu add for deepks on 20210813
#endif
#include "module_io/write_HS.h"
#include "module_elecstate/elecstate_lcao.h"

Force_LCAO_gamma::Force_LCAO_gamma()
{
}

Force_LCAO_gamma::~Force_LCAO_gamma()
{
}

// be called in force_lo.cpp
void Force_LCAO_gamma::ftable_gamma(const bool isforce,
                                    const bool isstress,
                                    const psi::Psi<double>* psid,
                                    Local_Orbital_Charge& loc,
                                    const elecstate::ElecState* pelec,
                                    ModuleBase::matrix& foverlap,
                                    ModuleBase::matrix& ftvnl_dphi,
                                    ModuleBase::matrix& fvnl_dbeta,
                                    ModuleBase::matrix& fvl_dphi,
                                    ModuleBase::matrix& soverlap,
                                    ModuleBase::matrix& stvnl_dphi,
                                    ModuleBase::matrix& svnl_dbeta,
#ifdef __DEEPKS
                                    ModuleBase::matrix& svl_dphi,
                                    ModuleBase::matrix& svnl_dalpha,
#else
                                    ModuleBase::matrix& svl_dphi,
#endif
		                            LCAO_gen_fixedH &gen_h, // mohan add 2024-04-02
                                    Gint_Gamma &gint_gamma,
									LCAO_Matrix &lm)
{
    ModuleBase::TITLE("Force_LCAO_gamma", "ftable");
    ModuleBase::timer::tick("Force_LCAO_gamma", "ftable_gamma");

    // get DM
    const elecstate::DensityMatrix<double,double>* DM
        = dynamic_cast<const elecstate::ElecStateLCAO<double>*>(pelec)->get_DM();

    this->ParaV = DM->get_paraV_pointer();
    //const Parallel_Orbitals* pv = loc.ParaV;

    // allocate DSloc_x, DSloc_y, DSloc_z
    // allocate DHloc_fixed_x, DHloc_fixed_y, DHloc_fixed_z
    this->allocate_gamma(*this->ParaV, gen_h, lm);

    // calculate the 'energy density matrix' here.
    this->cal_foverlap(isforce, isstress, psid, pelec, lm, foverlap, soverlap);

    // sum up the density matrix with different spin
    // DM->sum_DMR_spin();
    
    this->cal_ftvnl_dphi(DM, lm, isforce, isstress, ftvnl_dphi, stvnl_dphi);

    this->cal_fvnl_dbeta(DM, isforce, isstress, fvnl_dbeta, svnl_dbeta);

    this->cal_fvl_dphi(loc.DM, isforce, isstress, pelec->pot, gint_gamma, fvl_dphi, svl_dphi);

    // caoyu add for DeePKS
#ifdef __DEEPKS
    if (GlobalV::deepks_scf)
    {
        const std::vector<std::vector<double>>& dm_gamma = DM->get_DMK_vector();
        GlobalC::ld.cal_projected_DM(DM, GlobalC::ucell, GlobalC::ORB, GlobalC::GridD);
        GlobalC::ld.cal_descriptor();
        GlobalC::ld.cal_gedm(GlobalC::ucell.nat);
        GlobalC::ld.cal_f_delta_gamma(
				dm_gamma, 
				GlobalC::ucell, 
				GlobalC::ORB, 
				GlobalC::GridD, 
				isstress, 
				svnl_dalpha);

#ifdef __MPI
        Parallel_Reduce::reduce_all(GlobalC::ld.F_delta.c, GlobalC::ld.F_delta.nr * GlobalC::ld.F_delta.nc);
        if (isstress)
        {
            Parallel_Reduce::reduce_pool(svnl_dalpha.c, svnl_dalpha.nr * svnl_dalpha.nc);
        }
#endif

        if (GlobalV::deepks_out_unittest)
        {
            GlobalC::ld.print_dm(dm_gamma[0]);
            GlobalC::ld.check_projected_dm();
            GlobalC::ld.check_descriptor(GlobalC::ucell);
            GlobalC::ld.check_gedm();
            GlobalC::ld.add_v_delta(GlobalC::ucell, GlobalC::ORB, GlobalC::GridD);
            GlobalC::ld.check_v_delta();

            GlobalC::ld.cal_e_delta_band(dm_gamma);
            std::ofstream ofs("E_delta_bands.dat");
            ofs << std::setprecision(10) << GlobalC::ld.e_delta_band;
            std::ofstream ofs1("E_delta.dat");
            ofs1 << std::setprecision(10) << GlobalC::ld.E_delta;
            GlobalC::ld.check_f_delta(GlobalC::ucell.nat, svnl_dalpha);
        }
    }
#endif

    if (isforce)
    {
        Parallel_Reduce::reduce_pool(foverlap.c, foverlap.nr * foverlap.nc);
        Parallel_Reduce::reduce_pool(ftvnl_dphi.c, ftvnl_dphi.nr * ftvnl_dphi.nc);
        Parallel_Reduce::reduce_pool(fvnl_dbeta.c, fvnl_dbeta.nr * fvnl_dbeta.nc);
        Parallel_Reduce::reduce_pool(fvl_dphi.c, fvl_dphi.nr * fvl_dphi.nc);
    }
    if (isstress)
    {
        Parallel_Reduce::reduce_pool(soverlap.c, soverlap.nr * soverlap.nc);
        Parallel_Reduce::reduce_pool(stvnl_dphi.c, stvnl_dphi.nr * stvnl_dphi.nc);
        Parallel_Reduce::reduce_pool(svnl_dbeta.c, svnl_dbeta.nr * svnl_dbeta.nc);
        Parallel_Reduce::reduce_pool(svl_dphi.c, svl_dphi.nr * svl_dphi.nc);
    }

    // delete DSloc_x, DSloc_y, DSloc_z
    // delete DHloc_fixed_x, DHloc_fixed_y, DHloc_fixed_z
    this->finish_ftable_gamma(lm);

    ModuleBase::timer::tick("Force_LCAO_gamma", "ftable_gamma");
    return;
}

void Force_LCAO_gamma::allocate_gamma(
    const Parallel_Orbitals &pv,
    LCAO_gen_fixedH &gen_h,
    LCAO_Matrix &lm)
{
    ModuleBase::TITLE("Force_LCAO_gamma", "allocate_gamma");
    ModuleBase::timer::tick("Force_LCAO_gamma", "allocate_gamma");

    // need to calculate the derivative in build_ST_new
    bool cal_deri = true;
    this->ParaV = &pv;

    // calculate dS in LCAO
    // liaochen add on 2010/7/12
    // save the results in dense matrix by now.
    // pv.nloc: number of H elements in this proc.
    lm.DSloc_x = new double[pv.nloc];
    lm.DSloc_y = new double[pv.nloc];
    lm.DSloc_z = new double[pv.nloc];
    ModuleBase::GlobalFunc::ZEROS(lm.DSloc_x, pv.nloc);
    ModuleBase::GlobalFunc::ZEROS(lm.DSloc_y, pv.nloc);
    ModuleBase::GlobalFunc::ZEROS(lm.DSloc_z, pv.nloc);
    ModuleBase::Memory::record("Force::dS_GO", sizeof(double) * pv.nloc * 3);
    // allocate stress part in gamma_only-line, added by zhengdy-stress
    if (GlobalV::CAL_STRESS)
    {
        lm.DSloc_11 = new double[pv.nloc];
        lm.DSloc_12 = new double[pv.nloc];
        lm.DSloc_13 = new double[pv.nloc];
        lm.DSloc_22 = new double[pv.nloc];
        lm.DSloc_23 = new double[pv.nloc];
        lm.DSloc_33 = new double[pv.nloc];
        ModuleBase::GlobalFunc::ZEROS(lm.DSloc_11, pv.nloc);
        ModuleBase::GlobalFunc::ZEROS(lm.DSloc_12, pv.nloc);
        ModuleBase::GlobalFunc::ZEROS(lm.DSloc_13, pv.nloc);
        ModuleBase::GlobalFunc::ZEROS(lm.DSloc_22, pv.nloc);
        ModuleBase::GlobalFunc::ZEROS(lm.DSloc_23, pv.nloc);
        ModuleBase::GlobalFunc::ZEROS(lm.DSloc_33, pv.nloc);
        lm.DHloc_fixed_11 = new double[pv.nloc];
        lm.DHloc_fixed_12 = new double[pv.nloc];
        lm.DHloc_fixed_13 = new double[pv.nloc];
        lm.DHloc_fixed_22 = new double[pv.nloc];
        lm.DHloc_fixed_23 = new double[pv.nloc];
        lm.DHloc_fixed_33 = new double[pv.nloc];
        ModuleBase::GlobalFunc::ZEROS(lm.DHloc_fixed_11, pv.nloc);
        ModuleBase::GlobalFunc::ZEROS(lm.DHloc_fixed_12, pv.nloc);
        ModuleBase::GlobalFunc::ZEROS(lm.DHloc_fixed_13, pv.nloc);
        ModuleBase::GlobalFunc::ZEROS(lm.DHloc_fixed_22, pv.nloc);
        ModuleBase::GlobalFunc::ZEROS(lm.DHloc_fixed_23, pv.nloc);
        ModuleBase::GlobalFunc::ZEROS(lm.DHloc_fixed_33, pv.nloc);
        ModuleBase::Memory::record("Stress::dSH_GO", sizeof(double) * pv.nloc * 12);
    }
    // calculate dS in LCAO basis
    // ModuleBase::timer::tick("Force_LCAO_gamma","build_S_new");
    gen_h.build_ST_new('S', cal_deri, GlobalC::ucell, GlobalC::ORB, GlobalC::UOT, &(GlobalC::GridD), lm.Sloc.data());
    // ModuleBase::timer::tick("Force_LCAO_gamma","build_S_new");

    // calculate dT in LCAP
    // allocation dt
    // liaochen add on 2010/7/12
    lm.DHloc_fixed_x = new double[pv.nloc];
    lm.DHloc_fixed_y = new double[pv.nloc];
    lm.DHloc_fixed_z = new double[pv.nloc];
    ModuleBase::Memory::record("Force::dTVNL", sizeof(double) * pv.nloc * 3);
    ModuleBase::GlobalFunc::ZEROS(lm.DHloc_fixed_x, pv.nloc);
    ModuleBase::GlobalFunc::ZEROS(lm.DHloc_fixed_y, pv.nloc);
    ModuleBase::GlobalFunc::ZEROS(lm.DHloc_fixed_z, pv.nloc);

    // calculate dT
    // calculate T + VNL(P1) in LCAO basis
    // ModuleBase::timer::tick("Force_LCAO_gamma","build_T_new");
    gen_h.build_ST_new('T', cal_deri, GlobalC::ucell, GlobalC::ORB, GlobalC::UOT, &(GlobalC::GridD), lm.Hloc_fixed.data());
    // ModuleBase::timer::tick("Force_LCAO_gamma","build_T_new");
    // test_gamma(lm.DHloc_fixed_x, "dHloc_fixed_x T part");

    // ModuleBase::timer::tick("Force_LCAO_gamma","build_Nonlocal_mu");
    gen_h.build_Nonlocal_mu_new(lm.Hloc_fixed.data(), cal_deri, GlobalC::ucell, GlobalC::ORB, GlobalC::UOT, &(GlobalC::GridD));
    // ModuleBase::timer::tick("Force_LCAO_gamma","build_Nonlocal_mu");
    // test_gamma(lm.DHloc_fixed_x, "dHloc_fixed_x Vnl part");

    // calculate asynchronous S matrix to output for Hefei-NAMD
    if (INPUT.cal_syns)
    {
        cal_deri = false;

        lm.zeros_HSgamma('S');

        gen_h.build_ST_new('S', cal_deri, GlobalC::ucell, GlobalC::ORB, GlobalC::UOT, &(GlobalC::GridD), lm.Sloc.data(), INPUT.cal_syns, INPUT.dmax);

        bool bit = false; // LiuXh, 2017-03-21

		ModuleIO::save_mat(0, 
				lm.Hloc.data(), 
				GlobalV::NLOCAL, 
				bit, 
				GlobalV::out_ndigits, 
				0, 
				GlobalV::out_app_flag, 
				"H", 
				"data-" + std::to_string(0), 
				*this->ParaV, 
				GlobalV::DRANK);

		ModuleIO::save_mat(0, 
				lm.Sloc.data(), 
				GlobalV::NLOCAL, 
				bit, 
				GlobalV::out_ndigits, 
				0, 
				GlobalV::out_app_flag, 
				"S", 
				"data-" + std::to_string(0), 
				*this->ParaV, 
				GlobalV::DRANK);
	}

    ModuleBase::timer::tick("Force_LCAO_gamma", "allocate_gamma");
    return;
}

void Force_LCAO_gamma::finish_ftable_gamma(LCAO_Matrix &lm)
{
    delete[] lm.DSloc_x;
    delete[] lm.DSloc_y;
    delete[] lm.DSloc_z;
    delete[] lm.DHloc_fixed_x;
    delete[] lm.DHloc_fixed_y;
    delete[] lm.DHloc_fixed_z;
    if (GlobalV::CAL_STRESS) // added by zhengdy-stress
    {
        delete[] lm.DSloc_11;
        delete[] lm.DSloc_12;
        delete[] lm.DSloc_13;
        delete[] lm.DHloc_fixed_11;
        delete[] lm.DHloc_fixed_12;
        delete[] lm.DHloc_fixed_13;
        delete[] lm.DSloc_22;
        delete[] lm.DSloc_23;
        delete[] lm.DSloc_33;
        delete[] lm.DHloc_fixed_22;
        delete[] lm.DHloc_fixed_23;
        delete[] lm.DHloc_fixed_33;
    }
    return;
}

void Force_LCAO_gamma::test_gamma(double* mm, const std::string& name)
{
    std::cout << "\n PRINT " << name << std::endl;
    std::cout << std::setprecision(6) << std::endl;
    for (int i = 0; i < GlobalV::NLOCAL; i++)
    {
        for (int j = 0; j < GlobalV::NLOCAL; j++)
        {
            if (std::abs(mm[i * GlobalV::NLOCAL + j]) > 1.0e-5)
            {
                std::cout << std::setw(12) << mm[i * GlobalV::NLOCAL + j];
            }
            else
            {
                std::cout << std::setw(12) << "0";
            }
        }
        std::cout << std::endl;
    }
    return;
}

namespace StressTools
{
void stress_fill(const double& lat0_, const double& omega_, ModuleBase::matrix& stress_matrix)
{
    assert(omega_ > 0.0);
    double weight = lat0_ / omega_;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            if (j > i)
                stress_matrix(j, i) = stress_matrix(i, j);
            stress_matrix(i, j) *= weight;
        }
    }
}
} // namespace StressTools
