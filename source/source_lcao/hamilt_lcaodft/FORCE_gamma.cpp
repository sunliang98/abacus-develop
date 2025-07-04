#include "FORCE.h"
#include "source_base/memory.h"
#include "source_base/parallel_reduce.h"
#include "source_base/timer.h"
#include "source_cell/module_neighbor/sltk_grid_driver.h"
#include "source_pw/hamilt_pwdft/global.h"
#include "module_parameter/parameter.h"
#ifdef __MLALGO
#include "source_lcao/module_deepks/LCAO_deepks.h" //caoyu add for deepks on 20210813
#include "source_lcao/module_deepks/LCAO_deepks_io.h"
#endif
#include "source_cell/module_neighbor/sltk_grid_driver.h" //GridD
#include "source_estate/elecstate_lcao.h"
#include "source_lcao/hamilt_lcaodft/LCAO_domain.h"
#include "source_lcao/hamilt_lcaodft/pulay_force_stress.h"
#include "source_io/write_HS.h"

template <>
void Force_LCAO<double>::allocate(const UnitCell& ucell,
                                  const Grid_Driver& gd,
                                  const Parallel_Orbitals& pv,
                                  ForceStressArrays& fsr, // mohan add 2024-06-15
                                  const TwoCenterBundle& two_center_bundle,
                                  const LCAO_Orbitals& orb,
                                  const int& nks,
                                  const std::vector<ModuleBase::Vector3<double>>& kvec_d)
{
    ModuleBase::TITLE("Forces", "allocate");
    ModuleBase::timer::tick("Forces", "allocate");

    // need to calculate the derivative in build_ST_new
    bool cal_deri = true;
    this->ParaV = &pv;

    // calculate dS in LCAO
    // liaochen add on 2010/7/12
    // save the results in dense matrix by now.
    // pv.nloc: number of H elements in this proc.

    assert(pv.nloc > 0);
    fsr.DSloc_x = new double[pv.nloc];
    fsr.DSloc_y = new double[pv.nloc];
    fsr.DSloc_z = new double[pv.nloc];
    ModuleBase::GlobalFunc::ZEROS(fsr.DSloc_x, pv.nloc);
    ModuleBase::GlobalFunc::ZEROS(fsr.DSloc_y, pv.nloc);
    ModuleBase::GlobalFunc::ZEROS(fsr.DSloc_z, pv.nloc);
    ModuleBase::Memory::record("Force::dS_GO", sizeof(double) * pv.nloc * 3);
    // allocate stress part in gamma_only-line, added by zhengdy-stress
    if (PARAM.inp.cal_stress)
    {
        fsr.DSloc_11 = new double[pv.nloc];
        fsr.DSloc_12 = new double[pv.nloc];
        fsr.DSloc_13 = new double[pv.nloc];
        fsr.DSloc_22 = new double[pv.nloc];
        fsr.DSloc_23 = new double[pv.nloc];
        fsr.DSloc_33 = new double[pv.nloc];
        ModuleBase::GlobalFunc::ZEROS(fsr.DSloc_11, pv.nloc);
        ModuleBase::GlobalFunc::ZEROS(fsr.DSloc_12, pv.nloc);
        ModuleBase::GlobalFunc::ZEROS(fsr.DSloc_13, pv.nloc);
        ModuleBase::GlobalFunc::ZEROS(fsr.DSloc_22, pv.nloc);
        ModuleBase::GlobalFunc::ZEROS(fsr.DSloc_23, pv.nloc);
        ModuleBase::GlobalFunc::ZEROS(fsr.DSloc_33, pv.nloc);
        fsr.DHloc_fixed_11 = new double[pv.nloc];
        fsr.DHloc_fixed_12 = new double[pv.nloc];
        fsr.DHloc_fixed_13 = new double[pv.nloc];
        fsr.DHloc_fixed_22 = new double[pv.nloc];
        fsr.DHloc_fixed_23 = new double[pv.nloc];
        fsr.DHloc_fixed_33 = new double[pv.nloc];
        ModuleBase::GlobalFunc::ZEROS(fsr.DHloc_fixed_11, pv.nloc);
        ModuleBase::GlobalFunc::ZEROS(fsr.DHloc_fixed_12, pv.nloc);
        ModuleBase::GlobalFunc::ZEROS(fsr.DHloc_fixed_13, pv.nloc);
        ModuleBase::GlobalFunc::ZEROS(fsr.DHloc_fixed_22, pv.nloc);
        ModuleBase::GlobalFunc::ZEROS(fsr.DHloc_fixed_23, pv.nloc);
        ModuleBase::GlobalFunc::ZEROS(fsr.DHloc_fixed_33, pv.nloc);
        ModuleBase::Memory::record("Stress::dSH_GO", sizeof(double) * pv.nloc * 12);
    }
    // calculate dS in LCAO basis
    LCAO_domain::build_ST_new(fsr,
                              'S',
                              cal_deri,
                              PARAM.inp.cal_stress,
                              ucell,
                              orb,
                              pv,
                              two_center_bundle,
                              &gd,
                              nullptr);

    // calculate dT in LCAP
    // allocation dt
    // liaochen add on 2010/7/12
    fsr.DHloc_fixed_x = new double[pv.nloc];
    fsr.DHloc_fixed_y = new double[pv.nloc];
    fsr.DHloc_fixed_z = new double[pv.nloc];
    ModuleBase::Memory::record("Force::dTVNL", sizeof(double) * pv.nloc * 3);
    ModuleBase::GlobalFunc::ZEROS(fsr.DHloc_fixed_x, pv.nloc);
    ModuleBase::GlobalFunc::ZEROS(fsr.DHloc_fixed_y, pv.nloc);
    ModuleBase::GlobalFunc::ZEROS(fsr.DHloc_fixed_z, pv.nloc);

    // calculate dT
    // calculate T + VNL(P1) in LCAO basis
    LCAO_domain::build_ST_new(fsr,
                              'T',
                              cal_deri,
                              PARAM.inp.cal_stress,
                              ucell,
                              orb,
                              pv,
                              two_center_bundle,
                              &gd,
                              nullptr);

    // calculate asynchronous S matrix to output for Hefei-NAMD
    if (PARAM.inp.cal_syns)
    {
        cal_deri = false;
        ModuleBase::timer::tick("Forces", "allocate");
        ModuleBase::WARNING_QUIT("cal_syns", "this function has been broken and will be fixed later.");
    }

    ModuleBase::timer::tick("Forces", "allocate");
    return;
}

template <>
void Force_LCAO<double>::finish_ftable(ForceStressArrays& fsr)
{
    delete[] fsr.DSloc_x;
    delete[] fsr.DSloc_y;
    delete[] fsr.DSloc_z;
    delete[] fsr.DHloc_fixed_x;
    delete[] fsr.DHloc_fixed_y;
    delete[] fsr.DHloc_fixed_z;

    if (PARAM.inp.cal_stress) // added by zhengdy-stress
    {
        delete[] fsr.DSloc_11;
        delete[] fsr.DSloc_12;
        delete[] fsr.DSloc_13;
        delete[] fsr.DSloc_22;
        delete[] fsr.DSloc_23;
        delete[] fsr.DSloc_33;
        delete[] fsr.DHloc_fixed_11;
        delete[] fsr.DHloc_fixed_12;
        delete[] fsr.DHloc_fixed_13;
        delete[] fsr.DHloc_fixed_22;
        delete[] fsr.DHloc_fixed_23;
        delete[] fsr.DHloc_fixed_33;
    }
    return;
}

// be called in force_lo.cpp
template <>
void Force_LCAO<double>::ftable(const bool isforce,
                                const bool isstress,
                                ForceStressArrays& fsr, // mohan add 2024-06-16
                                const UnitCell& ucell,
                                const Grid_Driver& gd,
                                const psi::Psi<double>* psi,
                                const elecstate::ElecState* pelec,
                                ModuleBase::matrix& foverlap,
                                ModuleBase::matrix& ftvnl_dphi,
                                ModuleBase::matrix& fvnl_dbeta,
                                ModuleBase::matrix& fvl_dphi,
                                ModuleBase::matrix& soverlap,
                                ModuleBase::matrix& stvnl_dphi,
                                ModuleBase::matrix& svnl_dbeta,
                                ModuleBase::matrix& svl_dphi,
#ifdef __MLALGO
                                ModuleBase::matrix& fvnl_dalpha,
                                ModuleBase::matrix& svnl_dalpha,
                                LCAO_Deepks<double>& ld,
#endif
                                TGint<double>::type& gint,
                                const TwoCenterBundle& two_center_bundle,
                                const LCAO_Orbitals& orb,
                                const Parallel_Orbitals& pv,
                                const K_Vectors* kv,
                                Record_adj* ra)
{
    ModuleBase::TITLE("Forces", "ftable");
    ModuleBase::timer::tick("Forces", "ftable");

    // get DM
    const elecstate::DensityMatrix<double, double>* dm
        = dynamic_cast<const elecstate::ElecStateLCAO<double>*>(pelec)->get_DM();

    this->ParaV = dm->get_paraV_pointer();

    // allocate DSloc_x, DSloc_y, DSloc_z
    // allocate DHloc_fixed_x, DHloc_fixed_y, DHloc_fixed_z
    this->allocate(ucell, gd, pv, fsr, two_center_bundle, orb);

    const double* dSx[3] = {fsr.DSloc_x, fsr.DSloc_y, fsr.DSloc_z};
    const double* dSxy[6] = {fsr.DSloc_11, fsr.DSloc_12, fsr.DSloc_13, fsr.DSloc_22, fsr.DSloc_23, fsr.DSloc_33};
    // calculate the force related to 'energy density matrix'.
    PulayForceStress::cal_pulay_fs(
        foverlap,
        soverlap,
        this->cal_edm(pelec, *psi, *dm, *kv, pv, PARAM.inp.nspin, PARAM.inp.nbands, ucell, *ra),
        ucell,
        pv,
        dSx,
        dSxy,
        isforce,
        isstress);

    const double* dHx[3] = {fsr.DHloc_fixed_x, fsr.DHloc_fixed_y, fsr.DHloc_fixed_z};
    const double* dHxy[6] = {fsr.DHloc_fixed_11,
                             fsr.DHloc_fixed_12,
                             fsr.DHloc_fixed_13,
                             fsr.DHloc_fixed_22,
                             fsr.DHloc_fixed_23,
                             fsr.DHloc_fixed_33};
    // tvnl_dphi
    PulayForceStress::cal_pulay_fs(ftvnl_dphi, stvnl_dphi, *dm, ucell, pv, dHx, dHxy, isforce, isstress);

    // vl_dphi
    PulayForceStress::cal_pulay_fs(fvl_dphi,
                                   svl_dphi,
                                   *dm,
                                   ucell,
                                   pelec->pot,
                                   gint,
                                   isforce,
                                   isstress,
                                   false /*reset dm to gint*/);

#ifdef __MLALGO
    if (PARAM.inp.deepks_scf)
    {
        // No need to update E_delta here since it have been done in LCAO_Deepks_Interface in after_scf
        const int nks = 1;
        DeePKS_domain::cal_f_delta<double>(ld.dm_r,
                                           ucell,
                                           orb,
                                           gd,
                                           *this->ParaV,
                                           nks,
                                           kv->kvec_d,
                                           ld.phialpha,
                                           ld.gedm,
                                           ld.inl_index,
                                           fvnl_dalpha,
                                           isstress,
                                           svnl_dalpha);
    }
#endif

    if (isforce)
    {
        Parallel_Reduce::reduce_pool(foverlap.c, foverlap.nr * foverlap.nc);
        Parallel_Reduce::reduce_pool(ftvnl_dphi.c, ftvnl_dphi.nr * ftvnl_dphi.nc);
        Parallel_Reduce::reduce_pool(fvnl_dbeta.c, fvnl_dbeta.nr * fvnl_dbeta.nc);
        Parallel_Reduce::reduce_pool(fvl_dphi.c, fvl_dphi.nr * fvl_dphi.nc);
#ifdef __MLALGO
        Parallel_Reduce::reduce_pool(fvnl_dalpha.c, fvnl_dalpha.nr * fvnl_dalpha.nc);
#endif
    }
    if (isstress)
    {
        Parallel_Reduce::reduce_pool(soverlap.c, soverlap.nr * soverlap.nc);
        Parallel_Reduce::reduce_pool(stvnl_dphi.c, stvnl_dphi.nr * stvnl_dphi.nc);
        Parallel_Reduce::reduce_pool(svnl_dbeta.c, svnl_dbeta.nr * svnl_dbeta.nc);
        Parallel_Reduce::reduce_pool(svl_dphi.c, svl_dphi.nr * svl_dphi.nc);
#ifdef __MLALGO
        Parallel_Reduce::reduce_pool(svnl_dalpha.c, svnl_dalpha.nr * svnl_dalpha.nc);
#endif
    }

#ifdef __MLALGO
    if (PARAM.inp.deepks_scf && PARAM.inp.deepks_out_unittest)
    {
        std::ofstream ofs_f("F_delta.dat");
        std::ofstream ofs_s("stress_delta.dat");
        ofs_f << std::setprecision(10);
        ofs_s << std::setprecision(10);
        fvnl_dalpha.print(ofs_f);
        ofs_f.close();
        svnl_dalpha.print(ofs_s);
        ofs_s.close();
    }
#endif

    // delete DSloc_x, DSloc_y, DSloc_z
    // delete DHloc_fixed_x, DHloc_fixed_y, DHloc_fixed_z
    this->finish_ftable(fsr);

    ModuleBase::timer::tick("Forces", "ftable");
    return;
}
