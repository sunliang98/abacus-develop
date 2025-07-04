#include "spar_dh.h"

#include "module_parameter/parameter.h"
#include "source_lcao/hamilt_lcaodft/LCAO_domain.h"
#include "source_lcao/module_gint/temp_gint/gint_interface.h"
#include <vector>

void sparse_format::cal_dS(const UnitCell& ucell,
    const Parallel_Orbitals& pv,
    LCAO_HS_Arrays& HS_Arrays,
    const Grid_Driver& grid,
    const TwoCenterBundle& two_center_bundle,
    const LCAO_Orbitals& orb,
    const double& sparse_thr)
{
ModuleBase::TITLE("sparse_format", "cal_dS");

sparse_format::set_R_range(HS_Arrays.all_R_coor, grid);
const int nnr = pv.nnr;

ForceStressArrays fsr_dh;
fsr_dh.DHloc_fixedR_x = new double[nnr];
fsr_dh.DHloc_fixedR_y = new double[nnr];
fsr_dh.DHloc_fixedR_z = new double[nnr];
ModuleBase::GlobalFunc::ZEROS(fsr_dh.DHloc_fixedR_x, nnr);
ModuleBase::GlobalFunc::ZEROS(fsr_dh.DHloc_fixedR_y, nnr);
ModuleBase::GlobalFunc::ZEROS(fsr_dh.DHloc_fixedR_z, nnr);
// the pointers of dS is different from dH, use the dh pointers to reuse the print functions
fsr_dh.DSloc_Rx = fsr_dh.DHloc_fixedR_x;
fsr_dh.DSloc_Ry = fsr_dh.DHloc_fixedR_y;
fsr_dh.DSloc_Rz = fsr_dh.DHloc_fixedR_z;
// cal dS=<phi|dphi> in LCAO
const bool cal_deri = true;
const bool cal_stress = false;
LCAO_domain::build_ST_new(fsr_dh,
       'S',
       cal_deri,
       cal_stress,
       ucell,
       orb,
       pv,
       two_center_bundle,
       &grid,
       nullptr,
       false); // delete unused parameter lm.Hloc_fixedR

sparse_format::cal_dSTN_R(ucell,pv, HS_Arrays, fsr_dh, grid, orb.cutoffs(), 0, sparse_thr);
delete[] fsr_dh.DHloc_fixedR_x;
delete[] fsr_dh.DHloc_fixedR_y;
delete[] fsr_dh.DHloc_fixedR_z;
return;
}
void sparse_format::cal_dH(const UnitCell& ucell,
                           const Parallel_Orbitals& pv,
                           LCAO_HS_Arrays& HS_Arrays,
                           const Grid_Driver& grid,
                           const TwoCenterBundle& two_center_bundle,
                           const LCAO_Orbitals& orb,
                           const int& current_spin,
                           const double& sparse_thr,
                           const ModuleBase::matrix& v_eff,
                           Gint_k& gint_k)
{
    ModuleBase::TITLE("sparse_format", "cal_dH");

    sparse_format::set_R_range(HS_Arrays.all_R_coor, grid);

    const int nnr = pv.nnr;

    ForceStressArrays fsr_dh;

    fsr_dh.DHloc_fixedR_x = new double[nnr];
    fsr_dh.DHloc_fixedR_y = new double[nnr];
    fsr_dh.DHloc_fixedR_z = new double[nnr];

    ModuleBase::GlobalFunc::ZEROS(fsr_dh.DHloc_fixedR_x, nnr);
    ModuleBase::GlobalFunc::ZEROS(fsr_dh.DHloc_fixedR_y, nnr);
    ModuleBase::GlobalFunc::ZEROS(fsr_dh.DHloc_fixedR_z, nnr);
    // cal dT=<phi|kin|dphi> in LCAO
    // cal T + VNL(P1) in LCAO basis
    const bool cal_deri = true;
    const bool cal_stress = false;
    LCAO_domain::build_ST_new(fsr_dh,
                              'T',
                              cal_deri,
                              cal_stress,
                              ucell,
                              orb,
                              pv,
                              two_center_bundle,
                              &grid,
                              nullptr,
                              false); // delete unused parameter lm.Hloc_fixedR

    LCAO_domain::build_Nonlocal_mu_new(pv,
                                       fsr_dh,
                                       nullptr,
                                       true,
                                       ucell,
                                       orb,
                                       *(two_center_bundle.overlap_orb_beta),
                                       &grid);

    sparse_format::cal_dSTN_R(ucell,pv, HS_Arrays, fsr_dh, grid, orb.cutoffs(), current_spin, sparse_thr);

    delete[] fsr_dh.DHloc_fixedR_x;
    delete[] fsr_dh.DHloc_fixedR_y;
    delete[] fsr_dh.DHloc_fixedR_z;

    if(PARAM.inp.nspin==2)
    {
#ifdef __OLD_GINT
        gint_k.allocate_pvdpR();
        // note: some MPI process will not have grids when MPI cores are too
        // many, v_eff in these processes are empty
        const double* vr_eff1
            = v_eff.nc * v_eff.nr > 0 ? &(v_eff(current_spin, 0)) : nullptr;

        if (!PARAM.globalv.gamma_only_local) 
        {
            if (PARAM.inp.vl_in_h) 
            {
                Gint_inout inout(vr_eff1,
                                 current_spin,
                                 Gint_Tools::job_type::dvlocal);
                gint_k.cal_gint(&inout);
            }
        }
        gint_k.cal_dvlocal_R_sparseMatrix(current_spin, sparse_thr, HS_Arrays, &pv, ucell, grid);
        gint_k.destroy_pvdpR();
#else
        const double* vr_eff1
            = v_eff.nc * v_eff.nr > 0 ? &(v_eff(current_spin, 0)) : nullptr;
        if (!PARAM.globalv.gamma_only_local) 
        {
            ModuleGint::cal_dvlocal_R_sparseMatrix(
                PARAM.inp.nspin, PARAM.globalv.npol, current_spin, PARAM.globalv.nlocal,
                sparse_thr, vr_eff1, pv, ucell, grid, HS_Arrays);
        }
#endif
    }
    return;
}

void sparse_format::set_R_range(std::set<Abfs::Vector3_Order<int>>& all_R_coor, const Grid_Driver& grid)
{
    int RminX = int(-grid.getGlayerX_minus());
    int RminY = int(-grid.getGlayerY_minus());
    int RminZ = int(-grid.getGlayerZ_minus());

    int Rx = grid.getGlayerX() + grid.getGlayerX_minus();
    int Ry = grid.getGlayerY() + grid.getGlayerY_minus();
    int Rz = grid.getGlayerZ() + grid.getGlayerZ_minus();

    for (int ix = 0; ix < Rx; ix++)
    {
        for (int iy = 0; iy < Ry; iy++)
        {
            for (int iz = 0; iz < Rz; iz++)
            {
                Abfs::Vector3_Order<int> temp_R(ix + RminX, iy + RminY, iz + RminZ);
                all_R_coor.insert(temp_R);
            }
        }
    }

    return;
}

void sparse_format::cal_dSTN_R(const UnitCell& ucell,
                               const Parallel_Orbitals& pv,
                               LCAO_HS_Arrays& HS_Arrays,
                               ForceStressArrays& fsr,
                               const Grid_Driver& grid,
                               const std::vector<double>& orb_cutoff,
                               const int& current_spin,
                               const double& sparse_thr)
{
    ModuleBase::TITLE("sparse_format", "cal_dSTN_R");

    int index = 0;
    ModuleBase::Vector3<double> dtau, tau1, tau2;
    ModuleBase::Vector3<double> dtau1, dtau2, tau0;

    double temp_value_double;
    std::complex<double> temp_value_complex;

    for (int T1 = 0; T1 < ucell.ntype; ++T1)
    {
        Atom* atom1 = &ucell.atoms[T1];
        for (int I1 = 0; I1 < atom1->na; ++I1)
        {
            tau1 = atom1->tau[I1];
            grid.Find_atom(ucell, tau1, T1, I1);
            Atom* atom1 = &ucell.atoms[T1];
            const int start = ucell.itiaiw2iwt(T1, I1, 0);

            for (int ad = 0; ad < grid.getAdjacentNum() + 1; ++ad)
            {
                const int T2 = grid.getType(ad);
                const int I2 = grid.getNatom(ad);
                Atom* atom2 = &ucell.atoms[T2];

                tau2 = grid.getAdjacentTau(ad);
                dtau = tau2 - tau1;
                double distance = dtau.norm() * ucell.lat0;
                double rcut = orb_cutoff[T1] + orb_cutoff[T2];

                bool adj = false;

                if (distance < rcut)
                {
                    adj = true;
                }
                else if (distance >= rcut)
                {
                    for (int ad0 = 0; ad0 < grid.getAdjacentNum() + 1; ++ad0)
                    {
                        const int T0 = grid.getType(ad0);

                        tau0 = grid.getAdjacentTau(ad0);
                        dtau1 = tau0 - tau1;
                        dtau2 = tau0 - tau2;

                        double distance1 = dtau1.norm() * ucell.lat0;
                        double distance2 = dtau2.norm() * ucell.lat0;

                        double rcut1 = orb_cutoff[T1] + ucell.infoNL.Beta[T0].get_rcut_max();
                        double rcut2 = orb_cutoff[T2] + ucell.infoNL.Beta[T0].get_rcut_max();

                        if (distance1 < rcut1 && distance2 < rcut2)
                        {
                            adj = true;
                            break;
                        }
                    }
                }

                if (adj)
                {
                    const int start2 = ucell.itiaiw2iwt(T2, I2, 0);

                    Abfs::Vector3_Order<int> dR(grid.getBox(ad).x, grid.getBox(ad).y, grid.getBox(ad).z);

                    for (int ii = 0; ii < atom1->nw * PARAM.globalv.npol; ii++)
                    {
                        const int iw1_all = start + ii;
                        const int mu = pv.global2local_row(iw1_all);

                        if (mu < 0)
                        {
                            continue;
                        }

                        for (int jj = 0; jj < atom2->nw * PARAM.globalv.npol; jj++)
                        {
                            int iw2_all = start2 + jj;
                            const int nu = pv.global2local_col(iw2_all);

                            if (nu < 0)
                            {
                                continue;
                            }

                            if (PARAM.inp.nspin != 4)
                            {
                                temp_value_double = fsr.DHloc_fixedR_x[index];
                                if (std::abs(temp_value_double) > sparse_thr)
                                {
                                    HS_Arrays.dHRx_sparse[current_spin][dR][iw1_all][iw2_all] = temp_value_double;
                                }
                                temp_value_double = fsr.DHloc_fixedR_y[index];
                                if (std::abs(temp_value_double) > sparse_thr)
                                {
                                    HS_Arrays.dHRy_sparse[current_spin][dR][iw1_all][iw2_all] = temp_value_double;
                                }
                                temp_value_double = fsr.DHloc_fixedR_z[index];
                                if (std::abs(temp_value_double) > sparse_thr)
                                {
                                    HS_Arrays.dHRz_sparse[current_spin][dR][iw1_all][iw2_all] = temp_value_double;
                                }
                            }
                            else
                            {
                                ModuleBase::WARNING_QUIT("cal_dSTN_R", "nspin=4 with SOC is not supported yet.");
                            }
                            ++index;
                        }
                    }
                }
            }
        }
    }

    return;
}

void sparse_format::destroy_dH_R_sparse(LCAO_HS_Arrays& HS_Arrays)
{
    ModuleBase::TITLE("LCAO_domain", "destroy_dH_R_sparse");

    if (PARAM.inp.nspin != 4)
    {
        std::map<Abfs::Vector3_Order<int>, std::map<size_t, std::map<size_t, double>>> empty_dHRx_sparse_up;
        std::map<Abfs::Vector3_Order<int>, std::map<size_t, std::map<size_t, double>>> empty_dHRx_sparse_down;
        std::map<Abfs::Vector3_Order<int>, std::map<size_t, std::map<size_t, double>>> empty_dHRy_sparse_up;
        std::map<Abfs::Vector3_Order<int>, std::map<size_t, std::map<size_t, double>>> empty_dHRy_sparse_down;
        std::map<Abfs::Vector3_Order<int>, std::map<size_t, std::map<size_t, double>>> empty_dHRz_sparse_up;
        std::map<Abfs::Vector3_Order<int>, std::map<size_t, std::map<size_t, double>>> empty_dHRz_sparse_down;

        HS_Arrays.dHRx_sparse[0].swap(empty_dHRx_sparse_up);
        HS_Arrays.dHRx_sparse[1].swap(empty_dHRx_sparse_down);
        HS_Arrays.dHRy_sparse[0].swap(empty_dHRy_sparse_up);
        HS_Arrays.dHRy_sparse[1].swap(empty_dHRy_sparse_down);
        HS_Arrays.dHRz_sparse[0].swap(empty_dHRz_sparse_up);
        HS_Arrays.dHRz_sparse[1].swap(empty_dHRz_sparse_down);
    }
    else
    {
        std::map<Abfs::Vector3_Order<int>, std::map<size_t, std::map<size_t, std::complex<double>>>>
            empty_dHRx_soc_sparse;
        std::map<Abfs::Vector3_Order<int>, std::map<size_t, std::map<size_t, std::complex<double>>>>
            empty_dHRy_soc_sparse;
        std::map<Abfs::Vector3_Order<int>, std::map<size_t, std::map<size_t, std::complex<double>>>>
            empty_dHRz_soc_sparse;

        HS_Arrays.dHRx_soc_sparse.swap(empty_dHRx_soc_sparse);
        HS_Arrays.dHRy_soc_sparse.swap(empty_dHRy_soc_sparse);
        HS_Arrays.dHRz_soc_sparse.swap(empty_dHRz_soc_sparse);
    }

    return;
}
