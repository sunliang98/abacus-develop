#pragma once
#include <vector>
#include "module_hamilt_lcao/module_hcontainer/hcontainer.h"
#include "gint_type.h"
#include "gint_dvlocal.h"

namespace ModuleGint
{

void cal_gint_vl(
    const double* vr_eff,
    HContainer<double>* hR);

void cal_gint_vl(
    std::vector<const double*> vr_eff,
    HContainer<std::complex<double>>* hR);

void cal_gint_vl_metagga(
    const double* vr_eff,
    const double* vfork,
    HContainer<double>* hR);

void cal_gint_vl_metagga(
    std::vector<const double*> vr_eff,
    std::vector<const double*> vofk,
    HContainer<std::complex<double>>* hR);

void cal_gint_rho(
    const std::vector<HContainer<double>*>& dm_vec,
    const int nspin,
    double **rho,
    bool is_dm_symm = true);

void cal_gint_tau(        
    const std::vector<HContainer<double>*>& dm_vec,
    const int nspin,
    double**tau);

void cal_gint_fvl(
    const int nspin,
    const std::vector<const double*>& vr_eff,
    const std::vector<HContainer<double>*>& dm_vec,
    const bool isforce,
    const bool isstress,
    ModuleBase::matrix* fvl,
    ModuleBase::matrix* svl);

void cal_gint_fvl_meta(
    const int nspin,
    const std::vector<const double*>& vr_eff,
    const std::vector<const double*>& vofk,
    const std::vector<HContainer<double>*>& dm_vec,
    const bool isforce,
    const bool isstress,
    ModuleBase::matrix* fvl,
    ModuleBase::matrix* svl);

void cal_dvlocal_R_sparseMatrix(
    const int nspin,
    const int npol,
    const int current_spin,
    const int nlocal,
    const double sparse_thr,
    const double* vr_eff,
    const Parallel_Orbitals& pv,
    const UnitCell& ucell,
    const Grid_Driver& gdriver,
    LCAO_HS_Arrays& hs_arrays);


} // namespace ModuleGint