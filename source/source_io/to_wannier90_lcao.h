#ifndef TOWannier90_LCAO_H
#define TOWannier90_LCAO_H

#include "source_base/abfs-vector3_order.h"
#include "source_base/complexmatrix.h"
#include "source_base/global_function.h"
#include "source_base/global_variable.h"
#include "source_base/lapack_connector.h"
#include "source_base/matrix.h"
#include "source_base/matrix3.h"
#include "source_base/parallel_reduce.h"
#include "source_base/sph_bessel_recursive.h"
#include "source_base/timer.h"
#include "source_base/vector3.h"
#include "source_base/ylm.h"
#include "source_basis/module_ao/ORB_atomic_lm.h"
#include "source_basis/module_ao/ORB_gaunt_table.h"
#include "source_basis/module_ao/ORB_read.h"
#include "source_basis/module_ao/parallel_orbitals.h"
#include "source_cell/klist.h"
#include "source_cell/module_neighbor/sltk_grid_driver.h"
#include "source_lcao/hamilt_lcaodft/center2_orb-orb11.h"
#include "source_lcao/hamilt_lcaodft/center2_orb-orb21.h"
#include "source_lcao/hamilt_lcaodft/center2_orb.h"
#include "source_lcao/hamilt_lcaodft/wavefunc_in_pw.h"
#include "source_pw/hamilt_pwdft/global.h"
#include "source_psi/psi.h"
#include "single_R_io.h"
#include "to_wannier90.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <set>
#include <vector>

#ifdef __LCAO
#include "fR_overlap.h"
#include "source_base/abfs-vector3_order.h"
#include "source_base/math_lebedev_laikov.h"
#include "source_lcao/module_gint/grid_technique.h"
#include "source_lcao/module_hcontainer/hcontainer.h"

class Coordinate_3D
{
  public:
    double x = 0;
    double y = 0;
    double z = 0;

    Coordinate_3D(double x = 0.0, double y = 0.0, double z = 0.0) : x(x), y(y), z(z)
    {
    }

    bool operator<(const Coordinate_3D& other) const
    {
        const double threshold = 1e-8;
        if (std::abs(x - other.x) >= threshold)
            return x < other.x;
        if (std::abs(y - other.y) >= threshold)
            return y < other.y;
        return std::abs(z - other.z) >= threshold && z < other.z;
    }
};

class toWannier90_LCAO : public toWannier90
{
  public:
    toWannier90_LCAO(const bool& out_wannier_mmn,
                     const bool& out_wannier_amn,
                     const bool& out_wannier_unk,
                     const bool& out_wannier_eig,
                     const bool& out_wannier_wvfn_formatted,
                     const std::string& nnkpfile,
                     const std::string& wannier_spin,
                     const LCAO_Orbitals& orb
                     );
    ~toWannier90_LCAO();

    void calculate(const UnitCell& ucell,
                   const Grid_Driver& gd,
                   const ModuleBase::matrix& ekb,
                   const K_Vectors& kv,
                   const psi::Psi<std::complex<double>>& psi,
                   const Parallel_Orbitals* pv);

    void calculate(const UnitCell& ucell,
                   const Grid_Driver& gd,
                   const ModuleBase::matrix& ekb,
                   const K_Vectors& kv,
                   const psi::Psi<double>& psi,
                   const Parallel_Orbitals* pv)
    {
        throw std::logic_error("The wave function of toWannier90_LCAO_IN_PW is generally a std::complex<double> type.");
    }

    void cal_Amn(const UnitCell& ucell, const K_Vectors& kv, const psi::Psi<std::complex<double>>& psi);
    void cal_Mmn(const UnitCell& ucell, const K_Vectors& kv, const psi::Psi<std::complex<double>>& psi);
    void out_unk(const psi::Psi<std::complex<double>>& psi);

  protected:
    // Radial section of trial orbitals
    const int mesh_r = 1001; // unit is a.u.
    const double dr = 0.01;
    std::vector<std::vector<Numerical_Orbital_Lm>> A_orbs;

    const LCAO_Orbitals& orb_;

    // Use default element orbital information
    int orb_r_ntype = 0;

    ModuleBase::Sph_Bessel_Recursive::D2* psb_ = nullptr;
    ORB_gaunt_table MGT;
    double kmesh_times = 1;

    Numerical_Orbital_Lm orb_r;
    std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>> orbs;
    std::map<size_t, std::map<size_t, std::map<size_t, Center2_Orb::Orb11>>> center2_orb11_A;

    std::vector<ModuleBase::Vector3<double>> R_coor_car;
    std::vector<std::vector<std::vector<double>>> psi_psiA_R;

    std::vector<int> iw2it;
    std::vector<int> iw2ia;
    std::vector<int> iw2iL;
    std::vector<int> iw2iN;
    std::vector<int> iw2im;
    std::vector<int> iw2iorb;

    const Parallel_Orbitals* ParaV;

    void initialize_orb_table(const UnitCell& ucell);
    void produce_basis_orb();
    void set_R_coor(const UnitCell& ucell, const Grid_Driver& gd);
    void count_delta_k(const UnitCell& ucell, const K_Vectors& kv);

    std::vector<Coordinate_3D> delta_k_all;
    std::map<Coordinate_3D, int> delta_k_all_index;

    void unkdotkb(const UnitCell& ucell,
                  const K_Vectors& kv,
                  const psi::Psi<std::complex<double>>& psi_in,
                  const int& ik,
                  const int& ikb,
                  const ModuleBase::Vector3<double> G,
                  ModuleBase::ComplexMatrix& Mmn);

    void produce_trial_in_lcao();
    void construct_overlap_table_project();
    void cal_orbA_overlap_R(const UnitCell& ucell);

    void unkdotA(const K_Vectors& kv,
                 const psi::Psi<std::complex<double>>& psi_in,
                 const int& ik,
                 ModuleBase::ComplexMatrix& Amn);

    std::vector<FR_overlap<std::complex<double>>> FR;
};
#endif
#endif
