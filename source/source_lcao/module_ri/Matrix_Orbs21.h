//=======================
// AUTHOR : Peize Lin
// DATE :   2022-08-17
//=======================

#ifndef MATRIX_ORB21_H
#define MATRIX_ORB21_H

#include "source_base/element_basis_index.h"
#include "source_base/vector3.h"
#include "source_basis/module_ao/ORB_gaunt_table.h"
#include "source_basis/module_ao/ORB_read.h"
#include "source_lcao/hamilt_lcaodft/center2_orb-orb21.h"
#include "source_cell/unitcell.h"
#include <RI/global/Tensor.h>
#include <map>
#include <set>
#include <vector>

class Matrix_Orbs21
{
  public:
    // mode:
    //    1: <jYs lcaos|lcaos>  <abfs lcaos|lcaos>
    void init(const int mode,
              const UnitCell& ucell,
              const LCAO_Orbitals& orb,
              const double kmesh_times,  // extend Kcut, keep dK
              const double rmesh_times); // extend Rcut, keep dR

    void init_radial(const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& orb_A1,
                     const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& orb_A2,
                     const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& orb_B);
    void init_radial(const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>& orb_A1,
                     const LCAO_Orbitals& orb_A2,
                     const LCAO_Orbitals& orb_B);

    void init_radial_table();
    void init_radial_table(const std::map<size_t, std::map<size_t, std::set<double>>>& Rs); // unit: ucell.lat0

    enum class Matrix_Order
    {
        A1A2B,
        A1BA2,
        A2A1B,
        A2BA1,
        BA1A2,
        BA2A1
    };

    template <typename Tdata>
    RI::Tensor<Tdata> cal_overlap_matrix(const size_t TA,
                                         const size_t TB,
                                         const ModuleBase::Vector3<double>& tauA, // unit: ucell.lat0
                                         const ModuleBase::Vector3<double>& tauB, // unit: ucell.lat0
                                         const ModuleBase::Element_Basis_Index::IndexLNM& index_A1,
                                         const ModuleBase::Element_Basis_Index::IndexLNM& index_A2,
                                         const ModuleBase::Element_Basis_Index::IndexLNM& index_B,
                                         const Matrix_Order& matrix_order) const;
    template <typename Tdata>
    std::array<RI::Tensor<Tdata>, 3> cal_grad_overlap_matrix(
        const size_t TA,
        const size_t TB,
        const ModuleBase::Vector3<double>& tauA, // unit: ucell.lat0
        const ModuleBase::Vector3<double>& tauB, // unit: ucell.lat0
        const ModuleBase::Element_Basis_Index::IndexLNM& index_A1,
        const ModuleBase::Element_Basis_Index::IndexLNM& index_A2,
        const ModuleBase::Element_Basis_Index::IndexLNM& index_B,
        const Matrix_Order& matrix_order) const;

    template <typename Tdata>
    std::map<size_t, std::map<size_t, std::map<size_t, std::map<size_t, std::vector<RI::Tensor<Tdata>>>>>>
        cal_overlap_matrix_all(const UnitCell& ucell,
                               const ModuleBase::Element_Basis_Index::IndexLNM& index_A1,
                               const ModuleBase::Element_Basis_Index::IndexLNM& index_A2,
                               const ModuleBase::Element_Basis_Index::IndexLNM& index_B) const;

  private:
    ModuleBase::Sph_Bessel_Recursive::D2* psb_ = nullptr;
    ORB_gaunt_table MGT;
    const double lcao_dr_ = 0.01;
    double* lat0 = nullptr;                                                         // restore ucell.lat0
    std::map<size_t,                                                                // TA
             std::map<size_t,                                                       // TB
                      std::map<int,                                                 // LA1
                               std::map<size_t,                                     // NA1
                                        std::map<int,                               // LA2
                                                 std::map<size_t,                   // NA2
                                                          std::map<int,             // LB
                                                                   std::map<size_t, // NB
                                                                            Center2_Orb::Orb21>>>>>>>>
        center2_orb21_s;
    // this->center2_orb21_s[TA][TB][LA1][NA1][LA2][NA2][LB][NB]
};

#include "Matrix_Orbs21.hpp"

#endif
