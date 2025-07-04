#ifndef OUTPUT_MAT_SPARSE_H
#define OUTPUT_MAT_SPARSE_H

#include "source_basis/module_ao/parallel_orbitals.h"
#include "source_basis/module_nao/two_center_bundle.h"
#include "source_cell/klist.h"
#include "source_hamilt/hamilt.h"
#include "source_lcao/module_gint/gint_k.h"

namespace ModuleIO
{
/// @brief the output interface to write the sparse matrix of H, S, T, and r
template <typename T> 
void output_mat_sparse(const bool& out_mat_hsR,
                       const bool& out_mat_dh,
                       const bool& out_mat_ds,
                       const bool& out_mat_t,
                       const bool& out_mat_r,
                       const int& istep,
                       const ModuleBase::matrix& v_eff,
                       const Parallel_Orbitals& pv,
                       Gint_k& gint_k, // mohan add 2024-04-01
                       const TwoCenterBundle& two_center_bundle,
                       const LCAO_Orbitals& orb,
                       UnitCell& ucell,
                       const Grid_Driver& grid, // mohan add 2024-04-06
                       const K_Vectors& kv,
                       hamilt::Hamilt<T>* p_ham);
} // namespace ModuleIO

#endif // OUTPUT_MAT_SPARSE_H
