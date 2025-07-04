#ifndef WRITE_HS_SPARSE_H
#define WRITE_HS_SPARSE_H

#include "source_base/global_function.h"
#include "source_base/global_variable.h"
#include "source_basis/module_ao/parallel_orbitals.h"
#include "source_lcao/hamilt_lcaodft/LCAO_HS_arrays.hpp"

#include <string>

namespace ModuleIO
{

// jingan add 2021-6-4, modify 2021-12-2
void save_HSR_sparse(const int& istep,
                     const Parallel_Orbitals& pv,
                     LCAO_HS_Arrays& HS_Arrays,
                     const double& sparse_thr,
                     const bool& binary,
                     const std::string& SR_filename,
                     const std::string& HR_filename_up,
                     const std::string& HR_filename_down);

void save_dH_sparse(const int& istep,
                    const Parallel_Orbitals& pv,
                    LCAO_HS_Arrays& HS_Arrays,
                    const double& sparse_thr,
                    const bool& binary,
                    const std::string& fileflag = "h");

template <typename Tdata>
void save_sparse(const std::map<Abfs::Vector3_Order<int>, std::map<size_t, std::map<size_t, Tdata>>>& smat,
                 const std::set<Abfs::Vector3_Order<int>>& all_R_coor,
                 const double& sparse_thr,
                 const bool& binary,
                 const std::string& filename,
                 const Parallel_Orbitals& pv,
                 const std::string& label,
                 const int& istep = -1,
                 const bool& reduce = true);
} // namespace ModuleIO

#endif
