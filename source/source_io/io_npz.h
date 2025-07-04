#ifndef NPZ_IO_H
#define NPZ_IO_H

#include "source_basis/module_ao/parallel_orbitals.h"
#include "source_cell/unitcell.h"
#include "source_lcao/module_hcontainer/hcontainer.h"

#include <string>
#include <vector>

namespace ModuleIO
{

void read_mat_npz(const Parallel_Orbitals* paraV,
                  const UnitCell& ucell,
                  std::string& zipname,
                  hamilt::HContainer<double>& hR);

void output_mat_npz(const UnitCell& ucell, std::string& zipname, const hamilt::HContainer<double>& hR);

} // namespace ModuleIO

#endif // NPZ_IO_H
