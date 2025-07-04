#ifndef CAL_EDM_TDDFT_H
#define CAL_EDM_TDDFT_H

#include "source_basis/module_ao/parallel_orbitals.h"
#include "source_cell/klist.h"
#include "source_estate/elecstate_lcao.h"
#include "source_hamilt/hamilt.h"

namespace elecstate
{
void cal_edm_tddft(Parallel_Orbitals& pv,
                   elecstate::ElecState* pelec,
                   K_Vectors& kv,
                   hamilt::Hamilt<std::complex<double>>* p_hamilt);
} // namespace elecstate
#endif // CAL_EDM_TDDFT_H
