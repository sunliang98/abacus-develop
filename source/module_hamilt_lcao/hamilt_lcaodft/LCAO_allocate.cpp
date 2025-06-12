#include "module_base/timer.h"
#include "module_hamilt_lcao/hamilt_lcaodft/LCAO_domain.h"
#include "module_hamilt_lcao/module_deepks/LCAO_deepks.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"
#include "module_parameter/parameter.h"

namespace LCAO_domain
{
#ifdef __MLALGO
// It seems it is only related to DeePKS, so maybe we should move it to DeeKS_domain
template <typename T>
void DeePKS_init(const UnitCell& ucell,
                 Parallel_Orbitals& pv,
                 const int& nks,
                 const LCAO_Orbitals& orb,
                 LCAO_Deepks<T>& ld,
                 std::ofstream& ofs)
{
    ModuleBase::TITLE("LCAO_domain", "DeePKS_init");
    // preparation for DeePKS
    if (PARAM.inp.deepks_out_labels == 1 || PARAM.inp.deepks_scf)
    {
        // allocate relevant data structures for calculating descriptors
        std::vector<int> na;
        na.resize(ucell.ntype);
        for (int it = 0; it < ucell.ntype; it++)
        {
            na[it] = ucell.atoms[it].na;
        }

        ld.init(orb, ucell.nat, ucell.ntype, nks, pv, na, ofs);

        if (PARAM.inp.deepks_scf)
        {
            ld.allocate_V_delta(ucell.nat, nks);
        }
    }
    return;
}

template void DeePKS_init<double>(const UnitCell& ucell,
                                  Parallel_Orbitals& pv,
                                  const int& nks,
                                  const LCAO_Orbitals& orb,
                                  LCAO_Deepks<double>& ld,
                                  std::ofstream& ofs);

template void DeePKS_init<std::complex<double>>(const UnitCell& ucell,
                                                Parallel_Orbitals& pv,
                                                const int& nks,
                                                const LCAO_Orbitals& orb,
                                                LCAO_Deepks<std::complex<double>>& ld,
                                                std::ofstream& ofs);
#endif
} // namespace LCAO_domain
