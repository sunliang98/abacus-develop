#ifndef DOS_LCAO_H
#define DOS_LCAO_H

#include "module_io/nscf_fermi_surf.h"
#include "module_io/write_dos_lcao.h"
#include "module_elecstate/fp_energy.h"
#include "module_hamilt_general/hamilt.h"

namespace ModuleIO
{
    template<typename T>
		void out_dos_nao(
				const psi::Psi<T>* psi,
				LCAO_Matrix &lm,
				const Parallel_Orbitals &pv,
				const ModuleBase::matrix& ekb,
				const ModuleBase::matrix& wg,
				const double& dos_edelta_ev,
				const double& dos_scale,
				const double& dos_sigma,
				const K_Vectors& kv,
				const Parallel_Kpoints& Pkpoints,
				const UnitCell& ucell,
				const elecstate::efermi& eferm,
				int nbands,
				hamilt::Hamilt<T>* p_ham);
}

#endif
