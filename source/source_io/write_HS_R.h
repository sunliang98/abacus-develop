#ifndef WRITE_HS_R_H
#define WRITE_HS_R_H

#include "source_base/matrix.h"
#include "source_basis/module_nao/two_center_bundle.h"
#include "source_cell/klist.h"
#include "source_hamilt/hamilt.h"
#include "source_lcao/module_gint/gint_k.h"
#include "source_pw/hamilt_pwdft/global.h"

namespace ModuleIO
{
	using TAC = std::pair<int, std::array<int, 3>>;
	void output_HSR(const UnitCell& ucell,
			const int& istep,
			const ModuleBase::matrix& v_eff,
			const Parallel_Orbitals& pv,
			LCAO_HS_Arrays& HS_Arrays,
			const Grid_Driver& grid, // mohan add 2024-04-06
			const K_Vectors& kv,
			hamilt::Hamilt<std::complex<double>>* p_ham,
#ifdef __EXX
			const std::vector<std::map<int, std::map<TAC, RI::Tensor<double>>>>* Hexxd = nullptr,
			const std::vector<std::map<int, std::map<TAC, RI::Tensor<std::complex<double>>>>>* Hexxc = nullptr,
#endif
			const std::string& SR_filename = "srs1_nao.csr",
			const std::string& HR_filename_up = "hrs1_nao.csr",
			const std::string HR_filename_down = "hrs2_nao.csr",
			const bool& binary = false,
			const double& sparse_threshold = 1e-10); // LiuXh add 2019-07-15, modify in 2021-12-3

	void output_dHR(const int& istep,
			const ModuleBase::matrix& v_eff,
			Gint_k& gint_k, // mohan add 2024-04-01
			const UnitCell& ucell,
			const Parallel_Orbitals& pv,
			LCAO_HS_Arrays& HS_Arrays,
			const Grid_Driver& grid, // mohan add 2024-04-06
			const TwoCenterBundle& two_center_bundle,
			const LCAO_Orbitals& orb,
			const K_Vectors& kv,
			const bool& binary = false,
			const double& sparse_threshold = 1e-10);

	void output_dSR(const int& istep,
			const UnitCell& ucell,
			const Parallel_Orbitals& pv,
			LCAO_HS_Arrays& HS_Arrays,
			const Grid_Driver& grid, // mohan add 2024-04-06
			const TwoCenterBundle& two_center_bundle,
			const LCAO_Orbitals& orb,
			const K_Vectors& kv,
			const bool& binary = false,
			const double& sparse_thr = 1e-10);

	void output_TR(const int istep,
			const UnitCell& ucell,
			const Parallel_Orbitals& pv,
			LCAO_HS_Arrays& HS_Arrays,
			const Grid_Driver& grid,
			const TwoCenterBundle& two_center_bundle,
			const LCAO_Orbitals& orb,
			const std::string& TR_filename = "trs1_nao.csr",
			const bool& binary = false,
			const double& sparse_threshold = 1e-10);

	void output_SR(Parallel_Orbitals& pv,
			const Grid_Driver& grid,
			hamilt::Hamilt<std::complex<double>>* p_ham,
			const std::string& SR_filename = "srs1_nao.csr",
			const bool& binary = false,
			const double& sparse_threshold = 1e-10);
} // namespace ModuleIO

#endif
