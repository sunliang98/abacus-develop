//=======================
// AUTHOR : Peize Lin
#include "module_parameter/parameter.h"
// DATE :   2022-08-17
//=======================

#ifndef EXX_LRI_HPP
#define EXX_LRI_HPP

#include "Exx_LRI.h"
#include "RI_2D_Comm.h"
#include "RI_Util.h"
#include "source_lcao/module_ri/exx_abfs-construct_orbs.h"
#include "source_lcao/module_ri/exx_abfs-io.h"
#include "source_lcao/module_ri/conv_coulomb_pot_k.h"
#include "source_base/tool_title.h"
#include "source_base/timer.h"
#include "source_lcao/module_ri/serialization_cereal.h"
#include "source_lcao/module_ri/Mix_DMk_2D.h"
#include "source_basis/module_ao/parallel_orbitals.h"

#include <RI/distribute/Distribute_Equally.h>
#include <RI/global/Map_Operator-3.h>

#include <fstream>
#include <string>

template<typename Tdata>
void Exx_LRI<Tdata>::init(const MPI_Comm &mpi_comm_in,
						  const UnitCell &ucell,
						  const K_Vectors &kv_in,
						  const LCAO_Orbitals& orb)
{
	ModuleBase::TITLE("Exx_LRI","init");
	ModuleBase::timer::tick("Exx_LRI", "init");

	this->mpi_comm = mpi_comm_in;
	this->p_kv = &kv_in;
	this->orb_cutoff_ = orb.cutoffs();

	this->lcaos = Exx_Abfs::Construct_Orbs::change_orbs( orb, this->info.kmesh_times );

	const std::vector<std::vector<std::vector<Numerical_Orbital_Lm>>>
		abfs_same_atom = Exx_Abfs::Construct_Orbs::abfs_same_atom(ucell, orb, this->lcaos, this->info.kmesh_times, this->info.pca_threshold );
	if(this->info.files_abfs.empty())
		{ this->abfs = abfs_same_atom;}
	else
		{ this->abfs = Exx_Abfs::IO::construct_abfs( abfs_same_atom, orb, this->info.files_abfs, this->info.kmesh_times ); 	}
	Exx_Abfs::Construct_Orbs::print_orbs_size(ucell, this->abfs, GlobalV::ofs_running);

	for( size_t T=0; T!=this->abfs.size(); ++T )
		{ GlobalC::exx_info.info_ri.abfs_Lmax = std::max( GlobalC::exx_info.info_ri.abfs_Lmax, static_cast<int>(this->abfs[T].size())-1 ); }

	for(const auto &settings_list : this->info.coulomb_settings)
	{
		const std::map<Conv_Coulomb_Pot_K::Coulomb_Type, std::vector<std::map<std::string,std::string>>>
			coulomb_param_updated = RI_Util::update_coulomb_param(settings_list.second.second, ucell.omega, this->p_kv->get_nkstot_full());
		this->exx_objs[settings_list.first].abfs_ccp = Conv_Coulomb_Pot_K::cal_orbs_ccp(this->abfs, coulomb_param_updated, this->info.ccp_rmesh_times);
		this->exx_objs[settings_list.first].cv.set_orbitals(ucell, orb,
															this->lcaos, this->abfs, this->exx_objs[settings_list.first].abfs_ccp,
															this->info.kmesh_times, this->info.ccp_rmesh_times );
	}

	ModuleBase::timer::tick("Exx_LRI", "init");
}

template<typename Tdata>
void Exx_LRI<Tdata>::cal_exx_ions(const UnitCell& ucell,
								  const bool write_cv)
{
	ModuleBase::TITLE("Exx_LRI","cal_exx_ions");
	ModuleBase::timer::tick("Exx_LRI", "cal_exx_ions");

	// init_radial_table_ions( cal_atom_centres_core(atom_pairs_core_origin), atom_pairs_core_origin );

	// this->m_abfsabfs.init_radial_table(Rradial);
	// this->m_abfslcaos_lcaos.init_radial_table(Rradial);

	std::vector<TA> atoms(ucell.nat);
	for(int iat=0; iat<ucell.nat; ++iat)
		{ atoms[iat] = iat; }
	std::map<TA,TatomR> atoms_pos;
	for(int iat=0; iat<ucell.nat; ++iat)
		{ atoms_pos[iat] = RI_Util::Vector3_to_array3( ucell.atoms[ ucell.iat2it[iat] ].tau[ ucell.iat2ia[iat] ] ); }
	const std::array<TatomR,Ndim> latvec
		= {RI_Util::Vector3_to_array3(ucell.a1),
		   RI_Util::Vector3_to_array3(ucell.a2),
		   RI_Util::Vector3_to_array3(ucell.a3)};
	const std::array<Tcell,Ndim> period = {this->p_kv->nmp[0], this->p_kv->nmp[1], this->p_kv->nmp[2]};

	this->exx_lri.set_parallel(this->mpi_comm, atoms_pos, latvec, period);

	// std::max(3) for gamma_only, list_A2 should contain cell {-1,0,1}. In the future distribute will be neighbour.
	const std::array<Tcell,Ndim> period_Vs = LRI_CV_Tools::cal_latvec_range<Tcell>(1+this->info.ccp_rmesh_times, ucell, orb_cutoff_);
	const std::pair<std::vector<TA>, std::vector<std::vector<std::pair<TA,std::array<Tcell,Ndim>>>>>
		list_As_Vs = RI::Distribute_Equally::distribute_atoms_periods(this->mpi_comm, atoms, period_Vs, 2, false);

	std::map<TA,std::map<TAC,RI::Tensor<Tdata>>> Vs;
	std::map<TA, std::map<TAC, std::array<RI::Tensor<Tdata>, Ndim>>> dVs;
	for(const auto &settings_list : this->info.coulomb_settings)
	{
		std::map<TA,std::map<TAC,RI::Tensor<Tdata>>>
			Vs_temp = this->exx_objs[settings_list.first].cv.cal_Vs(ucell,
				list_As_Vs.first, list_As_Vs.second[0],
				{{"writable_Vws",true}});
		this->exx_objs[settings_list.first].cv.Vws = LRI_CV_Tools::get_CVws(ucell,Vs_temp);
		Vs = Vs.empty() ? Vs_temp : LRI_CV_Tools::add(Vs, Vs_temp);

		if(PARAM.inp.cal_force || PARAM.inp.cal_stress)
		{
			std::map<TA, std::map<TAC, std::array<RI::Tensor<Tdata>, Ndim>>>
				dVs_temp = this->exx_objs[settings_list.first].cv.cal_dVs(ucell,
					list_As_Vs.first, list_As_Vs.second[0],
					{{"writable_dVws",true}});
			this->exx_objs[settings_list.first].cv.dVws = LRI_CV_Tools::get_dCVws(ucell,dVs_temp);
			dVs = dVs.empty() ? dVs_temp : LRI_CV_Tools::add(dVs, dVs_temp);
		}
	}
	if (write_cv && GlobalV::MY_RANK == 0)
		{ LRI_CV_Tools::write_Vs_abf(Vs, PARAM.globalv.global_out_dir + "Vs"); }
	this->exx_lri.set_Vs(std::move(Vs), this->info.V_threshold);

	if(PARAM.inp.cal_force || PARAM.inp.cal_stress)
	{
		std::array<std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>, Ndim>
			dVs_order = LRI_CV_Tools::change_order(std::move(dVs));
		this->exx_lri.set_dVs(std::move(dVs_order), this->info.V_grad_threshold);
		if(PARAM.inp.cal_stress)
		{
			std::array<std::array<std::map<TA,std::map<TAC,RI::Tensor<Tdata>>>,3>,3> dVRs = LRI_CV_Tools::cal_dMRs(ucell,dVs_order);
			this->exx_lri.set_dVRs(std::move(dVRs), this->info.V_grad_R_threshold);
		}
	}

	const std::array<Tcell,Ndim> period_Cs = LRI_CV_Tools::cal_latvec_range<Tcell>(2, ucell,orb_cutoff_);
	const std::pair<std::vector<TA>, std::vector<std::vector<std::pair<TA,std::array<Tcell,Ndim>>>>>
		list_As_Cs = RI::Distribute_Equally::distribute_atoms_periods(this->mpi_comm, atoms, period_Cs, 2, false);

	std::map<TA,std::map<TAC,RI::Tensor<Tdata>>> Cs;
	std::map<TA, std::map<TAC, std::array<RI::Tensor<Tdata>, 3>>> dCs;
	for(const auto &settings_list : this->info.coulomb_settings)
	{
		if(settings_list.second.first)
		{
			std::pair<std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>,
				std::map<TA, std::map<TAC, std::array<RI::Tensor<Tdata>, 3>>>>
					Cs_dCs = this->exx_objs[settings_list.first].cv.cal_Cs_dCs(
						ucell,
						list_As_Cs.first, list_As_Cs.second[0],
						{{"cal_dC",PARAM.inp.cal_force||PARAM.inp.cal_stress},
						{"writable_Cws",true}, {"writable_dCws",true}, {"writable_Vws",false}, {"writable_dVws",false}});
			std::map<TA,std::map<TAC,RI::Tensor<Tdata>>> &Cs_temp = std::get<0>(Cs_dCs);
			this->exx_objs[settings_list.first].cv.Cws = LRI_CV_Tools::get_CVws(ucell,Cs_temp);
			Cs = Cs.empty() ? Cs_temp : LRI_CV_Tools::add(Cs, Cs_temp);

			if(PARAM.inp.cal_force || PARAM.inp.cal_stress)
			{
				std::map<TA, std::map<TAC, std::array<RI::Tensor<Tdata>, 3>>> &dCs_temp = std::get<1>(Cs_dCs);
				this->exx_objs[settings_list.first].cv.dCws = LRI_CV_Tools::get_dCVws(ucell,dCs_temp);
				dCs = dCs.empty() ? dCs_temp : LRI_CV_Tools::add(dCs, dCs_temp);
			}
		}
	}
	if (write_cv && GlobalV::MY_RANK == 0)
		{ LRI_CV_Tools::write_Cs_ao(Cs, PARAM.globalv.global_out_dir + "Cs"); }
	this->exx_lri.set_Cs(std::move(Cs), this->info.C_threshold);

	if(PARAM.inp.cal_force || PARAM.inp.cal_stress)
	{
		std::array<std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>, Ndim>
			dCs_order = LRI_CV_Tools::change_order(std::move(dCs));
		this->exx_lri.set_dCs(std::move(dCs_order), this->info.C_grad_threshold);
		if(PARAM.inp.cal_stress)
		{
			std::array<std::array<std::map<TA,std::map<TAC,RI::Tensor<Tdata>>>,3>,3> dCRs = LRI_CV_Tools::cal_dMRs(ucell,dCs_order);
			this->exx_lri.set_dCRs(std::move(dCRs), this->info.C_grad_R_threshold);
		}
	}
	ModuleBase::timer::tick("Exx_LRI", "cal_exx_ions");
}

template<typename Tdata>
void Exx_LRI<Tdata>::cal_exx_elec(const std::vector<std::map<TA, std::map<TAC, RI::Tensor<Tdata>>>>& Ds,
	const UnitCell& ucell,
	const Parallel_Orbitals& pv,
	const ModuleSymmetry::Symmetry_rotation* p_symrot)
{
	ModuleBase::TITLE("Exx_LRI","cal_exx_elec");
	ModuleBase::timer::tick("Exx_LRI", "cal_exx_elec");

	const std::vector<std::tuple<std::set<TA>, std::set<TA>>> judge = RI_2D_Comm::get_2D_judge(ucell,pv);

	if(p_symrot)
		{ this->exx_lri.set_symmetry(true, p_symrot->get_irreducible_sector()); }
	else
		{ this->exx_lri.set_symmetry(false, {}); }

	this->Hexxs.resize(PARAM.inp.nspin);
	this->Eexx = 0;
	for(int is=0; is<PARAM.inp.nspin; ++is)
	{
		const std::string suffix = ((PARAM.inp.cal_force || PARAM.inp.cal_stress) ? std::to_string(is) : "");

		this->exx_lri.set_Ds(Ds[is], this->info.dm_threshold, suffix);
		this->exx_lri.cal_Hs({ "","",suffix });

		if (!p_symrot)
		{
			this->Hexxs[is] = RI::Communicate_Tensors_Map_Judge::comm_map2_first(
				this->mpi_comm, std::move(this->exx_lri.Hs), std::get<0>(judge[is]), std::get<1>(judge[is]));
		}
		else
		{
			// reduce but not repeat
			auto Hs_a2D = this->exx_lri.post_2D.set_tensors_map2(this->exx_lri.Hs);
			// rotate locally without repeat
			Hs_a2D = p_symrot->restore_HR(ucell.symm, ucell.atoms, ucell.st, 'H', Hs_a2D);
			// cal energy using full Hs without repeat
			this->exx_lri.energy = this->exx_lri.post_2D.cal_energy(
				this->exx_lri.post_2D.saves["Ds_" + suffix],
				this->exx_lri.post_2D.set_tensors_map2(Hs_a2D));
			// get repeated full Hs for abacus
			this->Hexxs[is] = RI::Communicate_Tensors_Map_Judge::comm_map2_first(
				this->mpi_comm, std::move(Hs_a2D), std::get<0>(judge[is]), std::get<1>(judge[is]));
		}
		this->Eexx += std::real(this->exx_lri.energy);
		post_process_Hexx(this->Hexxs[is]);
	}
	this->Eexx = post_process_Eexx(this->Eexx);
	this->exx_lri.set_symmetry(false, {});
	ModuleBase::timer::tick("Exx_LRI", "cal_exx_elec");
}

template<typename Tdata>
void Exx_LRI<Tdata>::post_process_Hexx( std::map<TA, std::map<TAC, RI::Tensor<Tdata>>> &Hexxs_io ) const
{
	ModuleBase::TITLE("Exx_LRI","post_process_Hexx");
	constexpr Tdata frac = -1 * 2;								// why?	Hartree to Ry?
	const std::function<void(RI::Tensor<Tdata>&)>
		multiply_frac = [&frac](RI::Tensor<Tdata> &t)
		{ t = t*frac; };
	RI::Map_Operator::for_each( Hexxs_io, multiply_frac );
}

template<typename Tdata>
double Exx_LRI<Tdata>::post_process_Eexx(const double& Eexx_in) const
{
	ModuleBase::TITLE("Exx_LRI","post_process_Eexx");
	const double SPIN_multiple = std::map<int, double>{ {1,2}, {2,1}, {4,1} }.at(PARAM.inp.nspin);				// why?
	const double frac = -SPIN_multiple;
	return frac * Eexx_in;
}

/*
post_process_old
{
	// D
	const std::map<int,double> SPIN_multiple = {{1,0.5}, {2,1}, {4,1}};							// ???
	DR *= SPIN_multiple.at(NSPIN);

	// H
	HR *= -2;

	// E
	const std::map<int,double> SPIN_multiple = {{1,2}, {2,1}, {4,1}};							// ???
	energy *= SPIN_multiple.at(PARAM.inp.nspin);			// ?
	energy /= 2;					// /2 for Ry
}
*/

template<typename Tdata>
void Exx_LRI<Tdata>::cal_exx_force(const int& nat)
{
	ModuleBase::TITLE("Exx_LRI","cal_exx_force");
	ModuleBase::timer::tick("Exx_LRI", "cal_exx_force");

	this->force_exx.create(nat, Ndim);
	for(int is=0; is<PARAM.inp.nspin; ++is)
	{
		this->exx_lri.cal_force({"","",std::to_string(is),"",""});
		for(std::size_t idim=0; idim<Ndim; ++idim) {
			for(const auto &force_item : this->exx_lri.force[idim]) {
				this->force_exx(force_item.first, idim) += std::real(force_item.second);
					} 		}
	}

	const double SPIN_multiple = std::map<int,double>{{1,2}, {2,1}, {4,1}}.at(PARAM.inp.nspin);				// why?
	const double frac = -2 * SPIN_multiple;		// why?
	this->force_exx *= frac;
	ModuleBase::timer::tick("Exx_LRI", "cal_exx_force");
}


template<typename Tdata>
void Exx_LRI<Tdata>::cal_exx_stress(const double& omega, const double& lat0)
{
	ModuleBase::TITLE("Exx_LRI","cal_exx_stress");
	ModuleBase::timer::tick("Exx_LRI", "cal_exx_stress");

	this->stress_exx.create(Ndim, Ndim);
	for(int is=0; is<PARAM.inp.nspin; ++is)
	{
		this->exx_lri.cal_stress({"","",std::to_string(is),"",""});
		for(std::size_t idim0=0; idim0<Ndim; ++idim0) {
			for(std::size_t idim1=0; idim1<Ndim; ++idim1) {
				this->stress_exx(idim0,idim1) += std::real(this->exx_lri.stress(idim0,idim1));
				} 	}
	}

	const double SPIN_multiple = std::map<int,double>{{1,2}, {2,1}, {4,1}}.at(PARAM.inp.nspin);				// why?
	const double frac = 2 * SPIN_multiple / omega * lat0;		// why?
	this->stress_exx *= frac;

	ModuleBase::timer::tick("Exx_LRI", "cal_exx_stress");
}

/*
template<typename Tdata>
std::vector<std::vector<int>> Exx_LRI<Tdata>::get_abfs_nchis() const
{
	std::vector<std::vector<int>> abfs_nchis;
	for (const auto& abfs_T : this->abfs)
	{
		std::vector<int> abfs_nchi_T;
		for (const auto& abfs_L : abfs_T)
			{ abfs_nchi_T.push_back(abfs_L.size()); }
		abfs_nchis.push_back(abfs_nchi_T);
	}
	return abfs_nchis;
}
*/

#endif
