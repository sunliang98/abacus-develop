#include "nscf_band.h"
#include "module_base/global_function.h"
#include "module_base/global_variable.h"
#include "module_base/timer.h"
#include "module_base/tool_title.h"
#include "module_base/formatter_physfmt.h"

void ModuleIO::nscf_band(
	const int &is,
	const std::string &out_band_dir, 
	const int &nband,
	const double &fermie,
	const int &precision,
	const ModuleBase::matrix& ekb,
	const K_Vectors& kv,
	const Parallel_Kpoints* Pkpoints)
{
	ModuleBase::TITLE("ModuleIO","nscf_band");
	ModuleBase::timer::tick("ModuleIO", "nscf_band");
	// number of k points without spin; nspin = 1,2, nkstot = nkstot_np * nspin; 
	//                                  nspin = 4, nkstot = nkstot_np
	const int nkstot_np = Pkpoints->nkstot_np;
	const int nks_np = Pkpoints->nks_np;

#ifdef __MPI
	if(GlobalV::MY_RANK==0)
	{
		std::ofstream ofs(out_band_dir.c_str());//make the file clear!!
		ofs.close();
	}
	MPI_Barrier(MPI_COMM_WORLD);
	std::vector<double> klength;
	klength.resize(nkstot_np);
	klength[0] = 0.0;
	std::vector<ModuleBase::Vector3<double>> kvec_c_global;
	Pkpoints->gatherkvec(kv.kvec_c, kvec_c_global);
	for(int ik=0; ik<nkstot_np; ik++)
	{
		if (ik>0)
		{
			auto delta=kvec_c_global[ik]-kvec_c_global[ik-1];
			klength[ik] = klength[ik-1];
			klength[ik] += (kv.kl_segids[ik] == kv.kl_segids[ik-1]) ? delta.norm() : 0.0;
		}
		/* first find if present kpoint in present pool */
		if ( GlobalV::MY_POOL == Pkpoints->whichpool[ik] )
		{
			/* then get the local kpoint index, which starts definitly from 0 */
			const int ik_now = ik - Pkpoints->startk_pool[GlobalV::MY_POOL];
			/* if present kpoint corresponds the spin of the present one */
			assert( kv.isk[ik_now+is*nks_np] == is );
			if ( GlobalV::RANK_IN_POOL == 0)
			{
				formatter::PhysicalFmt physfmt; // create a physical formatter temporarily
				std::ofstream ofs(out_band_dir.c_str(), std::ios::app);
				physfmt.adjust_formatter_flexible(4, 0, false); // for integer
				ofs << physfmt.get_p_formatter()->format(ik+1);
				physfmt.adjust_formatter_flexible(precision, 4.0/double(precision), false); // for decimal
				ofs << physfmt.get_p_formatter()->format(klength[ik]);
				for(int ib = 0; ib < nband; ib++)
				{
					ofs << physfmt.get_p_formatter()->format((ekb(ik_now+is*nks_np, ib)-fermie) * ModuleBase::Ry_to_eV);
				}
				ofs << std::endl;
				ofs.close();	
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
	
	// old version
	/*
	for(int ip=0;ip<GlobalV::KPAR;ip++)
	{
		if(GlobalV::MY_POOL == ip && GlobalV::RANK_IN_POOL == 0)
		{
			std::ofstream ofs(out_band_dir.c_str(),ios::app);
			for(int ik=0;ik<nkstot_np;ik++)
			{
				ofs<<std::setw(12)<<ik;
				for(int ib = 0; ib < nband; ib++)
				{
					ofs <<std::setw(12)<< ekb[ik][ib] * ModuleBase::Ry_to_eV;
				}
				ofs<<std::endl;
			}
			ofs.close();
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
	*/
#else
//	std::cout<<"\n nband = "<<nband<<std::endl;
//	std::cout<<out_band_dir<<std::endl;
	formatter::PhysicalFmt physfmt; // create a physical formatter temporarily
	std::vector<double> klength;
	klength.resize(nkstot_np);
	klength[0] = 0.0;
	std::ofstream ofs(out_band_dir.c_str());
	for(int ik=0;ik<nkstot_np;ik++)
	{
		if (ik>0)
		{
			auto delta=kv.kvec_c[ik]-kv.kvec_c[ik-1];
			klength[ik] = klength[ik-1];
			klength[ik] += (kv.kl_segids[ik] == kv.kl_segids[ik-1]) ? delta.norm() : 0.0;
		}
		if( kv.isk[ik] == is)
		{
			physfmt.adjust_formatter_flexible(4, 0, false); // for integer
			ofs << physfmt.get_p_formatter()->format(ik+1);
			physfmt.adjust_formatter_flexible(precision, 4.0/double(precision), false); // for decimal
			ofs << physfmt.get_p_formatter()->format(klength[ik]); // add klength, in accordance with the MPI version
			for(int ibnd = 0; ibnd < nband; ibnd++)
			{
				ofs << physfmt.get_p_formatter()->format((ekb(ik, ibnd)-fermie) * ModuleBase::Ry_to_eV);
			}
			ofs << std::endl;
		}
	}
	ofs.close();
#endif

	ModuleBase::timer::tick("ModuleIO", "nscf_band");
	return;
}
