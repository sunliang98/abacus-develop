#include "dos.h"
#include "../src_pw/global.h"
#include "../src_pw/energy.h"
#include "../src_parallel/parallel_reduce.h"
#include "module_elecstate/elecstate.h"

void energy::print_occ(const elecstate::ElecState* pelec)
{
	
		std::stringstream ss;
		ss << GlobalV::global_out_dir << "istate.info" ;
		if(GlobalV::MY_RANK==0)
		{
			std::ofstream ofsi( ss.str().c_str() ); // clear istate.info
			ofsi.close();
		}
#ifdef __MPI
		for(int ip=0; ip<GlobalV::KPAR; ip++)
		{
			MPI_Barrier(MPI_COMM_WORLD);
			if( GlobalV::MY_POOL == ip )
			{
				if( GlobalV::RANK_IN_POOL != 0 || GlobalV::MY_STOGROUP != 0 ) continue;
#endif
				std::ofstream ofsi2( ss.str().c_str(), ios::app );
				if(GlobalV::NSPIN == 1||GlobalV::NSPIN == 4)
				{
					for (int ik = 0;ik < GlobalC::kv.nks;ik++)
					{
						ofsi2<<"BAND"
						<<std::setw(25)<<"Energy(ev)"
						<<std::setw(25)<<"Occupation"
#ifdef __MPI
						<<std::setw(25)<<"Kpoint = "<<GlobalC::Pkpoints.startk_pool[ip]+ik+1
#else
						<<std::setw(25)<<"Kpoint = "<<ik+1
#endif
						<<std::setw(25)<<"("<<GlobalC::kv.kvec_d[ik].x<<" "<<GlobalC::kv.kvec_d[ik].y<<" "<<GlobalC::kv.kvec_d[ik].z<<")"<<std::endl;
						for(int ib=0;ib<GlobalV::NBANDS;ib++)
						{
							ofsi2<<std::setw(6)<<ib+1<<std::setw(25)<<pelec->ekb(ik,ib)* ModuleBase::Ry_to_eV<<std::setw(25)<<pelec->wg(ik,ib)<<std::endl;
						}
						ofsi2 <<std::endl;
						ofsi2 <<std::endl;
					}
				}
				else
				{
					for (int ik = 0;ik < GlobalC::kv.nks/2;ik++)
					{
						ofsi2<<"BAND"
						<<std::setw(25)<<"Spin up Energy(ev)"
						<<std::setw(25)<<"Occupation"
						<<std::setw(25)<<"Spin down Energy(ev)"
						<<std::setw(25)<<"Occupation"
#ifdef __MPI
						<<std::setw(25)<<"Kpoint = "<<GlobalC::Pkpoints.startk_pool[ip]+ik+1
#else
						<<std::setw(25)<<"Kpoint = "<<ik+1
#endif
						<<std::setw(25)<<"("<<GlobalC::kv.kvec_d[ik].x<<" "<<GlobalC::kv.kvec_d[ik].y<<" "<<GlobalC::kv.kvec_d[ik].z<<")"<<std::endl;

						for(int ib=0;ib<GlobalV::NBANDS;ib++)
						{
							ofsi2<<std::setw(6)<<ib+1
							<<std::setw(25)<<pelec->ekb(ik, ib)* ModuleBase::Ry_to_eV
							<<std::setw(25)<<pelec->wg(ik,ib)
							<<std::setw(25)<<pelec->ekb((ik+GlobalC::kv.nks/2), ib)* ModuleBase::Ry_to_eV
							<<std::setw(25)<<pelec->wg(ik+GlobalC::kv.nks/2,ib)<<std::endl;
						}
						ofsi2 <<std::endl;
						ofsi2 <<std::endl;

					}
				}

				ofsi2.close();
#ifdef __MPI
			}
		}
#endif
}

void energy::perform_dos_pw(const elecstate::ElecState* pelec)
{
	ModuleBase::TITLE("energy","perform_dos_pw");

	if(out_dos !=0 || out_band !=0)
    {
        GlobalV::ofs_running << "\n\n\n\n";
        GlobalV::ofs_running << " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
        GlobalV::ofs_running << " |                                                                    |" << std::endl;
        GlobalV::ofs_running << " | Post-processing of data:                                           |" << std::endl;
        GlobalV::ofs_running << " | DOS (density of states) and bands will be output here.             |" << std::endl;
        GlobalV::ofs_running << " | If atomic orbitals are used, Mulliken charge analysis can be done. |" << std::endl;
        GlobalV::ofs_running << " | Also the .bxsf file containing fermi surface information can be    |" << std::endl;
        GlobalV::ofs_running << " | done here.                                                         |" << std::endl;
        GlobalV::ofs_running << " |                                                                    |" << std::endl;
        GlobalV::ofs_running << " <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
        GlobalV::ofs_running << "\n\n\n\n";
    }	
	
	int nspin0=1;
	if(GlobalV::NSPIN==2) nspin0=2;

	if(this->out_dos)
	{
//find energy range
		double emax = pelec->ekb(0, 0);
		double emin = pelec->ekb(0, 0);
		for(int ik=0; ik<GlobalC::kv.nks; ++ik)
		{
			for(int ib=0; ib<GlobalV::NBANDS; ++ib)
			{
				emax = std::max( emax, pelec->ekb(ik, ib) );
				emin = std::min( emin, pelec->ekb(ik, ib) );
			}
		}

#ifdef __MPI
		Parallel_Reduce::gather_max_double_all(emax);
		Parallel_Reduce::gather_min_double_all(emin);
#endif

		emax *= ModuleBase::Ry_to_eV;
		emin *= ModuleBase::Ry_to_eV;

		if(INPUT.dos_setemax)	emax = INPUT.dos_emax_ev;
		if(INPUT.dos_setemin)	emin = INPUT.dos_emin_ev;
		if(!INPUT.dos_setemax && !INPUT.dos_setemin)
		{
			//scale up a little bit so the end peaks are displaced better
			double delta=(emax-emin)*dos_scale;
			emax=emax+delta/2.0;
			emin=emin-delta/2.0;
		}

		ModuleBase::GlobalFunc::OUT(GlobalV::ofs_running,"minimal energy is (eV)", emin);
		ModuleBase::GlobalFunc::OUT(GlobalV::ofs_running,"maximal energy is (eV)", emax);
		// 		atom_arrange::set_sr_NL();
		//		atom_arrange::search( GlobalV::SEARCH_RADIUS );//qifeng-2019-01-21
		
//determine #. energy points	
		const double de_ev = this->dos_edelta_ev;
		//std::cout << de_ev;

		const int npoints = static_cast<int>(std::floor ( ( emax - emin ) / de_ev ));
		const int np=npoints;

	 	for(int is=0; is<nspin0; ++is)
	 	{
//DOS_ispin contains not smoothed dos
			 std::stringstream ss;
			 ss << GlobalV::global_out_dir << "DOS" << is+1;
			 std::stringstream ss1;
			 ss1 << GlobalV::global_out_dir << "DOS" << is+1 << "_smearing.dat";

			 Dos::calculate_dos(
					 is,
					 GlobalC::kv.isk,
					 ss.str(),
					 ss1.str(), 
					 this->dos_edelta_ev, 
					 emax, 
					 emin, 
					 GlobalC::kv.nks, GlobalC::kv.nkstot, GlobalC::kv.wk, pelec->wg, GlobalV::NBANDS, pelec->ekb );
	 	}

	}//out_dos=1
	if(this->out_band) //pengfei 2014-10-13
	{
		int nks=0;
		if(nspin0==1) 
		{
			nks = GlobalC::kv.nkstot;
		}
		else if(nspin0==2) 
		{
			nks = GlobalC::kv.nkstot/2;
		}

		for(int is=0; is<nspin0; is++)
		{
			std::stringstream ss2;
			ss2 << GlobalV::global_out_dir << "BANDS_" << is+1 << ".dat";
			GlobalV::ofs_running << "\n Output bands in file: " << ss2.str() << std::endl;
			Dos::nscf_band(is, ss2.str(), nks, GlobalV::NBANDS, this->ef*0, pelec->ekb);
		}

	}
}

