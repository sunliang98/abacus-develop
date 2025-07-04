#include "read_pseudo.h"
#include "module_parameter/parameter.h"
#include "source_base/global_file.h"
#include "source_cell/read_pp.h"
#include "source_cell/bcast_cell.h"
#include "source_base/element_elec_config.h"
#include "source_base/parallel_common.h"

#include <cstring> // Peize Lin fix bug about strcmp 2016-08-02

namespace elecstate {
void read_pseudo(std::ofstream& ofs, UnitCell& ucell) {
    // read in non-local pseudopotential and ouput the projectors.
    ofs << "\n\n";
    ofs << " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
    ofs << " |                                                                    |" << std::endl;
    ofs << " |                 #Read Pseudopotentials Files#                      |" << std::endl;
    ofs << " | ABACUS supports norm-conserving (NC) pseudopotentials for both     |" << std::endl;
    ofs << " | plane wave basis and numerical atomic orbital basis sets.          |" << std::endl;
    ofs << " | In addition, ABACUS supports ultrasoft pseudopotentials (USPP)     |" << std::endl;
    ofs << " | for plane wave basis set.                                          |" << std::endl;
    ofs << " |                                                                    |" << std::endl;
    ofs << " <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
    ofs << "\n";

    read_cell_pseudopots(PARAM.inp.pseudo_dir, ofs, ucell);

	if (GlobalV::MY_RANK == 0) 
	{
		for (int it = 0; it < ucell.ntype; it++) 
		{
			Atom* atom = &ucell.atoms[it];
			if (!(atom->label_orb.empty())) 
			{
                ucell.compare_atom_labels(atom->label_orb, atom->ncpp.psd);
            }
        }

		if (PARAM.inp.out_element_info) 
		{
			for (int i = 0; i < ucell.ntype; i++) 
			{
				ModuleBase::Global_File::make_dir_atom(ucell.atoms[i].label);
            }
			for (int it = 0; it < ucell.ntype; it++) 
			{
				Atom* atom = &ucell.atoms[it];
                std::stringstream ss;
                ss << PARAM.globalv.global_out_dir << atom->label << "/"
                   << atom->label << ".NONLOCAL";
                std::ofstream ofs(ss.str().c_str());

                ofs << "<HEADER>" << std::endl;
                ofs << std::setw(10) << atom->label << "\t"
                    << "label" << std::endl;
                ofs << std::setw(10) << atom->ncpp.pp_type << "\t"
                    << "Pseudopotential type" << std::endl;
                ofs << std::setw(10) << atom->ncpp.lmax << "\t"
                    << "lmax" << std::endl;
                ofs << "</HEADER>" << std::endl;

                ofs << "\n<DIJ>" << std::endl;
                ofs << std::setw(10) << atom->ncpp.nbeta << "\t"
                    << "nummber of projectors." << std::endl;
                for (int ib = 0; ib < atom->ncpp.nbeta; ib++) {
                    for (int ib2 = 0; ib2 < atom->ncpp.nbeta; ib2++) {
                        ofs << std::setw(10) << atom->ncpp.lll[ib] << " "
                            << atom->ncpp.lll[ib2] << " "
                            << atom->ncpp.dion(ib, ib2) << std::endl;
                    }
                }
                ofs << "</DIJ>" << std::endl;

                for (int i = 0; i < atom->ncpp.nbeta; i++) {
                    ofs << "<PP_BETA>" << std::endl;
                    ofs << std::setw(10) << i << "\t"
                        << "the index of projectors." << std::endl;
                    ofs << std::setw(10) << atom->ncpp.lll[i] << "\t"
                        << "the angular momentum." << std::endl;

                    // mohan add
                    // only keep the nonzero part.
                    int cut_mesh = atom->ncpp.mesh;
                    for (int j = atom->ncpp.mesh - 1; j >= 0; --j) {
                        if (std::abs(atom->ncpp.betar(i, j)) > 1.0e-10) {
                            cut_mesh = j;
                            break;
                        }
                    }
                    if (cut_mesh % 2 == 0) {
                        ++cut_mesh;
                    }

                    ofs << std::setw(10) << cut_mesh << "\t"
                        << "the number of mesh points." << std::endl;

                    for (int j = 0; j < cut_mesh; ++j) {
                        ofs << std::setw(15) << atom->ncpp.r[j] << std::setw(15)
                            << atom->ncpp.betar(i, j) << std::setw(15)
                            << atom->ncpp.rab[j] << std::endl;
                    }
                    ofs << "</PP_BETA>" << std::endl;
                }

                ofs.close();
            }
        }
    }

#ifdef __MPI
    unitcell::bcast_atoms_pseudo(ucell.atoms,ucell.ntype);
#endif

    for (int it = 0; it < ucell.ntype; it++) {
        if (ucell.atoms[0].ncpp.xc_func != ucell.atoms[it].ncpp.xc_func) {
            GlobalV::ofs_warning << "\n type " << ucell.atoms[0].label
                                 << " functional is " << ucell.atoms[0].ncpp.xc_func;

            GlobalV::ofs_warning << "\n type " << ucell.atoms[it].label
                                 << " functional is " << ucell.atoms[it].ncpp.xc_func
                                 << std::endl;

            ModuleBase::WARNING_QUIT("setup_cell",
                                     "All DFT functional must consistent.");
        }
    }

    // setup the total number of PAOs
    cal_natomwfc(ofs,ucell.natomwfc,ucell.ntype,ucell.atoms);

    // Calculate the information of atoms from the pseudopotential to set PARAM
    CalAtomsInfo ca;
    ca.cal_atoms_info(ucell.atoms, ucell.ntype, PARAM);

    // setup PARAM.globalv.nlocal
    cal_nwfc(ofs,ucell,ucell.atoms);

    // Check whether the number of valence is minimum
	if (GlobalV::MY_RANK == 0) 
	{
		int abtype = 0;
		for (int it = 0; it < ucell.ntype; it++) 
		{
			if (ModuleBase::MinZval.find(ucell.atoms[it].ncpp.psd)
					!= ModuleBase::MinZval.end()) 
			{
				if (ucell.atoms[it].ncpp.zv
						> ModuleBase::MinZval.at(ucell.atoms[it].ncpp.psd)) 
				{
                    abtype += 1;
					if (abtype == 1) 
					{
                        std::cout << "\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
                                     "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
                                     "%%%%%%%%%%%%%%%%%%%%%%%%%%"
                                  << std::endl;
                        ofs << "\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
                               "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
                               "%%%%%%%%%%%%%%%%%%%%%"
                            << std::endl;
                    }
                    std::cout << " Warning: the number of valence electrons in "
                                 "pseudopotential > "
                              << ModuleBase::MinZval.at(ucell.atoms[it].ncpp.psd);
                    std::cout << " for " << ucell.atoms[it].ncpp.psd << ": "
                              << ModuleBase::EleConfig.at(ucell.atoms[it].ncpp.psd)
                              << std::endl;
                    ofs << " Warning: the number of valence electrons in "
                           "pseudopotential > "
                        << ModuleBase::MinZval.at(ucell.atoms[it].ncpp.psd);
                    ofs << " for " << ucell.atoms[it].ncpp.psd << ": "
                        << ModuleBase::EleConfig.at(ucell.atoms[it].ncpp.psd)
                        << std::endl;
                }
            }
        }
		if (abtype > 0) 
		{
			std::cout << " Pseudopotentials with additional electrons can "
                         "yield (more) accurate outcomes, but may be "
                         "less efficient."
                      << std::endl;
            std::cout
                << " If you're confident that your chosen pseudopotential is "
                   "appropriate, you can safely ignore "
                   "this warning."
                << std::endl;
            std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
                         "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
                         "%%%%%%%%%%%%\n"
                      << std::endl;
            ofs << " Pseudopotentials with additional electrons can yield "
                   "(more) accurate outcomes, but may be less "
                   "efficient."
                << std::endl;
            ofs << " If you're confident that your chosen pseudopotential is "
                   "appropriate, you can safely ignore this "
                   "warning."
                << std::endl;
            ofs << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
                   "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
                   "%%%%%%%";
            ModuleBase::GlobalFunc::OUT(ofs, "");
        }
    }

    cal_meshx(ucell.meshx,ucell.atoms,ucell.ntype);

#ifdef __MPI
    Parallel_Common::bcast_int(ucell.meshx);
    Parallel_Common::bcast_int(ucell.natomwfc);
    Parallel_Common::bcast_int(ucell.lmax);
    Parallel_Common::bcast_int(ucell.lmax_ppwf);
#endif
}

//==========================================================
// Read pseudopotential according to the dir
//==========================================================
void read_cell_pseudopots(const std::string& pp_dir, std::ofstream& log, UnitCell& ucell)
{
    ModuleBase::TITLE("Elecstate", "read_cell_pseudopots");
    // setup reading log for pseudopot_upf
    std::stringstream ss;
    ss << PARAM.globalv.global_out_dir << "atom_pseudo.log";

    // Read in the atomic pseudo potentials
    std::string pp_address;
    for (int i = 0; i < ucell.ntype; i++)
    {
        Pseudopot_upf upf;
        upf.coulomb_potential = ucell.atoms[i].coulomb_potential;

        // mohan update 2010-09-12
        int error = 0;
        int error_ap = 0;

        if (GlobalV::MY_RANK == 0)
        {
            pp_address = pp_dir + ucell.pseudo_fn[i];
            error = upf.init_pseudo_reader(pp_address, ucell.pseudo_type[i], ucell.atoms[i].ncpp); // xiaohui add 2013-06-23

            if (error == 0) // mohan add 2021-04-16
            {
                if (ucell.atoms[i].flag_empty_element) // Peize Lin add for bsse 2021.04.07
                {
                    upf.set_empty_element(ucell.atoms[i].ncpp);
                }
                upf.set_upf_q(ucell.atoms[i].ncpp); // liuyu add 2023-09-21
                // average pseudopotential if needed
                error_ap = upf.average_p(PARAM.inp.soc_lambda, ucell.atoms[i].ncpp); // added by zhengdy 2020-10-20
            }
            ucell.atoms[i].coulomb_potential = upf.coulomb_potential;
        }

#ifdef __MPI
        Parallel_Common::bcast_int(error);
        Parallel_Common::bcast_int(error_ap);
        Parallel_Common::bcast_bool(ucell.atoms[i].coulomb_potential);
#endif

        if (error_ap)
        {
            ModuleBase::WARNING_QUIT("read_cell_pseudopots", "error when average the pseudopotential.");
        }

        if (error == 1)
        {
            std::cout << " Pseudopotential directory now is : " << pp_address << std::endl;
            GlobalV::ofs_warning << " Pseudopotential directory now is : " << pp_address << std::endl;
            ModuleBase::WARNING_QUIT("read_cell_pseudopots", "Couldn't find pseudopotential file.");
        }
        else if (error == 2)
        {
            ModuleBase::WARNING_QUIT("read_cell_pseudopots", "Pseudopotential data do not match.");
        }
        else if (error == 3)
        {
            ModuleBase::WARNING_QUIT(
                "read_cell_pseudopots",
                "Check the reference states in pseudopotential .vwr file.\n Also the norm of the read in pseudo wave "
                "functions\n explicitly please check S, P and D channels.\n If the norm of the wave function is \n "
                "unreasonable large (should be near 1.0), ABACUS would quit. \n The solution is to turn off the wave "
                "functions  \n and the corresponding non-local projectors together\n in .vwr pseudopotential file.");
        }
        else if (error == 4)
        {
            ModuleBase::WARNING_QUIT("read_cell_pseudopots", "Unknown pseudopotential type.");
        }

        if (GlobalV::MY_RANK == 0)
        {
		    upf.complete_default(ucell.atoms[i].ncpp);

            ModuleBase::GlobalFunc::OUT(log, "Pseudopotential file", ucell.pseudo_fn[i]);
            ModuleBase::GlobalFunc::OUT(log, "Pseudopotential type", ucell.atoms[i].ncpp.pp_type);
            ModuleBase::GlobalFunc::OUT(log, "Exchange-correlation functional", ucell.atoms[i].ncpp.xc_func);
            ModuleBase::GlobalFunc::OUT(log, "Nonlocal core correction", ucell.atoms[i].ncpp.nlcc);
            // ModuleBase::GlobalFunc::OUT(log, "spin orbital", ucell.atoms[i].has_so);
            ModuleBase::GlobalFunc::OUT(log, "Valence electrons", ucell.atoms[i].ncpp.zv);
            ModuleBase::GlobalFunc::OUT(log, "Lmax", ucell.atoms[i].ncpp.lmax);
            ModuleBase::GlobalFunc::OUT(log, "Number of zeta", ucell.atoms[i].ncpp.nchi);
            ModuleBase::GlobalFunc::OUT(log, "Number of projectors", ucell.atoms[i].ncpp.nbeta);
            for (int ib = 0; ib < ucell.atoms[i].ncpp.nbeta; ib++)
            {
                ModuleBase::GlobalFunc::OUT(log, "L of projector", ucell.atoms[i].ncpp.lll[ib]);
            }
            //			ModuleBase::GlobalFunc::OUT(log,"Grid Mesh Number", atoms[i].mesh);
            if (PARAM.inp.dft_functional != "default")
            {
                std::string xc_func1 = PARAM.inp.dft_functional;
                transform(xc_func1.begin(), xc_func1.end(), xc_func1.begin(), (::toupper));
                if (xc_func1 != ucell.atoms[i].ncpp.xc_func)
                {
                    std::cout << " NAME OF ELEMENT      : " << ucell.atoms[i].label << std::endl;
                    std::cout << " DFT FUNC. (PSEUDO)   : " << ucell.atoms[i].ncpp.xc_func << std::endl;
                    std::cout << " DFT FUNC. (SET TO)   : " << xc_func1 << std::endl; 
                    std::cout << " MAKE SURE THIS DFT FUNCTIONAL IS WHAT YOU NEED" << std::endl;


                    GlobalV::ofs_warning << " NAME OF ELEMENT      : " << ucell.atoms[i].label << std::endl;
                    GlobalV::ofs_warning << " DFT FUNC. (PSEUDO)   : " << ucell.atoms[i].ncpp.xc_func << std::endl;
                    GlobalV::ofs_warning << " DFT FUNC. (SET TO)   : " << xc_func1 << std::endl; 
                    GlobalV::ofs_warning << " MAKE SURE THIS DFT FUNCTIONAL IS WHAT YOU NEED" << std::endl;

                    ucell.atoms[i].ncpp.xc_func = xc_func1;
                    ModuleBase::GlobalFunc::OUT(log, "DFT functional set to", xc_func1);
                }
            }
        }
    }
    return;
}

void print_unitcell_pseudo(const std::string& fn, UnitCell& ucell)
{
    ModuleBase::TITLE("elecstate", "print_unitcell_pseudo");
    std::ofstream ofs(fn.c_str());

    ucell.print_cell(ofs);
    for (int i = 0; i < ucell.ntype; i++)
    {
        ucell.atoms[i].print_Atom(ofs);
    }

    ofs.close();
    return;
}

}
