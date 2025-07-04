#include "write_proj_band_lcao.h"

#include "module_parameter/parameter.h"
#include "source_base/global_function.h"
#include "source_base/global_variable.h"
#include "source_base/scalapack_connector.h"
#include "source_base/timer.h"
#include "source_cell/module_neighbor/sltk_atom_arrange.h"
#include "write_orb_info.h"
#include "source_lcao/hamilt_lcaodft/hamilt_lcao.h"

template<>
void ModuleIO::write_proj_band_lcao(
    const psi::Psi<double>* psi,
    const Parallel_Orbitals &pv,
    const elecstate::ElecState* pelec,
    const K_Vectors& kv,
    const UnitCell &ucell,
    hamilt::Hamilt<double>* p_ham)
{
    ModuleBase::TITLE("ModuleIO", "write_proj_band_lcao");
    ModuleBase::timer::tick("ModuleIO", "write_proj_band_lcao");

    GlobalV::ofs_running << "\n";
    GlobalV::ofs_running << " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
    GlobalV::ofs_running << " |                                                                    |" << std::endl;
    GlobalV::ofs_running << " |          #Print out projected bands (psi in double)#               |" << std::endl;
    GlobalV::ofs_running << " |                                                                    |" << std::endl;
    GlobalV::ofs_running << " <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
    GlobalV::ofs_running << "\n";

    // get the date pointer of SK
    const double* sk = dynamic_cast<const hamilt::HamiltLCAO<double, double>*>(p_ham)->getSk();

    int nspin0 = 1;
	if (PARAM.inp.nspin == 2) 
	{
		nspin0 = 2;
	}

	int nks = 0;
    if (nspin0 == 1)
    {
        nks = kv.get_nkstot();
    }
    else if (nspin0 == 2)
    {
        nks = kv.get_nkstot() / 2;
    }

    ModuleBase::ComplexMatrix weightk;
    ModuleBase::matrix weight;
    int NUM = PARAM.globalv.nlocal * PARAM.inp.nbands * nspin0;
    weightk.create(nspin0, PARAM.inp.nbands * PARAM.globalv.nlocal, true);
    weight.create(nspin0, PARAM.inp.nbands * PARAM.globalv.nlocal, true);


    for (int is = 0; is < nspin0; is++)
    {
        std::vector<ModuleBase::matrix> Mulk;
        Mulk.resize(1);
        Mulk[0].create(pv.ncol, pv.nrow);

        psi->fix_k(is);
        for (int i = 0; i < PARAM.inp.nbands; ++i)
        {
            const int NB = i + 1;

            const double one_float = 1.0, zero_float = 0.0;
            const int one_int = 1;
#ifdef __MPI
            const char T_char = 'T';
            pdgemv_(&T_char,
                &PARAM.globalv.nlocal,
                &PARAM.globalv.nlocal,
                &one_float,
                sk,
                &one_int,
                &one_int,
                pv.desc,
                psi->get_pointer(),
                &one_int,
                &NB,
                pv.desc,
                &one_int,
                &zero_float,
                Mulk[0].c,
                &one_int,
                &NB,
                pv.desc,
                &one_int);
#endif
            for (int j = 0; j < PARAM.globalv.nlocal; ++j)
            {

                if (pv.in_this_processor(j, i))
                {
                    const int ir = pv.global2local_row(j);
                    const int ic = pv.global2local_col(i);
                    weightk(is, i * PARAM.globalv.nlocal + j) = Mulk[0](ic, ir) * psi[0](ic, ir);
                }
            }
        } // ib
#ifdef __MPI
        MPI_Reduce(weightk.real().c, weight.c, NUM, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#endif

        if (GlobalV::MY_RANK == 0)
        {
            std::stringstream ps2;
            ps2 << PARAM.globalv.global_out_dir << "pbands" << is + 1 << ".xml";
            GlobalV::ofs_running << " Output projected bands are in file: " << ps2.str() << std::endl;
            std::ofstream out(ps2.str().c_str());

            out << "<pband>" << std::endl;
            out << "<nspin>" << PARAM.inp.nspin << "</nspin>" << std::endl;
            if (PARAM.inp.nspin == 4) 
			{
				out << "<norbitals>" << std::setw(2) << PARAM.globalv.nlocal / 2 << "</norbitals>" << std::endl;
			} 
			else 
			{
				out << "<norbitals>" << std::setw(2) << PARAM.globalv.nlocal << "</norbitals>" << std::endl;
			}
            out << "<band_structure nkpoints=\"" << nks << "\" nbands=\"" << PARAM.inp.nbands << "\" units=\"eV\">"
                << std::endl;
			for (int ib = 0; ib < PARAM.inp.nbands; ib++) 
			{
				out << " " << (pelec->ekb(is * nks, ib)) * ModuleBase::Ry_to_eV;
			}
			out << std::endl;
            out << "</band_structure>" << std::endl;

            for (int i = 0; i < ucell.nat; i++)
            {
                int a = ucell.iat2ia[i];
                int t = ucell.iat2it[i];
                Atom* atom1 = &ucell.atoms[t];
                const int s0 = ucell.itiaiw2iwt(t, a, 0);
                for (int j = 0; j < atom1->nw; ++j)
                {
                    const int L1 = atom1->iw2l[j];
                    const int N1 = atom1->iw2n[j];
                    const int m1 = atom1->iw2m[j];
                    const int w = ucell.itiaiw2iwt(t, a, j);

                    // out << "</energy_values>" <<std::endl;
                    out << "<orbital" << std::endl;
                    out << std::setw(6) << "index=\"" << std::setw(40) << w + 1 << "\"" << std::endl;
                    out << std::setw(5) << "atom_index=\"" << std::setw(40) << i + 1 << "\"" << std::endl;
                    out << std::setw(8) << "species=\"" << ucell.atoms[t].label << "\"" << std::endl;
                    out << std::setw(2) << "l=\"" << std::setw(40) << L1 << "\"" << std::endl;
                    out << std::setw(2) << "m=\"" << std::setw(40) << m1 << "\"" << std::endl;
                    out << std::setw(2) << "z=\"" << std::setw(40) << N1 + 1 << "\"" << std::endl;
                    out << ">" << std::endl;
                    out << "<data>" << std::endl;
                    for (int ib = 0; ib < PARAM.inp.nbands; ib++)
                    {
                        if (PARAM.inp.nspin == 1 || PARAM.inp.nspin == 2) {
                            out << std::setw(13) << weight(is, ib * PARAM.globalv.nlocal + w);
                        } else if (PARAM.inp.nspin == 4)
                        {
                            int w0 = w - s0;
                            out << std::setw(13)
                                << weight(is, ib * PARAM.globalv.nlocal + s0 + 2 * w0)
                                + weight(is, ib * PARAM.globalv.nlocal + s0 + 2 * w0 + 1);
                        }
                    }
                    out << std::endl;
                    out << "</data>" << std::endl;
                    out << "</orbital>" << std::endl;
                } // j
            } // i

            out << "</pband>" << std::endl;
            out.close();
        }
    } // is
    ModuleIO::write_orb_info(&ucell);

    ModuleBase::timer::tick("ModuleIO", "write_proj_band_lcao");
    return;
}

template<>
void ModuleIO::write_proj_band_lcao(
    const psi::Psi<std::complex<double>>* psi,
    const Parallel_Orbitals &pv,
    const elecstate::ElecState* pelec,
    const K_Vectors& kv,
    const UnitCell& ucell,
    hamilt::Hamilt<std::complex<double>>* p_ham)
{
    ModuleBase::TITLE("ModuleIO", "write_proj_band_lcao");
    ModuleBase::timer::tick("ModuleIO", "write_proj_band_lcao");

    GlobalV::ofs_running << "\n";
    GlobalV::ofs_running << " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
    GlobalV::ofs_running << " |                                                                    |" << std::endl;
    GlobalV::ofs_running << " |       #Print out projected bands (psi in complex<double>)#         |" << std::endl;
    GlobalV::ofs_running << " |                                                                    |" << std::endl;
    GlobalV::ofs_running << " <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
    GlobalV::ofs_running << "\n";

    int nspin0 = 1;
	if (PARAM.inp.nspin == 2) 
	{
		nspin0 = 2;
	}
	int nks = 0;
    if (nspin0 == 1)
    {
        nks = kv.get_nkstot();
    }
    else if (nspin0 == 2)
    {
        nks = kv.get_nkstot() / 2;
    }

    ModuleBase::ComplexMatrix weightk;
    ModuleBase::matrix weight;
    int NUM = PARAM.globalv.nlocal * PARAM.inp.nbands * kv.get_nks();
    weightk.create(kv.get_nks(), PARAM.inp.nbands * PARAM.globalv.nlocal, true);
    weight.create(kv.get_nks(), PARAM.inp.nbands * PARAM.globalv.nlocal, true);

    for (int is = 0; is < nspin0; is++)
    {
            std::vector<ModuleBase::ComplexMatrix> Mulk;
            Mulk.resize(1);
            Mulk[0].create(pv.ncol, pv.nrow);

            for (int ik = 0; ik < kv.get_nks(); ik++)
            {
                if (is == kv.isk[ik])
                {
                    // calculate SK for current k point
                    const std::complex<double>* sk = nullptr;
                    if (PARAM.inp.nspin == 4)
                    {
                        dynamic_cast<hamilt::HamiltLCAO<std::complex<double>, std::complex<double>>*>(p_ham)->updateSk(ik, 1);
                        sk = dynamic_cast<const hamilt::HamiltLCAO<std::complex<double>, std::complex<double>>*>(p_ham)->getSk();
                    }
                    else
                    {
                        dynamic_cast<hamilt::HamiltLCAO<std::complex<double>, double>*>(p_ham)->updateSk(ik, 1);
                        sk = dynamic_cast<const hamilt::HamiltLCAO<std::complex<double>, double>*>(p_ham)->getSk();
                    }

                    // calculate Mulk
                    psi->fix_k(ik);
                    psi::Psi<std::complex<double>> Dwfc(1, 
                                                        psi->get_nbands(),
                                                        psi->get_nbasis(),
                                                        psi->get_nbasis(),
                                                        true);

                    std::complex<double>* p_dwfc = Dwfc.get_pointer();
                    for (int index = 0; index < Dwfc.size(); ++index)
                    {
                        p_dwfc[index] = conj(psi->get_pointer()[index]);
                    }

                    for (int i = 0; i < PARAM.inp.nbands; ++i)
                    {
                        const int NB = i + 1;

                        const double one_float[2] = { 1.0, 0.0 };
                        const double zero_float[2] = { 0.0, 0.0 };
                        const int one_int = 1;
                        const char T_char = 'T';
#ifdef __MPI
                        pzgemv_(&T_char,
                            &PARAM.globalv.nlocal,
                            &PARAM.globalv.nlocal,
                            &one_float[0],
                            sk,
                            &one_int,
                            &one_int,
                            pv.desc,
                            p_dwfc,
                            &one_int,
                            &NB,
                            pv.desc,
                            &one_int,
                            &zero_float[0],
                            Mulk[0].c,
                            &one_int,
                            &NB,
                            pv.desc,
                            &one_int);
#endif
                        for (int j = 0; j < PARAM.globalv.nlocal; ++j)
                        {

                            if (pv.in_this_processor(j, i))
                            {

                                const int ir = pv.global2local_row(j);
                                const int ic = pv.global2local_col(i);

                                weightk(ik, i * PARAM.globalv.nlocal + j) = Mulk[0](ic, ir) * psi[0](ic, ir);
                            }
                        }

                    } // ib

                } // if
            } // ik
#ifdef __MPI
        MPI_Reduce(weightk.real().c, weight.c, NUM, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#endif

        if (GlobalV::MY_RANK == 0)
        {
            std::stringstream ps2;
            ps2 << PARAM.globalv.global_out_dir << "pbands" << is + 1 << ".xml";
            GlobalV::ofs_running << " Output projected bands in file: " << ps2.str() << std::endl;
            std::ofstream out(ps2.str().c_str());

            out << "<pband>" << std::endl;
            out << "<nspin>" << PARAM.inp.nspin << "</nspin>" << std::endl;
            if (PARAM.inp.nspin == 4)
            {
                out << "<norbitals>" << std::setw(2) << PARAM.globalv.nlocal / 2 << "</norbitals>" << std::endl;
            }
            else
            {
                out << "<norbitals>" << std::setw(2) << PARAM.globalv.nlocal << "</norbitals>" << std::endl;
            }

            out << "<band_structure nkpoints=\"" << nks << "\" nbands=\"" << PARAM.inp.nbands << "\" units=\"eV\">"
                << std::endl;

			for (int ik = 0; ik < nks; ik++)
			{
				for (int ib = 0; ib < PARAM.inp.nbands; ib++) {
					out << " " << (pelec->ekb(ik + is * nks, ib)) * ModuleBase::Ry_to_eV;
}
				out << std::endl;
			}
            out << "</band_structure>" << std::endl;

            for (int i = 0; i < ucell.nat; i++)
            {
                int a = ucell.iat2ia[i];
                int t = ucell.iat2it[i];
                Atom* atom1 = &ucell.atoms[t];
                const int s0 = ucell.itiaiw2iwt(t, a, 0);
                for (int j = 0; j < atom1->nw; ++j)
                {
                    const int L1 = atom1->iw2l[j];
                    const int N1 = atom1->iw2n[j];
                    const int m1 = atom1->iw2m[j];
                    const int w = ucell.itiaiw2iwt(t, a, j);

                    // out << "</energy_values>" <<std::endl;
                    out << "<orbital" << std::endl;
                    out << std::setw(6) << "index=\"" << std::setw(40) << w + 1 << "\"" << std::endl;
                    out << std::setw(5) << "atom_index=\"" << std::setw(40) << i + 1 << "\"" << std::endl;
                    out << std::setw(8) << "species=\"" << ucell.atoms[t].label << "\"" << std::endl;
                    out << std::setw(2) << "l=\"" << std::setw(40) << L1 << "\"" << std::endl;
                    out << std::setw(2) << "m=\"" << std::setw(40) << m1 << "\"" << std::endl;
                    out << std::setw(2) << "z=\"" << std::setw(40) << N1 + 1 << "\"" << std::endl;
                    out << ">" << std::endl;
                    out << "<data>" << std::endl;
					for (int ik = 0; ik < nks; ik++)
					{
						for (int ib = 0; ib < PARAM.inp.nbands; ib++)
						{
							if (PARAM.inp.nspin == 1)
							{
								out << std::setw(13) << weight(ik, ib * PARAM.globalv.nlocal + w);
							}
							else if (PARAM.inp.nspin == 2)
							{
								out << std::setw(13) << weight(ik + nks * is, ib * PARAM.globalv.nlocal + w);
							}
							else if (PARAM.inp.nspin == 4)
							{
								int w0 = w - s0;
								out << std::setw(13)
									<< weight(ik, ib * PARAM.globalv.nlocal + s0 + 2 * w0)
									+ weight(ik, ib * PARAM.globalv.nlocal + s0 + 2 * w0 + 1);
							}
						}
						out << std::endl;
					}
                    out << "</data>" << std::endl;
                    out << "</orbital>" << std::endl;
                } // j
            } // i

            out << "</pband>" << std::endl;
            out.close();
        }
    } // is
    ModuleIO::write_orb_info(&ucell);

    ModuleBase::timer::tick("ModuleIO", "write_proj_band_lcao");
    return;
}
