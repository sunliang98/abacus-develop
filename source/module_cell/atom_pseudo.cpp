#include "atom_pseudo.h"
#include "module_parameter/parameter.h"

#include "module_parameter/parameter.h"
Atom_pseudo::Atom_pseudo()
{
}

Atom_pseudo::~Atom_pseudo()
{
}

// mohan add 2021-05-07
void Atom_pseudo::set_d_so(ModuleBase::ComplexMatrix& d_so_in,
                           const int& nproj_in,
                           const int& nproj_in_so,
                           const bool has_so)
{
    if (this->lmax < -1 || this->lmax > 20)
    {
        ModuleBase::WARNING_QUIT("Numerical_Nonlocal", "bad input of lmax : should be between -1 and 20");
    }

    this->nproj = nproj_in;
    this->nproj_soc = nproj_in_so;
    int spin_dimension = 4;

    // optimize
    for (int is = 0; is < spin_dimension; is++)
    {
        this->non_zero_count_soc[is] = 0;
        this->index1_soc[is] = std::vector<int>(nproj_soc * nproj_soc, 0);
        this->index2_soc[is] = std::vector<int>(nproj_soc * nproj_soc, 0);
    }

    if (!has_so)
    {
        this->d_real.create(nproj_soc + 1, nproj_soc + 1);
        this->d_so.create(spin_dimension, nproj_soc + 1, nproj_soc + 1); // for noncollinear-spin only case

        // calculate the number of non-zero elements in dion
        for (int L1 = 0; L1 < nproj_soc; L1++)
        {
            for (int L2 = 0; L2 < nproj_soc; L2++)
            {
                this->d_real(L1, L2) = d_so_in(L1, L2).real();
                if (std::fabs(d_real(L1, L2)) > 1.0e-8)
                {
                    this->index1_soc[0][non_zero_count_soc[0]] = L1;
                    this->index2_soc[0][non_zero_count_soc[0]] = L2;
                    this->non_zero_count_soc[0]++;
                }
                // for noncollinear-spin only case
                this->d_so(0, L1, L2) = d_so_in(L1, L2);
                this->d_so(3, L1, L2) = d_so_in(L1, L2);
                if (std::fabs(d_real(L1, L2)) > 1.0e-8)
                {
                    this->index1_soc[3][non_zero_count_soc[3]] = L1;
                    this->index2_soc[3][non_zero_count_soc[3]] = L2;
                    this->non_zero_count_soc[3]++;
                }
            }
        }
    }
    else // zhengdy-soc
    {
        this->d_so.create(spin_dimension, nproj_soc + 1, nproj_soc + 1);
        //		std::cout << "lmax=" << lmax << std::endl;

        if (this->lmax > -1)
        {
            if (PARAM.inp.lspinorb)
            {
                int is = 0;
                for (int is1 = 0; is1 < 2; is1++)
                {
                    for (int is2 = 0; is2 < 2; is2++)
                    {
                        for (int L1 = 0; L1 < nproj_soc; L1++)
                        {
                            for (int L2 = 0; L2 < nproj_soc; L2++)
                            {
                                this->d_so(is, L1, L2) = d_so_in(L1 + nproj_soc * is1, L2 + nproj_soc * is2);

                                if (fabs(this->d_so(is, L1, L2).real()) > 1.0e-8
                                    || fabs(this->d_so(is, L1, L2).imag()) > 1.0e-8)
                                {
                                    //									std::cout << "tt in atom is=" << is << " L1=" <<
                                    //L1
                                    //<< " L2="
                                    //									<< L2 << " " << d_so(is, L1, L2) << std::endl;

                                    this->index1_soc[is][non_zero_count_soc[is]] = L1;
                                    this->index2_soc[is][non_zero_count_soc[is]] = L2;
                                    this->non_zero_count_soc[is]++;
                                }
                            }
                        }
                        is++;
                    }
                }
            }
            else
            {
                int is = 0;
                for (int is1 = 0; is1 < 2; is1++)
                {
                    for (int is2 = 0; is2 < 2; is2++)
                    {
                        if (is >= PARAM.inp.nspin) {
                            break;
}
                        for (int L1 = 0; L1 < nproj_soc; L1++)
                        {
                            for (int L2 = 0; L2 < nproj_soc; L2++)
                            {
                                if (is == 1 || is == 2)
                                {
                                    this->d_so(is, L1, L2) = std::complex<double>(0.0, 0.0);
                                }
                                else
                                {
                                    this->d_so(is, L1, L2) = d_so_in(L1 + nproj_soc * is1, L2 + nproj_soc * is2);
                                }
                                if (std::abs(this->d_so(is, L1, L2).real()) > 1.0e-8
                                    || std::abs(this->d_so(is, L1, L2).imag()) > 1.0e-8)
                                {
                                    this->index1_soc[is][non_zero_count_soc[is]] = L1;
                                    this->index2_soc[is][non_zero_count_soc[is]] = L2;
                                    this->non_zero_count_soc[is]++;
                                }
                            }
                        }
                        is++;
                    }
                }
            }
        }
    }
    // 2016-07-19 end, LiuXh

    return;
}

#include "module_base/parallel_common.h"
#ifdef __MPI

void Atom_pseudo::bcast_atom_pseudo()
{
    ModuleBase::TITLE("Atom_pseudo", "bcast_atom_pseudo");
    // == pseudo_h ==
    // int
    Parallel_Common::bcast_int(lmax);
    Parallel_Common::bcast_int(mesh);
    Parallel_Common::bcast_int(nchi);
    Parallel_Common::bcast_int(nbeta);
    Parallel_Common::bcast_int(nv);
    Parallel_Common::bcast_double(zv);

    // double
    Parallel_Common::bcast_double(etotps);
    Parallel_Common::bcast_double(ecutwfc);
    Parallel_Common::bcast_double(ecutrho);

    // bool
    Parallel_Common::bcast_bool(tvanp);
    Parallel_Common::bcast_bool(nlcc);
    Parallel_Common::bcast_bool(has_so);

    // std::string
    Parallel_Common::bcast_string(psd);
    Parallel_Common::bcast_string(pp_type);
    Parallel_Common::bcast_string(xc_func);

    if (GlobalV::MY_RANK != 0)
    {
        jjj = std::vector<double>(nbeta, 0.0);
        els = std::vector<std::string>(nchi, "");
        lchi = std::vector<int>(nchi, 0);
        oc = std::vector<double>(nchi, 0.0);
        jchi = std::vector<double>(nchi, 0.0);
        nn = std::vector<int>(nchi, 0);
    }

    Parallel_Common::bcast_double(jjj.data(), nbeta);
    Parallel_Common::bcast_string(els.data(), nchi);
    Parallel_Common::bcast_int(lchi.data(), nchi);
    Parallel_Common::bcast_double(oc.data(), nchi);
    Parallel_Common::bcast_double(jchi.data(), nchi);
    Parallel_Common::bcast_int(nn.data(), nchi);
    // == end of pseudo_h

    // == pseudo_atom ==
    Parallel_Common::bcast_int(msh);
    Parallel_Common::bcast_double(rcut);
    if (GlobalV::MY_RANK != 0)
    {
        assert(mesh != 0);
        r = std::vector<double>(mesh, 0.0);
        rab = std::vector<double>(mesh, 0.0);
        rho_atc = std::vector<double>(mesh, 0.0);
        rho_at = std::vector<double>(mesh, 0.0);
        chi.create(nchi, mesh);
    }

    Parallel_Common::bcast_double(r.data(), mesh);
    Parallel_Common::bcast_double(rab.data(), mesh);
    Parallel_Common::bcast_double(rho_atc.data(), mesh);
    Parallel_Common::bcast_double(rho_at.data(), mesh);
    Parallel_Common::bcast_double(chi.c, nchi * mesh);
    // == end of pseudo_atom ==

    // == pseudo_vl ==
    if (GlobalV::MY_RANK != 0)
    {
        vloc_at = std::vector<double>(mesh, 0.0);
    }
    Parallel_Common::bcast_double(vloc_at.data(), mesh);
    // == end of pseudo_vl ==

    // == pseudo ==
    if (nbeta == 0) {
        return;
}

    if (GlobalV::MY_RANK != 0)
    {
        lll = std::vector<int>(nbeta, 0);
    }
    Parallel_Common::bcast_int(lll.data(), nbeta);
    Parallel_Common::bcast_int(kkbeta);
    Parallel_Common::bcast_int(nh);

    int nr, nc;
    if (GlobalV::MY_RANK == 0)
    {
        nr = betar.nr;
        nc = betar.nc;
    }
    Parallel_Common::bcast_int(nr);
    Parallel_Common::bcast_int(nc);

    if (GlobalV::MY_RANK != 0)
    {
        betar.create(nr, nc);
        dion.create(nbeta, nbeta);
    }

    // below two 'bcast_double' lines of codes seem to have bugs,
    // on some computers, the code will stuck here for ever.
    // mohan note 2021-04-28
    Parallel_Common::bcast_double(dion.c, nbeta * nbeta);
    Parallel_Common::bcast_double(betar.c, nr * nc);
    // == end of psesudo_nc ==

    // uspp   liuyu 2023-10-03
    if (tvanp)
    {
        Parallel_Common::bcast_int(nqlc);
        if (GlobalV::MY_RANK != 0)
        {
            qfuncl.create(nqlc, nbeta * (nbeta + 1) / 2, mesh);
        }
        const int dim = nqlc * nbeta * (nbeta + 1) / 2 * mesh;
        Parallel_Common::bcast_double(qfuncl.ptr, dim);

        if (GlobalV::MY_RANK != 0)
        {
            qqq.create(nbeta, nbeta);
        }
        Parallel_Common::bcast_double(qqq.c, nbeta * nbeta);
    }

    return;
}

#endif
