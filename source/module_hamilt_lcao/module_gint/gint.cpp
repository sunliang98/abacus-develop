#include "gint.h"

#include "module_parameter/parameter.h"
#if ((defined __CUDA))
#include "gint_force_gpu.h"
#include "gint_rho_gpu.h"
#include "gint_vl_gpu.h"
#endif

#include "module_base/memory.h"
#include "module_base/timer.h"
#include "module_basis/module_ao/ORB_read.h"
#include "module_hamilt_lcao/module_hcontainer/hcontainer_funcs.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __MKL
#include <mkl_service.h>
#endif

Gint::~Gint() {

    delete this->hRGint;
    delete this->hRGintCd;
    for (int is = 0; is < this->DMRGint.size(); is++) {
        delete this->DMRGint[is];
    }
#ifdef __MPI
    delete this->DMRGint_full;
#endif
}

void Gint::cal_gint(Gint_inout* inout) {
    ModuleBase::TITLE("Gint_interface", "cal_gint");
    ModuleBase::timer::tick("Gint_interface", "cal_gint");
    // In multi-process environments,
    // some processes may not be allocated any data.
    if (this->gridt->get_init_malloced() == false) {
        ModuleBase::WARNING_QUIT("Gint_interface::cal_gint",
                                 "gridt has not been allocated yet!");
    }
    if (this->gridt->max_atom > 0) {
#ifdef __CUDA
        if (PARAM.inp.device == "gpu"
            && (inout->job == Gint_Tools::job_type::vlocal
                || inout->job == Gint_Tools::job_type::rho
                || inout->job == Gint_Tools::job_type::force)) {
            if (inout->job == Gint_Tools::job_type::vlocal) {
                gpu_vlocal_interface(inout);
            } else if (inout->job == Gint_Tools::job_type::rho) {
                gpu_rho_interface(inout);
            } else if (inout->job == Gint_Tools::job_type::force) {
                gpu_force_interface(inout);
            }
        } else
#endif
        {
#ifdef __MKL
            const int mkl_threads = mkl_get_max_threads();
            mkl_set_num_threads(mkl_threads);
#endif
            {
                if (inout->job == Gint_Tools::job_type::vlocal) {
                    gint_kernel_vlocal(inout);
                } else if (inout->job == Gint_Tools::job_type::dvlocal) {
                    gint_kernel_dvlocal(inout);
                } else if (inout->job == Gint_Tools::job_type::vlocal_meta) {
                    gint_kernel_vlocal_meta(inout);
                } else if (inout->job == Gint_Tools::job_type::rho) {
                    gint_kernel_rho(inout);
                } else if (inout->job == Gint_Tools::job_type::tau) {
                    gint_kernel_tau(inout);
                } else if (inout->job == Gint_Tools::job_type::force) {
                    gint_kernel_force(inout);
                } else if (inout->job == Gint_Tools::job_type::force_meta) {
                    gint_kernel_force_meta(inout);
                }
            }
        }
    }
    ModuleBase::timer::tick("Gint_interface", "cal_gint");
    return;
}
void Gint::prep_grid(const Grid_Technique& gt,
                     const int& nbx_in,
                     const int& nby_in,
                     const int& nbz_in,
                     const int& nbz_start_in,
                     const int& ncxyz_in,
                     const int& bx_in,
                     const int& by_in,
                     const int& bz_in,
                     const int& bxyz_in,
                     const int& nbxx_in,
                     const int& ny_in,
                     const int& nplane_in,
                     const int& startz_current_in,
                     const UnitCell* ucell_in,
                     const LCAO_Orbitals* orb_in) {
    ModuleBase::TITLE(GlobalV::ofs_running, "Gint_k", "prep_grid");

    this->gridt = &gt;
    this->nbx = nbx_in;
    this->nby = nby_in;
    this->nbz = nbz_in;
    this->ncxyz = ncxyz_in;
    this->nbz_start = nbz_start_in;
    this->bx = bx_in;
    this->by = by_in;
    this->bz = bz_in;
    this->bxyz = bxyz_in;
    this->nbxx = nbxx_in;
    this->ny = ny_in;
    this->nplane = nplane_in;
    this->startz_current = startz_current_in;
    this->ucell = ucell_in;
    assert(nbx > 0);
    assert(nby > 0);
    assert(nbz >= 0);
    assert(ncxyz > 0);
    assert(bx > 0);
    assert(by > 0);
    assert(bz > 0);
    assert(bxyz > 0);
    assert(nbxx >= 0);
    assert(ny > 0);
    assert(nplane >= 0);
    assert(startz_current >= 0);
    assert(this->ucell->omega > 0.0);

    return;
}

void Gint::initialize_pvpR(const UnitCell& ucell_in, Grid_Driver* gd) {
    ModuleBase::TITLE("Gint", "initialize_pvpR");

    int npol = 1;
    // there is the only resize code of DMRGint
    if (this->DMRGint.size() == 0) {
        this->DMRGint.resize(PARAM.inp.nspin);
    }
    if (PARAM.inp.nspin != 4) {
        if (this->hRGint != nullptr) {
            delete this->hRGint;
        }
        this->hRGint = new hamilt::HContainer<double>(ucell_in.nat);
    } else {
        npol = 2;
        if (this->hRGintCd != nullptr) {
            delete this->hRGintCd;
        }
        this->hRGintCd
            = new hamilt::HContainer<std::complex<double>>(ucell_in.nat);
        for (int is = 0; is < PARAM.inp.nspin; is++) {
            if (this->DMRGint[is] != nullptr) {
                delete this->DMRGint[is];
            }
            this->DMRGint[is] = new hamilt::HContainer<double>(ucell_in.nat);
        }
#ifdef __MPI
        if (this->DMRGint_full != nullptr) {
            delete this->DMRGint_full;
        }
        this->DMRGint_full = new hamilt::HContainer<double>(ucell_in.nat);
#endif
    }

    // prepare the row_index and col_index for construct AtomPairs, they are
    // same, name as orb_index
    std::vector<int> orb_index(ucell_in.nat + 1);
    orb_index[0] = 0;
    for (int i = 1; i < orb_index.size(); i++) {
        int type = ucell_in.iat2it[i - 1];
        orb_index[i] = orb_index[i - 1] + ucell_in.atoms[type].nw;
    }
    std::vector<int> orb_index_npol;
    if (npol == 2) {
        orb_index_npol.resize(ucell_in.nat + 1);
        orb_index_npol[0] = 0;
        for (int i = 1; i < orb_index_npol.size(); i++) {
            int type = ucell_in.iat2it[i - 1];
            orb_index_npol[i]
                = orb_index_npol[i - 1] + ucell_in.atoms[type].nw * npol;
        }
    }

    if (PARAM.globalv.gamma_only_local && PARAM.inp.nspin != 4) {
        this->hRGint->fix_gamma();
    }
    for (int T1 = 0; T1 < ucell_in.ntype; ++T1) {
        const Atom* atom1 = &(ucell_in.atoms[T1]);
        for (int I1 = 0; I1 < atom1->na; ++I1) {
            auto& tau1 = atom1->tau[I1];

            gd->Find_atom(ucell_in, tau1, T1, I1);

            const int iat1 = ucell_in.itia2iat(T1, I1);

            // for grid integration (on FFT box),
            // we only need to consider <phi_i | phi_j>,

            // whether this atom is in this processor.
            if (this->gridt->in_this_processor[iat1]) {
                for (int ad = 0; ad < gd->getAdjacentNum() + 1; ++ad) {
                    const int T2 = gd->getType(ad);
                    const int I2 = gd->getNatom(ad);
                    const int iat2 = ucell_in.itia2iat(T2, I2);
                    const Atom* atom2 = &(ucell_in.atoms[T2]);

                    // NOTE: hRGint wil save total number of atom pairs,
                    // if only upper triangle is saved, the lower triangle will
                    // be lost in 2D-block parallelization. if the adjacent atom
                    // is in this processor.
                    if (this->gridt->in_this_processor[iat2]) {
                        ModuleBase::Vector3<double> dtau
                            = gd->getAdjacentTau(ad) - tau1;
                        double distance = dtau.norm() * ucell_in.lat0;
                        double rcut
                            = this->gridt->rcuts[T1] + this->gridt->rcuts[T2];

                        // if(distance < rcut)
                        //  mohan reset this 2013-07-02 in Princeton
                        //  we should make absolutely sure that the distance is
                        //  smaller than rcuts[it] this should be consistant
                        //  with LCAO_nnr::cal_nnrg function typical example : 7
                        //  Bohr cutoff Si orbital in 14 Bohr length of cell.
                        //  distance = 7.0000000000000000
                        //  rcuts[it] = 7.0000000000000008
                        if (distance < rcut - 1.0e-15) {
                            // calculate R index
                            auto& R_index = gd->getBox(ad);
                            // insert this atom-pair into this->hRGint
                            if (npol == 1) {
                                hamilt::AtomPair<double> tmp_atom_pair(
                                    iat1,
                                    iat2,
                                    R_index.x,
                                    R_index.y,
                                    R_index.z,
                                    orb_index.data(),
                                    orb_index.data(),
                                    ucell_in.nat);
                                this->hRGint->insert_pair(tmp_atom_pair);
                            } else {
                                // HR is complex and size is nw * npol
                                hamilt::AtomPair<std::complex<double>>
                                    tmp_atom_pair(iat1,
                                                  iat2,
                                                  R_index.x,
                                                  R_index.y,
                                                  R_index.z,
                                                  orb_index_npol.data(),
                                                  orb_index_npol.data(),
                                                  ucell_in.nat);
                                this->hRGintCd->insert_pair(tmp_atom_pair);
                                // DMR is double now and size is nw
                                hamilt::AtomPair<double> tmp_dmR(
                                    iat1,
                                    iat2,
                                    R_index.x,
                                    R_index.y,
                                    R_index.z,
                                    orb_index.data(),
                                    orb_index.data(),
                                    ucell_in.nat);
                                for (int is = 0; is < this->DMRGint.size();
                                     is++) {
                                    this->DMRGint[is]->insert_pair(tmp_dmR);
                                }
#ifdef __MPI
                                hamilt::AtomPair<double> tmp_dmR_full(
                                    iat1,
                                    iat2,
                                    R_index.x,
                                    R_index.y,
                                    R_index.z,
                                    orb_index_npol.data(),
                                    orb_index_npol.data(),
                                    ucell_in.nat);
                                // tmp DMR for transfer
                                this->DMRGint_full->insert_pair(tmp_dmR_full);
#endif
                            }
                        }
                    } // end iat2
                }     // end ad
            }         // end iat
        }             // end I1
    }                 // end T1
    if (npol == 1) {
        this->hRGint->allocate(nullptr, false);
        ModuleBase::Memory::record("Gint::hRGint",
                                   this->hRGint->get_memory_size());
        // initialize DMRGint with hRGint when NSPIN != 4
        for (int is = 0; is < this->DMRGint.size(); is++) {
            if (this->DMRGint[is] != nullptr) {
                delete this->DMRGint[is];
            }
            this->DMRGint[is] = new hamilt::HContainer<double>(*this->hRGint);
        }
        ModuleBase::Memory::record("Gint::DMRGint",
                                   this->DMRGint[0]->get_memory_size()
                                       * this->DMRGint.size());
    } else {
        this->hRGintCd->allocate(nullptr, 0);
        ModuleBase::Memory::record("Gint::hRGintCd",
                                   this->hRGintCd->get_memory_size());
        for (int is = 0; is < this->DMRGint.size(); is++) {
            this->DMRGint[is]->allocate(nullptr, false);
        }
        ModuleBase::Memory::record("Gint::DMRGint",
                                   this->DMRGint[0]->get_memory_size()
                                       * this->DMRGint.size());
#ifdef __MPI
        this->DMRGint_full->allocate(nullptr, false);
        ModuleBase::Memory::record("Gint::DMRGint_full",
                                   this->DMRGint_full->get_memory_size());
#endif
    }
}

void Gint::transfer_DM2DtoGrid(std::vector<hamilt::HContainer<double>*> DM2D) {
    ModuleBase::TITLE("Gint", "transfer_DMR");

    // To check whether input parameter DM2D has been initialized
#ifdef __DEBUG
    assert(!DM2D.empty()
           && "Input parameter DM2D has not been initialized while calling "
              "function transfer_DM2DtoGrid!");
#endif

    ModuleBase::timer::tick("Gint", "transfer_DMR");
    if (PARAM.inp.nspin != 4) {
        for (int is = 0; is < this->DMRGint.size(); is++) {
#ifdef __MPI
            hamilt::transferParallels2Serials(*DM2D[is], DMRGint[is]);
#else
            this->DMRGint[is]->set_zero();
            this->DMRGint[is]->add(*DM2D[is]);
#endif
        }
    } else // NSPIN=4 case
    {
#ifdef __MPI
        hamilt::transferParallels2Serials(*DM2D[0], this->DMRGint_full);
#else
        this->DMRGint_full = DM2D[0];
#endif
        std::vector<double*> tmp_pointer(4, nullptr);
        for (int iap = 0; iap < this->DMRGint_full->size_atom_pairs(); ++iap) {
            auto& ap = this->DMRGint_full->get_atom_pair(iap);
            int iat1 = ap.get_atom_i();
            int iat2 = ap.get_atom_j();
            for (int ir = 0; ir < ap.get_R_size(); ++ir) {
                const ModuleBase::Vector3<int> r_index = ap.get_R_index(ir);
                for (int is = 0; is < 4; is++) {
                    tmp_pointer[is] = this->DMRGint[is]
                                          ->find_matrix(iat1, iat2, r_index)
                                          ->get_pointer();
                }
                double* data_full = ap.get_pointer(ir);
                for (int irow = 0; irow < ap.get_row_size(); irow += 2) {
                    for (int icol = 0; icol < ap.get_col_size(); icol += 2) {
                        *(tmp_pointer[0])++ = data_full[icol];
                        *(tmp_pointer[1])++ = data_full[icol + 1];
                    }
                    data_full += ap.get_col_size();
                    for (int icol = 0; icol < ap.get_col_size(); icol += 2) {
                        *(tmp_pointer[2])++ = data_full[icol];
                        *(tmp_pointer[3])++ = data_full[icol + 1];
                    }
                    data_full += ap.get_col_size();
                }
            }
        }
    }
    ModuleBase::timer::tick("Gint", "transfer_DMR");
}
