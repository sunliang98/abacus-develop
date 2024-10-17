#ifdef __EXX
#include "operator_lr_exx.h"
#include "module_lr/dm_trans/dm_trans.h"
#include "module_lr/utils/lr_util.h"
#include "module_lr/utils/lr_util_print.h"
#include "module_lr/ri_benchmark/ri_benchmark.h"
namespace LR
{
    template<typename T>
    void OperatorLREXX<T>::allocate_Ds_onebase()
    {
        ModuleBase::TITLE("OperatorLREXX", "allocate_Ds_onebase");
        this->Ds_onebase.resize(this->nspin_solve);
        for (int is = 0;is < this->nspin_solve;++is) {
            for (int iat1 = 0;iat1 < ucell.nat;++iat1) {
                const int it1 = ucell.iat2it[iat1];
                for (int iat2 = 0;iat2 < ucell.nat;++iat2) {
                    const int it2=ucell.iat2it[iat2];
                    for (auto cell : this->BvK_cells) {
                        this->Ds_onebase[is][iat1][std::make_pair(iat2, cell)] = aims_nbasis.empty() ?
                            RI::Tensor<T>({ static_cast<size_t>(ucell.atoms[it1].nw),  static_cast<size_t>(ucell.atoms[it2].nw) }) :
                            RI::Tensor<T>({ static_cast<size_t>( aims_nbasis[it1]),  static_cast<size_t>( aims_nbasis[it2]) });
                    }
                }
            }
        }
    }

    template<>
    void OperatorLREXX<double>::cal_DM_onebase(const int io, const int iv, const int ik, const int is) const
    {
        ModuleBase::TITLE("OperatorLREXX", "cal_DM_onebase");
        assert(ik == 0);
        for (auto cell : this->BvK_cells)
        {
            for (int it1 = 0;it1 < ucell.ntype;++it1)
                for (int ia1 = 0; ia1 < ucell.atoms[it1].na; ++ia1)
                    for (int it2 = 0;it2 < ucell.ntype;++it2)
                        for (int ia2 = 0;ia2 < ucell.atoms[it2].na;++ia2)
                        {
                            int iat1 = ucell.itia2iat(it1, ia1);
                            int iat2 = ucell.itia2iat(it2, ia2);
                            auto& D2d = this->Ds_onebase[is][iat1][std::make_pair(iat2, cell)];
                            const int nw1 = aims_nbasis.empty() ? ucell.atoms[it1].nw : aims_nbasis[it1];
                            const int nw2 = aims_nbasis.empty() ? ucell.atoms[it2].nw : aims_nbasis[it2];
                            for (int iw1 = 0;iw1 < nw1;++iw1)
                                for (int iw2 = 0;iw2 < nw2;++iw2)
                                    if(this->pmat->in_this_processor(ucell.itiaiw2iwt(it1, ia1, iw1), ucell.itiaiw2iwt(it2, ia2, iw2)))
                                        D2d(iw1, iw2) = this->psi_ks_full(ik, io, ucell.itiaiw2iwt(it1, ia1, iw1)) * this->psi_ks_full(ik, nocc + iv, ucell.itiaiw2iwt(it2, ia2, iw2));
                        }
        }
    }

    template<>
    void OperatorLREXX<std::complex<double>>::cal_DM_onebase(const int io, const int iv, const int ik, const int is) const
    {
        ModuleBase::TITLE("OperatorLREXX", "cal_DM_onebase");
        for (auto cell : this->BvK_cells)
        {
            std::complex<double> frac = RI::Global_Func::convert<std::complex<double>>(std::exp(
                -ModuleBase::TWO_PI * ModuleBase::IMAG_UNIT * (this->kv.kvec_c.at(ik) * (RI_Util::array3_to_Vector3(cell) * GlobalC::ucell.latvec))));
            for (int it1 = 0;it1 < ucell.ntype;++it1)
                for (int ia1 = 0; ia1 < ucell.atoms[it1].na; ++ia1)
                    for (int it2 = 0;it2 < ucell.ntype;++it2)
                        for (int ia2 = 0;ia2 < ucell.atoms[it2].na;++ia2)
                        {
                            int iat1 = ucell.itia2iat(it1, ia1);
                            int iat2 = ucell.itia2iat(it2, ia2);
                            auto& D2d = this->Ds_onebase[is][iat1][std::make_pair(iat2, cell)];
                            const int nw1 = aims_nbasis.empty() ? ucell.atoms[it1].nw : aims_nbasis[it1];
                            const int nw2 = aims_nbasis.empty() ? ucell.atoms[it2].nw : aims_nbasis[it2];
                            for (int iw1 = 0;iw1 < nw1;++iw1)
                                for (int iw2 = 0;iw2 < nw2;++iw2)
                                    if(this->pmat->in_this_processor(ucell.itiaiw2iwt(it1, ia1, iw1), ucell.itiaiw2iwt(it2, ia2, iw2)))
                                        D2d(iw1, iw2) = frac * std::conj(this->psi_ks_full(ik, io, ucell.itiaiw2iwt(it1, ia1, iw1))) * this->psi_ks_full(ik, nocc + iv, ucell.itiaiw2iwt(it2, ia2, iw2));
                        }
        }
    }

    template<typename T>
    void OperatorLREXX<T>::act(const psi::Psi<T>& psi_in, psi::Psi<T>& psi_out, const int nbands) const
    {
        ModuleBase::TITLE("OperatorLREXX", "act");

        assert(nbands <= psi_in.get_nbands());
        const int& nk = this->kv.get_nks() / this->nspin;
        psi::Psi<T> psi_in_bfirst = LR_Util::k1_to_bfirst_wrapper(psi_in, nk, this->pX->get_local_size());
        psi::Psi<T> psi_out_bfirst = LR_Util::k1_to_bfirst_wrapper(psi_out, nk, this->pX->get_local_size());

        // convert parallel info to LibRI interfaces
        std::vector<std::tuple<std::set<TA>, std::set<TA>>> judge = RI_2D_Comm::get_2D_judge(*this->pmat);
        for (int ib = 0;ib < nbands;++ib)
        {
            psi_out_bfirst.fix_b(ib);
            // suppose Cs，Vs, have already been calculated in the ion-step of ground state, 
            // DM_trans(k) and DM_trans(R) has already been calculated from psi_in in OperatorLRHxc::act
            // but int RI_benchmark, DM_trans(k) should be first calculated here
            if (cal_dm_trans)
            {
#ifdef __MPI
                std::vector<container::Tensor>  dm_trans_2d = cal_dm_trans_pblas(psi_in_bfirst, *pX, *psi_ks, *pc, naos, nocc, nvirt, *pmat);
                if (this->tdm_sym) for (auto& t : dm_trans_2d) LR_Util::matsym(t.data<T>(), naos, *pmat);
#else
                std::vector<container::Tensor>  dm_trans_2d = cal_dm_trans_blas(psi_in_bfirst, *psi_ks, nocc, nvirt);
                if (this->tdm_sym) for (auto& t : dm_trans_2d) LR_Util::matsym(t.data<T>(), naos);
#endif
                // tensor to vector, then set DMK
                for (int ik = 0;ik < nk;++ik) { this->DM_trans[ib]->set_DMK_pointer(ik, dm_trans_2d[ik].data<T>()); }
            }

            // 1. set_Ds (once)
            // convert to vector<T*> for the interface of RI_2D_Comm::split_m2D_ktoR (interface will be unified to ct::Tensor)
            std::vector<std::vector<T>> DMk_trans_vector = this->DM_trans[ib]->get_DMK_vector();
            // assert(DMk_trans_vector.size() == nk);
            std::vector<const std::vector<T>*> DMk_trans_pointer(nk);
            for (int ik = 0;ik < nk;++ik) {DMk_trans_pointer[ik] = &DMk_trans_vector[ik];}
            // if multi-k, DM_trans(TR=double) -> Ds_trans(TR=T=complex<double>)
            std::vector<std::map<TA, std::map<TAC, RI::Tensor<T>>>> Ds_trans =
                aims_nbasis.empty() ? 
                RI_2D_Comm::split_m2D_ktoR<T>(this->kv, DMk_trans_pointer, *this->pmat, this->nspin_solve)
                : RI_Benchmark::split_Ds(DMk_trans_vector, aims_nbasis, ucell); //0.5 will be multiplied
            // LR_Util::print_CV(Ds_trans[0], "Ds_trans in OperatorLREXX", 1e-10);
            // 2. cal_Hs
            auto lri = this->exx_lri.lock();
            for (int is = 0;is < nspin_solve;++is)
            {
                // LR_Util::print_CV(Ds_trans[is], "Ds_trans in OperatorLREXX", 1e-10);
                lri->exx_lri.set_Ds(std::move(Ds_trans[is]), lri->info.dm_threshold);
                lri->exx_lri.cal_Hs();
                lri->Hexxs[is] = RI::Communicate_Tensors_Map_Judge::comm_map2_first(
                    lri->mpi_comm, std::move(lri->exx_lri.Hs), std::get<0>(judge[is]), std::get<1>(judge[is]));
                lri->post_process_Hexx(lri->Hexxs[is]);
            }

            // 3. set [AX]_iak = DM_onbase * Hexxs for each occ-virt pair and each k-point
            // caution: parrallel

            for (int io = 0;io < this->nocc;++io) {
                for (int iv = 0;iv < this->nvirt;++iv) {
                    for (int ik = 0;ik < nk;++ik) {
                        for (int is = 0;is < this->nspin_solve;++is)
                            {
                                this->cal_DM_onebase(io, iv, ik, is);       //set Ds_onebase for all e-h pairs (not only on this processor)
                                // LR_Util::print_CV(Ds_onebase[is], "Ds_onebase of occ " + std::to_string(io) + ", virtual " + std::to_string(iv) + " in OperatorLREXX", 1e-10);
                                const T& ene= 2 * alpha * //minus for exchange(but here plus is right, why?), 2 for Hartree to Ry
                                    lri->exx_lri.post_2D.cal_energy(this->Ds_onebase[is], lri->Hexxs[is]);
                                if(this->pX->in_this_processor(iv, io)) { 
                                    psi_out_bfirst(ik, this->pX->global2local_col(io) * this->pX->get_row_size() + this->pX->global2local_row(iv)) += ene;
}
                            }
}
}
}
        }
    }
    template class OperatorLREXX<double>;
    template class OperatorLREXX<std::complex<double>>;
}
#endif