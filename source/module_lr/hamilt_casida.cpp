#include "hamilt_casida.h"
namespace LR
{
    template<typename T>
    std::vector<T> HamiltCasidaLR<T>::matrix()
    {
        ModuleBase::TITLE("HamiltCasidaLR", "matrix");
        int npairs = this->nocc * this->nvirt;
        std::vector<T> Amat_full(this->nk * npairs * this->nk * npairs, 0.0);
        for (int ik = 0;ik < this->nk;++ik)
            for (int j = 0;j < nocc;++j)
                for (int b = 0;b < nvirt;++b)
                {//calculate A^{ai} for each bj
                    int bj = j * nvirt + b;
                    int kbj = ik * npairs + bj;
                    psi::Psi<T> X_bj(1, 1, this->nk * this->pX->get_local_size()); // k1-first, like in iterative solver
                    X_bj.zero_out();
                    // X_bj(0, 0, lj * this->pX->get_row_size() + lb) = this->one();
                    int lj = this->pX->global2local_col(j);
                    int lb = this->pX->global2local_row(b);
                    if (this->pX->in_this_processor(b, j)) X_bj(0, 0, ik * this->pX->get_local_size() + lj * this->pX->get_row_size() + lb) = this->one();
                    psi::Psi<T> A_aibj(1, 1, this->nk * this->pX->get_local_size()); // k1-first
                    A_aibj.zero_out();

                    hamilt::Operator<T>* node(this->ops);
                    while (node != nullptr)
                    {   // act() on and return the k1-first type of psi
                        node->act(X_bj, A_aibj, 1);
                        node = (hamilt::Operator<T>*)(node->next_op);
                    }
                    // reduce ai for a fixed bj
                    A_aibj.fix_kb(0, 0);
#ifdef __MPI
                    for (int ik_ai = 0;ik_ai < this->nk;++ik_ai)
                        LR_Util::gather_2d_to_full(*this->pX, &A_aibj.get_pointer()[ik_ai * this->pX->get_local_size()],
                            Amat_full.data() + kbj * this->nk * npairs /*col, bj*/ + ik_ai * npairs/*row, ai*/,
                            false, this->nvirt, this->nocc);
#endif
                }
        // output Amat
        std::cout << "Amat_full:" << std::endl;
        for (int i = 0;i < this->nk * npairs;++i)
        {
            for (int j = 0;j < this->nk * npairs;++j)
            {
                std::cout << Amat_full[i * this->nk * npairs + j] << " ";
            }
            std::cout << std::endl;
        }
        return Amat_full;
    }

    template<> double HamiltCasidaLR<double>::one() { return 1.0; }
    template<> std::complex<double> HamiltCasidaLR<std::complex<double>>::one() { return std::complex<double>(1.0, 0.0); }

    template class HamiltCasidaLR<double>;
    template class HamiltCasidaLR<std::complex<double>>;
}