#include "hcontainer_funcs.h"
#include "source_base/libm/libm.h"

namespace hamilt
{
/**
 * @brief calculate the Hk matrix with specific k vector
 * @param hR the HContainer of <I,J,R> atom pairs
 * @param hk the data pointer of Hk matrix, the size of hk would be nrow * ncol
 * @param kvec_d_in the k vector in Direct coordinate
 * @param hk_ld the leading dimension number of hk, ncol for row-major, nrow for column-major
 * @param hk_type the data-type of hk, 0 is row-major, 1 is column-major
*/
template<typename TR>
void folding_HR(const hamilt::HContainer<TR>& hR,
                std::complex<double>* hk,
                const ModuleBase::Vector3<double>& kvec_d_in,
                const int hk_ld,
                const int hk_type)
{
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < hR.size_atom_pairs(); ++i)
    {
        hamilt::AtomPair<TR>& tmp = hR.get_atom_pair(i);
        const int row_size = tmp.get_row_size();
        const int col_size = tmp.get_col_size();
        // copy hk to hk_type
        // hk_tmp is row-major and stored contiguously in memory,
        // so copy hr to hk_tmp is faster than copy hr to hk
        std::vector<std::complex<double>> hk_mat_tmp(row_size * col_size, 0);

        // copy hr to hk_tmp
        for(int ir = 0; ir < tmp.get_R_size(); ++ir)
        {
            const ModuleBase::Vector3<int> r_index = tmp.get_R_index(ir);
            TR* hr_mat = tmp.get_pointer(ir);
            // cal k_phase
            // if TK==std::complex<double>, kphase is e^{ikR}
            const ModuleBase::Vector3<double> dR(r_index.x, r_index.y, r_index.z);
            const double arg = (kvec_d_in * dR) * ModuleBase::TWO_PI;
            double sinp, cosp;
            ModuleBase::libm::sincos(arg, &sinp, &cosp);
            std::complex<double> kphase = std::complex<double>(cosp, sinp);
            
            for(int i = 0; i < row_size * col_size; ++i)
            {
                hk_mat_tmp[i] += kphase * hr_mat[i];
            }
        }

        // copy hk_tmp to hk
        if (hk_type == 0)
        {
            std::complex<double>* hk_mat = hk + tmp.get_begin_row() * hk_ld + tmp.get_begin_col();
            for(int irow = 0; irow < row_size; ++irow)
            {
                for(int icol = 0; icol < col_size; ++icol)
                {
                    hk_mat[irow * hk_ld + icol] += hk_mat_tmp[irow * col_size + icol];
                }
            }
        }
        else if(hk_type == 1)
        {
            std::complex<double>* hk_mat = hk + tmp.get_begin_col() * hk_ld + tmp.get_begin_row();
            for(int icol = 0; icol < col_size; ++icol)
            {
                for(int irow = 0; irow < row_size; ++irow)
                {
                    hk_mat[icol * hk_ld + irow] += hk_mat_tmp[irow * col_size + icol];
                }
            }
        }
    }
    /*for (int i = 0; i < hR.size_R_loop(); ++i)
    {
        // get R index
        int rx, ry, rz;
        hR.loop_R(i, rx, ry, rz);
        // only deal with current_R for hR
        hR.fix_R(rx, ry, rz);

        // cal k_phase
        // if TK==std::complex<double>, kphase is e^{ikR}
        const ModuleBase::Vector3<double> dR(rx, ry, rz);
        const double arg = (kvec_d_in * dR) * ModuleBase::TWO_PI;
        double sinp, cosp;
        ModuleBase::libm::sincos(arg, &sinp, &cosp);
        std::complex<double> kphase = std::complex<double>(cosp, sinp);

        // loop_atom_pairs
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int j = 0; j < hR.size_atom_pairs(); ++j)
        {
            // Hk += HR * e^ikR
            hR.get_atom_pair(j).add_to_matrix(hk, ncol, kphase, hk_type);
        }
    }*/
}

// template instantiation
template void folding_HR<std::complex<double>>(const hamilt::HContainer<std::complex<double>>& hR,
                                            std::complex<double>* hk,
                                            const ModuleBase::Vector3<double>& kvec_d_in,
                                            const int ncol,
                                            const int hk_type);
template void folding_HR<double>(const hamilt::HContainer<double>& hR,
                                std::complex<double>* hk,
                                const ModuleBase::Vector3<double>& kvec_d_in,
                                const int ncol,
                                const int hk_type);
// special case for double
void folding_HR(const hamilt::HContainer<double>& hR,
                double* hk,
                const ModuleBase::Vector3<double>& kvec_d_in,
                const int hk_ld,
                const int hk_type)
{
// in ABACUS, this function works with gamma-only case.
// hR should be R=(0,0,0) only. 
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < hR.size_atom_pairs(); ++i)
    {
        // cal k_phase
        // if TK==double, kphase is 1.0
        double kphase = 1.0;

        // Hk = HR 
        hR.get_atom_pair(i).add_to_matrix(hk, hk_ld  , kphase, hk_type);
    }
}

} // namespace hamilt