#include "kernels/cuda/cuda_tools.cuh"
#include "kernels/cuda/vbatch_matrix_mul.cuh"
#include "module_base/ylm.h"
#include "module_hamilt_lcao/module_gint/gint_rho.h"
#include "module_hamilt_lcao/module_gint/gint_tools.h"
#include "module_hamilt_lcao/module_gint/kernels/cuda/gint_rho.cuh"

namespace GintKernel
{

void gint_gamma_rho_gpu(const hamilt::HContainer<double>* dm,
                        const int nczp,
                        const double* ylmcoef_now,
                        const double dr,
                        const double* rcut,
                        const Grid_Technique& gridt,
                        const UnitCell& ucell,
                        double* rho)
{
    const int nbz = gridt.nbzp;
    const int lgd = gridt.lgd;
    const int max_size = gridt.max_atom;
    double* dm_matrix_h = new double[lgd * lgd];

    checkCuda(cudaMemset(gridt.rho_g, 0, gridt.ncxyz * sizeof(double)));

    // retrieve the density matrix on the host
    ModuleBase::GlobalFunc::ZEROS(dm_matrix_h, lgd * lgd);
    for (int iat1 = 0; iat1 < ucell.nat; iat1++)
    {
        for (int iat2 = 0; iat2 < ucell.nat; iat2++)
        {
            int it1 = ucell.iat2it[iat1];
            int it2 = ucell.iat2it[iat2];
            int lo1
                = gridt.trace_lo[ucell.itiaiw2iwt(it1, ucell.iat2ia[iat1], 0)];
            int lo2
                = gridt.trace_lo[ucell.itiaiw2iwt(it2, ucell.iat2ia[iat2], 0)];

            hamilt::AtomPair<double>* tmp_ap = dm->find_pair(iat1, iat2);
            int orb_index = 0;
            if (tmp_ap == NULL)
            {
                continue;
            }
            for (int orb_i = 0; orb_i < tmp_ap->get_row_size(); orb_i++)
            {
                for (int orb_j = 0; orb_j < tmp_ap->get_col_size(); orb_j++)
                {
                    dm_matrix_h[(lo1 + orb_i) * lgd + (lo2 + orb_j)]
                        = tmp_ap->get_pointer(0)[orb_index];
                    orb_index++;
                }
            }
        }
    }

    // transfer the density matrix to the device
    double* dm_matrix_g;
    checkCuda(cudaMalloc((void**)&dm_matrix_g, lgd * lgd * sizeof(double)));
    checkCuda(cudaMemcpy(dm_matrix_g,
                         dm_matrix_h,
                         lgd * lgd * sizeof(double),
                         cudaMemcpyHostToDevice));

    for (int i = 0; i < gridt.nstreams; i++)
    {
        checkCuda(cudaStreamSynchronize(gridt.streams[i]));
    }

    // calculate the rho for every nbz bigcells
    int iter_num = 0;
    for (int i = 0; i < gridt.nbx; i++)
    {
        for (int j = 0; j < gridt.nby; j++)
        {
            // get stream id
            int stream_num = iter_num % gridt.nstreams;

            // psi_input contains data used to generate the psi values.
            // The suffix "_g" indicates that the data is stored in the GPU,
            // otherwise it is stored in the host.
            double* input_double
                = &gridt.psi_dbl_gbl[gridt.psi_size_max * stream_num * 5];
            int* input_int
                = &gridt.psi_int_gbl[gridt.psi_size_max * stream_num * 2];
            double* input_double_g
                = &gridt.psi_dbl_gbl_g[gridt.psi_size_max * stream_num * 5];
            int* input_int_g
                = &gridt.psi_int_gbl_g[gridt.psi_size_max * stream_num * 2];

            // num_psir represents the number of atoms in each bigcell.
            int* num_psir = &gridt.num_psir_gbl[nbz * stream_num];
            int* num_psir_g = &gridt.num_psir_gbl_g[nbz * stream_num];

            // ap_alpha represents the coefficient alpha in the
            // expression alpha * mat_DM * mat_psir.
            double* ap_alpha
                = &gridt.alpha_global[gridt.atom_pair_nbz * stream_num];
            double* ap_alpha_g
                = &gridt.alpha_global_g[gridt.atom_pair_nbz * stream_num];

            // m, n, k, lda, ldb, ldc in matrix multiplication
            int* atom_pair_A_m
                = &gridt.l_info_global[gridt.atom_pair_nbz * stream_num];
            int* atom_pair_B_n
                = &gridt.r_info_global[gridt.atom_pair_nbz * stream_num];
            int* atom_pair_k
                = &gridt.k_info_global[gridt.atom_pair_nbz * stream_num];
            int* atom_pair_lda
                = &gridt.lda_info_global[gridt.atom_pair_nbz * stream_num];
            int* atom_pair_ldb
                = &gridt.ldb_info_global[gridt.atom_pair_nbz * stream_num];
            int* atom_pair_ldc
                = &gridt.ldc_info_global[gridt.atom_pair_nbz * stream_num];

            int* atom_pair_A_m_g
                = &gridt.l_info_global_g[gridt.atom_pair_nbz * stream_num];
            int* atom_pair_B_n_g
                = &gridt.r_info_global_g[gridt.atom_pair_nbz * stream_num];
            int* atom_pair_k_g
                = &gridt.k_info_global_g[gridt.atom_pair_nbz * stream_num];
            int* atom_pair_lda_g
                = &gridt.lda_info_gbl_g[gridt.atom_pair_nbz * stream_num];
            int* atom_pair_ldb_g
                = &gridt.ldb_info_gbl_g[gridt.atom_pair_nbz * stream_num];
            int* atom_pair_ldc_g
                = &gridt.ldc_info_gbl_g[gridt.atom_pair_nbz * stream_num];

            // matrix A, B, C used in matrix multiplication
            double** matrix_A
                = &gridt.ap_left_gbl[gridt.atom_pair_nbz * stream_num];
            double** matrix_B
                = &gridt.ap_right_gbl[gridt.atom_pair_nbz * stream_num];
            double** matrix_C
                = &gridt.ap_output_gbl[gridt.atom_pair_nbz * stream_num];

            double** matrix_A_g
                = &gridt.ap_left_gbl_g[gridt.atom_pair_nbz * stream_num];
            double** matrix_B_g
                = &gridt.ap_right_gbl_g[gridt.atom_pair_nbz * stream_num];
            double** matrix_C_g
                = &gridt.ap_output_gbl_g[gridt.atom_pair_nbz * stream_num];

            // psir_ylm_left_g is used to store the psi values.
            // psir_r_g is used to store psir_dm, which is the product
            // of mat_DM * mat_psir.
            double* psir_ylm_left_g
                = &gridt.left_global_g[gridt.psir_size * stream_num];
            double* psir_r_g
                = &gridt.right_global_g[gridt.psir_size * stream_num];
            double* rho_g = gridt.rho_g;

            // variables for dot product psir * psir_dm
            int dot_count = 0;
            int* vec_len = &gridt.vec_len[gridt.num_mcell * stream_num];
            double** vec_l = &gridt.vec_l[gridt.num_mcell * stream_num];
            double** vec_r = &gridt.vec_r[gridt.num_mcell * stream_num];
            double** dot_product
                = &gridt.dot_product[gridt.num_mcell * stream_num];

            int* vec_len_g = &gridt.vec_len_g[gridt.num_mcell * stream_num];
            double** vec_l_g = &gridt.vec_l_g[gridt.num_mcell * stream_num];
            double** vec_r_g = &gridt.vec_r_g[gridt.num_mcell * stream_num];
            double** dot_product_g
                = &gridt.dot_product_g[gridt.num_mcell * stream_num];

            int max_m = 0;
            int max_n = 0;
            int atom_pair_num = 0;

            checkCuda(cudaStreamSynchronize(gridt.streams[stream_num]));

            // generate GPU tasks, including the calculation of psir, matrix
            // multiplication, and dot product
            gtask_rho(gridt,
                      i,
                      j,
                      max_size,
                      nczp,
                      ucell,
                      rcut,
                      input_double,
                      input_int,
                      num_psir,
                      lgd,
                      psir_ylm_left_g,
                      psir_r_g,
                      dm_matrix_g,
                      ap_alpha,
                      atom_pair_A_m,
                      atom_pair_B_n,
                      atom_pair_k,
                      atom_pair_lda,
                      atom_pair_ldb,
                      atom_pair_ldc,
                      matrix_A,
                      matrix_B,
                      matrix_C,
                      max_m,
                      max_n,
                      atom_pair_num,
                      rho_g,
                      vec_l,
                      vec_r,
                      dot_product,
                      vec_len,
                      dot_count);

            // Copying data from host to device
            checkCuda(cudaMemcpyAsync(input_double_g,
                                      input_double,
                                      gridt.psi_size_max * 5 * sizeof(double),
                                      cudaMemcpyHostToDevice,
                                      gridt.streams[stream_num]));
            checkCuda(cudaMemcpyAsync(input_int_g,
                                      input_int,
                                      gridt.psi_size_max * 2 * sizeof(int),
                                      cudaMemcpyHostToDevice,
                                      gridt.streams[stream_num]));
            checkCuda(cudaMemcpyAsync(num_psir_g,
                                      num_psir,
                                      nbz * sizeof(int),
                                      cudaMemcpyHostToDevice,
                                      gridt.streams[stream_num]));

            checkCuda(cudaMemcpyAsync(ap_alpha_g,
                                      ap_alpha,
                                      gridt.atom_pair_nbz * sizeof(double),
                                      cudaMemcpyHostToDevice,
                                      gridt.streams[stream_num]));
            checkCuda(cudaMemcpyAsync(atom_pair_A_m_g,
                                      atom_pair_A_m,
                                      gridt.atom_pair_nbz * sizeof(int),
                                      cudaMemcpyHostToDevice,
                                      gridt.streams[stream_num]));
            checkCuda(cudaMemcpyAsync(atom_pair_B_n_g,
                                      atom_pair_B_n,
                                      gridt.atom_pair_nbz * sizeof(int),
                                      cudaMemcpyHostToDevice,
                                      gridt.streams[stream_num]));
            checkCuda(cudaMemcpyAsync(atom_pair_k_g,
                                      atom_pair_k,
                                      gridt.atom_pair_nbz * sizeof(int),
                                      cudaMemcpyHostToDevice,
                                      gridt.streams[stream_num]));
            checkCuda(cudaMemcpyAsync(atom_pair_lda_g,
                                      atom_pair_lda,
                                      gridt.atom_pair_nbz * sizeof(int),
                                      cudaMemcpyHostToDevice,
                                      gridt.streams[stream_num]));
            checkCuda(cudaMemcpyAsync(atom_pair_ldb_g,
                                      atom_pair_ldb,
                                      gridt.atom_pair_nbz * sizeof(int),
                                      cudaMemcpyHostToDevice,
                                      gridt.streams[stream_num]));
            checkCuda(cudaMemcpyAsync(atom_pair_ldc_g,
                                      atom_pair_ldc,
                                      gridt.atom_pair_nbz * sizeof(int),
                                      cudaMemcpyHostToDevice,
                                      gridt.streams[stream_num]));

            checkCuda(cudaMemcpyAsync(matrix_A_g,
                                      matrix_A,
                                      gridt.atom_pair_nbz * sizeof(double*),
                                      cudaMemcpyHostToDevice,
                                      gridt.streams[stream_num]));
            checkCuda(cudaMemcpyAsync(matrix_B_g,
                                      matrix_B,
                                      gridt.atom_pair_nbz * sizeof(double*),
                                      cudaMemcpyHostToDevice,
                                      gridt.streams[stream_num]));
            checkCuda(cudaMemcpyAsync(matrix_C_g,
                                      matrix_C,
                                      gridt.atom_pair_nbz * sizeof(double*),
                                      cudaMemcpyHostToDevice,
                                      gridt.streams[stream_num]));

            checkCuda(cudaMemcpyAsync(vec_len_g,
                                      vec_len,
                                      gridt.num_mcell * sizeof(int),
                                      cudaMemcpyHostToDevice,
                                      gridt.streams[stream_num]));
            checkCuda(cudaMemcpyAsync(vec_l_g,
                                      vec_l,
                                      gridt.num_mcell * sizeof(double*),
                                      cudaMemcpyHostToDevice,
                                      gridt.streams[stream_num]));
            checkCuda(cudaMemcpyAsync(vec_r_g,
                                      vec_r,
                                      gridt.num_mcell * sizeof(double*),
                                      cudaMemcpyHostToDevice,
                                      gridt.streams[stream_num]));
            checkCuda(cudaMemcpyAsync(dot_product_g,
                                      dot_product,
                                      gridt.num_mcell * sizeof(double*),
                                      cudaMemcpyHostToDevice,
                                      gridt.streams[stream_num]));

            checkCuda(cudaMemsetAsync(psir_ylm_left_g,
                                      0,
                                      gridt.psir_size * sizeof(double),
                                      gridt.streams[stream_num]));
            checkCuda(cudaMemsetAsync(psir_r_g,
                                      0,
                                      gridt.psir_size * sizeof(double),
                                      gridt.streams[stream_num]));

            // Launching kernel to calculate psi
            dim3 grid_psi(nbz, 8);
            dim3 block_psi(64);
            get_psi<<<grid_psi, block_psi, 0, gridt.streams[stream_num]>>>(
                gridt.ylmcoef_g,
                dr,
                gridt.bxyz,
                ucell.nwmax,
                input_double_g,
                input_int_g,
                num_psir_g,
                gridt.psi_size_max_z,
                gridt.atom_nwl_g,
                gridt.atom_new_g,
                gridt.atom_ylm_g,
                gridt.atom_nw_g,
                gridt.nr_max,
                gridt.psi_u_g,
                psir_ylm_left_g);
            checkCudaLastError();

            // Performing matrix multiplication alpha * mat_dm * mat_psir
            gridt.fastest_matrix_mul(max_m,
                                     max_n,
                                     atom_pair_A_m_g,
                                     atom_pair_B_n_g,
                                     atom_pair_k_g,
                                     matrix_A_g,
                                     atom_pair_lda_g,
                                     matrix_B_g,
                                     atom_pair_ldb_g,
                                     matrix_C_g,
                                     atom_pair_ldc_g,
                                     atom_pair_num,
                                     gridt.streams[stream_num],
                                     ap_alpha_g);

            // Launching kernel to calculate dot product psir * psir_dm
            dim3 grid_dot(64);
            dim3 block_dot(64);
            int incx = 1;
            int incy = 1;
            psir_dot<<<grid_dot, block_dot, 0, gridt.streams[stream_num]>>>(
                vec_len_g,
                vec_l_g,
                incx,
                vec_r_g,
                incy,
                dot_product_g,
                dot_count);

            iter_num++;
        }
    }

    // Synchronizing streams
    for (int i = 0; i < gridt.nstreams; i++)
    {
        checkCuda(cudaStreamSynchronize(gridt.streams[i]));
    }

    // Copy rho from device to host
    checkCuda(cudaMemcpy(rho,
                         gridt.rho_g,
                         nczp * gridt.ncx * gridt.ncy * sizeof(double),
                         cudaMemcpyDeviceToHost));

    // free the memory
    checkCuda(cudaFree(dm_matrix_g));
    delete[] dm_matrix_h;
}

} // namespace GintKernel
