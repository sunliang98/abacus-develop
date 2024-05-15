#include "gint_rho.h"
#include "module_base/ylm.h"
#include "module_hamilt_lcao/module_gint/gint_tools.h"
#include "omp.h"
namespace GintKernel
{

void gtask_rho(const Grid_Technique& gridt,
               const int i,
               const int j,
               const int max_size,
               const int nczp,
               const UnitCell& ucell,
               const double* rcut,
               double* input_double,
               int* input_int,
               int* num_psir,
               const int lgd,
               double* const psir_ylm_g,
               double* const psir_dm_g,
               double* const dm_matrix_g,
               double* mat_alpha,
               int* mat_m,
               int* mat_n,
               int* mat_k,
               int* mat_lda,
               int* mat_ldb,
               int* mat_ldc,
               double** mat_A,
               double** mat_B,
               double** mat_C,
               int& max_m,
               int& max_n,
               int& atom_pair_num,
               double* rho_g,
               double** vec_l,
               double** vec_r,
               double** dot_product,
               int* vec_len,
               int& dot_count)
{
    const int grid_index_ij = i * gridt.nby * gridt.nbzp + j * gridt.nbzp;
    const int nwmax = ucell.nwmax;
    const int psi_size_max = max_size * gridt.bxyz;

    // record whether mat_psir is a zero matrix or not.
    bool* gpu_mat_cal_flag = new bool[max_size * gridt.nbzp];

    for (int i = 0; i < max_size * gridt.nbzp; i++)
    {
        gpu_mat_cal_flag[i] = false;
    }
    dot_count = 0;

    // generate data for calculating psir
    for (int z_index = 0; z_index < gridt.nbzp; z_index++)
    {
        int num_get_psi = 0;
        int grid_index = grid_index_ij + z_index;
        int num_psi_pos = psi_size_max * z_index;
        int calc_flag_index = max_size * z_index;
        int bcell_start_index = gridt.bcell_start[grid_index];
        int na_grid = gridt.how_many_atoms[grid_index];

        for (int id = 0; id < na_grid; id++)
        {
            int ib = 0;
            int mcell_index = bcell_start_index + id;
            int imcell = gridt.which_bigcell[mcell_index];
            int iat = gridt.which_atom[mcell_index];
            int it_temp = ucell.iat2it[iat];
            int start_ind_grid = gridt.start_ind[grid_index];

            for (int bx_index = 0; bx_index < gridt.bx; bx_index++)
            {
                for (int by_index = 0; by_index < gridt.by; by_index++)
                {
                    for (int bz_index = 0; bz_index < gridt.bz; bz_index++)
                    {
                        double dr_temp[3];
                        dr_temp[0] = gridt.meshcell_pos[ib][0]
                                     + gridt.meshball_positions[imcell][0]
                                     - gridt.tau_in_bigcell[iat][0];
                        dr_temp[1] = gridt.meshcell_pos[ib][1]
                                     + gridt.meshball_positions[imcell][1]
                                     - gridt.tau_in_bigcell[iat][1];
                        dr_temp[2] = gridt.meshcell_pos[ib][2]
                                     + gridt.meshball_positions[imcell][2]
                                     - gridt.tau_in_bigcell[iat][2];

                        double distance = sqrt(dr_temp[0] * dr_temp[0]
                                               + dr_temp[1] * dr_temp[1]
                                               + dr_temp[2] * dr_temp[2]);
                        if (distance <= rcut[it_temp])
                        {
                            gpu_mat_cal_flag[calc_flag_index + id] = true;
                            int pos_temp_double = num_psi_pos + num_get_psi;
                            int pos_temp_int = pos_temp_double * 2;
                            pos_temp_double *= 5;
                            if (distance < 1.0E-9)
                            {
                                distance += 1.0E-9;
                            }
                            input_double[pos_temp_double]
                                = dr_temp[0] / distance;
                            input_double[pos_temp_double + 1]
                                = dr_temp[1] / distance;
                            input_double[pos_temp_double + 2]
                                = dr_temp[2] / distance;
                            input_double[pos_temp_double + 3] = distance;

                            input_int[pos_temp_int] = it_temp; // atom type
                            input_int[pos_temp_int + 1]
                                = (z_index * gridt.bxyz + ib) * max_size * nwmax
                                  + id * nwmax; // psir index in psir_ylm
                            num_get_psi++;
                        }
                        ib++;
                    }
                }
            }
        }
        num_psir[z_index] = num_get_psi;
    }

    int tid = 0;
    max_m = 0;
    max_n = 0;

    // generate matrix multiplication tasks
    for (int z_index = 0; z_index < gridt.nbzp; z_index++)
    {
        int grid_index = grid_index_ij + z_index;
        int calc_flag_index = max_size * z_index;
        int bcell_start_index = gridt.bcell_start[grid_index];
        int bcell_start_psir = z_index * gridt.bxyz * max_size * nwmax;

        for (int atom1 = 0; atom1 < gridt.how_many_atoms[grid_index]; atom1++)
        {
            if (!gpu_mat_cal_flag[calc_flag_index + atom1])
            {
                continue;
            }
            int mcell_index1 = bcell_start_index + atom1;
            int iat1 = gridt.which_atom[mcell_index1];
            int it1 = ucell.iat2it[iat1];
            int lo1
                = gridt.trace_lo[ucell.itiaiw2iwt(it1, ucell.iat2ia[iat1], 0)];
            int nw1 = ucell.atoms[it1].nw;

            for (int atom2 = atom1; atom2 < gridt.how_many_atoms[grid_index];
                 atom2++)
            {
                if (!gpu_mat_cal_flag[calc_flag_index + atom2])
                {
                    continue;
                }
                int mcell_index2 = bcell_start_index + atom2;
                int iat2 = gridt.which_atom[mcell_index2];
                int it2 = ucell.iat2it[iat2];
                int lo2 = gridt.trace_lo[ucell.itiaiw2iwt(it2,
                                                          ucell.iat2ia[iat2],
                                                          0)];
                int nw2 = ucell.atoms[it2].nw;

                int mat_A_idx = bcell_start_psir + atom2 * nwmax;
                int mat_B_idx = lgd * lo1 + lo2;
                int mat_C_idx = bcell_start_psir + atom1 * nwmax;

                mat_alpha[tid] = atom2 == atom1 ? 1 : 2;
                mat_m[tid] = gridt.bxyz;
                mat_n[tid] = nw1;
                mat_k[tid] = nw2;
                mat_lda[tid] = nwmax * max_size;
                mat_ldb[tid] = lgd;
                mat_ldc[tid] = nwmax * max_size;
                mat_A[tid] = psir_ylm_g + mat_A_idx;
                mat_B[tid] = dm_matrix_g + mat_B_idx;
                mat_C[tid] = psir_dm_g + mat_C_idx;

                if (mat_m[tid] > max_m)
                {
                    max_m = mat_m[tid];
                }

                if (mat_n[tid] > max_n)
                {
                    max_n = mat_n[tid];
                }

                tid++;
            }
        }

        // generate vec dot product tasks
        int* vindex = Gint_Tools::get_vindex(gridt.bxyz,
                                             gridt.bx,
                                             gridt.by,
                                             gridt.bz,
                                             nczp,
                                             gridt.start_ind[grid_index],
                                             gridt.ncy * nczp);
        for (int i = 0; i < gridt.bxyz; i++)
        {
            vec_l[dot_count]
                = psir_ylm_g + (bcell_start_psir + i * max_size * nwmax);
            vec_r[dot_count]
                = psir_dm_g + (bcell_start_psir + i * max_size * nwmax);
            dot_product[dot_count] = rho_g + vindex[i];
            vec_len[dot_count] = nwmax * max_size;
            dot_count++;
        }
    }
    atom_pair_num = tid;

    delete[] gpu_mat_cal_flag;
}

} // namespace GintKernel