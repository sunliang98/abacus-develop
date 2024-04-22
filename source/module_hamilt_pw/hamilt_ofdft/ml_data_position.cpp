#include "ml_data.h"

void ML_data::get_r_matrix(const UnitCell& ucell,
                           ModulePW::PW_Basis* pw_rho,
                           const double rcut,
                           const int n_max,
                           std::vector<std::vector<std::vector<double>>>& r_matrix)
{
    double* height = new double[3];
    height[0] = ucell.omega / ((ucell.a2 ^ ucell.a3).norm() * ucell.lat0 * ucell.lat0);
    height[1] = ucell.omega / ((ucell.a3 ^ ucell.a1).norm() * ucell.lat0 * ucell.lat0);
    height[2] = ucell.omega / ((ucell.a1 ^ ucell.a2).norm() * ucell.lat0 * ucell.lat0);

    std::cout << "volume: " << ucell.omega << std::endl;
    std::cout << "Height: " << height[0] << ", " << height[1] << ", " << height[2] << std::endl;
    int* search_range = new int[3];
    for (int i = 0; i < 3; ++i)
    {
        search_range[i] = std::ceil(rcut / height[i]);
    }
    std::cout << "Search range: " << search_range[0] << ", " << search_range[1] << ", " << search_range[2] << std::endl;

    ModuleBase::Vector3<double> r_cartesian;
    ModuleBase::Vector3<double> r_direct;
    ModuleBase::Vector3<double> move_direct;
    ModuleBase::Vector3<double> R_row;
    bool found = false;
    int n_found = 0;
    int found_max = 0;

    for (int ix = 0; ix < pw_rho->nx; ++ix)
    {
        r_direct[0] = 1. / pw_rho->nx * ix;
        for (int iy = 0; iy < pw_rho->ny; ++iy)
        {
            r_direct[1] = 1. / pw_rho->ny * iy;
            for (int iz = 0; iz < pw_rho->nz; ++iz)
            {
                r_direct[2] = 1. / pw_rho->nz * iz;
                r_cartesian = r_direct * ucell.latvec * ucell.lat0;
                n_found = 0;

                for (int it = 0; it < ucell.ntype; ++it)
                {
                    Atom* atom = &ucell.atoms[it];
                    for (int ia = 0; ia < atom->na; ++ia)
                    {
                        for (int i = -search_range[0]; i < search_range[0] + 1; ++i)
                        {
                            move_direct[0] = i;
                            for (int j = -search_range[1]; j < search_range[1] + 1; ++j)
                            {
                                move_direct[1] = j;
                                found = false;
                                for (int k = -search_range[2]; k < search_range[2] + 1; ++k)
                                {
                                    move_direct[2] = k;
                                    R_row = move_direct * ucell.latvec * ucell.lat0 + atom->tau[ia] * ucell.lat0
                                            - r_cartesian;
                                    double R_row_norm = R_row.norm();
                                    if (R_row_norm <= rcut)
                                    {
                                        if (n_found >= n_max)
                                        {
                                            std::cout << "Warning: n_found >= n_max" << std::endl;
                                            break;
                                        }
                                        double s = this->soft(R_row_norm, rcut);
                                        int r_index = ix * pw_rho->ny * pw_rho->nz + iy * pw_rho->nz + iz;
                                        r_matrix[r_index][n_found][0] = s;
                                        if (R_row_norm == 0)
                                        {
                                            for (int nn = 0; nn < 3; ++nn)
                                            {
                                                r_matrix[r_index][n_found][nn + 1] = 0;
                                            }
                                        }
                                        else
                                        {
                                            for (int nn = 0; nn < 3; ++nn)
                                            {
                                                r_matrix[r_index][n_found][nn + 1] = s * R_row[nn] / R_row_norm;
                                            }
                                        }
                                        n_found++;
                                        found = true;
                                    }
                                    else if (found)
                                    {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
                if (n_found > found_max)
                {
                    found_max = n_found;
                }
            }
        }
    }
    std::cout << "found max = " << found_max << std::endl;

    delete[] height;
    delete[] search_range;
}

double ML_data::soft(const double norm, const double r_cut)
{
    double u = norm / r_cut;
    if (u <= 1)
    {
        return std::pow(u, 3.) * (-6. * std::pow(u, 2.) + 15. * u - 10.) + 1.;
    }
    else
    {
        return 0.;
    }
}
