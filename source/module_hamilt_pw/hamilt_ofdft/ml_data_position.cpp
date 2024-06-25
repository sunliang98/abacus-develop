#include "ml_data.h"

void ML_data::get_r_matrix(const UnitCell& ucell,
                           ModulePW::PW_Basis* pw_rho,
                           std::vector<double>& r_min_vector)
{
    int* search_range = new int[3];
    for (int i = 0; i < 3; ++i)
    {
        search_range[i] = 1;
    }
    // std::cout << "Search range: " << search_range[0] << ", " << search_range[1] << ", " << search_range[2] << std::endl;

    double r0 = std::pow(ucell.omega / ucell.nat, 1. / 3.);
    std::cout << "r0 = " << r0 << std::endl;

    ModuleBase::Vector3<double> r_cartesian;
    ModuleBase::Vector3<double> r_direct;
    ModuleBase::Vector3<double> move_direct;
    ModuleBase::Vector3<double> R_row;
    bool found = false;

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
                double r_min = 100000.;

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
                                    if (R_row_norm < r_min)
                                    {
                                        r_min = R_row_norm;
                                        found = true;
                                    }
                                    // else if (found)
                                    // {
                                    //     break;
                                    // }
                                }
                            }
                        }
                    }
                }
                int r_index = ix * pw_rho->ny * pw_rho->nz + iy * pw_rho->nz + iz;
                r_min_vector[r_index] = r_min / r0;
            }
        }
    }

    delete[] search_range;
}