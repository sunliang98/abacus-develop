//wenfei 2022-1-11
//This file contains subroutines for calculating pdm,
#include "module_parameter/parameter.h"
//which is defind as sum_mu,nu rho_mu,nu (<chi_mu|alpha><alpha|chi_nu>);
//as well as gdmx, which is the gradient of pdm, defined as
//sum_mu,nu rho_mu,nu d/dX(<chi_mu|alpha><alpha|chi_nu>)

//It also contains subroutines for printing pdm and gdmx
//for checking purpose

//There are 6 subroutines in this file:
//1. cal_projected_DM, which is used for calculating pdm for gamma point calculation
//2. cal_projected_DM_k, counterpart of 1, for multi-k
//3. check_projected_dm, which prints pdm to descriptor.dat

//4. cal_gdmx, calculating gdmx (and optionally gdm_epsl for stress) for gamma point
//5. cal_gdmx_k, counterpart of 3, for multi-k
//6. check_gdmx, which prints gdmx to a series of .dat files

#ifdef __DEEPKS

#include "LCAO_deepks.h"
#include "module_base/vector3.h"
#include "module_base/timer.h"
#include "module_base/constants.h"
#include "module_hamilt_lcao/module_hcontainer/atom_pair.h"
#include "module_base/libm/libm.h"

void LCAO_Deepks::read_projected_DM(bool read_pdm_file, bool is_equiv, const Numerical_Orbital& alpha)
{
    if (read_pdm_file && !this->init_pdm) //for DeePKS NSCF calculation 
    {
        int pdm_size = 0;
        if(!is_equiv)
        {
            pdm_size = (this->lmaxd * 2 + 1) * (this->lmaxd * 2 + 1);
        }
        else
        {
            int nproj = 0;
            for(int il = 0; il < this->lmaxd + 1; il++)
            {
                nproj += (2 * il + 1) * alpha.getNchi(il);
            }
            pdm_size = nproj * nproj;
        }

        const std::string file_projdm = PARAM.globalv.global_out_dir + "deepks_projdm.dat";
        std::ifstream ifs(file_projdm.c_str());

        if (!ifs)
        {
            ModuleBase::WARNING_QUIT("LCAO_Deepks::cal_projected_DM", "Cannot find the file deepks_projdm.dat");
        }
        for(int inl=0;inl<this->inlmax;inl++)
        {
            for(int ind=0;ind<pdm_size;ind++)
            {
                double c;
			    ifs >> c;
                pdm[inl][ind] = c;
            }
        }
        this->init_pdm = true;
    }
}

//this subroutine performs the calculation of projected density matrices
//pdm_m,m'=\sum_{mu,nu} rho_{mu,nu} <chi_mu|alpha_m><alpha_m'|chi_nu>
void LCAO_Deepks::cal_projected_DM(const elecstate::DensityMatrix<double, double>* dm, 
    const UnitCell &ucell,
    const LCAO_Orbitals &orb,
    Grid_Driver& GridD)
{
    ModuleBase::TITLE("LCAO_Deepks", "cal_projected_DM");

    // if pdm has been initialized, skip the calculation
    if(this->init_pdm)
    {
        this->init_pdm = false;
        return;
    }
    int pdm_size = 0;
    if(!PARAM.inp.deepks_equiv)
    {
        pdm_size = (this->lmaxd * 2 + 1) * (this->lmaxd * 2 + 1);
    }
    else
    {
        int nproj = 0;
        for(int il = 0; il < this->lmaxd + 1; il++)
        {
            nproj += (2 * il + 1) * orb.Alpha[0].getNchi(il);
        }
        pdm_size = nproj * nproj;
    }

    //if(dm.size() == 0 || dm[0].size() == 0)
    //{
    //    return;
    //}
    ModuleBase::timer::tick("LCAO_Deepks","cal_projected_DM");

    for(int inl=0;inl<inlmax;inl++)
    {
        ModuleBase::GlobalFunc::ZEROS(pdm[inl],pdm_size);
    }

    const double Rcut_Alpha = orb.Alpha[0].getRcut();
    int nrow = this->pv->nrow;
    for (int T0 = 0; T0 < ucell.ntype; T0++)
    {
		Atom* atom0 = &ucell.atoms[T0]; 
        for (int I0 =0; I0< atom0->na; I0++)
        {
            const int iat = ucell.itia2iat(T0,I0);
            const ModuleBase::Vector3<double> tau0 = atom0->tau[I0];
            GridD.Find_atom(ucell, atom0->tau[I0] ,T0, I0);

            //trace alpha orbital
            std::vector<int> trace_alpha_row;
            std::vector<int> trace_alpha_col;
            if(!PARAM.inp.deepks_equiv)
            {
                int ib=0;
                for (int L0 = 0; L0 <= orb.Alpha[0].getLmax();++L0)
                {
                    for (int N0 = 0;N0 < orb.Alpha[0].getNchi(L0);++N0)
                    {
                        const int inl = this->inl_index[T0](I0, L0, N0);
                        const int nm = 2*L0+1;
                
                        for (int m1=0; m1<nm; ++m1) // m1 = 1 for s, 3 for p, 5 for d
                        {
                            for (int m2=0; m2<nm; ++m2) // m1 = 1 for s, 3 for p, 5 for d
                            {
                                trace_alpha_row.push_back(ib+m1);
                                trace_alpha_col.push_back(ib+m2);
                            }
                        }
                        ib+=nm;
                    }
                }
            }
            else
            {
                int nproj = 0;
                for(int il = 0; il < this->lmaxd + 1; il++)
                {
                    nproj += (2 * il + 1) * orb.Alpha[0].getNchi(il);
                }
                for(int iproj = 0; iproj < nproj; iproj ++)
                {
                    for(int jproj = 0; jproj < nproj; jproj ++)
                    {
                        trace_alpha_row.push_back(iproj);
                        trace_alpha_col.push_back(jproj);
                    }
                }
            }
            const int trace_alpha_size = trace_alpha_row.size();

            for (int ad1=0; ad1<GridD.getAdjacentNum()+1 ; ++ad1)
            {
                const int T1 = GridD.getType(ad1);
                const int I1 = GridD.getNatom(ad1);
                const int ibt1 = ucell.itia2iat(T1,I1);
                const ModuleBase::Vector3<double> tau1 = GridD.getAdjacentTau(ad1);
				const Atom* atom1 = &ucell.atoms[T1];
				const int nw1_tot = atom1->nw*PARAM.globalv.npol;
				const double Rcut_AO1 = orb.Phi[T1].getRcut(); 
                const double dist1 = (tau1-tau0).norm() * ucell.lat0;
                if (dist1 >= Rcut_Alpha + Rcut_AO1)
                {
                    continue;
                }

                auto row_indexes = pv->get_indexes_row(ibt1);
                const int row_size = row_indexes.size();
                if(row_size == 0) { continue;
}

                // no possible to unexist key
                std::vector<double> s_1t(trace_alpha_size * row_size);
                std::vector<double> g_1dmt(trace_alpha_size * row_size, 0.0);
                for(int irow=0;irow<row_size;irow++)
                {
                    const double* row_ptr = this->nlm_save[iat][ad1][row_indexes[irow]][0].data();
                    
                    for(int i=0;i<trace_alpha_size;i++)
                    {
                        s_1t[i * row_size + irow] = row_ptr[trace_alpha_row[i]];
                    }
                }

				for (int ad2=0; ad2 < GridD.getAdjacentNum()+1 ; ad2++)
				{
					const int T2 = GridD.getType(ad2);
					const int I2 = GridD.getNatom(ad2);
                    const int ibt2 = ucell.itia2iat(T2,I2);
					const ModuleBase::Vector3<double> tau2 = GridD.getAdjacentTau(ad2);
					const Atom* atom2 = &ucell.atoms[T2];
					const int nw2_tot = atom2->nw*PARAM.globalv.npol;
					
					const double Rcut_AO2 = orb.Phi[T2].getRcut();
                	const double dist2 = (tau2-tau0).norm() * ucell.lat0;

					if (dist2 >= Rcut_Alpha + Rcut_AO2)
					{
						continue;
					}

                    auto col_indexes = pv->get_indexes_col(ibt2);
                    const int col_size = col_indexes.size();
                    if(col_size == 0) { continue;
}

                    std::vector<double> s_2t(trace_alpha_size * col_size);
                    // no possible to unexist key
                    for(int icol=0;icol<col_size;icol++)
                    {
                        const double* col_ptr = this->nlm_save[iat][ad2][col_indexes[icol]][0].data();
                        for(int i=0;i<trace_alpha_size;i++)
                        {
                            s_2t[i * col_size + icol] = col_ptr[trace_alpha_col[i]];
                        }
                    }
                    // prepare DM_gamma from DMR
                    std::vector<double> dm_array(row_size*col_size, 0.0);
                    const double* dm_current = nullptr;
                    for(int is=0;is<dm->get_DMR_vector().size();is++)
                    {
                        auto* tmp = dm->get_DMR_vector()[is]->find_matrix(ibt1, ibt2, 0, 0, 0);
                        if(tmp == nullptr)
                        {
                            // in case of no deepks_scf but out_deepks_label, size of DMR would mismatch with deepks-orbitals
                            dm_current = nullptr; 
                            break;
                        }
                        dm_current = tmp->get_pointer();
                        for(int idm=0;idm<row_size*col_size;idm++)
                        {
                            dm_array[idm] += dm_current[idm];
                        }
                    }
                    if(dm_current == nullptr) { continue; //skip the long range DM pair more than nonlocal term
}
                    dm_current = dm_array.data();
                    //dgemm for s_2t and dm_current to get g_1dmt
                    constexpr char transa='T', transb='N';
                    const double gemm_alpha = 1.0, gemm_beta = 1.0;
                    dgemm_(
                        &transa, &transb, 
                        &row_size, 
                        &trace_alpha_size, 
                        &col_size, 
                        &gemm_alpha, 
                        dm_current, 
                        &col_size,
                        s_2t.data(),  
                        &col_size,     
                        &gemm_beta,      
                        g_1dmt.data(),    
                        &row_size);
				}//ad2
                // do dot of g_1dmt and s_1t to get orbital_pdm_shell
                if(!PARAM.inp.deepks_equiv)
                {
                    int ib=0, index=0, inc=1;
                    for (int L0 = 0; L0 <= orb.Alpha[0].getLmax();++L0)
                    {
                        for (int N0 = 0;N0 < orb.Alpha[0].getNchi(L0);++N0)
                        {
                            const int inl = this->inl_index[T0](I0, L0, N0);
                            const int nm = 2*L0+1;
                    
                            for (int m1=0; m1<nm; ++m1) // m1 = 1 for s, 3 for p, 5 for d
                            {
                                for (int m2=0; m2<nm; ++m2) // m1 = 1 for s, 3 for p, 5 for d
                                {
                                    int ind = m1*nm + m2;
                                    pdm[inl][ind] += 
                                        ddot_(&row_size, g_1dmt.data()+index*row_size, &inc, s_1t.data()+index*row_size, &inc);
                                    index++;
                                }
                            }
                            ib+=nm;
                        }
                    }
                }
                else
                {
                    int index = 0, inc=1;
                    int nproj = 0;
                    for(int il = 0; il < this->lmaxd + 1; il++)
                    {
                        nproj += (2 * il + 1) * orb.Alpha[0].getNchi(il);
                    }
                    for(int iproj = 0; iproj < nproj; iproj ++)
                    {
                        for(int jproj = 0; jproj < nproj; jproj ++)
                        {
                            pdm[iat][iproj * nproj + jproj] += 
                                ddot_(&row_size, g_1dmt.data()+index*row_size, &inc, s_1t.data()+index*row_size, &inc);
                            index ++;
                        }
                    }
                }
			}//ad1
        }//I0
    }//T0

#ifdef __MPI
    allsum_deepks(this->inlmax,pdm_size,this->pdm);
#endif
    ModuleBase::timer::tick("LCAO_Deepks","cal_projected_DM"); 
    return;
}

void LCAO_Deepks::cal_projected_DM_k(const elecstate::DensityMatrix<std::complex<double>, double>* dm,
    const UnitCell &ucell,
    const LCAO_Orbitals &orb,
    Grid_Driver& GridD)
{
    // if pdm has been initialized, skip the calculation
    if(this->init_pdm)
    {
        this->init_pdm = false;
        return;
    }

    int pdm_size = 0;
    if(!PARAM.inp.deepks_equiv)
    {
        pdm_size = (this->lmaxd * 2 + 1) * (this->lmaxd * 2 + 1);
    }
    else
    {
        int nproj = 0;
        for(int il = 0; il < this->lmaxd + 1; il++)
        {
            nproj += (2 * il + 1) * orb.Alpha[0].getNchi(il);
        }
        pdm_size = nproj * nproj;        
    }

    //check for skipping
    //if(dm.size() == 0 || dm[0].size() == 0)
    //{
    //    return;
    //}
    ModuleBase::timer::tick("LCAO_Deepks","cal_projected_DM_k");

    for(int inl=0;inl<inlmax;inl++)
    {
        ModuleBase::GlobalFunc::ZEROS(pdm[inl],pdm_size);
    }

    const double Rcut_Alpha = orb.Alpha[0].getRcut();
    for (int T0 = 0; T0 < ucell.ntype; T0++)
    {
		Atom* atom0 = &ucell.atoms[T0]; 
        for (int I0 =0; I0< atom0->na; I0++)
        {
            const int iat = ucell.itia2iat(T0,I0);
            const ModuleBase::Vector3<double> tau0 = atom0->tau[I0];
            GridD.Find_atom(ucell, atom0->tau[I0] ,T0, I0);

            //trace alpha orbital
            std::vector<int> trace_alpha_row;
            std::vector<int> trace_alpha_col;
            if(!PARAM.inp.deepks_equiv)
            {
                int ib=0;
                for (int L0 = 0; L0 <= orb.Alpha[0].getLmax();++L0)
                {
                    for (int N0 = 0;N0 < orb.Alpha[0].getNchi(L0);++N0)
                    {
                        const int inl = this->inl_index[T0](I0, L0, N0);
                        const int nm = 2*L0+1;
                
                        for (int m1=0; m1<nm; ++m1) // m1 = 1 for s, 3 for p, 5 for d
                        {
                            for (int m2=0; m2<nm; ++m2) // m1 = 1 for s, 3 for p, 5 for d
                            {
                                trace_alpha_row.push_back(ib+m1);
                                trace_alpha_col.push_back(ib+m2);
                            }
                        }
                        ib+=nm;
                    }
                }
            }
            else
            {
                int nproj = 0;
                for(int il = 0; il < this->lmaxd + 1; il++)
                {
                    nproj += (2 * il + 1) * orb.Alpha[0].getNchi(il);
                }
                for(int iproj = 0; iproj < nproj; iproj ++)
                {
                    for(int jproj = 0; jproj < nproj; jproj ++)
                    {
                        trace_alpha_row.push_back(iproj);
                        trace_alpha_col.push_back(jproj);
                    }
                }                
            }
            const int trace_alpha_size = trace_alpha_row.size();

            for (int ad1=0; ad1<GridD.getAdjacentNum()+1 ; ++ad1)
            {
                const int T1 = GridD.getType(ad1);
                const int I1 = GridD.getNatom(ad1);
                const int ibt1 = ucell.itia2iat(T1,I1);
                const int start1 = ucell.itiaiw2iwt(T1, I1, 0);
                const ModuleBase::Vector3<double> tau1 = GridD.getAdjacentTau(ad1);
				const Atom* atom1 = &ucell.atoms[T1];
				const int nw1_tot = atom1->nw*PARAM.globalv.npol;
				const double Rcut_AO1 = orb.Phi[T1].getRcut();
                const double dist1 = (tau1-tau0).norm() * ucell.lat0;
                if (dist1 >= Rcut_Alpha + Rcut_AO1)
                {
                    continue;
                }

                ModuleBase::Vector3<double> dR1(GridD.getBox(ad1).x, GridD.getBox(ad1).y, GridD.getBox(ad1).z); 

                auto row_indexes = pv->get_indexes_row(ibt1);
                const int row_size = row_indexes.size();
                if(row_size == 0) { continue;
}

                key_tuple key_1(ibt1,dR1.x,dR1.y,dR1.z);
                if(this->nlm_save_k[iat].find(key_1) == this->nlm_save_k[iat].end()) { continue;
}
                std::vector<double> s_1t(trace_alpha_size * row_size);
                std::vector<double> g_1dmt(trace_alpha_size * row_size, 0.0);
                for(int irow=0;irow<row_size;irow++)
                {
                    const double* row_ptr = this->nlm_save_k[iat][key_1][row_indexes[irow]][0].data();
                    for(int i=0;i<trace_alpha_size;i++)
                    {
                        s_1t[i * row_size + irow] = row_ptr[trace_alpha_row[i]];
                    }
                }

				for (int ad2=0; ad2 < GridD.getAdjacentNum()+1 ; ad2++)
				{
					const int T2 = GridD.getType(ad2);
					const int I2 = GridD.getNatom(ad2);
                    const int ibt2 = ucell.itia2iat(T2,I2);
					const ModuleBase::Vector3<double> tau2 = GridD.getAdjacentTau(ad2);
					const Atom* atom2 = &ucell.atoms[T2];
					const int nw2_tot = atom2->nw*PARAM.globalv.npol;
                    ModuleBase::Vector3<double> dR2(GridD.getBox(ad2).x, GridD.getBox(ad2).y, GridD.getBox(ad2).z);
					
					const double Rcut_AO2 = orb.Phi[T2].getRcut();
                	const double dist2 = (tau2-tau0).norm() * ucell.lat0;

					if (dist2 >= Rcut_Alpha + Rcut_AO2)
					{
						continue;
					}

                    auto col_indexes = pv->get_indexes_col(ibt2);
                    const int col_size = col_indexes.size();
                    if(col_size == 0) { continue;
}

                    std::vector<double> s_2t(trace_alpha_size * col_size);
                    key_tuple key_2(ibt2,dR2.x,dR2.y,dR2.z);
                    if(this->nlm_save_k[iat].find(key_2) == this->nlm_save_k[iat].end()) { continue;
}
                    for(int icol=0;icol<col_size;icol++)
                    {
                        const double* col_ptr = this->nlm_save_k[iat][key_2][col_indexes[icol]][0].data();
                        for(int i=0;i<trace_alpha_size;i++)
                        {
                            s_2t[i * col_size + icol] = col_ptr[trace_alpha_col[i]];
                        }
                    }
                    // prepare DM_gamma from DMR
                    std::vector<double> dm_array(row_size*col_size, 0.0);
                    const double* dm_current;
                    for(int is=0;is<dm->get_DMR_vector().size();is++)
                    {
                        auto tmp_matrix = dm->get_DMR_vector()[is]->find_matrix(ibt1, ibt2, (dR2-dR1).x, (dR2-dR1).y, (dR2-dR1).z);
                        if(tmp_matrix == nullptr)
                        {
                            dm_current = nullptr;
                            break;
                        }
                        dm_current = tmp_matrix->get_pointer();
                        for(int idm=0;idm<row_size*col_size;idm++)
                        {
                            dm_array[idm] += dm_current[idm];
                        }
                    }
                    if(dm_current == nullptr) { continue;
}
                    dm_current = dm_array.data();
                    //dgemm for s_2t and dm_current to get g_1dmt
                    constexpr char transa='T', transb='N';
                    const double gemm_alpha = 1.0, gemm_beta = 1.0;
                    dgemm_(
                        &transa, &transb, 
                        &row_size, 
                        &trace_alpha_size, 
                        &col_size, 
                        &gemm_alpha, 
                        dm_current, 
                        &col_size,
                        s_2t.data(),  
                        &col_size,     
                        &gemm_beta,      
                        g_1dmt.data(),    
                        &row_size);
				}//ad2
                // do dot of g_1dmt and s_1t to get orbital_pdm_shell
                if(!PARAM.inp.deepks_equiv)
                {
                    int ib=0, index=0, inc=1;
                    for (int L0 = 0; L0 <= orb.Alpha[0].getLmax();++L0)
                    {
                        for (int N0 = 0;N0 < orb.Alpha[0].getNchi(L0);++N0)
                        {
                            const int inl = this->inl_index[T0](I0, L0, N0);
                            const int nm = 2*L0+1;
                    
                            for (int m1=0; m1<nm; ++m1) // m1 = 1 for s, 3 for p, 5 for d
                            {
                                for (int m2=0; m2<nm; ++m2) // m1 = 1 for s, 3 for p, 5 for d
                                {
                                    int ind = m1*nm + m2;
                                    pdm[inl][ind] += 
                                        ddot_(&row_size, g_1dmt.data()+index*row_size, &inc, s_1t.data()+index*row_size, &inc);
                                    index++;
                                }
                            }
                            ib+=nm;
                        }
                    }
                }
                else
                {
                    int index = 0, inc=1;
                    int nproj = 0;
                    for(int il = 0; il < this->lmaxd + 1; il++)
                    {
                        nproj += (2 * il + 1) * orb.Alpha[0].getNchi(il);
                    }
                    for(int iproj = 0; iproj < nproj; iproj ++)
                    {
                        for(int jproj = 0; jproj < nproj; jproj ++)
                        {
                            pdm[iat][iproj * nproj + jproj] += 
                                ddot_(&row_size, g_1dmt.data()+index*row_size, &inc, s_1t.data()+index*row_size, &inc);
                            index ++;
                        }
                    }                    
                }
			}//ad1
        }//I0
    }//T0

#ifdef __MPI
    allsum_deepks(this->inlmax,pdm_size,this->pdm);
#endif
    ModuleBase::timer::tick("LCAO_Deepks","cal_projected_DM_k");
    return;
    
}

void LCAO_Deepks::check_projected_dm()
{
    const std::string file_projdm = PARAM.globalv.global_out_dir + "deepks_projdm.dat";
    std::ofstream ofs(file_projdm.c_str());

    const int pdm_size = (this->lmaxd * 2 + 1) * (this->lmaxd * 2 + 1);
    ofs<<std::setprecision(10);
    for(int inl=0;inl<inlmax;inl++)
    {
        for(int ind=0;ind<pdm_size;ind++)
        {
            ofs << pdm[inl][ind] << " ";
        }
        ofs << std::endl;
    }
}

#endif
