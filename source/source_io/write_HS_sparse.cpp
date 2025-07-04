#include "write_HS_sparse.h"

#include "module_parameter/parameter.h"
#include "source_base/parallel_reduce.h"
#include "source_base/timer.h"
#include "source_lcao/module_tddft/td_velocity.h"
#include "source_pw/hamilt_pwdft/global.h"
#include "single_R_io.h"

void ModuleIO::save_HSR_sparse(const int& istep,
                               const Parallel_Orbitals& pv,
                               LCAO_HS_Arrays& HS_Arrays,
                               const double& sparse_thr,
                               const bool& binary,
                               const std::string& SR_filename,
                               const std::string& HR_filename_up,
                               const std::string& HR_filename_down = "") {
    ModuleBase::TITLE("ModuleIO", "save_HSR_sparse");
    ModuleBase::timer::tick("ModuleIO", "save_HSR_sparse");

    auto& all_R_coor_ptr = HS_Arrays.all_R_coor;
    auto& output_R_coor_ptr = HS_Arrays.output_R_coor;
    auto& HR_sparse_ptr = HS_Arrays.HR_sparse;
    auto& SR_sparse_ptr = HS_Arrays.SR_sparse;
    auto& HR_soc_sparse_ptr = HS_Arrays.HR_soc_sparse;
    auto& SR_soc_sparse_ptr = HS_Arrays.SR_soc_sparse;

    int total_R_num = all_R_coor_ptr.size();
    int output_R_number = 0;
    int* H_nonzero_num[2] = {nullptr, nullptr};
    int* S_nonzero_num = nullptr;
    int step = istep;

    S_nonzero_num = new int[total_R_num];
    ModuleBase::GlobalFunc::ZEROS(S_nonzero_num, total_R_num);

    int spin_loop = 1;
    if (PARAM.inp.nspin == 2) {
        spin_loop = 2;
    }

    for (int ispin = 0; ispin < spin_loop; ++ispin) {
        H_nonzero_num[ispin] = new int[total_R_num];
        ModuleBase::GlobalFunc::ZEROS(H_nonzero_num[ispin], total_R_num);
    }

    int count = 0;
    for (auto& R_coor: all_R_coor_ptr) {
        if (PARAM.inp.nspin != 4) {
            for (int ispin = 0; ispin < spin_loop; ++ispin) {
                if (TD_Velocity::tddft_velocity) {
                    auto iter
                        = TD_Velocity::td_vel_op->HR_sparse_td_vel[ispin].find(
                            R_coor);
                    if (iter
                        != TD_Velocity::td_vel_op->HR_sparse_td_vel[ispin]
                               .end()) {
                        for (auto& row_loop: iter->second) {
                            H_nonzero_num[ispin][count]
                                += row_loop.second.size();
                        }
                    }
                } else {
                    auto iter = HR_sparse_ptr[ispin].find(R_coor);
                    if (iter != HR_sparse_ptr[ispin].end()) {
                        for (auto& row_loop: iter->second) {
                            H_nonzero_num[ispin][count]
                                += row_loop.second.size();
                        }
                    }
                }
            }

            auto iter = SR_sparse_ptr.find(R_coor);
            if (iter != SR_sparse_ptr.end()) {
                for (auto& row_loop: iter->second) {
                    S_nonzero_num[count] += row_loop.second.size();
                }
            }
        } else {
            auto iter = HR_soc_sparse_ptr.find(R_coor);
            if (iter != HR_soc_sparse_ptr.end()) {
                for (auto& row_loop: iter->second) {
                    H_nonzero_num[0][count] += row_loop.second.size();
                }
            }

            iter = SR_soc_sparse_ptr.find(R_coor);
            if (iter != SR_soc_sparse_ptr.end()) {
                for (auto& row_loop: iter->second) {
                    S_nonzero_num[count] += row_loop.second.size();
                }
            }
        }

        count++;
    }

    Parallel_Reduce::reduce_all(S_nonzero_num, total_R_num);
    for (int ispin = 0; ispin < spin_loop; ++ispin) {
        Parallel_Reduce::reduce_all(H_nonzero_num[ispin], total_R_num);
    }

    if (PARAM.inp.nspin == 2) {
        for (int index = 0; index < total_R_num; ++index) {
            if (H_nonzero_num[0][index] != 0 || H_nonzero_num[1][index] != 0
                || S_nonzero_num[index] != 0) {
                output_R_number++;
            }
        }
    } else {
        for (int index = 0; index < total_R_num; ++index) {
            if (H_nonzero_num[0][index] != 0 || S_nonzero_num[index] != 0) {
                output_R_number++;
            }
        }
    }

    std::stringstream ssh[2];
    std::stringstream sss;
    if (PARAM.inp.calculation == "md" && !PARAM.inp.out_app_flag) {
        ssh[0] << PARAM.globalv.global_matrix_dir << step << "_" << HR_filename_up;
        ssh[1] << PARAM.globalv.global_matrix_dir << step << "_" << HR_filename_down;
        sss << PARAM.globalv.global_matrix_dir << step << "_" << SR_filename;
    } else {
        ssh[0] << PARAM.globalv.global_out_dir << HR_filename_up;
        ssh[1] << PARAM.globalv.global_out_dir << HR_filename_down;
        sss << PARAM.globalv.global_out_dir << SR_filename;
    }

    GlobalV::ofs_running << " The output filename is " << ssh[0].str() << std::endl;
    GlobalV::ofs_running << " The output filename is " << ssh[1].str() << std::endl;
    GlobalV::ofs_running << " The output filename is " << sss.str() << std::endl;

    std::ofstream g1[2];
    std::ofstream g2;

    if (GlobalV::DRANK == 0) {
        if (binary) {
            int nlocal = PARAM.globalv.nlocal;
            for (int ispin = 0; ispin < spin_loop; ++ispin) {
                if (PARAM.inp.calculation == "md" && PARAM.inp.out_app_flag
                    && step) {
                    g1[ispin].open(ssh[ispin].str().c_str(),
                                   std::ios::binary | std::ios::app);
                } else {
                    g1[ispin].open(ssh[ispin].str().c_str(), std::ios::binary);
                }
                g1[ispin].write(reinterpret_cast<char*>(&step), sizeof(int));
                g1[ispin].write(reinterpret_cast<char*>(&nlocal),
                                sizeof(int));
                g1[ispin].write(reinterpret_cast<char*>(&output_R_number),
                                sizeof(int));
            }

            if (PARAM.inp.calculation == "md" && PARAM.inp.out_app_flag && step) {
                g2.open(sss.str().c_str(), std::ios::binary | std::ios::app);
            } else {
                g2.open(sss.str().c_str(), std::ios::binary);
            }
            g2.write(reinterpret_cast<char*>(&step), sizeof(int));
            g2.write(reinterpret_cast<char*>(&nlocal), sizeof(int));
            g2.write(reinterpret_cast<char*>(&output_R_number), sizeof(int));
        } else {
            for (int ispin = 0; ispin < spin_loop; ++ispin) {
                if (PARAM.inp.calculation == "md" && PARAM.inp.out_app_flag
                    && step) {
                    g1[ispin].open(ssh[ispin].str().c_str(), std::ios::app);
                } else {
                    g1[ispin].open(ssh[ispin].str().c_str());
                }
                g1[ispin] << "STEP: " << step << std::endl;
                g1[ispin] << "Matrix Dimension of H(R): " << PARAM.globalv.nlocal
                          << std::endl;
                g1[ispin] << "Matrix number of H(R): " << output_R_number
                          << std::endl;
            }

            if (PARAM.inp.calculation == "md" && PARAM.inp.out_app_flag && step) {
                g2.open(sss.str().c_str(), std::ios::app);
            } else {
                g2.open(sss.str().c_str());
            }
            g2 << "STEP: " << step << std::endl;
            g2 << "Matrix Dimension of S(R): " << PARAM.globalv.nlocal << std::endl;
            g2 << "Matrix number of S(R): " << output_R_number << std::endl;
        }
    }

    output_R_coor_ptr.clear();

    count = 0;
    for (auto& R_coor: all_R_coor_ptr) {
        int dRx = R_coor.x;
        int dRy = R_coor.y;
        int dRz = R_coor.z;

        if (PARAM.inp.nspin == 2) {
            if (H_nonzero_num[0][count] == 0 && H_nonzero_num[1][count] == 0
                && S_nonzero_num[count] == 0) {
                count++;
                continue;
            }
        } else {
            if (H_nonzero_num[0][count] == 0 && S_nonzero_num[count] == 0) {
                count++;
                continue;
            }
        }

        output_R_coor_ptr.insert(R_coor);

        if (GlobalV::DRANK == 0) {
            if (binary) {
                for (int ispin = 0; ispin < spin_loop; ++ispin) {
                    g1[ispin].write(reinterpret_cast<char*>(&dRx), sizeof(int));
                    g1[ispin].write(reinterpret_cast<char*>(&dRy), sizeof(int));
                    g1[ispin].write(reinterpret_cast<char*>(&dRz), sizeof(int));
                    g1[ispin].write(
                        reinterpret_cast<char*>(&H_nonzero_num[ispin][count]),
                        sizeof(int));
                }

                g2.write(reinterpret_cast<char*>(&dRx), sizeof(int));
                g2.write(reinterpret_cast<char*>(&dRy), sizeof(int));
                g2.write(reinterpret_cast<char*>(&dRz), sizeof(int));
                g2.write(reinterpret_cast<char*>(&S_nonzero_num[count]),
                         sizeof(int));
            } else {
                for (int ispin = 0; ispin < spin_loop; ++ispin) {
                    g1[ispin] << dRx << " " << dRy << " " << dRz << " "
                              << H_nonzero_num[ispin][count] << std::endl;
                }
                g2 << dRx << " " << dRy << " " << dRz << " "
                   << S_nonzero_num[count] << std::endl;
            }
        }

        for (int ispin = 0; ispin < spin_loop; ++ispin) {
            if (H_nonzero_num[ispin][count] == 0) {
                // if (GlobalV::DRANK == 0)
                // {
                //     if (!binary)
                //     {
                //         g1[ispin] << std::endl;
                //         g1[ispin] << std::endl;
                //         for (int index = 0; index < PARAM.globalv.nlocal+1;
                //         ++index)
                //         {
                //             g1[ispin] << 0 << " ";
                //         }
                //         g1[ispin] << std::endl;
                //     }
                // }
            } else {
                if (PARAM.inp.nspin != 4) {
                    if (TD_Velocity::tddft_velocity) {
                        output_single_R(g1[ispin],
                                        TD_Velocity::td_vel_op
                                            ->HR_sparse_td_vel[ispin][R_coor],
                                        sparse_thr,
                                        binary,
                                        pv);
                    } else {
                        output_single_R(g1[ispin],
                                        HR_sparse_ptr[ispin][R_coor],
                                        sparse_thr,
                                        binary,
                                        pv);
                    }
                } else {
                    output_single_R(g1[ispin],
                                    HR_soc_sparse_ptr[R_coor],
                                    sparse_thr,
                                    binary,
                                    pv);
                }
            }
        }

        if (S_nonzero_num[count] == 0) {
            // if (!binary)
            // {
            //     if (GlobalV::DRANK == 0)
            //     {
            //         g2 << std::endl;
            //         g2 << std::endl;
            //         for (int index = 0; index < PARAM.globalv.nlocal+1; ++index)
            //         {
            //             g2 << 0 << " ";
            //         }
            //         g2 << std::endl;
            //     }
            // }
        } else {
            if (PARAM.inp.nspin != 4) {
                output_single_R(g2,
                                SR_sparse_ptr[R_coor],
                                sparse_thr,
                                binary,
                                pv);
            } else {
                output_single_R(g2,
                                SR_soc_sparse_ptr[R_coor],
                                sparse_thr,
                                binary,
                                pv);
            }
        }

        count++;
    }

    if (GlobalV::DRANK == 0) {
        for (int ispin = 0; ispin < spin_loop; ++ispin) {
            g1[ispin].close();
        }
        g2.close();
    }

    for (int ispin = 0; ispin < spin_loop; ++ispin) {
        delete[] H_nonzero_num[ispin];
        H_nonzero_num[ispin] = nullptr;
    }
    delete[] S_nonzero_num;
    S_nonzero_num = nullptr;

    ModuleBase::timer::tick("ModuleIO", "save_HSR_sparse");
    return;
}

void ModuleIO::save_dH_sparse(const int& istep,
                              const Parallel_Orbitals& pv,
                              LCAO_HS_Arrays& HS_Arrays,
                              const double& sparse_thr,
                              const bool& binary,
                              const std::string& fileflag) {
    ModuleBase::TITLE("ModuleIO", "save_dH_sparse");
    ModuleBase::timer::tick("ModuleIO", "save_dH_sparse");

    auto& all_R_coor_ptr = HS_Arrays.all_R_coor;
    auto& output_R_coor_ptr = HS_Arrays.output_R_coor;
    auto& dHRx_sparse_ptr = HS_Arrays.dHRx_sparse;
    auto& dHRx_soc_sparse_ptr = HS_Arrays.dHRx_soc_sparse;
    auto& dHRy_sparse_ptr = HS_Arrays.dHRy_sparse;
    auto& dHRy_soc_sparse_ptr = HS_Arrays.dHRy_soc_sparse;
    auto& dHRz_sparse_ptr = HS_Arrays.dHRz_sparse;
    auto& dHRz_soc_sparse_ptr = HS_Arrays.dHRz_soc_sparse;

    int total_R_num = all_R_coor_ptr.size();
    int output_R_number = 0;
    int* dHx_nonzero_num[2] = {nullptr, nullptr};
    int* dHy_nonzero_num[2] = {nullptr, nullptr};
    int* dHz_nonzero_num[2] = {nullptr, nullptr};
    int step = istep;

    int spin_loop = 1;
    if (PARAM.inp.nspin == 2) {
        spin_loop = 2;
    }

    for (int ispin = 0; ispin < spin_loop; ++ispin) {
        dHx_nonzero_num[ispin] = new int[total_R_num];
        ModuleBase::GlobalFunc::ZEROS(dHx_nonzero_num[ispin], total_R_num);
        dHy_nonzero_num[ispin] = new int[total_R_num];
        ModuleBase::GlobalFunc::ZEROS(dHy_nonzero_num[ispin], total_R_num);
        dHz_nonzero_num[ispin] = new int[total_R_num];
        ModuleBase::GlobalFunc::ZEROS(dHz_nonzero_num[ispin], total_R_num);
    }

    int count = 0;
    for (auto& R_coor: all_R_coor_ptr) {
        if (PARAM.inp.nspin != 4) {
            for (int ispin = 0; ispin < spin_loop; ++ispin) {
                auto iter1 = dHRx_sparse_ptr[ispin].find(R_coor);
                if (iter1 != dHRx_sparse_ptr[ispin].end()) {
                    for (auto& row_loop: iter1->second) {
                        dHx_nonzero_num[ispin][count] += row_loop.second.size();
                    }
                }

                auto iter2 = dHRy_sparse_ptr[ispin].find(R_coor);
                if (iter2 != dHRy_sparse_ptr[ispin].end()) {
                    for (auto& row_loop: iter2->second) {
                        dHy_nonzero_num[ispin][count] += row_loop.second.size();
                    }
                }

                auto iter3 = dHRz_sparse_ptr[ispin].find(R_coor);
                if (iter3 != dHRz_sparse_ptr[ispin].end()) {
                    for (auto& row_loop: iter3->second) {
                        dHz_nonzero_num[ispin][count] += row_loop.second.size();
                    }
                }
            }
        } else {
            auto iter = dHRx_soc_sparse_ptr.find(R_coor);
            if (iter != dHRx_soc_sparse_ptr.end()) {
                for (auto& row_loop: iter->second) {
                    dHx_nonzero_num[0][count] += row_loop.second.size();
                }
            }
        }

        count++;
    }

    for (int ispin = 0; ispin < spin_loop; ++ispin) {
        Parallel_Reduce::reduce_all(dHx_nonzero_num[ispin], total_R_num);
        Parallel_Reduce::reduce_all(dHy_nonzero_num[ispin], total_R_num);
        Parallel_Reduce::reduce_all(dHz_nonzero_num[ispin], total_R_num);
    }

	if (PARAM.inp.nspin == 2) 
	{
		for (int index = 0; index < total_R_num; ++index) 
		{
			if (dHx_nonzero_num[0][index] != 0 || dHx_nonzero_num[1][index] != 0
					|| dHy_nonzero_num[0][index] != 0
					|| dHy_nonzero_num[1][index] != 0
					|| dHz_nonzero_num[0][index] != 0
					|| dHz_nonzero_num[1][index] != 0) 
			{
				output_R_number++;
			}
		}
	} else 
	{
		for (int index = 0; index < total_R_num; ++index) 
		{
			if (dHx_nonzero_num[0][index] != 0 || dHy_nonzero_num[0][index] != 0
					|| dHz_nonzero_num[0][index] != 0) 
			{
				output_R_number++;
			}
		}
    }

    std::stringstream sshx[2];
    std::stringstream sshy[2];
    std::stringstream sshz[2];

	if (PARAM.inp.calculation == "md" && !PARAM.inp.out_app_flag) 
	{
		sshx[0] << PARAM.globalv.global_matrix_dir
			<< "d"<<fileflag<<"rxs1g" << step << "_nao.csr";
		sshx[1] << PARAM.globalv.global_matrix_dir
			<< "d"<<fileflag<<"rxs2g" << step << "_nao.csr";
		sshy[0] << PARAM.globalv.global_matrix_dir
			<< "d"<<fileflag<<"rys1g" << step << "_nao.csr";
		sshy[1] << PARAM.globalv.global_matrix_dir
			<< "d"<<fileflag<<"rys2g" << step << "_nao.csr";
		sshz[0] << PARAM.globalv.global_matrix_dir
			<< "d"<<fileflag<<"rzs1g" << step << "_nao.csr";
		sshz[1] << PARAM.globalv.global_matrix_dir
			<< "d"<<fileflag<<"rzs2g" << step << "_nao.csr";
	} 
	else 
	{
		sshx[0] << PARAM.globalv.global_out_dir << "d"<<fileflag<<"rxs1_nao.csr";
        sshx[1] << PARAM.globalv.global_out_dir << "d"<<fileflag<<"rxs2_nao.csr";
        sshy[0] << PARAM.globalv.global_out_dir << "d"<<fileflag<<"rys1_nao.csr";
        sshy[1] << PARAM.globalv.global_out_dir << "d"<<fileflag<<"rys2_nao.csr";
        sshz[0] << PARAM.globalv.global_out_dir << "d"<<fileflag<<"rzs1_nao.csr";
        sshz[1] << PARAM.globalv.global_out_dir << "d"<<fileflag<<"rzs2_nao.csr";
    }
    std::ofstream g1x[2];
    std::ofstream g1y[2];
    std::ofstream g1z[2];

	if (GlobalV::DRANK == 0) 
	{
		if (binary) // binary format 
		{
			int nlocal = PARAM.globalv.nlocal;
			for (int ispin = 0; ispin < spin_loop; ++ispin) 
			{
				if (PARAM.inp.calculation == "md" && PARAM.inp.out_app_flag
						&& step) 
				{
					g1x[ispin].open(sshx[ispin].str().c_str(),
                                    std::ios::binary | std::ios::app);
                    g1y[ispin].open(sshy[ispin].str().c_str(),
                                    std::ios::binary | std::ios::app);
                    g1z[ispin].open(sshz[ispin].str().c_str(),
                                    std::ios::binary | std::ios::app);
				} 
				else 
				{
                    g1x[ispin].open(sshx[ispin].str().c_str(),std::ios::binary);
                    g1y[ispin].open(sshy[ispin].str().c_str(),std::ios::binary);
                    g1z[ispin].open(sshz[ispin].str().c_str(),std::ios::binary);
                }

                g1x[ispin].write(reinterpret_cast<char*>(&step), sizeof(int));
                g1x[ispin].write(reinterpret_cast<char*>(&nlocal),
                                 sizeof(int));
                g1x[ispin].write(reinterpret_cast<char*>(&output_R_number),
                                 sizeof(int));

                g1y[ispin].write(reinterpret_cast<char*>(&step), sizeof(int));
                g1y[ispin].write(reinterpret_cast<char*>(&nlocal),
                                 sizeof(int));
                g1y[ispin].write(reinterpret_cast<char*>(&output_R_number),
                                 sizeof(int));

                g1z[ispin].write(reinterpret_cast<char*>(&step), sizeof(int));
                g1z[ispin].write(reinterpret_cast<char*>(&nlocal),
                                 sizeof(int));
                g1z[ispin].write(reinterpret_cast<char*>(&output_R_number),
                                 sizeof(int));
            }
		} 
		else 
		{
			for (int ispin = 0; ispin < spin_loop; ++ispin) 
			{
				if (PARAM.inp.calculation == "md" && PARAM.inp.out_app_flag && step) 
				{
					g1x[ispin].open(sshx[ispin].str().c_str(), std::ios::app);
                    g1y[ispin].open(sshy[ispin].str().c_str(), std::ios::app);
                    g1z[ispin].open(sshz[ispin].str().c_str(), std::ios::app);
				} 
				else 
				{
					GlobalV::ofs_running << " dH/dRx data are in file: " << sshx[ispin].str() << std::endl;
					GlobalV::ofs_running << " dH/dRy data are in file: " << sshy[ispin].str() << std::endl;
					GlobalV::ofs_running << " dH/dRz data are in file: " << sshz[ispin].str() << std::endl;
                    g1x[ispin].open(sshx[ispin].str().c_str());
                    g1y[ispin].open(sshy[ispin].str().c_str());
                    g1z[ispin].open(sshz[ispin].str().c_str());
                }

                g1x[ispin] << "STEP: " << step << std::endl;
                g1x[ispin] << "Matrix Dimension of dHx(R): " << PARAM.globalv.nlocal
                           << std::endl;
                g1x[ispin] << "Matrix number of dHx(R): " << output_R_number
                           << std::endl;

                g1y[ispin] << "STEP: " << step << std::endl;
                g1y[ispin] << "Matrix Dimension of dHy(R): " << PARAM.globalv.nlocal
                           << std::endl;
                g1y[ispin] << "Matrix number of dHy(R): " << output_R_number
                           << std::endl;

                g1z[ispin] << "STEP: " << step << std::endl;
                g1z[ispin] << "Matrix Dimension of dHz(R): " << PARAM.globalv.nlocal
                           << std::endl;
                g1z[ispin] << "Matrix number of dHz(R): " << output_R_number
                           << std::endl;
            }
        }
    }

    output_R_coor_ptr.clear();

    count = 0;
    for (auto& R_coor: all_R_coor_ptr) {
        int dRx = R_coor.x;
        int dRy = R_coor.y;
        int dRz = R_coor.z;

        if (PARAM.inp.nspin == 2) {
            if (dHx_nonzero_num[0][count] == 0 && dHx_nonzero_num[1][count] == 0
                && dHy_nonzero_num[0][count] == 0
                && dHy_nonzero_num[1][count] == 0
                && dHz_nonzero_num[0][count] == 0
                && dHz_nonzero_num[1][count] == 0) {
                count++;
                continue;
            }
        } else {
            if (dHx_nonzero_num[0][count] == 0 && dHy_nonzero_num[0][count] == 0
                && dHz_nonzero_num[0][count] == 0) {
                count++;
                continue;
            }
        }

        output_R_coor_ptr.insert(R_coor);

        if (GlobalV::DRANK == 0) {
            if (binary) {
                for (int ispin = 0; ispin < spin_loop; ++ispin) {
                    g1x[ispin].write(reinterpret_cast<char*>(&dRx),
                                     sizeof(int));
                    g1x[ispin].write(reinterpret_cast<char*>(&dRy),
                                     sizeof(int));
                    g1x[ispin].write(reinterpret_cast<char*>(&dRz),
                                     sizeof(int));
                    g1x[ispin].write(
                        reinterpret_cast<char*>(&dHx_nonzero_num[ispin][count]),
                        sizeof(int));

                    g1y[ispin].write(reinterpret_cast<char*>(&dRx),
                                     sizeof(int));
                    g1y[ispin].write(reinterpret_cast<char*>(&dRy),
                                     sizeof(int));
                    g1y[ispin].write(reinterpret_cast<char*>(&dRz),
                                     sizeof(int));
                    g1y[ispin].write(
                        reinterpret_cast<char*>(&dHy_nonzero_num[ispin][count]),
                        sizeof(int));

                    g1z[ispin].write(reinterpret_cast<char*>(&dRx),
                                     sizeof(int));
                    g1z[ispin].write(reinterpret_cast<char*>(&dRy),
                                     sizeof(int));
                    g1z[ispin].write(reinterpret_cast<char*>(&dRz),
                                     sizeof(int));
                    g1z[ispin].write(
                        reinterpret_cast<char*>(&dHz_nonzero_num[ispin][count]),
                        sizeof(int));
                }
            } else {
                for (int ispin = 0; ispin < spin_loop; ++ispin) {
                    g1x[ispin] << dRx << " " << dRy << " " << dRz << " "
                               << dHx_nonzero_num[ispin][count] << std::endl;
                    g1y[ispin] << dRx << " " << dRy << " " << dRz << " "
                               << dHy_nonzero_num[ispin][count] << std::endl;
                    g1z[ispin] << dRx << " " << dRy << " " << dRz << " "
                               << dHz_nonzero_num[ispin][count] << std::endl;
                }
            }
        }

        for (int ispin = 0; ispin < spin_loop; ++ispin) {
            if (dHx_nonzero_num[ispin][count] > 0) {
                if (PARAM.inp.nspin != 4) {
                    output_single_R(g1x[ispin],
                                    dHRx_sparse_ptr[ispin][R_coor],
                                    sparse_thr,
                                    binary,
                                    pv);
                } else {
                    output_single_R(g1x[ispin],
                                    dHRx_soc_sparse_ptr[R_coor],
                                    sparse_thr,
                                    binary,
                                    pv);
                }
            }
            if (dHy_nonzero_num[ispin][count] > 0) {
                if (PARAM.inp.nspin != 4) {
                    output_single_R(g1y[ispin],
                                    dHRy_sparse_ptr[ispin][R_coor],
                                    sparse_thr,
                                    binary,
                                    pv);
                } else {
                    output_single_R(g1y[ispin],
                                    dHRy_soc_sparse_ptr[R_coor],
                                    sparse_thr,
                                    binary,
                                    pv);
                }
            }
            if (dHz_nonzero_num[ispin][count] > 0) {
                if (PARAM.inp.nspin != 4) {
                    output_single_R(g1z[ispin],
                                    dHRz_sparse_ptr[ispin][R_coor],
                                    sparse_thr,
                                    binary,
                                    pv);
                } else {
                    output_single_R(g1z[ispin],
                                    dHRz_soc_sparse_ptr[R_coor],
                                    sparse_thr,
                                    binary,
                                    pv);
                }
            }
        }

        count++;
    }

    if (GlobalV::DRANK == 0) {
        for (int ispin = 0; ispin < spin_loop; ++ispin) {
            g1x[ispin].close();
        }
        for (int ispin = 0; ispin < spin_loop; ++ispin) {
            g1y[ispin].close();
        }
        for (int ispin = 0; ispin < spin_loop; ++ispin) {
            g1z[ispin].close();
        }
    }

    for (int ispin = 0; ispin < spin_loop; ++ispin) {
        delete[] dHx_nonzero_num[ispin];
        dHx_nonzero_num[ispin] = nullptr;
        delete[] dHy_nonzero_num[ispin];
        dHy_nonzero_num[ispin] = nullptr;
        delete[] dHz_nonzero_num[ispin];
        dHz_nonzero_num[ispin] = nullptr;
    }

    ModuleBase::timer::tick("ModuleIO", "save_dH_sparse");
    return;
}

template <typename Tdata>
void ModuleIO::save_sparse(
    const std::map<Abfs::Vector3_Order<int>,
                   std::map<size_t, std::map<size_t, Tdata>>>& smat,
    const std::set<Abfs::Vector3_Order<int>>& all_R_coor,
    const double& sparse_thr,
    const bool& binary,
    const std::string& filename,
    const Parallel_Orbitals& pv,
    const std::string& label,
    const int& istep,
    const bool& reduce) {
    ModuleBase::TITLE("ModuleIO", "save_sparse");
    ModuleBase::timer::tick("ModuleIO", "save_sparse");

    int total_R_num = all_R_coor.size();
    std::vector<int> nonzero_num(total_R_num, 0);
    int count = 0;
    for (auto& R_coor: all_R_coor) {
        auto iter = smat.find(R_coor);
        if (iter != smat.end()) {
            for (auto& row_loop: iter->second) {
                nonzero_num[count] += row_loop.second.size();
            }
        }
        ++count;
    }
    if (reduce) {
        Parallel_Reduce::reduce_all(nonzero_num.data(), total_R_num);
    }

    int output_R_number = 0;
    for (int index = 0; index < total_R_num; ++index) {
        if (nonzero_num[index] != 0) {
            ++output_R_number;
        }
    }

    std::stringstream sss;
    sss << filename;
    std::ofstream ofs;
    if (!reduce || GlobalV::DRANK == 0) {
        if (binary) {
            int nlocal = PARAM.globalv.nlocal;
            if (PARAM.inp.calculation == "md" && PARAM.inp.out_app_flag
                && istep) {
                ofs.open(sss.str().c_str(), std::ios::binary | std::ios::app);
            } else {
                ofs.open(sss.str().c_str(), std::ios::binary);
            }
            ofs.write(reinterpret_cast<char*>(0), sizeof(int));
            ofs.write(reinterpret_cast<char*>(&nlocal), sizeof(int));
            ofs.write(reinterpret_cast<char*>(&output_R_number), sizeof(int));
        } else {
            if (PARAM.inp.calculation == "md" && PARAM.inp.out_app_flag
                && istep) {
                ofs.open(sss.str().c_str(), std::ios::app);
            } else {
                ofs.open(sss.str().c_str());
            }
            ofs << "STEP: " << std::max(istep, 0) << std::endl;
            ofs << "Matrix Dimension of " + label + "(R): " << PARAM.globalv.nlocal
                << std::endl;
            ofs << "Matrix number of " + label + "(R): " << output_R_number
                << std::endl;
        }
    }

    count = 0;
    for (auto& R_coor: all_R_coor) {
        int dRx = R_coor.x;
        int dRy = R_coor.y;
        int dRz = R_coor.z;

        if (nonzero_num[count] == 0) {
            count++;
            continue;
        }

        if (!reduce || GlobalV::DRANK == 0) {
            if (binary) {
                ofs.write(reinterpret_cast<char*>(&dRx), sizeof(int));
                ofs.write(reinterpret_cast<char*>(&dRy), sizeof(int));
                ofs.write(reinterpret_cast<char*>(&dRz), sizeof(int));
                ofs.write(reinterpret_cast<char*>(&nonzero_num[count]),
                          sizeof(int));
            } else {
                ofs << dRx << " " << dRy << " " << dRz << " "
                    << nonzero_num[count] << std::endl;
            }
        }

        if (smat.count(R_coor))
        {
            output_single_R(ofs, smat.at(R_coor), sparse_thr, binary, pv, reduce);
        }
        else
        {
            std::map<size_t, std::map<size_t, Tdata>> empty_map;
            output_single_R(ofs, empty_map, sparse_thr, binary, pv, reduce);
        }
        ++count;
    }
    if (!reduce || GlobalV::DRANK == 0) {
        ofs.close();
    }

    ModuleBase::timer::tick("ModuleIO", "save_sparse");
}

template void ModuleIO::save_sparse<double>(
    const std::map<Abfs::Vector3_Order<int>,
                   std::map<size_t, std::map<size_t, double>>>&,
    const std::set<Abfs::Vector3_Order<int>>&,
    const double&,
    const bool&,
    const std::string&,
    const Parallel_Orbitals&,
    const std::string&,
    const int&,
    const bool&);

template void ModuleIO::save_sparse<std::complex<double>>(
    const std::map<Abfs::Vector3_Order<int>,
                   std::map<size_t, std::map<size_t, std::complex<double>>>>&,
    const std::set<Abfs::Vector3_Order<int>>&,
    const double&,
    const bool&,
    const std::string&,
    const Parallel_Orbitals&,
    const std::string&,
    const int&,
    const bool&);
