#include "gint.h"
#include "module_base/memory.h"
#include "module_parameter/parameter.h"
#include "module_base/timer.h"

void Gint::gint_kernel_vlocal(Gint_inout* inout) {
    ModuleBase::TITLE("Gint_interface", "cal_gint_vlocal");
    ModuleBase::timer::tick("Gint_interface", "cal_gint_vlocal");
    const UnitCell& ucell = *this->ucell;
    const int max_size = this->gridt->max_atom;
    const int lgd = this->gridt->lgd;
    const int nnrg = this->gridt->nnrg;
    const int ncyz = this->ny * this->nplane;
    const double dv = ucell.omega / this->ncxyz;
    const double delta_r = this->gridt->dr_uniform;
    if (!PARAM.globalv.gamma_only_local) {
        if (!pvpR_alloc_flag) {
            ModuleBase::WARNING_QUIT("Gint_interface::cal_gint",
                                     "pvpR has not been allocated yet!");
        } else {
            ModuleBase::GlobalFunc::ZEROS(this->pvpR_reduced[inout->ispin], nnrg);
        }
    }


#pragma omp parallel 
{   /**
     * @brief When in OpenMP, it points to a newly allocated memory,
    */
    hamilt::HContainer<double>* hRGint_thread;
    double* pvpR_thread; 
    if (PARAM.globalv.gamma_only_local) {
        hRGint_thread = new hamilt::HContainer<double>(*this->hRGint);
    } else {
        pvpR_thread = new double[nnrg];
        ModuleBase::GlobalFunc::ZEROS(pvpR_thread, nnrg);
    }
    std::vector<int> block_iw(max_size,0);
    std::vector<int> block_index(max_size+1,0);
    std::vector<int> block_size(max_size,0);
    std::vector<double> vldr3(this->bxyz,0.0);
    #pragma omp for
    for (int grid_index = 0; grid_index < this->nbxx; grid_index++) {
        const int na_grid = this->gridt->how_many_atoms[grid_index];
        if (na_grid == 0) {
            continue;
        }
        /**
         * @brief Prepare block information
        */
        ModuleBase::Array_Pool<bool> cal_flag(this->bxyz,max_size);

        Gint_Tools::get_gint_vldr3(vldr3.data(),
                                    inout->vl,
                                    this->bxyz,
                                    this->bx,
                                    this->by,
                                    this->bz,
                                    this->nplane,
                                    this->gridt->start_ind[grid_index],
                                    ncyz,
                                    dv);

        Gint_Tools::get_block_info(*this->gridt, this->bxyz, na_grid, grid_index, 
                                            block_iw.data(), block_index.data(), block_size.data(), cal_flag.get_ptr_2D());

        /**
         * @brief Evaluate psi and dpsi on grids
        */
        const int LD_pool = block_index[na_grid];
        ModuleBase::Array_Pool<double> psir_ylm(this->bxyz, LD_pool);
	    Gint_Tools::cal_psir_ylm(*this->gridt, 
            this->bxyz, na_grid, grid_index, delta_r,
            block_index.data(), block_size.data(), 
            cal_flag.get_ptr_2D(),psir_ylm.get_ptr_2D());
        
	//calculating f_mu(r) = v(r)*psi_mu(r)*dv
        const ModuleBase::Array_Pool<double> psir_vlbr3 = Gint_Tools::get_psir_vlbr3(
                this->bxyz, na_grid, LD_pool, block_index.data(), 
                cal_flag.get_ptr_2D(), vldr3.data(), psir_ylm.get_ptr_2D());

	//integrate (psi_mu*v(r)*dv) * psi_nu on grid
	//and accumulates to the corresponding element in Hamiltonian
        if(PARAM.globalv.gamma_only_local)
        {
            this->cal_meshball_vlocal_gamma(
                na_grid, LD_pool, block_iw.data(), block_size.data(), block_index.data(), grid_index, 
                cal_flag.get_ptr_2D(),psir_ylm.get_ptr_2D(), psir_vlbr3.get_ptr_2D(),
                hRGint_thread);
        }
        else
        {
            this->cal_meshball_vlocal_k(
                na_grid, LD_pool, grid_index, block_size.data(), block_index.data(), block_iw.data(), 
                cal_flag.get_ptr_2D(),psir_ylm.get_ptr_2D(), psir_vlbr3.get_ptr_2D(),
                pvpR_thread,ucell);
        }

    }
    if (PARAM.globalv.gamma_only_local) {
        {
        #pragma omp critical(gint_gamma)
            BlasConnector::axpy(this->hRGint->get_nnr(),
                                1.0,
                                hRGint_thread->get_wrapper(),
                                1,
                                this->hRGint->get_wrapper(),
                                1);
        delete hRGint_thread;
        }
    } else {
        {
        #pragma omp critical(gint_k)
            BlasConnector::axpy(nnrg,
                                1.0,
                                pvpR_thread,
                                1,
                                this->pvpR_reduced[inout->ispin],
                                1);
        }
        delete[] pvpR_thread;
    }
    ModuleBase::TITLE("Gint_interface", "cal_gint_vlocal");
    ModuleBase::timer::tick("Gint_interface", "cal_gint_vlocal");
}
}
void Gint::gint_kernel_dvlocal(Gint_inout* inout) {
    ModuleBase::TITLE("Gint_interface", "cal_gint_dvlocal");
    ModuleBase::timer::tick("Gint_interface", "cal_gint_dvlocal");
    const UnitCell& ucell = *this->ucell;
    const int max_size = this->gridt->max_atom;
    const int lgd = this->gridt->lgd;
    const int nnrg = this->gridt->nnrg;
    const int ncyz = this->ny * this->nplane;
    const double dv = ucell.omega / this->ncxyz;
    const double delta_r = this->gridt->dr_uniform;

    if (PARAM.globalv.gamma_only_local) {
        ModuleBase::WARNING_QUIT("Gint_interface::cal_gint","dvlocal only for k point!");
    }

#pragma omp parallel 
{
    ModuleBase::GlobalFunc::ZEROS(this->pvdpRx_reduced[inout->ispin],nnrg);
    ModuleBase::GlobalFunc::ZEROS(this->pvdpRy_reduced[inout->ispin],nnrg);
    ModuleBase::GlobalFunc::ZEROS(this->pvdpRz_reduced[inout->ispin],nnrg);
    double* pvdpRx_thread = new double[nnrg];
    ModuleBase::GlobalFunc::ZEROS(pvdpRx_thread, nnrg);
    double* pvdpRy_thread = new double[nnrg];
    ModuleBase::GlobalFunc::ZEROS(pvdpRy_thread, nnrg);
    double* pvdpRz_thread = new double[nnrg];
    ModuleBase::GlobalFunc::ZEROS(pvdpRz_thread, nnrg);
    std::vector<int> block_iw(max_size,0);
    std::vector<int> block_index(max_size+1,0);
    std::vector<int> block_size(max_size,0);
    std::vector<double> vldr3(this->bxyz,0.0);
#pragma omp for
    for (int grid_index = 0; grid_index < this->nbxx; grid_index++) {
        const int na_grid = this->gridt->how_many_atoms[grid_index];
        if (na_grid == 0) {
            continue;
        }
        Gint_Tools::get_gint_vldr3(vldr3.data(),
                                    inout->vl,
                                    this->bxyz,
                                    this->bx,
                                    this->by,
                                    this->bz,
                                    this->nplane,
                                    this->gridt->start_ind[grid_index],
                                    ncyz,
                                    dv);
    //prepare block information
        ModuleBase::Array_Pool<bool> cal_flag(this->bxyz,max_size);
        Gint_Tools::get_block_info(*this->gridt, this->bxyz, na_grid, grid_index, 
                                    block_iw.data(), block_index.data(), block_size.data(), cal_flag.get_ptr_2D());
        
	//evaluate psi and dpsi on grids
        const int LD_pool = block_index[na_grid];

        ModuleBase::Array_Pool<double> psir_ylm(this->bxyz, LD_pool);
        ModuleBase::Array_Pool<double> dpsir_ylm_x(this->bxyz, LD_pool);
        ModuleBase::Array_Pool<double> dpsir_ylm_y(this->bxyz, LD_pool);
        ModuleBase::Array_Pool<double> dpsir_ylm_z(this->bxyz, LD_pool);
        Gint_Tools::cal_dpsir_ylm(*this->gridt, this->bxyz, na_grid, grid_index, delta_r, 
                                    block_index.data(), block_size.data(), cal_flag.get_ptr_2D(),psir_ylm.get_ptr_2D(),
                                    dpsir_ylm_x.get_ptr_2D(), dpsir_ylm_y.get_ptr_2D(), dpsir_ylm_z.get_ptr_2D());

	//calculating f_mu(r) = v(r)*psi_mu(r)*dv
        const ModuleBase::Array_Pool<double> psir_vlbr3 = Gint_Tools::get_psir_vlbr3(
                this->bxyz, na_grid, LD_pool, block_index.data(), cal_flag.get_ptr_2D(), vldr3.data(), psir_ylm.get_ptr_2D());

	//integrate (psi_mu*v(r)*dv) * psi_nu on grid
	//and accumulates to the corresponding element in Hamiltonian
        this->cal_meshball_vlocal_k(na_grid, LD_pool, grid_index, block_size.data(), block_index.data(),
                                    block_iw.data(), cal_flag.get_ptr_2D(),psir_vlbr3.get_ptr_2D(),
                                    dpsir_ylm_x.get_ptr_2D(), pvdpRx_thread,ucell);
        this->cal_meshball_vlocal_k(na_grid, LD_pool, grid_index, block_size.data(), block_index.data(),
                                    block_iw.data(), cal_flag.get_ptr_2D(),psir_vlbr3.get_ptr_2D(),
                                    dpsir_ylm_y.get_ptr_2D(), pvdpRy_thread,ucell);
	    this->cal_meshball_vlocal_k(na_grid, LD_pool, grid_index, block_size.data(), block_index.data(), 
                                    block_iw.data(), cal_flag.get_ptr_2D(),psir_vlbr3.get_ptr_2D(),
                                    dpsir_ylm_z.get_ptr_2D(), pvdpRz_thread,ucell);
    }
    #pragma omp critical(gint_k)
    {
        BlasConnector::axpy(nnrg,
                            1.0,
                            pvdpRx_thread,
                            1,
                            this->pvdpRx_reduced[inout->ispin],
                            1);
        BlasConnector::axpy(nnrg,
                            1.0,
                            pvdpRy_thread,
                            1,
                            this->pvdpRy_reduced[inout->ispin],
                            1);
        BlasConnector::axpy(nnrg,
                            1.0,
                            pvdpRz_thread,
                            1,
                            this->pvdpRz_reduced[inout->ispin],
                            1);
    }
    delete[] pvdpRx_thread;
    delete[] pvdpRy_thread;
    delete[] pvdpRz_thread;
}
    ModuleBase::TITLE("Gint_interface", "cal_gint_dvlocal");
    ModuleBase::timer::tick("Gint_interface", "cal_gint_dvlocal");
}

void Gint::gint_kernel_vlocal_meta(Gint_inout* inout) {
    ModuleBase::TITLE("Gint_interface", "cal_gint_vlocal_meta");
    ModuleBase::timer::tick("Gint_interface", "cal_gint_vlocal_meta");
    const UnitCell& ucell = *this->ucell;
    const int max_size = this->gridt->max_atom;
    const int lgd = this->gridt->lgd;
    const int nnrg = this->gridt->nnrg;
    const int ncyz = this->ny * this->nplane;
    const double dv = ucell.omega / this->ncxyz;
    const double delta_r = this->gridt->dr_uniform;

    if (!PARAM.globalv.gamma_only_local) {
        if (!pvpR_alloc_flag) {
            ModuleBase::WARNING_QUIT("Gint_interface::cal_gint",
                                     "pvpR has not been allocated yet!");
        } else {
            ModuleBase::GlobalFunc::ZEROS(this->pvpR_reduced[inout->ispin], nnrg);
        }
    }
    
#pragma omp parallel
{
    // define HContainer here to reference.
    //Under the condition of gamma_only, hRGint will be instantiated.
    hamilt::HContainer<double>* hRGint_thread;
    double* pvpR_thread; 
    if (PARAM.globalv.gamma_only_local)
    {
        hRGint_thread =new hamilt::HContainer<double>(*this->hRGint);
    }else{
    //use vector instead of new-delete to avoid memory leak.
        pvpR_thread = new double[nnrg];
        ModuleBase::GlobalFunc::ZEROS(pvpR_thread, nnrg);
    }
    std::vector<int> block_iw(max_size,0);
    std::vector<int> block_index(max_size+1,0);
    std::vector<int> block_size(max_size,0);
    std::vector<double> vldr3(this->bxyz,0.0);
    std::vector<double> vkdr3(this->bxyz,0.0);
#pragma omp for
    for (int grid_index = 0; grid_index < this->nbxx; grid_index++) {
        const int na_grid = this->gridt->how_many_atoms[grid_index];
        if (na_grid == 0) {
            continue;
        }
        Gint_Tools::get_gint_vldr3(vldr3.data(),
                                inout->vl,
                                this->bxyz,
                                this->bx,
                                this->by,
                                this->bz,
                                this->nplane,
                                this->gridt->start_ind[grid_index],
                                ncyz,
                                dv);
        Gint_Tools::get_gint_vldr3(vkdr3.data(),
                                    inout->vofk,
                                    this->bxyz,
                                    this->bx,
                                    this->by,
                                    this->bz,
                                    this->nplane,
                                    this->gridt->start_ind[grid_index],
                                    ncyz,
                                    dv);
    //prepare block information
        ModuleBase::Array_Pool<bool> cal_flag(this->bxyz,max_size);
	    Gint_Tools::get_block_info(*this->gridt, this->bxyz, na_grid, grid_index, 
                                    block_iw.data(), block_index.data(), block_size.data(), cal_flag.get_ptr_2D());

    //evaluate psi and dpsi on grids
        const int LD_pool = block_index[na_grid];
        ModuleBase::Array_Pool<double> psir_ylm(this->bxyz, LD_pool);
        ModuleBase::Array_Pool<double> dpsir_ylm_x(this->bxyz, LD_pool);
        ModuleBase::Array_Pool<double> dpsir_ylm_y(this->bxyz, LD_pool);
        ModuleBase::Array_Pool<double> dpsir_ylm_z(this->bxyz, LD_pool);

        Gint_Tools::cal_dpsir_ylm(*this->gridt,
            this->bxyz, na_grid, grid_index, delta_r,
            block_index.data(), block_size.data(), 
            cal_flag.get_ptr_2D(),
            psir_ylm.get_ptr_2D(),
            dpsir_ylm_x.get_ptr_2D(),
            dpsir_ylm_y.get_ptr_2D(),
            dpsir_ylm_z.get_ptr_2D()
        );
	
	//calculating f_mu(r) = v(r)*psi_mu(r)*dv
	    const ModuleBase::Array_Pool<double> psir_vlbr3 = Gint_Tools::get_psir_vlbr3(
		    	this->bxyz, na_grid, LD_pool, block_index.data(), cal_flag.get_ptr_2D(), vldr3.data(), psir_ylm.get_ptr_2D());

	//calculating df_mu(r) = vofk(r) * dpsi_mu(r) * dv
	    const ModuleBase::Array_Pool<double> dpsix_vlbr3 = Gint_Tools::get_psir_vlbr3(
			this->bxyz, na_grid, LD_pool, block_index.data(), cal_flag.get_ptr_2D(), vkdr3.data(), dpsir_ylm_x.get_ptr_2D());
	    const ModuleBase::Array_Pool<double> dpsiy_vlbr3 = Gint_Tools::get_psir_vlbr3(
			this->bxyz, na_grid, LD_pool, block_index.data(), cal_flag.get_ptr_2D(), vkdr3.data(), dpsir_ylm_y.get_ptr_2D());	
	    const ModuleBase::Array_Pool<double> dpsiz_vlbr3 = Gint_Tools::get_psir_vlbr3(
			this->bxyz, na_grid, LD_pool, block_index.data(), cal_flag.get_ptr_2D(), vkdr3.data(), dpsir_ylm_z.get_ptr_2D());

        if(PARAM.globalv.gamma_only_local)
        {
            //integrate (psi_mu*v(r)*dv) * psi_nu on grid
            //and accumulates to the corresponding element in Hamiltonian
            this->cal_meshball_vlocal_gamma(
                na_grid, LD_pool, block_iw.data(), block_size.data(), block_index.data(), grid_index, cal_flag.get_ptr_2D(),
                psir_ylm.get_ptr_2D(), psir_vlbr3.get_ptr_2D(), hRGint_thread);
            //integrate (d/dx_i psi_mu*vk(r)*dv) * (d/dx_i psi_nu) on grid (x_i=x,y,z)
            //and accumulates to the corresponding element in Hamiltonian
            this->cal_meshball_vlocal_gamma(
                na_grid, LD_pool, block_iw.data(), block_size.data(), block_index.data(), grid_index, cal_flag.get_ptr_2D(),
                dpsir_ylm_x.get_ptr_2D(), dpsix_vlbr3.get_ptr_2D(), hRGint_thread);
            this->cal_meshball_vlocal_gamma(
                na_grid, LD_pool, block_iw.data(), block_size.data(), block_index.data(), grid_index, cal_flag.get_ptr_2D(),
                dpsir_ylm_y.get_ptr_2D(), dpsiy_vlbr3.get_ptr_2D(), hRGint_thread);
            this->cal_meshball_vlocal_gamma(
                na_grid, LD_pool, block_iw.data(), block_size.data(), block_index.data(), grid_index, cal_flag.get_ptr_2D(),
                dpsir_ylm_z.get_ptr_2D(), dpsiz_vlbr3.get_ptr_2D(), hRGint_thread);
        }
        else
        {
            this->cal_meshball_vlocal_k(
                na_grid, LD_pool, grid_index, block_size.data(), block_index.data(), block_iw.data(), cal_flag.get_ptr_2D(),
                psir_ylm.get_ptr_2D(), psir_vlbr3.get_ptr_2D(), pvpR_thread,ucell);
            this->cal_meshball_vlocal_k(
                na_grid, LD_pool, grid_index, block_size.data(), block_index.data(), block_iw.data(), cal_flag.get_ptr_2D(),
                dpsir_ylm_x.get_ptr_2D(), dpsix_vlbr3.get_ptr_2D(), pvpR_thread,ucell);
            this->cal_meshball_vlocal_k(
                na_grid, LD_pool, grid_index, block_size.data(), block_index.data(), block_iw.data(), cal_flag.get_ptr_2D(),
                dpsir_ylm_y.get_ptr_2D(), dpsiy_vlbr3.get_ptr_2D(), pvpR_thread,ucell);
            this->cal_meshball_vlocal_k(
                na_grid, LD_pool, grid_index, block_size.data(), block_index.data(), block_iw.data(), cal_flag.get_ptr_2D(),
                dpsir_ylm_z.get_ptr_2D(), dpsiz_vlbr3.get_ptr_2D(), pvpR_thread,ucell);
        }
    }
    if (PARAM.globalv.gamma_only_local) {
#pragma omp critical(gint_gamma)
        {
            BlasConnector::axpy(this->hRGint->get_nnr(),
                                1.0,
                                hRGint_thread->get_wrapper(),
                                1,
                                this->hRGint->get_wrapper(),
                                1);
        }
        delete hRGint_thread;
    }
    else{
#pragma omp critical(gint_k)
        {
            BlasConnector::axpy(nnrg,
                                1.0,
                                pvpR_thread,
                                1,
                                pvpR_reduced[inout->ispin],
                                1);
        }
        delete[] pvpR_thread;
    }
    
}
    ModuleBase::TITLE("Gint_interface", "cal_gint_vlocal_meta");
    ModuleBase::timer::tick("Gint_interface", "cal_gint_vlocal_meta");
}