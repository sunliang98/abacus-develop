#include "parallel_grid.h"

#include "source_base/parallel_global.h"
#include "module_parameter/parameter.h"
Parallel_Grid::Parallel_Grid()
{
    this->allocate = false;
    this->allocate_final_scf = false; // LiuXh add 20180619
}

Parallel_Grid::~Parallel_Grid()
{
    if (this->allocate || this->allocate_final_scf) // LiuXh add 20180619
    {
        for (int ip = 0; ip < GlobalV::KPAR; ip++)
        {
            delete[] numz[ip];
            delete[] startz[ip];
            delete[] whichpro[ip];
            delete[] whichpro_loc[ip];
        }
        delete[] numz;
        delete[] startz;
        delete[] whichpro;
        delete[] whichpro_loc;
        delete[] nproc_in_pool;
    }
}

void Parallel_Grid::init(const int& ncx_in,
                         const int& ncy_in,
                         const int& ncz_in,
                         const int& nczp_in,
                         const int& nrxx_in,
                         const int& nbz_in,
                         const int& bz_in)
{

    ModuleBase::TITLE("Parallel_Grid", "init");

    this->ncx = ncx_in;
    this->ncy = ncy_in;
    this->ncz = ncz_in;
    this->nczp = nczp_in;
    this->nrxx = nrxx_in;
    this->nbz = nbz_in;
    this->bz = bz_in;

    if (nczp < 0)
    {
        GlobalV::ofs_warning << " nczp = " << nczp << std::endl;
        ModuleBase::WARNING_QUIT("Parallel_Grid::init", "nczp<0");
    }

    assert(ncx > 0);
    assert(ncy > 0);
    assert(ncz > 0);

    this->ncxy = ncx * ncy;
    this->ncxyz = ncxy * ncz;

#ifndef __MPI
    return;
#endif

    // enable to call this function again liuyu 2023-03-10
    if (this->allocate)
    {
        for (int ip = 0; ip < GlobalV::KPAR; ip++)
        {
            delete[] numz[ip];
            delete[] startz[ip];
            delete[] whichpro[ip];
            delete[] whichpro_loc[ip];
        }
        delete[] numz;
        delete[] startz;
        delete[] whichpro;
        delete[] whichpro_loc;
        delete[] nproc_in_pool;
        this->allocate = false;
    }

    // (2)
    assert(allocate == false);
    assert(GlobalV::KPAR > 0);

    this->nproc_in_pool = new int[GlobalV::KPAR];
    int nprocgroup;
    if (PARAM.inp.esolver_type == "sdft")
    {
        nprocgroup = GlobalV::NPROC_IN_BNDGROUP;
    }
    else
    {
        nprocgroup = GlobalV::NPROC;
    }

    const int remain_pro = nprocgroup % GlobalV::KPAR;
    for (int i = 0; i < GlobalV::KPAR; i++)
    {
        nproc_in_pool[i] = nprocgroup / GlobalV::KPAR;
        if (i < remain_pro)
        {
            this->nproc_in_pool[i]++;
        }
    }

    this->numz = new int*[GlobalV::KPAR];
    this->startz = new int*[GlobalV::KPAR];
    this->whichpro = new int*[GlobalV::KPAR];
    this->whichpro_loc = new int*[GlobalV::KPAR];

    for (int ip = 0; ip < GlobalV::KPAR; ip++)
    {
        const int nproc = nproc_in_pool[ip];
        this->numz[ip] = new int[nproc];
        this->startz[ip] = new int[nproc];
        this->whichpro[ip] = new int[this->ncz];
        this->whichpro_loc[ip] = new int[this->ncz];
        ModuleBase::GlobalFunc::ZEROS(this->numz[ip], nproc);
        ModuleBase::GlobalFunc::ZEROS(this->startz[ip], nproc);
        ModuleBase::GlobalFunc::ZEROS(this->whichpro[ip], this->ncz);
        ModuleBase::GlobalFunc::ZEROS(this->whichpro_loc[ip], this->ncz);
    }

    this->allocate = true;
    this->z_distribution();

    return;
}

void Parallel_Grid::z_distribution()
{
    assert(allocate);

    int* startp = new int[GlobalV::KPAR];
    startp[0] = 0;
    for (int ip = 0; ip < GlobalV::KPAR; ip++)
    {
        //		GlobalV::ofs_running << "\n now POOL=" << ip;
        const int nproc = nproc_in_pool[ip];

        if (ip > 0)
        {
            startp[ip] = startp[ip - 1] + nproc_in_pool[ip - 1];
        }

        // (1) how many z on each 'proc' in each 'pool'
        for (int iz = 0; iz < nbz; iz++)
        {
            const int proc = iz % nproc;
            numz[ip][proc] += bz;
        }

        //		for(int proc=0; proc<nproc; proc++)
        //		{
        //			GlobalV::ofs_running << "\n proc=" << proc << " numz=" << numz[ip][proc];
        //		}

        // (2) start position of z in each 'proc' in each 'pool'
        startz[ip][0] = 0;
        for (int proc = 1; proc < nproc; proc++)
        {
            startz[ip][proc] = startz[ip][proc - 1] + numz[ip][proc - 1];
        }

        //		for(int proc=0; proc<nproc; proc++)
        //		{
        //			GlobalV::ofs_running << "\n proc=" << proc << " startz=" << startz[ip][proc];
        //		}

        // (3) each z belongs to which 'proc' ( global index )
        for (int iz = 0; iz < ncz; iz++)
        {
            for (int proc = 0; proc < nproc; proc++)
            {
                if (iz >= startz[ip][nproc - 1])
                {
                    whichpro[ip][iz] = startp[ip] + nproc - 1;
                    whichpro_loc[ip][iz] = nproc - 1;
                    break;
                }
                else if (iz >= startz[ip][proc] && iz < startz[ip][proc + 1])
                {
                    whichpro[ip][iz] = startp[ip] + proc;
                    whichpro_loc[ip][iz] = proc;
                    break;
                }
            }
        }

        //		for(int iz=0; iz<ncz; iz++)
        //		{
        //			GlobalV::ofs_running << "\n iz=" << iz << " whichpro=" << whichpro[ip][iz];
        //		}
    }

    delete[] startp;
    return;
}

#ifdef __MPI
void Parallel_Grid::bcast(const double* const data_global, double* data_local, const int& rank) const
{
    std::vector<double> zpiece(ncxy);
    for (int iz = 0; iz < this->ncz; ++iz)
    {
        ModuleBase::GlobalFunc::ZEROS(zpiece.data(), ncxy);
        if (rank == 0)
        {
            for (int ix = 0; ix < ncx; ix++)
            {
                for (int iy = 0; iy < ncy; iy++)
                {
                    const int ixy = ix * ncy + iy;
                    zpiece[ixy] = data_global[ixy * ncz + iz];
                }
            }
        }
        this->zpiece_to_all(zpiece.data(), iz, data_local);
    }
}

void Parallel_Grid::zpiece_to_all(double* zpiece, const int& iz, double* rho) const
{
    if (PARAM.inp.esolver_type == "sdft")
    {
        this->zpiece_to_stogroup(zpiece, iz, rho);
        return;
    }
    assert(allocate);
    // ModuleBase::TITLE("Parallel_Grid","zpiece_to_all");
    MPI_Status ierror;

    const int znow = iz - this->startz[GlobalV::MY_POOL][GlobalV::RANK_IN_POOL];
    const int proc = this->whichpro[GlobalV::MY_POOL][iz];

    if (GlobalV::MY_POOL == 0)
    {
        // case 1: the first part of rho in processor 0.
        // and send zpeice to to other pools.
        if (proc == 0 && GlobalV::MY_RANK == 0)
        {
            for (int ir = 0; ir < ncxy; ir++)
            {
                rho[ir * nczp + znow] = zpiece[ir];
            }
            for (int ipool = 1; ipool < GlobalV::KPAR; ipool++)
            {
                MPI_Send(zpiece, ncxy, MPI_DOUBLE, this->whichpro[ipool][iz], iz, MPI_COMM_WORLD);
            }
        }

        // case 2: processor n (n!=0) receive rho from processor 0.
        // and the receive tag is iz.
        else if (proc == GlobalV::RANK_IN_POOL)
        {
            MPI_Recv(zpiece, ncxy, MPI_DOUBLE, 0, iz, MPI_COMM_WORLD, &ierror);
            for (int ir = 0; ir < ncxy; ir++)
            {
                rho[ir * nczp + znow] = zpiece[ir];
            }
        }

        // case 2: > first part rho: processor 0 send the rho
        // to all pools. The tag is iz, because processor may
        // send more than once, and the only tag to distinguish
        // them is iz.
        else if (GlobalV::RANK_IN_POOL == 0)
        {
            for (int ipool = 0; ipool < GlobalV::KPAR; ipool++)
            {
                MPI_Send(zpiece, ncxy, MPI_DOUBLE, this->whichpro[ipool][iz], iz, MPI_COMM_WORLD);
            }
        }
    } // GlobalV::MY_POOL == 0
    else
    {
        // GlobalV::ofs_running << "\n Receive charge density iz=" << iz << std::endl;
        //  the processors in other pools always receive rho from
        //  processor 0. the tag is 'iz'
        if (proc == GlobalV::MY_RANK)
        {
            MPI_Recv(zpiece, ncxy, MPI_DOUBLE, 0, iz, MPI_COMM_WORLD, &ierror);
            for (int ir = 0; ir < ncxy; ir++)
            {
                rho[ir * nczp + znow] = zpiece[ir];
            }
        }
    }

    // GlobalV::ofs_running << "\n iz = " << iz << " Done.";
    return;
}
#endif

#ifdef __MPI
void Parallel_Grid::zpiece_to_stogroup(double* zpiece, const int& iz, double* rho) const
{
    assert(allocate);
    // TITLE("Parallel_Grid","zpiece_to_all");
    MPI_Status ierror;

    const int znow = iz - this->startz[GlobalV::MY_POOL][GlobalV::RANK_IN_POOL];
    const int proc = this->whichpro[GlobalV::MY_POOL][iz];

    if (GlobalV::MY_POOL == 0)
    {
        // case 1: the first part of rho in processor 0.
        // and send zpeice to to other pools.
        if (proc == 0 && GlobalV::RANK_IN_BPGROUP == 0)
        {
            for (int ir = 0; ir < ncxy; ir++)
            {
                rho[ir * nczp + znow] = zpiece[ir];
            }
            for (int ipool = 1; ipool < GlobalV::KPAR; ipool++)
            {
                MPI_Send(zpiece, ncxy, MPI_DOUBLE, this->whichpro[ipool][iz], iz, INT_BGROUP);
            }
        }

        // case 2: processor n (n!=0) receive rho from processor 0.
        // and the receive tag is iz.
        else if (proc == GlobalV::RANK_IN_POOL)
        {
            MPI_Recv(zpiece, ncxy, MPI_DOUBLE, 0, iz, INT_BGROUP, &ierror);
            for (int ir = 0; ir < ncxy; ir++)
            {
                rho[ir * nczp + znow] = zpiece[ir];
            }
        }

        // case 2: > first part rho: processor 0 send the rho
        // to all pools. The tag is iz, because processor may
        // send more than once, and the only tag to distinguish
        // them is iz.
        else if (GlobalV::RANK_IN_POOL == 0)
        {
            for (int ipool = 0; ipool < GlobalV::KPAR; ipool++)
            {
                MPI_Send(zpiece, ncxy, MPI_DOUBLE, this->whichpro[ipool][iz], iz, INT_BGROUP);
            }
        }
    } // MY_POOL == 0
    else
    {
        // ofs_running << "\n Receive charge density iz=" << iz << endl;
        //  the processors in other pools always receive rho from
        //  processor 0. the tag is 'iz'
        if (proc == GlobalV::RANK_IN_BPGROUP)
        {
            MPI_Recv(zpiece, ncxy, MPI_DOUBLE, 0, iz, INT_BGROUP, &ierror);
            for (int ir = 0; ir < ncxy; ir++)
            {
                rho[ir * nczp + znow] = zpiece[ir];
            }
        }
    }

    // ofs_running << "\n iz = " << iz << " Done.";
    return;
}
void Parallel_Grid::reduce(double* rhotot, const double* const rhoin, const bool reduce_all_pool) const
{
    // ModuleBase::TITLE("Parallel_Grid","reduce");

    // if not the first pool, wait here until processpr 0
    // send the Barrier command.
    if (!reduce_all_pool && GlobalV::MY_POOL != 0)
    {
        return;
    }

    double* zpiece = new double[this->ncxy];

    for (int iz = 0; iz < this->ncz; iz++)
    {
        const int znow = iz - this->startz[GlobalV::MY_POOL][GlobalV::RANK_IN_POOL];
        const int proc = this->whichpro[GlobalV::MY_POOL][iz];
        const int proc_loc = this->whichpro_loc[GlobalV::MY_POOL][iz]; // Obtain the local processor index in the pool
        ModuleBase::GlobalFunc::ZEROS(zpiece, this->ncxy);
        int tag = iz;
        MPI_Status ierror;

        // Local processor 0 collects data from all other processors in the pool
        // proc = proc_loc if GlobalV::MY_POOL == 0
        if (proc_loc == GlobalV::RANK_IN_POOL)
        {
            for (int ir = 0; ir < ncxy; ir++)
            {
                zpiece[ir] = rhoin[ir * this->nczp + znow];
            }
            // Send data to the root of the pool
            if (GlobalV::RANK_IN_POOL != 0)
            {
                MPI_Send(zpiece, ncxy, MPI_DOUBLE, 0, tag, POOL_WORLD);
            }
        }

        // The root of the pool receives data from other processors
        if (GlobalV::RANK_IN_POOL == 0 && proc_loc != GlobalV::RANK_IN_POOL)
        {
            MPI_Recv(zpiece, ncxy, MPI_DOUBLE, proc_loc, tag, POOL_WORLD, &ierror);
        }

        if (GlobalV::RANK_IN_POOL == 0)
        {
            for (int ixy = 0; ixy < this->ncxy; ++ixy)
            {
                rhotot[ixy * ncz + iz] = zpiece[ixy];
            }
        }
    }

    delete[] zpiece;

    return;
}
#endif

void Parallel_Grid::init_final_scf(const int& ncx_in,
                                   const int& ncy_in,
                                   const int& ncz_in,
                                   const int& nczp_in,
                                   const int& nrxx_in,
                                   const int& nbz_in,
                                   const int& bz_in)
{

    ModuleBase::TITLE("Parallel_Grid", "init");

    this->ncx = ncx_in;
    this->ncy = ncy_in;
    this->ncz = ncz_in;
    this->nczp = nczp_in;
    this->nrxx = nrxx_in;
    this->nbz = nbz_in;
    this->bz = bz_in;

    if (nczp < 0)
    {
        GlobalV::ofs_warning << " nczp = " << nczp << std::endl;
        ModuleBase::WARNING_QUIT("Parallel_Grid::init", "nczp<0");
    }

    assert(ncx > 0);
    assert(ncy > 0);
    assert(ncz > 0);

    this->ncxy = ncx * ncy;
    this->ncxyz = ncxy * ncz;

#ifndef __MPI
    return;
#endif

    // (2)
    assert(allocate_final_scf == false);
    assert(GlobalV::KPAR > 0);

    this->nproc_in_pool = new int[GlobalV::KPAR];
    const int remain_pro = GlobalV::NPROC % GlobalV::KPAR;
    for (int i = 0; i < GlobalV::KPAR; i++)
    {
        nproc_in_pool[i] = GlobalV::NPROC / GlobalV::KPAR;
        if (i < remain_pro)
        {
            this->nproc_in_pool[i]++;
        }
    }

    this->numz = new int*[GlobalV::KPAR];
    this->startz = new int*[GlobalV::KPAR];
    this->whichpro = new int*[GlobalV::KPAR];

    for (int ip = 0; ip < GlobalV::KPAR; ip++)
    {
        const int nproc = nproc_in_pool[ip];
        this->numz[ip] = new int[nproc];
        this->startz[ip] = new int[nproc];
        this->whichpro[ip] = new int[this->ncz];
        ModuleBase::GlobalFunc::ZEROS(this->numz[ip], nproc);
        ModuleBase::GlobalFunc::ZEROS(this->startz[ip], nproc);
        ModuleBase::GlobalFunc::ZEROS(this->whichpro[ip], this->ncz);
    }

    this->allocate_final_scf = true;
    this->z_distribution();

    return;
}
