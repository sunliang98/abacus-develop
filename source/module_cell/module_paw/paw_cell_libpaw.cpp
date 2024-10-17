#include "module_base/tool_quit.h"
#include "module_base/tool_title.h"
#include "module_parameter/parameter.h"
#include "paw_cell.h"
#include "module_base/global_variable.h"
#include "module_base/parallel_common.h"

// The subroutines here are used to gather information from the main ABACUS program
// 1. ecut, ecutpaw : kinetic energy cutoff of the planewave basis set
// there will be one coarse grid for density/potential, and a fine grid for PAW
// the unit is in Hartree
// 2. rprimd, gprimd : real and reciprocal space lattice vectors, respectively
// unit for rprimd is in Bohr, and for gprimd is in Bohr^-1
// 3. gmet : reciprocal space metric (bohr^-2)
// 4. ucvol : volume of unit cell (Bohr^3)
// 5. ngfft, ngfftdg : dimension of FFT grids of the corase and fine grids
// 6. natom, ntypat, typat: #. atoms, #. element types
// and typat records the type of each atom
// 7. xred : coordinate of each atom, in terms of rprimd (namely, direct coordinate)
// 8. filename_list : filename of the PAW xml files for each element

// Cutoff energies, sets ecut and ecutpaw
void Paw_Cell::set_libpaw_ecut(const double ecut_in, const double ecutpaw_in)
{
    ModuleBase::TITLE("Paw_Cell", "set_libpaw_ecut");
    ecut = ecut_in;
    ecutpaw = ecutpaw_in;
}

// inverse of 3 by 3 matrix, needed by set_libpaw_cell to calculate gprimd
// adapted from m_symtk/matr3inv of ABINIT

void matr3inv(std::vector<double>& mat_in, std::vector<double>& mat_out)
{

    assert(mat_in.size() == 9);
    assert(mat_out.size() == 9);

    double t1 = mat_in[4] * mat_in[8] - mat_in[7] * mat_in[5];
    double t2 = mat_in[7] * mat_in[2] - mat_in[1] * mat_in[8];
    double t3 = mat_in[1] * mat_in[5] - mat_in[4] * mat_in[2];
    double det = mat_in[0] * t1 + mat_in[3] * t2 + mat_in[6] * t3;

    double dd;
    if (std::abs(det) > 1e-16)
    {
        dd = 1.0 / det;
    }
    else
    {
        ModuleBase::WARNING_QUIT("matr3inv", "matrix is singular!");
    }

    mat_out[0] = t1 * dd;
    mat_out[3] = t2 * dd;
    mat_out[6] = t3 * dd;
    mat_out[1] = (mat_in[6] * mat_in[5] - mat_in[3] * mat_in[8]) * dd;
    mat_out[4] = (mat_in[0] * mat_in[8] - mat_in[6] * mat_in[2]) * dd;
    mat_out[7] = (mat_in[3] * mat_in[2] - mat_in[0] * mat_in[5]) * dd;
    mat_out[2] = (mat_in[3] * mat_in[7] - mat_in[6] * mat_in[4]) * dd;
    mat_out[5] = (mat_in[6] * mat_in[1] - mat_in[0] * mat_in[7]) * dd;
    mat_out[8] = (mat_in[0] * mat_in[4] - mat_in[3] * mat_in[1]) * dd;
}

// calculates G = A^T A for 3 by 3 matrix
// G_ij = sum_k A_ki A_kj

void mattmat(std::vector<double>& mat_in, std::vector<double>& mat_out)
{
    mat_out[0] = mat_in[0] * mat_in[0] + mat_in[1] * mat_in[1] + mat_in[2] * mat_in[2];
    mat_out[1] = mat_in[0] * mat_in[3] + mat_in[1] * mat_in[4] + mat_in[2] * mat_in[5];
    mat_out[2] = mat_in[0] * mat_in[6] + mat_in[1] * mat_in[7] + mat_in[2] * mat_in[8];
    mat_out[3] = mat_in[3] * mat_in[0] + mat_in[4] * mat_in[1] + mat_in[5] * mat_in[2];
    mat_out[4] = mat_in[3] * mat_in[3] + mat_in[4] * mat_in[4] + mat_in[5] * mat_in[5];
    mat_out[5] = mat_in[3] * mat_in[6] + mat_in[4] * mat_in[7] + mat_in[5] * mat_in[8];
    mat_out[6] = mat_in[6] * mat_in[0] + mat_in[7] * mat_in[1] + mat_in[8] * mat_in[2];
    mat_out[7] = mat_in[6] * mat_in[3] + mat_in[7] * mat_in[4] + mat_in[8] * mat_in[5];
    mat_out[8] = mat_in[6] * mat_in[6] + mat_in[7] * mat_in[7] + mat_in[8] * mat_in[8];
}

// Sets rprimd, gprimd, gmet and ucvol
// Only real space lattice vector needed, others are to be calculated
void Paw_Cell::set_libpaw_cell(const ModuleBase::Matrix3 latvec, const double lat0)
{
    ModuleBase::TITLE("Paw_Cell", "set_libpaw_cell");

    rprimd.resize(9);
    gprimd.resize(9);
    gmet.resize(9);

    rprimd[0] = latvec.e11 * lat0;
    rprimd[1] = latvec.e12 * lat0;
    rprimd[2] = latvec.e13 * lat0;
    rprimd[3] = latvec.e21 * lat0;
    rprimd[4] = latvec.e22 * lat0;
    rprimd[5] = latvec.e23 * lat0;
    rprimd[6] = latvec.e31 * lat0;
    rprimd[7] = latvec.e32 * lat0;
    rprimd[8] = latvec.e33 * lat0;

    // calculating gprimd, gmet and ucvol, adapted from m_geometry/metric of ABINIT

    // Compute first dimensional primitive translation vectors in reciprocal space
    // gprimd from rprimd
    // Then, computes metrics for real and recip space rmet and gmet using length
    // dimensional primitive translation vectors in columns of rprimd(3,3) and gprimd(3,3).
    //  gprimd is the inverse transpose of rprimd.
    //  i.e. $ rmet_{i,j}= \sum_k ( rprimd_{k,i}*rprimd_{k,j} )  $
    //       $ gmet_{i,j}= \sum_k ( gprimd_{k,i}*gprimd_{k,j} )  $
    // Also computes unit cell volume ucvol in $\textrm{bohr}^3$

    // Compute unit cell volume
    // ucvol=rprimd(1,1)*(rprimd(2,2)*rprimd(3,3)-rprimd(3,2)*rprimd(2,3))+&
    //       rprimd(2,1)*(rprimd(3,2)*rprimd(1,3)-rprimd(1,2)*rprimd(3,3))+&
    //       rprimd(3,1)*(rprimd(1,2)*rprimd(2,3)-rprimd(2,2)*rprimd(1,3))
    ucvol = rprimd[0] * (rprimd[4] * rprimd[8] - rprimd[7] * rprimd[5])
            + rprimd[3] * (rprimd[7] * rprimd[2] - rprimd[1] * rprimd[8])
            + rprimd[6] * (rprimd[1] * rprimd[5] - rprimd[4] * rprimd[2]);
    // ucvol = det3r(rprimd)

    if (std::abs(ucvol) < 1e-12)
    {
        ModuleBase::WARNING_QUIT("set_libpaw_cell", "ucvol vanishingly small");
    }

    // ABACUS allows negative volume, but it seems ABINIT do not
    // I am not quite sure why, but to avoid possible complications I will follow ABINIT here
    // as the user can always just exchange two axis to make it positive
    if (ucvol < 0)
    {
        ModuleBase::WARNING_QUIT("set_libpaw_cell", "ucvol negative, one way to solve is to exchange two cell axis");
    }

    // Generate gprimd
    matr3inv(rprimd, gprimd);

    // Compute reciprocal space metric.
    mattmat(gprimd, gmet);
}

// FFT grid information, sets ngfft and ngfftdg
void Paw_Cell::set_libpaw_fft(const int nx_in, const int ny_in, const int nz_in,
        const int nxdg_in, const int nydg_in, const int nzdg_in,
        const int * start_z_in, const int * num_z_in)
{
    ModuleBase::TITLE("Paw_Cell", "set_libpaw_fft");
    ngfft.resize(3);
    ngfftdg.resize(3);

    ngfft[0] = nx_in;
    ngfft[1] = ny_in;
    ngfft[2] = nz_in;

    nx = nx_in;
    ny = ny_in;
    nz = nz_in;

    ngfftdg[0] = nxdg_in;
    ngfftdg[1] = nydg_in;
    ngfftdg[2] = nzdg_in;
    nfft = ngfftdg[0]*ngfftdg[1]*ngfftdg[2];

#ifdef __MPI
    start_z.resize(GlobalV::NPROC);
    num_z.resize(GlobalV::NPROC);
    for(int iproc = 0; iproc < GlobalV::NPROC; iproc ++)
    {
        start_z[iproc] = start_z_in[iproc];
        num_z[iproc] = num_z_in[iproc];
    }
#endif
}

// Sets natom, ntypat, typat and xred
// !!!!!!!Note : the index stored in typat here will start from 1, not 0 !!!!!!
void Paw_Cell::set_libpaw_atom(const int natom_in, const int ntypat_in, const int* typat_in, const double* xred_in)
{
    ModuleBase::TITLE("Paw_Cell", "set_libpaw_atom");
    natom = natom_in;
    ntypat = ntypat_in;

    typat.resize(natom);
    xred.resize(3 * natom);
    epsatm.resize(ntypat);
    
    for(int iat = 0; iat < natom; iat ++)
    {
        typat[iat] = typat_in[iat];
        for(int j = 0; j < 3; j ++)
        {
          xred[3 * iat + j] = xred_in[3 * iat + j];
        }
    }
}

void Paw_Cell::mix_dij(const int iat, double*dij_paw)
{
    double mixing_beta = 0.1;

    std::cout << "mixing_beta : " << mixing_beta << std::endl;

    const int it = atom_type[iat];
    const int nproj = paw_element_list[it].get_mstates();
    const int size_dij = nproj * (nproj+1) / 2;
    for(int i = 0; i < size_dij * nspden; i ++)
    {  
        if(!first_iter) { dij_paw[i] = dij_save[iat][i] * (1.0 - mixing_beta) + dij_paw[i] * mixing_beta;
}

        if(count > 30) { dij_paw[i] = dij_save[iat][i];
}

        dij_save[iat][i] = dij_paw[i];
    }

    first_iter = false;
    count ++;
}

// Sets filename_list
// I'm going to read directly from STRU file
void Paw_Cell::set_libpaw_files()
{
    ModuleBase::TITLE("Paw_Cell", "set_libpaw_files");

    filename_list = new char[ntypat*264];
    if(GlobalV::MY_RANK == 0)
    {
        std::ifstream ifa(PARAM.globalv.global_in_stru.c_str(), std::ios::in);
        if (!ifa)
        {
            ModuleBase::WARNING_QUIT("set_libpaw_files", "can not open stru file");
        }

        std::string line;
        while(!ifa.eof())
        {
            getline(ifa,line);
            if (line.find("PAW_FILES") != std::string::npos) { break;
}
        }

        for(int i = 0; i < ntypat*264; i++)
        {
            filename_list[i] = ' ';
        }
        for(int i = 0; i < ntypat; i++)
        {
            std::string filename;
            ifa >> filename;
            for(int j = 0; j < filename.size(); j++)
            {
                filename_list[264*i+j] = filename[j];
            }
        }
    }
#ifdef __MPI
    Parallel_Common::bcast_char(filename_list,ntypat*264);
#endif
}

void Paw_Cell::set_libpaw_xc(const int xclevel_in, const int ixc_in)
{
    ModuleBase::TITLE("Paw_Cell", "set_libpaw_xc");
    xclevel = xclevel_in;
    ixc = ixc_in;
}

void Paw_Cell::set_nspin(const int nspin_in)
{
    ModuleBase::TITLE("Paw_Cell", "set_nspin");
    nspden = nspin_in;
    nsppol = nspin_in;
}

extern "C"
{
    void prepare_libpaw_(double&,double&,double*,double*,double*,double&,int*,int*,
    //                   ecut    ecutpaw gmet    rprimd  gprimd  ucvol   ngfft ngfftdg
        int&,int&,int*,double*,int&,int&,char*,int&,int&,double*);
    //  natom ntypat typat xred ixc xclevel filename_list nspden nsppol

    void get_vloc_ncoret_(int*,   int&,int&, int&,  double*,double*,double*,double&,double*,double*,double*);
    //                    ngfftdg,nfft,natom,ntypat,rprimd, gprimd, gmet,   ucvol,  xred,   vloc,   ncoret

    void set_rhoij_(int&, int&,     int&,      int&,  int*,       double*);
    //              iatom nrhoijsel size_rhoij nspden rhoijselect rhoijp

    void get_nhat_(int&, int&, double*, int*, int&, int&, double*, double*, double&, double*, double*);
    //             natom,ntypat,xred,   ngfft,nfft,nspden,gprimd,  rprimd,  ucvol,   nhat,    nhatgr

    void calculate_dij_(int&, int&, int&, int&,   int&, int&,  double*, double&, double*, double*, double*, double&);
    //                  natom,ntypat,ixc, xclevel,nfft, nspden,xred,    ucvol,   gprimd,  vks,     vxc,     epawdc

    void get_dij_(int&, int&, int&, double*);
    //            iatom,size_dij,nspden,dij

    void get_sij_(int&, int&, double*);

    void init_rho_(int&,  int*,   int&,int&, int&,  double*,double*,double*,double&,double*,double*);
    //             nspden,ngfftdg,nfft,natom,ntypat,rprimd, gprimd, gmet,   ucvol,  xred,   rho

    void paw_force_(int&, int&,  int*, int*,   int&,double*,double*,double*,double*,double*,double&,double*,double*,double*,int&,  double*,double*);
    //              natom,ntypat,typat,ngfftdg,nfft,nhat,   xred,   rprimd, gprimd, gmet,   ucvol,  vtrial, vxc,    rhor,   nspden,grad,   stress
}

void Paw_Cell::prepare_paw()
{
    prepare_libpaw_(ecut, ecutpaw, gmet.data(), rprimd.data(), gprimd.data(), ucvol,
            ngfft.data(), ngfftdg.data(), natom, ntypat, typat.data(), xred.data(),
            ixc, xclevel, filename_list, nspden, nsppol, epsatm.data());
}

void Paw_Cell::get_vloc_ncoret(double* vloc, double* ncoret)
{
    double * vloc_tmp, * ncoret_tmp;
    vloc_tmp = new double[nfft];
    ncoret_tmp = new double[nfft];

#ifdef __MPI
    if(GlobalV::RANK_IN_POOL == 0)
    {
        get_vloc_ncoret_(ngfftdg.data(), nfft, natom, ntypat, rprimd.data(), gprimd.data(),
                gmet.data(), ucvol, xred.data(), ncoret_tmp, vloc_tmp);
    }
    Parallel_Common::bcast_double(vloc_tmp,nfft);
    Parallel_Common::bcast_double(ncoret_tmp,nfft);
#else
    get_vloc_ncoret_(ngfftdg.data(), nfft, natom, ntypat, rprimd.data(), gprimd.data(),
            gmet.data(), ucvol, xred.data(), ncoret_tmp, vloc_tmp);
#endif

    for(int ix = 0; ix < nx; ix ++)
    {
        for(int iy = 0; iy < ny; iy ++)
        {
#ifdef __MPI
            for(int iz = 0; iz < num_z[GlobalV::RANK_IN_POOL]; iz ++)
            {
                int ind_c = ix*ny*num_z[GlobalV::RANK_IN_POOL] + iy*num_z[GlobalV::RANK_IN_POOL] + iz;
                int ind_fortran = (iz+start_z[GlobalV::RANK_IN_POOL])*ny*nx + iy*nx + ix;

                vloc[ind_c] = vloc_tmp[ind_fortran];
                ncoret[ind_c] = ncoret_tmp[ind_fortran];
            }
#else
            for(int iz = 0; iz < nz; iz ++)
            {
                int ind_c = ix*ny*nz + iy*nz + iz;
                int ind_fortran = iz*ny*nx + iy*nx + ix;

                vloc[ind_c] = vloc_tmp[ind_fortran];
                ncoret[ind_c] = ncoret_tmp[ind_fortran];
            }
#endif
        }
    }

    delete[] vloc_tmp;
    delete[] ncoret_tmp;
}

void Paw_Cell::set_rhoij(int iat, int nrhoijsel, int size_rhoij, int* rhoijselect, double* rhoijp)
{
    int iat_fortran = iat + 1; //Fortran index starts from 1 !!!
    set_rhoij_(iat_fortran,nrhoijsel,size_rhoij,nspden,rhoijselect,rhoijp);
}

void Paw_Cell::get_nhat(double** nhat, double* nhatgr)
{
    ModuleBase::TITLE("Paw_Cell", "get_nhat");

    nhatgr = new double[3*nfft];
    double* nhat_tmp;
    nhat_tmp = new double[nfft*nspden];

#ifdef __MPI
    if(GlobalV::RANK_IN_POOL == 0)
    {
        get_nhat_(natom,ntypat,xred.data(),ngfft.data(),nfft,nspden,gprimd.data(),rprimd.data(),
                ucvol,nhat_tmp,nhatgr); 
    }
    Parallel_Common::bcast_double(nhat_tmp,nfft*nspden);
#else
    get_nhat_(natom,ntypat,xred.data(),ngfft.data(),nfft,nspden,gprimd.data(),rprimd.data(),
            ucvol,nhat_tmp,nhatgr);
#endif

    // I'm not sure about this yet !!!
    // need to check for nspin = 2 later
    // Fortran is column major, and rhor is of dimension (nfft, nspden)
    // so presumably should be this way m
    for(int ix = 0; ix < nx; ix ++)
    {
        for(int iy = 0; iy < ny; iy ++)
        {
#ifdef __MPI
            for(int iz = 0; iz < num_z[GlobalV::RANK_IN_POOL]; iz ++)
            {
                int ind_c = ix*ny*num_z[GlobalV::RANK_IN_POOL] + iy*num_z[GlobalV::RANK_IN_POOL] + iz;
                int ind_fortran = (iz+start_z[GlobalV::RANK_IN_POOL])*ny*nx + iy*nx + ix;

                if(nspden == 2)
                {
                    nhat[0][ind_c] = nhat_tmp[ind_fortran+nfft];
                    nhat[1][ind_c] = nhat_tmp[ind_fortran] - nhat_tmp[ind_fortran+nfft];
                }
                else
                {
                    nhat[0][ind_c] = nhat_tmp[ind_fortran];
                }
            }
#else
            for(int iz = 0; iz < nz; iz ++)
            {
                int ind_c = ix*ny*nz + iy*nz + iz;
                int ind_fortran = iz*ny*nx + iy*nx + ix;

                if(nspden == 2)
                {
                    nhat[0][ind_c] = nhat_tmp[ind_fortran+nfft];
                    nhat[1][ind_c] = nhat_tmp[ind_fortran] - nhat_tmp[ind_fortran+nfft];
                }
                else
                {
                    nhat[0][ind_c] = nhat_tmp[ind_fortran];
                }
            }
#endif
        }
    }
    delete[] nhat_tmp;
    delete[] nhatgr;
}

void Paw_Cell::calculate_dij(double* vks, double* vxc)
{
    ModuleBase::TITLE("Paw_Cell", "calculate_dij");
    double * vks_hartree, * vxc_hartree;

#ifdef __MPI
    double * vks_collected, * vxc_collected;
    if(GlobalV::RANK_IN_POOL == 0)
    {
        vks_hartree = new double[nspden * nfft];
        vxc_hartree = new double[nspden * nfft];
        vks_collected = new double[nspden * nfft];
        vxc_collected = new double[nspden * nfft];
    }

    // Collecting vks and vxc from all processes; I hope there could be a better way
    // but this is what I can think of right now.
    const int nxy = nx * ny;
    const int nrxx = nxy * num_z[GlobalV::RANK_IN_POOL];

    for(int is = 0; is < nspden; is ++)
    {
        double * vks_send = new double[num_z[GlobalV::RANK_IN_POOL]];
        double * vxc_send = new double[num_z[GlobalV::RANK_IN_POOL]];
        double * vks_receive = new double[nz];
        double * vxc_receive = new double[nz];
        for(int ixy = 0; ixy < nxy; ixy ++)
        {
            for(int iz = 0; iz < num_z[GlobalV::RANK_IN_POOL]; iz++)
            {
                vks_send[iz] = vks[(ixy*num_z[GlobalV::RANK_IN_POOL] + iz) + is*nrxx];
                vxc_send[iz] = vxc[(ixy*num_z[GlobalV::RANK_IN_POOL] + iz) + is*nrxx];
            }

            MPI_Gatherv(vks_send,num_z[GlobalV::RANK_IN_POOL],MPI_DOUBLE,vks_receive,num_z.data(),start_z.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
            MPI_Gatherv(vxc_send,num_z[GlobalV::RANK_IN_POOL],MPI_DOUBLE,vxc_receive,num_z.data(),start_z.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);

            if(GlobalV::RANK_IN_POOL == 0)
            {
                for(int iz = 0; iz < nz; iz ++)
                {
                    vks_collected[(ixy*nz + iz) + is*nfft] = vks_receive[iz];
                    vxc_collected[(ixy*nz + iz) + is*nfft] = vxc_receive[iz];
                }
            }
        }
        delete[] vks_send;
        delete[] vxc_send;
        delete[] vks_receive;
        delete[] vxc_receive;
    
        if(GlobalV::RANK_IN_POOL == 0)
        {
            for(int ix = 0; ix < nx; ix ++)
            {
                for(int iy = 0; iy < ny; iy ++)
                {
                    for(int iz = 0; iz < nz; iz ++)
                    {
                        int ind_c = (ix*ny*nz + iy*nz + iz) + is * nfft;
                        int ind_fortran = iz*ny*nx + iy*nx + ix + is*nfft;
                        vks_hartree[ind_fortran] = vks_collected[ind_c] / 2.0;
                        vxc_hartree[ind_fortran] = vxc_collected[ind_c] / 2.0;
                    }
                }
            }        
        }
    }

    if(GlobalV::RANK_IN_POOL == 0)
    {
        calculate_dij_(natom,ntypat,ixc,xclevel,nfft,nspden,xred.data(),ucvol,gprimd.data(),vks_hartree,vxc_hartree,epawdc);
    }

    if(GlobalV::RANK_IN_POOL == 0)
    {
        delete[] vks_hartree;
        delete[] vxc_hartree;
        delete[] vks_collected;
        delete[] vxc_collected;
    }

#else
    vks_hartree = new double[nspden * nfft];
    vxc_hartree = new double[nspden * nfft];
    for(int is = 0; is < nspden; is ++)
    {
        for(int ix = 0; ix < nx; ix ++)
        {
            for(int iy = 0; iy < ny; iy ++)
            {
                for(int iz = 0; iz < nz; iz ++)
                {
                    int ind_c = ix*ny*nz + iy*nz + iz + is*nfft;
                    int ind_fortran = is*nfft + iz*ny*nx + iy*nx + ix;
                    vks_hartree[ind_fortran] = vks[ind_c] / 2.0;
                    vxc_hartree[ind_fortran] = vxc[ind_c] / 2.0;
                }
            }
        }
    }
    calculate_dij_(natom,ntypat,ixc,xclevel,nfft,nspden,xred.data(),ucvol,gprimd.data(),vks_hartree,vxc_hartree,epawdc);
    delete[] vks_hartree;
    delete[] vxc_hartree;
#endif
}

void Paw_Cell::extract_dij(int iat, int size_dij, double* dij)
{
    int iat_fortran = iat + 1;
    get_dij_(iat_fortran,size_dij,nspden,dij);
}

void Paw_Cell::extract_sij(int it, int size_sij, double* sij)
{
    int it_fortran = it + 1;
    get_sij_(it_fortran,size_sij,sij);
}

void Paw_Cell::init_rho(double ** rho)
{
    double* rho_tmp;
    rho_tmp = new double[nfft*nspden];

#ifdef __MPI
    if(GlobalV::RANK_IN_POOL == 0)
    {
        init_rho_(nspden, ngfftdg.data(), nfft, natom, ntypat, rprimd.data(), gprimd.data(),
                gmet.data(), ucvol, xred.data(), rho_tmp);      
    }
    Parallel_Common::bcast_double(rho_tmp,nfft*nspden);
#else
    init_rho_(nspden, ngfftdg.data(), nfft, natom, ntypat, rprimd.data(), gprimd.data(),
            gmet.data(), ucvol, xred.data(), rho_tmp);
#endif

#ifdef __MPI
    // I'm not sure about this yet !!!
    // need to check for nspin = 2 later
    // Fortran is column major, and rhor is of dimension (nfft, nspden)
    // so presumably should be this way
    for(int ix = 0; ix < nx; ix ++)
    {
        for(int iy = 0; iy < ny; iy ++)
        {
            for(int iz = 0; iz < num_z[GlobalV::RANK_IN_POOL]; iz ++)
            {
                int ind_c = ix*ny*num_z[GlobalV::RANK_IN_POOL] + iy*num_z[GlobalV::RANK_IN_POOL] + iz;
                int ind_fortran = (iz+start_z[GlobalV::RANK_IN_POOL])*ny*nx + iy*nx + ix;

                if(nspden == 2)
                {
                    rho[0][ind_c] = rho_tmp[ind_fortran+nfft];
                    rho[1][ind_c] = rho_tmp[ind_fortran] - rho_tmp[ind_fortran+nfft];
                }
                else
                {
                    rho[0][ind_c] = rho_tmp[ind_fortran];
                }
            }
        }
    }

#else
    for(int ix = 0; ix < nx; ix ++)
    {
        for(int iy = 0; iy < ny; iy ++)
        {
            for(int iz = 0; iz < nz; iz ++)
            {
                int ind_c = ix*ny*nz + iy*nz + iz;
                int ind_fortran = iz*ny*nx + iy*nx + ix;

                if(nspden == 2)
                {
                    rho[0][ind_c] = rho_tmp[ind_fortran+nfft];
                    rho[1][ind_c] = rho_tmp[ind_fortran] - rho_tmp[ind_fortran+nfft];
                }
                else
                {
                    rho[0][ind_c] = rho_tmp[ind_fortran];
                }
            }
        }
    }
#endif
    delete[] rho_tmp;
}

void Paw_Cell::set_dij()
{
    for(int iat = 0; iat < nat; iat ++)
    {
        const int it = atom_type[iat];
        const int nproj = paw_element_list[it].get_mstates();
        const int size_dij = nproj * (nproj+1) / 2;

        double* dij_libpaw = new double[size_dij * nspden];
        double** dij;
        dij = new double*[nspden];
        for(int is = 0; is < nspden; is ++)
        {
           dij[is] = new double[nproj * nproj];
        }

#ifdef __MPI
        if(GlobalV::RANK_IN_POOL == 0)
        {
            extract_dij(iat,size_dij,dij_libpaw);
            //mix_dij(iat,dij_libpaw);
        }
        Parallel_Common::bcast_double(dij_libpaw,size_dij*nspden);
#else
        extract_dij(iat,size_dij,dij_libpaw);
        //mix_dij(iat,dij_libpaw);
#endif

        for(int is = 0; is < nspden; is ++)
        {
            for(int jproj = 0; jproj < nproj; jproj ++)
            {
                for(int iproj = jproj; iproj < nproj; iproj ++)
                {
                    const int ind = iproj * (iproj+1) / 2 + jproj + is*nproj * (nproj+1) / 2;
                    dij[is][iproj*nproj+jproj] = dij_libpaw[ind] * 2.0; //hartree to rydberg
                    dij[is][jproj*nproj+iproj] = dij_libpaw[ind] * 2.0;
                }
            }
        }
        paw_atom_list[iat].set_dij(dij);

        delete[] dij_libpaw;
        for(int is = 0; is < nspden; is ++)
        {
            delete[] dij[is];
        }
        delete[] dij;
    }
}

void Paw_Cell::set_sij()
{
    int at_ind = 0;
    for(int it = 0; it < ntypat; it ++)
    {   
        const int nproj = paw_element_list[it].get_mstates();
        const int size_sij = nproj * (nproj+1) / 2 * nspden;
        double* sij_libpaw = new double[size_sij];
        double* sij = new double[nproj * nproj];

#ifdef __MPI
        if(GlobalV::RANK_IN_POOL == 0) { extract_sij(it,size_sij,sij_libpaw);
}
        Parallel_Common::bcast_double(sij_libpaw,size_sij*nspden);
#else
        extract_sij(it,size_sij,sij_libpaw);
#endif

        for(int jproj = 0; jproj < nproj; jproj ++)
        {
            for(int iproj = jproj; iproj < nproj; iproj ++)
            {
                const int ind = iproj * (iproj+1) / 2 + jproj;
                sij[iproj*nproj+jproj] = sij_libpaw[ind];
                sij[jproj*nproj+iproj] = sij_libpaw[ind];
            }
        }
        const int na = nat_type[it];
        for(int ia = 0; ia < na; ia ++)
        {
            int iat = atom_map[at_ind];
            at_ind ++;
            paw_atom_list[iat].set_sij(sij);
        }

        delete[] sij_libpaw;
        delete[] sij;
    }
}

double Paw_Cell::calculate_ecore()
{
    double charge = 0.0;
    double esum = 0.0;
    for(int ia = 0; ia < natom; ia ++)
    {
        const int it = atom_type[ia];
        esum += epsatm[it];
        charge += paw_element_list[it].get_zval();
    }
    return esum * charge / ucvol * 2.0;
}

void Paw_Cell::calculate_force(double* vks, double* vxc, double * rhor, double* force)
{
    ModuleBase::TITLE("Paw_Cell", "calculate_force");
    double * vks_hartree, * vxc_hartree, * rhor_fortran;
    double * grad, * nlstr;

    grad = new double[3*natom];
    nlstr = new double[6];

    for(int i = 0; i < 3*natom; i ++)
    {
        grad[i] = 0.0;
    }
    for(int i = 0; i < 6; i ++)
    {
        nlstr[i] = 0.0;
    }

#ifdef __MPI
    double * vks_collected, * vxc_collected, *rhor_collected;
    if(GlobalV::RANK_IN_POOL == 0)
    {
        vks_hartree = new double[nspden * nfft];
        vxc_hartree = new double[nspden * nfft];
        rhor_fortran = new double[nfft];
        vks_collected = new double[nspden * nfft];
        vxc_collected = new double[nspden * nfft];
        rhor_collected = new double[nfft];
    }

    // Collecting vks and vxc from all processes; I hope there could be a better way
    // but this is what I can think of right now.
    const int nxy = nx * ny;
    const int nrxx = nxy * num_z[GlobalV::RANK_IN_POOL];

    for(int is = 0; is < nspden; is ++)
    {
        double * vks_send = new double[num_z[GlobalV::RANK_IN_POOL]];
        double * vxc_send = new double[num_z[GlobalV::RANK_IN_POOL]];
        double * vks_receive = new double[nz];
        double * vxc_receive = new double[nz];
        for(int ixy = 0; ixy < nxy; ixy ++)
        {
            for(int iz = 0; iz < num_z[GlobalV::RANK_IN_POOL]; iz++)
            {
                vks_send[iz] = vks[(ixy*num_z[GlobalV::RANK_IN_POOL] + iz) + is*nrxx];
                vxc_send[iz] = vxc[(ixy*num_z[GlobalV::RANK_IN_POOL] + iz) + is*nrxx];
            }

            MPI_Gatherv(vks_send,num_z[GlobalV::RANK_IN_POOL],MPI_DOUBLE,vks_receive,num_z.data(),start_z.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
            MPI_Gatherv(vxc_send,num_z[GlobalV::RANK_IN_POOL],MPI_DOUBLE,vxc_receive,num_z.data(),start_z.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);

            if(GlobalV::RANK_IN_POOL == 0)
            {
                for(int iz = 0; iz < nz; iz ++)
                {
                    vks_collected[(ixy*nz + iz) + is*nfft] = vks_receive[iz];
                    vxc_collected[(ixy*nz + iz) + is*nfft] = vxc_receive[iz];
                }
            }
        }
        delete[] vks_send;
        delete[] vxc_send;
        delete[] vks_receive;
        delete[] vxc_receive;
    
        if(GlobalV::RANK_IN_POOL == 0)
        {
            for(int ix = 0; ix < nx; ix ++)
            {
                for(int iy = 0; iy < ny; iy ++)
                {
                    for(int iz = 0; iz < nz; iz ++)
                    {
                        int ind_c = (ix*ny*nz + iy*nz + iz) + is * nfft;
                        int ind_fortran = iz*ny*nx + iy*nx + ix + is*nfft;
                        vks_hartree[ind_fortran] = vks_collected[ind_c] / 2.0;
                        vxc_hartree[ind_fortran] = vxc_collected[ind_c] / 2.0;
                    }
                }
            }        
        }
    }

    double* rhor_send = new double[num_z[GlobalV::RANK_IN_POOL]];
    double* rhor_receive = new double[nz];
    for(int ixy = 0; ixy < nxy; ixy ++)
    {
        for(int iz = 0; iz < num_z[GlobalV::RANK_IN_POOL]; iz++)
        {
            rhor_send[iz] = rhor[ixy*num_z[GlobalV::RANK_IN_POOL] + iz];
        }

        MPI_Gatherv(rhor_send,num_z[GlobalV::RANK_IN_POOL],MPI_DOUBLE,rhor_receive,num_z.data(),start_z.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);

        if(GlobalV::RANK_IN_POOL == 0)
        {
            for(int iz = 0; iz < nz; iz ++)
            {
                rhor_collected[ixy*nz + iz] = rhor_receive[iz];
            }
        }
    }
    delete[] rhor_send;
    delete[] rhor_receive;

    if(GlobalV::RANK_IN_POOL == 0)
    {
        for(int ix = 0; ix < nx; ix ++)
        {
            for(int iy = 0; iy < ny; iy ++)
            {
                for(int iz = 0; iz < nz; iz ++)
                {
                    int ind_c = ix*ny*nz + iy*nz + iz;
                    int ind_fortran = iz*ny*nx + iy*nx + ix;
                    rhor_fortran[ind_fortran] = rhor_collected[ind_c];
                    rhor_fortran[ind_fortran] = rhor_collected[ind_c];
                }
            }
        }        
    }

    if(GlobalV::RANK_IN_POOL == 0)
    {
        double* nhatgr;
        nhatgr = new double[3*nfft];
        double* nhat_tmp;
        nhat_tmp = new double[nfft*nspden];

        get_nhat_(natom,ntypat,xred.data(),ngfft.data(),nfft,nspden,gprimd.data(),rprimd.data(),
                ucvol,nhat_tmp,nhatgr);

        paw_force_(natom,ntypat,typat.data(),ngfft.data(),nfft,nhat_tmp,xred.data(),
                rprimd.data(),gprimd.data(),gmet.data(),ucvol,vks_hartree,vxc_hartree,rhor_fortran,nspden,grad,nlstr);

        delete[] nhatgr;
        delete[] nhat_tmp;
    }

    Parallel_Common::bcast_double(grad,3*natom);
    Parallel_Common::bcast_double(nlstr,6);

    if(GlobalV::RANK_IN_POOL == 0)
    {
        delete[] vks_hartree;
        delete[] vxc_hartree;
        delete[] vks_collected;
        delete[] vxc_collected;
        delete[] rhor_collected;
        delete[] rhor_fortran;
    }

#else
    vks_hartree = new double[nspden * nfft];
    vxc_hartree = new double[nspden * nfft];
    rhor_fortran = new double[nfft];
    for(int is = 0; is < nspden; is ++)
    {
        for(int ix = 0; ix < nx; ix ++)
        {
            for(int iy = 0; iy < ny; iy ++)
            {
                for(int iz = 0; iz < nz; iz ++)
                {
                    int ind_c = ix*ny*nz + iy*nz + iz + is*nfft;
                    int ind_fortran = is*nfft + iz*ny*nx + iy*nx + ix;
                    vks_hartree[ind_fortran] = vks[ind_c] / 2.0;
                    vxc_hartree[ind_fortran] = vxc[ind_c] / 2.0;
                }
            }
        }
    }
    for(int ix = 0; ix < nx; ix ++)
    {
        for(int iy = 0; iy < ny; iy ++)
        {
            for(int iz = 0; iz < nz; iz ++)
            {
                int ind_c = ix*ny*nz + iy*nz + iz;
                int ind_fortran = iz*ny*nx + iy*nx + ix;
                rhor_fortran[ind_fortran] = rhor[ind_c];
                rhor_fortran[ind_fortran] = rhor[ind_c];                
            }
        }
    }
    double* nhatgr;
    nhatgr = new double[3*nfft];
    double* nhat_tmp;
    nhat_tmp = new double[nfft*nspden];

    get_nhat_(natom,ntypat,xred.data(),ngfft.data(),nfft,nspden,gprimd.data(),rprimd.data(),
            ucvol,nhat_tmp,nhatgr);

    paw_force_(natom,ntypat,typat.data(),ngfft.data(),nfft,nhat_tmp,xred.data(),
            rprimd.data(),gprimd.data(),gmet.data(),ucvol,vks_hartree,vxc_hartree,rhor_fortran,nspden,grad,nlstr);

    delete[] nhatgr;
    delete[] nhat_tmp;

    delete[] vks_hartree;
    delete[] vxc_hartree;
    delete[] rhor_fortran;
#endif

    for(int i = 0; i < 3*natom; i++)
    {
        force[i] = 0.0;
    }

    for(int iat = 0; iat < natom; iat ++)
    {
        for(int mu = 0; mu < 3; mu ++)
        {
            force[3*iat+mu] -= (grad[3*iat] * gprimd[mu] +
                grad[3*iat+1] * gprimd[mu+3] + grad[3*iat+2] * gprimd[mu+6]);
        }
    }

    delete[] grad;
    delete[] nlstr;

}