#include "pw_basis_k.h"

#include "source_base/constants.h"
#include "source_base/memory.h"
#include "source_base/timer.h"
#include "module_parameter/parameter.h"

#include <utility>
namespace ModulePW
{

PW_Basis_K::PW_Basis_K()
{
    classname = "PW_Basis_K";
    this->fft_bundle.setfft(this->device, this->precision);
}
PW_Basis_K::~PW_Basis_K()
{
    delete[] kvec_d;
    delete[] kvec_c;
    delete[] npwk;
    delete[] igl2isz_k;
    delete[] igl2ig_k;
    delete[] gk2;
#if defined(__CUDA) || defined(__ROCM)
    if (this->device == "gpu")
    {
        delmem_sd_op()(this->s_kvec_c);
        delmem_sd_op()(this->s_gcar);
        delmem_sd_op()(this->s_gk2);
        delmem_dd_op()(this->d_gcar);
        delmem_dd_op()(this->d_gk2);
        delmem_dd_op()(this->d_kvec_c);
        delmem_int_op()(this->ig2ixyz_k);
        delmem_int_op()(this->d_igl2isz_k);
    }
    else
    {
#endif
        delmem_sh_op()(this->s_kvec_c);
        delmem_sh_op()(this->s_gcar);
        delmem_sh_op()(this->s_gk2);
        // There's no need to delete double pointers while in a CPU environment.
#if defined(__CUDA) || defined(__ROCM)
    }
#endif
}

void PW_Basis_K::initparameters(const bool gamma_only_in,
                                const double gk_ecut_in,
                                const int nks_in,                             // number of k points in this pool
                                const ModuleBase::Vector3<double>* kvec_d_in, // Direct coordinates of k points
                                const int distribution_type_in,
                                const bool xprime_in)
{
    this->nks = nks_in;
    delete[] this->kvec_d;
    this->kvec_d = new ModuleBase::Vector3<double>[nks];
    delete[] this->kvec_c;
    this->kvec_c = new ModuleBase::Vector3<double>[nks];

    double kmaxmod = 0;
    for (int ik = 0; ik < this->nks; ++ik)
    {
        this->kvec_d[ik] = kvec_d_in[ik];
        this->kvec_c[ik] = this->kvec_d[ik] * this->G;
        double kmod = sqrt(this->kvec_c[ik] * this->kvec_c[ik]);
        if (kmod > kmaxmod)
        {
            kmaxmod = kmod;
        }
    }
    this->gk_ecut = gk_ecut_in / this->tpiba2;
    this->ggecut = pow(sqrt(this->gk_ecut) + kmaxmod, 2);
    if (this->ggecut > this->gridecut_lat)
    {
        this->ggecut = this->gridecut_lat;
        this->gk_ecut = pow(sqrt(this->ggecut) - kmaxmod, 2);
    }

    this->gamma_only = gamma_only_in;
    if (kmaxmod > 0)
    {
        this->gamma_only = false; // if it is not the gamma point, we do not use gamma_only
    }
    this->xprime = xprime_in;
    this->fftny = this->ny;
    this->fftnx = this->nx;
    if (this->gamma_only)
    {
        if (this->xprime)
        {
            this->fftnx = int(this->nx / 2) + 1;
        }
        else
        {
            this->fftny = int(this->ny / 2) + 1;
        }
    }
    this->fftnz = this->nz;
    this->fftnxy = this->fftnx * this->fftny;
    this->fftnxyz = this->fftnxy * this->fftnz;
    this->distribution_type = distribution_type_in;
#if defined(__CUDA) || defined(__ROCM)
    if (this->device == "gpu")
    {
        if (this->float_data_)
        {
            resmem_sd_op()(this->s_kvec_c, this->nks * 3);
            castmem_d2s_h2d_op()(this->s_kvec_c, reinterpret_cast<double*>(&this->kvec_c[0][0]), this->nks * 3);
        }
        resmem_dd_op()(this->d_kvec_c, this->nks * 3);
        syncmem_d2d_h2d_op()(this->d_kvec_c, reinterpret_cast<double*>(&this->kvec_c[0][0]), this->nks * 3);
    }
    else
    {
#endif
        if (this->float_data_)
        {
            resmem_sh_op()(this->s_kvec_c, this->nks * 3);
            castmem_d2s_h2h_op()(this->s_kvec_c, reinterpret_cast<double*>(&this->kvec_c[0][0]), this->nks * 3);
        }
        this->d_kvec_c = reinterpret_cast<double*>(&this->kvec_c[0][0]);
        // There's no need to allocate double pointers while in a CPU environment.
#if defined(__CUDA) || defined(__ROCM)
    }
#endif
}

void PW_Basis_K::setupIndGk()
{
    // count npwk
    this->npwk_max = 0;
    delete[] this->npwk;
    this->npwk = new int[this->nks];
    for (int ik = 0; ik < this->nks; ik++)
    {
        int ng = 0;
        for (int ig = 0; ig < this->npw; ig++)
        {
            const double gk2 = this->cal_GplusK_cartesian(ik, ig).norm2();
            if (gk2 <= this->gk_ecut)
            {
                ++ng;
            }
        }
        this->npwk[ik] = ng;
        ModuleBase::CHECK_WARNING_QUIT((ng == 0),
                                       "pw_basis_k.cpp",
                                       PARAM.inp.calculation,
                                       "Current core has no plane waves! Please reduce the cores.");
        if (this->npwk_max < ng)
        {
            this->npwk_max = ng;
        }
    }

    // get igl2isz_k and igl2ig_k
    if (this->npwk_max <= 0)
    {
        return;
    }

    delete[] igl2isz_k;
    this->igl2isz_k = new int[this->nks * this->npwk_max];
    delete[] igl2ig_k;
    this->igl2ig_k = new int[this->nks * this->npwk_max];
    for (int ik = 0; ik < this->nks; ik++)
    {
        int igl = 0;
        for (int ig = 0; ig < this->npw; ig++)
        {
            const double gk2 = this->cal_GplusK_cartesian(ik, ig).norm2();
            if (gk2 <= this->gk_ecut)
            {
                this->igl2isz_k[ik * npwk_max + igl] = this->ig2isz[ig];
                this->igl2ig_k[ik * npwk_max + igl] = ig;
                ++igl;
            }
        }
    }
#if defined(__CUDA) || defined(__ROCM)
    if (this->device == "gpu")
    {
        resmem_int_op()(this->d_igl2isz_k, this->npwk_max * this->nks);
        syncmem_int_h2d_op()(this->d_igl2isz_k, this->igl2isz_k, this->npwk_max * this->nks);
    }
#endif
    this->get_ig2ixyz_k();
    return;
}

///
/// distribute plane wave basis and real-space grids to different processors
/// set up maps for fft and create arrays for MPI_Alltoall
/// set up ffts
///
void PW_Basis_K::setuptransform()
{
    ModuleBase::timer::tick(this->classname, "setuptransform");
    this->distribute_r();
    this->distribute_g();
    this->getstartgr();
    this->setupIndGk();
    this->fft_bundle.clear();
    std::string fft_device = this->device;
#if defined(__DSP)
    fft_device = "dsp";
#endif
    this->fft_bundle.setfft(fft_device, this->precision);
    if (this->xprime)
    {
        this->fft_bundle.initfft(this->nx,
                                 this->ny,
                                 this->nz,
                                 this->lix,
                                 this->rix,
                                 this->nst,
                                 this->nplane,
                                 this->poolnproc,
                                 this->gamma_only,
                                 this->xprime);
    }
    else
    {
        this->fft_bundle.initfft(this->nx,
                                 this->ny,
                                 this->nz,
                                 this->liy,
                                 this->riy,
                                 this->nst,
                                 this->nplane,
                                 this->poolnproc,
                                 this->gamma_only,
                                 this->xprime);
    }
    this->fft_bundle.setupFFT();
    ModuleBase::timer::tick(this->classname, "setuptransform");
}

void PW_Basis_K::collect_local_pw(const double& erf_ecut_in, const double& erf_height_in, const double& erf_sigma_in)
{
    this->erf_ecut = erf_ecut_in;
    this->erf_height = erf_height_in;
    this->erf_sigma = erf_sigma_in;
    if (this->npwk_max <= 0)
    {
        return;
    }
    delete[] gk2;
    delete[] gcar;
    this->gk2 = new double[this->npwk_max * this->nks];
    this->gcar = new ModuleBase::Vector3<double>[this->npwk_max * this->nks];
    ModuleBase::Memory::record("PW_B_K::gk2", sizeof(double) * this->npwk_max * this->nks);
    ModuleBase::Memory::record("PW_B_K::gcar", sizeof(ModuleBase::Vector3<double>) * this->npwk_max * this->nks);

    ModuleBase::Vector3<double> f;
    for (int ik = 0; ik < this->nks; ++ik)
    {
        ModuleBase::Vector3<double> kv = this->kvec_d[ik];
        for (int igl = 0; igl < this->npwk[ik]; ++igl)
        {
            int isz = this->igl2isz_k[ik * npwk_max + igl];
            int iz = isz % this->nz;
            int is = isz / this->nz;
            int ixy = this->is2fftixy[is];
            int ix = ixy / this->fftny;
            int iy = ixy % this->fftny;
            if (ix >= int(this->nx / 2) + 1)
            {
                ix -= this->nx;
            }
            if (iy >= int(this->ny / 2) + 1)
            {
                iy -= this->ny;
            }
            if (iz >= int(this->nz / 2) + 1)
            {
                iz -= this->nz;
            }
            f.x = ix;
            f.y = iy;
            f.z = iz;

            this->gcar[ik * npwk_max + igl] = f * this->G;
            double temp_gk2 = (f + kv) * (this->GGT * (f + kv));
            if (erf_height > 0)
            {
                this->gk2[ik * npwk_max + igl]
                    = temp_gk2 + erf_height / tpiba2 * (1.0 + std::erf((temp_gk2 * tpiba2 - erf_ecut) / erf_sigma));
            }
            else
            {
                this->gk2[ik * npwk_max + igl] = temp_gk2;
            }
        }
    }
#if defined(__CUDA) || defined(__ROCM)
    if (this->device == "gpu")
    {
        if (this->float_data_)
        {
            resmem_sd_op()(this->s_gk2, this->npwk_max * this->nks);
            resmem_sd_op()(this->s_gcar, this->npwk_max * this->nks * 3);
            castmem_d2s_h2d_op()(this->s_gk2, this->gk2, this->npwk_max * this->nks);
            castmem_d2s_h2d_op()(this->s_gcar,
                                 reinterpret_cast<double*>(&this->gcar[0][0]),
                                 this->npwk_max * this->nks * 3);
        }
        if (this->double_data_)
        {
            resmem_dd_op()(this->d_gk2, this->npwk_max * this->nks);
            resmem_dd_op()(this->d_gcar, this->npwk_max * this->nks * 3);
            syncmem_d2d_h2d_op()(this->d_gk2, this->gk2, this->npwk_max * this->nks);
            syncmem_d2d_h2d_op()(this->d_gcar,
                                 reinterpret_cast<double*>(&this->gcar[0][0]),
                                 this->npwk_max * this->nks * 3);
        }
    }
    else
    {
#endif
        if (this->float_data_)
        {
            resmem_sh_op()(this->s_gk2, this->npwk_max * this->nks, "PW_B_K::s_gk2");
            resmem_sh_op()(this->s_gcar, this->npwk_max * this->nks * 3, "PW_B_K::s_gcar");
            castmem_d2s_h2h_op()(this->s_gk2, this->gk2, this->npwk_max * this->nks);
            castmem_d2s_h2h_op()(this->s_gcar,
                                 reinterpret_cast<double*>(&this->gcar[0][0]),
                                 this->npwk_max * this->nks * 3);
        }
        if (this->double_data_)
        {
            this->d_gcar = reinterpret_cast<double*>(&this->gcar[0][0]);
            this->d_gk2 = this->gk2;
        }
        // There's no need to allocate double pointers while in a CPU environment.
#if defined(__CUDA) || defined(__ROCM)
    }
#endif
}

ModuleBase::Vector3<double> PW_Basis_K::cal_GplusK_cartesian(const int ik, const int ig) const
{
    int isz = this->ig2isz[ig];
    int iz = isz % this->nz;
    int is = isz / this->nz;
    int ix = this->is2fftixy[is] / this->fftny;
    int iy = this->is2fftixy[is] % this->fftny;
    if (ix >= int(this->nx / 2) + 1)
    {
        ix -= this->nx;
    }
    if (iy >= int(this->ny / 2) + 1)
    {
        iy -= this->ny;
    }
    if (iz >= int(this->nz / 2) + 1)
    {
        iz -= this->nz;
    }
    ModuleBase::Vector3<double> f;
    f.x = ix;
    f.y = iy;
    f.z = iz;
    f = f * this->G;
    ModuleBase::Vector3<double> g_temp_ = this->kvec_c[ik] + f;
    return g_temp_;
}

double& PW_Basis_K::getgk2(const int ik, const int igl) const
{
    return this->gk2[ik * this->npwk_max + igl];
}

ModuleBase::Vector3<double>& PW_Basis_K::getgcar(const int ik, const int igl) const
{
    return this->gcar[ik * this->npwk_max + igl];
}

ModuleBase::Vector3<double> PW_Basis_K::getgdirect(const int ik, const int igl) const
{
    ModuleBase::Vector3<double> f = this->latvec * this->gcar[ik * this->npwk_max + igl];
    f.x = std::round(f.x);
    f.y = std::round(f.y);
    f.z = std::round(f.z);
    return f;
}

ModuleBase::Vector3<double> PW_Basis_K::getgpluskcar(const int ik, const int igl) const
{
    return this->gcar[ik * this->npwk_max + igl] + this->kvec_c[ik];
}
int& PW_Basis_K::getigl2isz(const int ik, const int igl) const
{
    return this->igl2isz_k[ik * this->npwk_max + igl];
}
int& PW_Basis_K::getigl2ig(const int ik, const int igl) const
{
    return this->igl2ig_k[ik * this->npwk_max + igl];
}

void PW_Basis_K::get_ig2ixyz_k()
{
#if not defined(__DSP)
    if (this->device != "gpu")
    {
        // only GPU need to get ig2ixyz_k
        return;
    }
#endif
    ig2ixyz_k_cpu.resize(this->npwk_max * this->nks);
    ModuleBase::Memory::record("PW_B_K::ig2ixyz", sizeof(int) * this->npwk_max * this->nks);
    assert(gamma_only == false); // We only finish non-gamma_only fft on GPU temperarily.
    for (int ik = 0; ik < this->nks; ++ik)
    {
        for (int igl = 0; igl < this->npwk[ik]; ++igl)
        {
            int isz = this->igl2isz_k[igl + ik * npwk_max];
            int iz = isz % this->nz;
            int is = isz / this->nz;
            int ixy = this->is2fftixy[is];
            int iy = ixy % this->ny;
            int ix = ixy / this->ny;
            ig2ixyz_k_cpu[igl + ik * npwk_max] = iz + iy * nz + ix * ny * nz;
        }
    }
    resmem_int_op()(ig2ixyz_k, this->npwk_max * this->nks);
    syncmem_int_h2d_op()(this->ig2ixyz_k, ig2ixyz_k_cpu.data(), this->npwk_max * this->nks);
}

std::vector<int> PW_Basis_K::get_ig2ix(const int ik) const
{
    std::vector<int> ig_to_ix;
    ig_to_ix.resize(npwk[ik]);

    for (int ig = 0; ig < npwk[ik]; ig++)
    {
        int isz = this->igl2isz_k[ig + ik * npwk_max];
        int is = isz / this->nz;
        int ixy = this->is2fftixy[is];
        int ix = ixy / this->ny;
        if (ix < (nx / 2) + 1)
        {
            ix += nx;
        }
        ig_to_ix[ig] = ix;
    }
    return ig_to_ix;
}

std::vector<int> PW_Basis_K::get_ig2iy(const int ik) const
{
    std::vector<int> ig_to_iy;
    ig_to_iy.resize(npwk[ik]);

    for (int ig = 0; ig < npwk[ik]; ig++)
    {
        int isz = this->igl2isz_k[ig + ik * npwk_max];
        int is = isz / this->nz;
        int ixy = this->is2fftixy[is];
        int iy = ixy % this->ny;
        if (iy < (ny / 2) + 1)
        {
            iy += ny;
        }
        ig_to_iy[ig] = iy;
    }
    return ig_to_iy;
}

std::vector<int> PW_Basis_K::get_ig2iz(const int ik) const
{
    std::vector<int> ig_to_iz;
    ig_to_iz.resize(npwk[ik]);

    for (int ig = 0; ig < npwk[ik]; ig++)
    {
        int isz = this->igl2isz_k[ig + ik * npwk_max];
        int iz = isz % this->nz;
        if (iz < (nz / 2) + 1)
        {
            iz += nz;
        }
        ig_to_iz[ig] = iz;
    }
    return ig_to_iz;
}

template <>
float* PW_Basis_K::get_kvec_c_data() const
{
    return this->s_kvec_c;
}
template <>
double* PW_Basis_K::get_kvec_c_data() const
{
    return this->d_kvec_c;
}

template <>
float* PW_Basis_K::get_gcar_data() const
{
    return this->s_gcar;
}
template <>
double* PW_Basis_K::get_gcar_data() const
{
    return this->d_gcar;
}

template <>
float* PW_Basis_K::get_gk2_data() const
{
    return this->s_gk2;
}
template <>
double* PW_Basis_K::get_gk2_data() const
{
    return this->d_gk2;
}

} // namespace ModulePW