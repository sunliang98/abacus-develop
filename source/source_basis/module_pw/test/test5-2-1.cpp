#include "../pw_basis.h"
#ifdef __MPI
#include "test_tool.h"
#include "source_base/parallel_global.h"
#include "mpi.h"
#endif
#include "source_base/global_function.h"
#include "source_base/constants.h"
#include "pw_test.h"
extern int nproc_in_pool,rank_in_pool;
using namespace std;

TEST_F(PWTEST,test5_2_1)
{
    cout<<"dividemthd 1, gamma_only: on, full_pw: true, full_pw_dim: 1, check gcar,gdirect,gg,istot2ixy,ig2isz"<<endl;
    //--------------------------------------------------
    ModuleBase::Matrix3 latvec(1,2,0,2,1,1,0,0,5);
    bool gamma_only = true;
    double wfcecut = 70;
    double lat0 = 2;
    int distribution_type = 1;
    bool xprime = false;
    bool full_pw = true;
    int full_pw_dim = 1;
    //--------------------------------------------------

    ModulePW::PW_Basis pwtest(device_flag, precision_flag);
#ifdef __MPI
    pwtest.initmpi(nproc_in_pool, rank_in_pool, POOL_WORLD);
#endif
    pwtest.setfullpw(full_pw, full_pw_dim);
    pwtest.initgrids(lat0, latvec, wfcecut);
    pwtest.initparameters(gamma_only, wfcecut, distribution_type,xprime);
    pwtest.setuptransform();
    pwtest.collect_local_pw();
    pwtest.collect_uniqgg();
    ModuleBase::Matrix3 GT,G,GGT;
    GT = latvec.Inverse();
	G  = GT.Transpose();
	GGT = G * GT;
    double tpiba2 = ModuleBase::TWO_PI * ModuleBase::TWO_PI / lat0 / lat0;
    double ggecut = wfcecut / tpiba2;

    //ref
    const int totnpw_ref = 3240;
    const int totnst_ref = 120;
    const int fftnx_ref = 15;
    const int ny_ref = 15;
    const int fftny_ref = 8;
    const int fftnz_ref = 27;

    //some results for different number of processors
    int npw_per_ref[12][12]={
        {3240},
        {1620,1620},
        {1080,1080,1080},
        {810,810,810,810},
        {648,648,648,648,648},
        {540,540,540,540,540,540},
        {459,459,459,459,459,459,486},
        {405,405,405,405,405,405,405,405},
        {378,378,351,351,351,351,351,351,378},
        {324,324,324,324,324,324,324,324,324,324},
        {297,297,297,297,297,297,297,297,297,297,270},
        {270,270,270,270,270,270,270,270,270,270,270,270}
    };
    int nst_per_ref[12][12]={
        {120},
        {60,60},
        {40,40,40},
        {30,30,30,30},
        {24,24,24,24,24},
        {20,20,20,20,20,20},
        {17,17,17,17,17,17,18},
        {15,15,15,15,15,15,15,15},
        {14,14,13,13,13,13,13,13,14},
        {12,12,12,12,12,12,12,12,12,12},
        {11,11,11,11,11,11,11,11,11,11,10},
        {10,10,10,10,10,10,10,10,10,10,10,10}
    };
    int *npw_per = nullptr;
    if(rank_in_pool == 0)
    {
        npw_per = new int [nproc_in_pool];
    }
#ifdef __MPI
    MPI_Gather(&pwtest.npw,1,MPI_INT,npw_per,1,MPI_INT,0,POOL_WORLD);
#else
    if(rank_in_pool == 0) npw_per[0] = pwtest.npw;
#endif
    if(rank_in_pool == 0)
    {
        if(nproc_in_pool <= 12)
        {
            for(int ip = 0 ; ip < nproc_in_pool ; ++ip)
            {
                EXPECT_EQ(npw_per_ref[nproc_in_pool-1][ip], npw_per[ip]);
                EXPECT_EQ(nst_per_ref[nproc_in_pool-1][ip], pwtest.nst_per[ip]);
            }
        }
        else
        {
            cout<<"Please use mpi processors no more than 12."<<endl;
        }
        delete []npw_per;
    }

    //results
    int tot_npw = 0;
#ifdef __MPI
    MPI_Allreduce(&pwtest.npw, &tot_npw, 1, MPI_INT, MPI_SUM, POOL_WORLD);
#else
    tot_npw = pwtest.npw;
#endif
    EXPECT_EQ(pwtest.fftnx, fftnx_ref);
    EXPECT_EQ(pwtest.fftny, fftny_ref);
    EXPECT_EQ(pwtest.ny, ny_ref);
    EXPECT_EQ(pwtest.fftnz, fftnz_ref);
    EXPECT_EQ(tot_npw, totnpw_ref);
    EXPECT_EQ(pwtest.npwtot, totnpw_ref);
    EXPECT_EQ(pwtest.nstot,totnst_ref);
    EXPECT_EQ(pwtest.nxyz, fftnx_ref*ny_ref*fftnz_ref);

    int *tmpx = new int[pwtest.fftnx*pwtest.fftny*pwtest.fftnz];
    int *tmpy = new int[pwtest.fftnx*pwtest.fftny*pwtest.fftnz];
    int *tmpz = new int[pwtest.fftnx*pwtest.fftny*pwtest.fftnz];
    ModuleBase::GlobalFunc::ZEROS(tmpx,pwtest.fftnx*pwtest.fftny*pwtest.fftnz);
    ModuleBase::GlobalFunc::ZEROS(tmpy,pwtest.fftnx*pwtest.fftny*pwtest.fftnz);
    ModuleBase::GlobalFunc::ZEROS(tmpz,pwtest.fftnx*pwtest.fftny*pwtest.fftnz);
    
    int * startnst = new int [nproc_in_pool];
    startnst[0] = 0;
    for(int ip = 1 ; ip < nproc_in_pool; ++ip)
    {
        startnst[ip] = startnst[ip-1] + pwtest.nst_per[ip-1];
    }

    for(int ig = 0 ; ig < pwtest.npw; ++ig)
    {
        int istot = pwtest.ig2isz[ig] / pwtest.fftnz + startnst[rank_in_pool];
        // int is = pwtest.ig2isz[ig] / pwtest.fftnz;
        int iz = pwtest.ig2isz[ig] % pwtest.fftnz;
        int iy = pwtest.istot2ixy[istot] % pwtest.ny;
        int ix = pwtest.istot2ixy[istot] / pwtest.ny;
        // int iy = pwtest.is2fftixy[is] % pwtest.fftny;
        // int ix = pwtest.is2fftixy[is] / pwtest.fftny;

        tmpx[iz+(iy+ix*pwtest.fftny)*pwtest.fftnz] = int(pwtest.gdirect[ig].x);
        tmpy[iz+(iy+ix*pwtest.fftny)*pwtest.fftnz] = int(pwtest.gdirect[ig].y);
        tmpz[iz+(iy+ix*pwtest.fftny)*pwtest.fftnz] = int(pwtest.gdirect[ig].z);
    }
#ifdef __MPI
    MPI_Allreduce(MPI_IN_PLACE,tmpx,pwtest.fftnxyz,MPI_INT,MPI_SUM,POOL_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,tmpy,pwtest.fftnxyz,MPI_INT,MPI_SUM,POOL_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,tmpz,pwtest.fftnxyz,MPI_INT,MPI_SUM,POOL_WORLD);
#endif
    if(rank_in_pool==0)
    {
        for(int iz = 0 ; iz < pwtest.fftnz; ++iz)
        {
            for(int iy = 0 ; iy < pwtest.fftny ; ++iy)
            {
                for(int ix = 0 ; ix < pwtest.fftnx ; ++ix)
                {
                    ModuleBase::Vector3<double> f;
                    f.x = ix;
                    f.y = iy;
                    f.z = iz;
                    if(iz >= int(pwtest.nz/2) +1) f.z -= pwtest.nz;
                    if(ix >= int(pwtest.nx/2) +1) f.x -= pwtest.nx;
                    double modulus = f * (GGT * f);
                    if (modulus <= ggecut)
                    {
                        EXPECT_EQ(tmpx[iz + iy*pwtest.fftnz + ix*pwtest.fftny*pwtest.fftnz], int(f.x));
                        EXPECT_EQ(tmpy[iz + iy*pwtest.fftnz + ix*pwtest.fftny*pwtest.fftnz], int(f.y));
                        EXPECT_EQ(tmpz[iz + iy*pwtest.fftnz + ix*pwtest.fftny*pwtest.fftnz], int(f.z));
                    }
                    
                }
            }
        }
    }
    for(int ig = 0 ;ig < pwtest.npw ; ++ig)
    {
        ModuleBase::Vector3<double> f;
        f.x = pwtest.gdirect[ig].x;
        f.y = pwtest.gdirect[ig].y;
        f.z = pwtest.gdirect[ig].z;
        ModuleBase::Vector3<double> gcar;
        gcar = f * G;
        double modulus = f*GGT*f;
        EXPECT_NEAR(gcar.x,pwtest.gcar[ig].x,1e-6);
        EXPECT_NEAR(gcar.y,pwtest.gcar[ig].y,1e-6);
        EXPECT_NEAR(gcar.z,pwtest.gcar[ig].z,1e-6);
        EXPECT_NEAR(modulus,pwtest.gg[ig],1e-6);
        EXPECT_NEAR(pwtest.gg[ig], pwtest.gg_uniq[pwtest.ig2igg[ig]],1e-8);
    }
    for(int igg = 1 ; igg < pwtest.ngg ; ++igg)
    {
        EXPECT_GT(pwtest.gg_uniq[igg], pwtest.gg_uniq[igg-1]);
    }
    if(pwtest.ig_gge0 >= 0) {EXPECT_NEAR(0.0, pwtest.gg[pwtest.ig_gge0], 1e-8);}
    delete [] startnst;
    delete [] tmpx;
    delete [] tmpy;
    delete [] tmpz;
}