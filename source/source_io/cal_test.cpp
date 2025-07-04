#include "source_pw/hamilt_pwdft/global.h"
#include "source_base/global_function.h"
#define private public
#include "module_parameter/parameter.h"
#undef private
#include "source_base/global_variable.h"
#include "source_base/memory.h"
#include "cal_test.h"

double Cal_Test::mporter=0.0;

// about charge density
double Cal_Test::mrho=0.0;
double Cal_Test::mrho_save=0.0;
double Cal_Test::mrho_core=0.0;

// about pulay mixing.
double Cal_Test::mRrho=0.0;
double Cal_Test::mdRrho=0.0;
double Cal_Test::mdrho=0.0;
double Cal_Test::mrho_save2=0.0;

// about potential on FFT grid.
double Cal_Test::mvltot=0.0;
double Cal_Test::mvr=0.0;
double Cal_Test::mvrs=0.0;
double Cal_Test::mvrs1=0.0;
double Cal_Test::mvnew=0.0;

// about charge in g space.
double Cal_Test::mrhog=0.0;
double Cal_Test::mrhog_save=0.0;
double Cal_Test::mrhog_core=0.0;

// others
double Cal_Test::mhs=0.0;
double Cal_Test::mwf=0.0;
double Cal_Test::mnonzero=0.0;
double Cal_Test::mspar_hsrho=0.0;
// plane waves
double Cal_Test::mgvec=0.0;
double Cal_Test::mig2fftw=0.0;
double Cal_Test::mig2fftc=0.0;
double Cal_Test::mgg=0.0;
double Cal_Test::mig123=0.0;
double Cal_Test::mstrucFac=0.0;
double Cal_Test::meigts123=0.0;

double Cal_Test::mtot=0.0;

void Cal_Test::test_memory(const int nat,
						   const int ntype,
						   const ModuleBase::Matrix3& GGT,
						   const ModulePW::PW_Basis* rhopw, 
						   const ModulePW::PW_Basis_K* wfcpw, 
						   const std::string chr_mixing_mode, 
						   const int chr_mixing_ndim)
{
	ModuleBase::TITLE("Cal_Test","test_memory");

	const int ngmw = Cal_Test::cal_np(GGT,wfcpw->ggecut, rhopw->nx, rhopw->ny, rhopw->nz);
	const int ngmc = Cal_Test::cal_np(GGT,rhopw->ggecut, rhopw->nx, rhopw->ny, rhopw->nz);

//  const int ecut_wfc = INPUT.ecutwfc;
//  const int ecut_chg = INPUT.ecutrho;

//	const int ngmw = Cal_Test::cal_np(ecut_wfc, rhopw->nx, rhopw->ny, rhopw->nz);
//	const int ngmc = Cal_Test::cal_np(ecut_chg, rhopw->nx, rhopw->ny, rhopw->nz);

	std::cout << " number of atoms = " << nat << std::endl;
	std::cout << " plane wave number for wave functions = " << ngmw << std::endl;
	std::cout << " plane wave number for chage density  = " << ngmc << std::endl;

	mporter = ModuleBase::Memory::calculate_mem ( rhopw->nxyz, "double");

	mrho = mporter;
	mrho_save = mrho;
	mrho_core = mrho;

	// (2) memory for charge mixing
	std::cout << " Mixing mode = " << chr_mixing_mode << std::endl;
	if(chr_mixing_mode == "pulay")
	{
		std::cout << " Mixing dimension = " << chr_mixing_ndim << std::endl;
		mRrho = chr_mixing_ndim * mrho;
		mdRrho = (chr_mixing_ndim-1) * mrho;
		mdrho = (chr_mixing_ndim-1) * mrho;
		mrho_save2 = mrho;
//		std::cout << " Memory for pulay mixing: " << mrho << " MB" << std::endl;	
	}

	mvltot = mrho;
	mvr = mrho;
	mvrs = mrho;
	mvrs1 = mrho;
	mvnew = mrho;

	mrhog = ModuleBase::Memory::calculate_mem( ngmc, "cdouble");
	mrhog_save = ModuleBase::Memory::calculate_mem( ngmc, "cdouble");
	mrhog_core = ModuleBase::Memory::calculate_mem( ngmc, "cdouble"); 
	
	mhs = ModuleBase::Memory::calculate_mem( PARAM.globalv.nlocal*PARAM.globalv.nlocal, "double" );
	mwf = ModuleBase::Memory::calculate_mem( PARAM.globalv.nlocal*PARAM.inp.nbands, "double" );
	mnonzero = ModuleBase::Memory::calculate_mem( PARAM.globalv.nlocal*(PARAM.globalv.nlocal+1)/2, "bool");
// mohan comment out 2021-02-11
//	mspar_hsrho = Memory::calculate_mem( Hnnz*3, "double");
	
	mgvec = ModuleBase::Memory::calculate_mem( ngmc * 3 * 2, "double" );
	mig2fftw = ModuleBase::Memory::calculate_mem( ngmw , "int");  
	mig2fftc = ModuleBase::Memory::calculate_mem( ngmc , "int");  
	mgg = ModuleBase::Memory::calculate_mem( ngmc, "double");
	mig123 = ModuleBase::Memory::calculate_mem( ngmc*3, "int");
	mstrucFac = ModuleBase::Memory::calculate_mem( ntype*ngmc, "cdouble");
	meigts123 = ModuleBase::Memory::calculate_mem( nat * (2*rhopw->nx+1+2*rhopw->ny+1+2*rhopw->nz+1), "cdouble");

	//(3) Memory for H,S matrix.
	std::cout << " NLOCAL = " << PARAM.globalv.nlocal << std::endl;
	std::cout << " NBANDS = " << PARAM.inp.nbands << std::endl;

	std::cout << " Memory for H,S matrix ( " 
		<< PARAM.globalv.nlocal << ", "
		<< PARAM.globalv.nlocal << ") = "
		<< mhs << " MB" << std::endl;
	
	//(4) Memory for wave functions.
	std::cout << " Memory for wave functions ( " 
		<< PARAM.globalv.nlocal << ", "
		<< PARAM.inp.nbands << ") = "
		<< mwf << " MB" << std::endl;

	print_mem(1);
//	print_mem(8);
//	print_mem(16);

//	if(nat > 200)
//	{
//		print_mem(32);
//		print_mem(64);
//	}

	return;
}

//! compute the number of plane waves
int Cal_Test::cal_np(const ModuleBase::Matrix3& GGT,
				     const double &ggcut, 
					 const int &n1, 
					 const int &n2, 
					 const int &n3)
{

/*
    std::cout << "ggcut=" << ggcut << std::endl;
    std::cout << "n1=" << n1 << std::endl;
    std::cout << "n2=" << n2 << std::endl;
    std::cout << "n3=" << n3 << std::endl;
*/

    assert(n1>=0);
    assert(n2>=0);
    assert(n3>=0);

	int ibox[3]={0};

	// set the center at origin point.
	ibox[0] = int(n1 / 2.0) + 1;
	ibox[1] = int(n2 / 2.0) + 1;
	ibox[2] = int(n3 / 2.0) + 1;
	// get the number of plane wave within 'gcut'
	int ng = 0;
	for (int i = -ibox[0]; i <= ibox[0]; i++)
	{
		for (int j = -ibox[1]; j <= ibox[1]; j++)
		{
			for (int k = -ibox[2]; k <= ibox[2]; k++)
			{
				ModuleBase::Vector3<double> f(i,j,k);
				// g2= |f|^2 in the unit of (2Pi/lat0)^2
				double g2 = f * (GGT * f);

				// gcut is from input.
				if (g2 <= ggcut)
				{
					ng++;
				}
			}
		}
	}
	return ng;
}

void Cal_Test::print_mem(const int &nproc)
{
	std::cout << " ========================: " << std::endl;
	mtot = 0.0;

	mtot += mporter + mrho + mrho_save + mrho_core + mRrho +
	mdRrho + mdrho + mrho_save2 + mvltot + mvr +
	mvrs + mvrs1 + mvnew + mrhog + mrhog_save + mrhog_core +
	mgvec + mgg + mig2fftw + mig2fftc + mig123 +
	mstrucFac + meigts123; 
	mtot += mwf + mhs;

	std::cout << " If you use " << nproc << " processors: " << std::endl;
	std::cout << " MEMORY FOR porter       : " << std::setw(15) << mporter/nproc << " MB" << std::endl;
	std::cout << " MEMORY FOR rho          : " << std::setw(15) << mrho/nproc << " MB" << std::endl;
	std::cout << " MEMORY FOR rho_save     : " << std::setw(15) << mrho_save/nproc << " MB" << std::endl;
	std::cout << " MEMORY FOR rho_core     : " << std::setw(15) << mrho_core/nproc << " MB" << std::endl;
	std::cout << " MEMORY FOR Rrho         : " << std::setw(15) << mRrho/nproc << " MB" << std::endl;
	std::cout << " MEMORY FOR dRrho        : " << std::setw(15) << mdRrho/nproc << " MB" << std::endl;
	std::cout << " MEMORY FOR drho         : " << std::setw(15) << mdrho/nproc << " MB" << std::endl;
	std::cout << " MEMORY FOR rho_save2    : " << std::setw(15) << mrho_save2/nproc << " MB" << std::endl;
	std::cout << " MEMORY FOR vltot        : " << std::setw(15) << mvltot/nproc << " MB" << std::endl;
	std::cout << " MEMORY FOR vr           : " << std::setw(15) << mvr/nproc << " MB" << std::endl;
	std::cout << " MEMORY FOR vrs          : " << std::setw(15) << mvrs/nproc << " MB" << std::endl;
	std::cout << " MEMORY FOR vrs1         : " << std::setw(15) << mvrs1/nproc << " MB" << std::endl;
	std::cout << " MEMORY FOR vrnew        : " << std::setw(15) << mvnew/nproc << " MB" << std::endl;
	std::cout << " MEMORY FOR rhog         : " << std::setw(15) << mrhog/nproc << " MB" << std::endl;
	std::cout << " MEMORY FOR rhog_save    : " << std::setw(15) << mrhog_save/nproc << " MB" << std::endl;
	std::cout << " MEMORY FOR rhog_core    : " << std::setw(15) << mrhog_core/nproc << " MB" << std::endl;
	std::cout << " MEMORY FOR H, S matrix  : " << std::setw(15) << mhs/nproc  << " MB" << std::endl;
	std::cout << " MEMORY FOR wave function: " << std::setw(15) << mwf/nproc  << " MB" << std::endl;
	std::cout << " MEMORY FOR spar H,S,rho : " << std::setw(15) << mspar_hsrho  << " MB" << std::endl;
	std::cout << " MEMORY FOR nonzero      : " << std::setw(15) << mnonzero << " MB" << std::endl;
	std::cout << " MEMORY FOR g vectors    : " << std::setw(15) << mgvec/nproc  << " MB" << std::endl;
	std::cout << " MEMORY FOR gg           : " << std::setw(15) << mgg/nproc << " MB" << std::endl;
	std::cout << " MEMORY FOR fftw index   : " << std::setw(15) << mig2fftw/nproc << " MB" << std::endl;
	std::cout << " MEMORY FOR fftc index   : " << std::setw(15) << mig2fftc/nproc << " MB" << std::endl;
	std::cout << " MEMORY FOR ig123        : " << std::setw(15) << mig123/nproc << " MB" << std::endl;
	std::cout << " MEMORY FOR strucFac     : " << std::setw(15) << mstrucFac/nproc << " MB" << std::endl;
	std::cout << " MEMORY FOR eigts1,2,3   : " << std::setw(15) << meigts123/nproc << " MB" << std::endl;
	std::cout << " TOTAL MEMORY            : " << std::setw(15) << mtot/nproc << " MB" << std::endl;
	
	std::cout << " MEMORY FOR nonzero      : " << std::setw(15) 
     << (double)PARAM.globalv.nlocal*(PARAM.globalv.nlocal+1)/1028/1028/2.0/nproc 
     << " MB" << std::endl; 
}
