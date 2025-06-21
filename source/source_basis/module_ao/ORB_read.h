#ifndef LCAO_ORBITALS_H
#define LCAO_ORBITALS_H

#include "ORB_atomic.h"
#include "ORB_atomic_lm.h"
#include "ORB_nonlocal.h"

////////////////////////////////////////////////////////////
/// advices for reconstructions:
/// -------------------------------
/// each set of orbitals should have: lmax, dr, dk, rmax, lmax, etc.
///
/// the orbitals include : NAO, non-local projectors, descriptors, etc.
///
/// mohan note 2021-02-13
///////////////////////////////////////////////////////////

class LCAO_Orbitals
{
	public:

	LCAO_Orbitals();
	~LCAO_Orbitals();

    void init(
        std::ofstream& ofs_in,
        const int& ntype,
        const std::string& orbital_dir,
        const std::string* orbital_file,
        const std::string& descriptor_file,
        const int& lmax,
        const double& lcao_ecut_in,
        const double& lcao_dk_in,
        const double& lcao_dr_in,
        const double& lcao_rmax_in,
        const bool& deepks_setorb,
        const int& out_mat_r,
        const bool& force_flag,
        const int& my_rank
    );

	void Read_Orbitals(
		std::ofstream &ofs_in, // mohan add 2021-05-07
		const int &ntype_in,
		const int &lmax_in,
		const bool &deepks_setorb, //  mohan add 2021-04-25
		const int &out_mat_r, // mohan add 2021-04-26
		const bool &force_flag, // mohan add 2021-05-07
		const int &my_rank); // mohan add 2021-04-26

	void Read_PAO(
		std::ofstream &ofs_in,
		const int& it,
		const bool &force_flag, // mohan add 2021-05-07
		const int& my_rank); // mohan add 2021-04-26



	void Read_Descriptor(
		std::ofstream &ofs_in,
		const bool &force_flag, // mohan add 2021-05-07
		const int &my_rank);	//caoyu add 2020-3-16

#ifdef __MPI
	void bcast_files(const int &ntype_in, const int &my_rank);
#endif

	const double& get_ecutwfc() const {return ecutwfc;}
	const int& get_kmesh() const{return kmesh;}
	const double& get_dk() const {return dk;}
	const double& get_dR() const {return dR;}
	const double& get_Rmax() const {return Rmax;}
	const int& get_lmax() const {return lmax;}
	const int& get_lmax_d() const { return lmax_d; }		///<lmax of descriptor basis
	const int& get_nchimax() const {return nchimax;}
	const int& get_nchimax_d() const { return nchimax_d; }	///<nchimax of descriptor basis
	const int& get_ntype() const {return ntype;}
	const double& get_dr_uniform() const { return dr_uniform; }

	//caoyu add 2021-05-24
	const double& get_rcutmax_Phi() const { return rcutmax_Phi; }

    std::vector<double> cutoffs() const;

	/// numerical atomic orbitals
	Numerical_Orbital* Phi;
	
	
	//caoyu add 2021-3-10
	/// descriptor bases, saved as one-type atom orbital
	Numerical_Orbital* Alpha;

	// initialized in input.cpp
	double ecutwfc;
	double dk;
	double dR;
	double Rmax;
	
	double dr_uniform;

	// initalized in UnitCell
	// assume ntype < 20.
	bool read_in_flag;
	std::vector<std::string> orbital_file;
	std::vector<std::string> nonlocal_file;
	std::string descriptor_file;	//caoyu add 2020-3-16

private:

	int ntype; // number of elements
	int kmesh; // number of points on kmesh

	int lmax;
	int nchimax;

	int lmax_d;	//max l of descriptor orbitals
	int nchimax_d;	//max number of descriptor orbitals per l

	double rcutmax_Phi;	//caoyu add 2021-05-24

	void read_orb_file(
		std::ofstream &ofs_in,
		std::ifstream &ifs, 
		const int &it, 
		int &lmax, 
		int &nchimax, 
		Numerical_Orbital* ao,
		const bool &force_flag, // mohan add 2021-05-07
		const int &my_rank);	//caoyu add 2021-04-26

    friend class TwoCenterBundle; // for the sake of TwoCenterBundle::to_LCAO_Orbitals
};

#endif
