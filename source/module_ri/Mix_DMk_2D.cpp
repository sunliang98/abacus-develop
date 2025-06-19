//=======================
// AUTHOR : Peize Lin
// DATE :   2023-05-09
//=======================

#include "Mix_DMk_2D.h"
#include "source_base/tool_title.h"

Mix_DMk_2D &Mix_DMk_2D::set_nks(const int nks, const bool gamma_only_in)
{
    ModuleBase::TITLE("Mix_DMk_2D", "set_nks");
    this->gamma_only = gamma_only_in;
    if (this->gamma_only)
		this->mix_DMk_gamma.resize(nks);
	else
		this->mix_DMk_k.resize(nks);
	return *this;
}

Mix_DMk_2D &Mix_DMk_2D::set_mixing(Base_Mixing::Mixing* mixing_in)
{
	ModuleBase::TITLE("Mix_DMk_2D","set_mixing");
	if(this->gamma_only)
        for (Mix_Matrix<std::vector<double>>& mix_one : this->mix_DMk_gamma)
			mix_one.init(mixing_in);
	else
        for (Mix_Matrix<std::vector<std::complex<double>>>& mix_one : this->mix_DMk_k)
			mix_one.init(mixing_in);
	return *this;
}

Mix_DMk_2D &Mix_DMk_2D::set_mixing_beta(const double mixing_beta)
{
	ModuleBase::TITLE("Mix_DMk_2D","set_mixing_beta");
	if(this->gamma_only)
        for (Mix_Matrix<std::vector<double>>& mix_one : this->mix_DMk_gamma)
			mix_one.mixing_beta = mixing_beta;
	else
        for (Mix_Matrix<std::vector<std::complex<double>>>& mix_one : this->mix_DMk_k)
			mix_one.mixing_beta = mixing_beta;
	return *this;
}

void Mix_DMk_2D::mix(const std::vector<std::vector<double>>& dm, const bool flag_restart)
{
	ModuleBase::TITLE("Mix_DMk_2D","mix");
	assert(this->mix_DMk_gamma.size() == dm.size());
	for(int ik=0; ik<dm.size(); ++ik)
		this->mix_DMk_gamma[ik].mix(dm[ik], flag_restart);
}
void Mix_DMk_2D::mix(const std::vector<std::vector<std::complex<double>>>& dm, const bool flag_restart)
{
	ModuleBase::TITLE("Mix_DMk_2D","mix");
	assert(this->mix_DMk_k.size() == dm.size());
	for(int ik=0; ik<dm.size(); ++ik)
		this->mix_DMk_k[ik].mix(dm[ik], flag_restart);
}

std::vector<const std::vector<double>*> Mix_DMk_2D::get_DMk_gamma_out() const
{
    std::vector<const std::vector<double>*> DMk_out(this->mix_DMk_gamma.size());
	for(int ik=0; ik<this->mix_DMk_gamma.size(); ++ik)
		DMk_out[ik] = &this->mix_DMk_gamma[ik].get_data_out();
	return DMk_out;
}
std::vector<const std::vector<std::complex<double>>*> Mix_DMk_2D::get_DMk_k_out() const
{
    std::vector<const std::vector<std::complex<double>>*> DMk_out(this->mix_DMk_k.size());
	for(int ik=0; ik<this->mix_DMk_k.size(); ++ik)
		DMk_out[ik] = &this->mix_DMk_k[ik].get_data_out();
	return DMk_out;
}