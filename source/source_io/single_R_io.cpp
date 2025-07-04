#include "single_R_io.h"
#include "source_base/parallel_reduce.h"
#include "module_parameter/parameter.h"
#include "source_base/global_function.h"
#include "source_base/global_variable.h"

inline void write_data(std::ofstream& ofs, const double& data)
{
    ofs << " " << std::fixed << std::scientific << std::setprecision(8) << data;
}
inline void write_data(std::ofstream& ofs, const std::complex<double>& data)
{
    ofs << " (" << std::fixed << std::scientific << std::setprecision(8) << data.real() << ","
        << std::fixed << std::scientific << std::setprecision(8) << data.imag() << ")";
}

template<typename T>
void ModuleIO::output_single_R(std::ofstream& ofs,
    const std::map<size_t, std::map<size_t, T>>& XR,
    const double& sparse_threshold,
    const bool& binary,
    const Parallel_Orbitals& pv,
    const bool& reduce)
{
    T* line = nullptr;
    std::vector<int> indptr;
    indptr.reserve(PARAM.globalv.nlocal + 1);
    indptr.push_back(0);

    std::stringstream tem1;
    tem1 << PARAM.globalv.global_out_dir << std::to_string(GlobalV::DRANK) + "temp_sparse_indices.dat";
    std::ofstream ofs_tem1;
    std::ifstream ifs_tem1;

    if (!reduce || GlobalV::DRANK == 0)
    {
        if (binary)
        {
            ofs_tem1.open(tem1.str().c_str(), std::ios::binary);
        }
        else
        {
            ofs_tem1.open(tem1.str().c_str());
        }
    }

    line = new T[PARAM.globalv.nlocal];
    for(int row = 0; row < PARAM.globalv.nlocal; ++row)
    {
        ModuleBase::GlobalFunc::ZEROS(line, PARAM.globalv.nlocal);

        if (!reduce || pv.global2local_row(row) >= 0)
        {
            auto iter = XR.find(row);
            if (iter != XR.end())
            {
                for (auto &value : iter->second)
                {
                    line[value.first] = value.second;
                }
            }
        }

		if (reduce) 
		{
			Parallel_Reduce::reduce_all(line, PARAM.globalv.nlocal);
		}

        if (!reduce || GlobalV::DRANK == 0)
        {
            int nonzeros_count = 0;
            for (int col = 0; col < PARAM.globalv.nlocal; ++col)
            {
                if (std::abs(line[col]) > sparse_threshold)
                {
                    if (binary)
                    {
                        ofs.write(reinterpret_cast<char*>(&line[col]), sizeof(T));
                        ofs_tem1.write(reinterpret_cast<char *>(&col), sizeof(int));
                    }
                    else
                    {
                        write_data(ofs, line[col]);
                        ofs_tem1 << " " << col;
                    }

                    nonzeros_count++;

                }

            }
            nonzeros_count += indptr.back();
            indptr.push_back(nonzeros_count);
        }
    }

    delete[] line;

    if (!reduce || GlobalV::DRANK == 0)
    {
        if (binary)
        {
            ofs_tem1.close();
            ifs_tem1.open(tem1.str().c_str(), std::ios::binary);
            ofs << ifs_tem1.rdbuf();
            ifs_tem1.close();
            for (auto &i : indptr)
            {
                ofs.write(reinterpret_cast<char *>(&i), sizeof(int));
            }
        }
        else
        {
            ofs << std::endl;
            ofs_tem1 << std::endl;
            ofs_tem1.close();
            ifs_tem1.open(tem1.str().c_str());
            ofs << ifs_tem1.rdbuf();
            ifs_tem1.close();
            for (auto &i : indptr)
            {
                ofs << " " << i;
            }
            ofs << std::endl;
        }

        std::remove(tem1.str().c_str());
    }
}

template void ModuleIO::output_single_R<double>(std::ofstream& ofs,
    const std::map<size_t, std::map<size_t, double>>& XR,
    const double& sparse_threshold,
    const bool& binary,
    const Parallel_Orbitals& pv,
    const bool& reduce);

template void ModuleIO::output_single_R<std::complex<double>>(std::ofstream& ofs,
    const std::map<size_t, std::map<size_t, std::complex<double>>>& XR,
    const double& sparse_threshold,
    const bool& binary,
    const Parallel_Orbitals& pv,
    const bool& reduce);
