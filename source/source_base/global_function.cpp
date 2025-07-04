//==========================================================
// AUTHOR : mohan
//==========================================================
#include "global_function.h"

#include "source_base/parallel_common.h"
#include "global_file.h"

//==========================================================
// USE FILE timer.h
// ONLY :  output time after quit.
//==========================================================
#include "memory.h"
#include "timer.h"

#include <fstream>
#include <iostream>
#include <string>
namespace ModuleBase
{
namespace GlobalFunc
{

void NOTE(const std::string &words)
{
    return;
    if (GlobalV::ofs_running)
    {
        // GlobalV::ofs_running << " *********************************************************************************"
        // << std::endl;
        GlobalV::ofs_running << " !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                             << std::endl;
        GlobalV::ofs_running << " " << words << std::endl;
        GlobalV::ofs_running << " !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                             << std::endl;
    }
}

void NEW_PART(const std::string &words)
{
    GlobalV::ofs_running << "\n ><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><" << std::endl;
    GlobalV::ofs_running << "\n " << words << std::endl;
    GlobalV::ofs_running << "\n ><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><\n" << std::endl;
    return;
}

//==========================================================
// GLOBAL FUNCTION :
// NAME : OUT( output date for checking )
//==========================================================
void OUT(std::ofstream &ofs, const std::string &name)
{
    ofs << "\n" << std::setw(18) << name << std::endl;
    return;
}

//==========================================================
// GLOBAL FUNCTION :
// NAME : MAKE_DIR( make dir ,using system function)
//==========================================================
void MAKE_DIR(const std::string &fn)
{
    //	ModuleBase::TITLE("global_function","MAKE_DIR");
    if (GlobalV::MY_RANK == 0)
    {
        std::stringstream ss;
        ss << " test -d " << fn << " || mkdir " << fn;
        //----------------------------------------------------------
        // EXPLAIN : 'system' function return '0' if success
        //----------------------------------------------------------
        if (system(ss.str().c_str()))
        {
            ModuleBase::WARNING_QUIT("MAKE_DIR", fn);
        }
    }
    return;
}

void DONE(std::ofstream &ofs, const std::string &description, const bool only_rank0)
{
    if (only_rank0)
    {
        if (GlobalV::MY_RANK == 0)
        {
            //       ofs << " ---------------------------------------------------------------------------------\n";
            ofs << " DONE : " << description;
            ofs << " Time : " << ModuleBase::timer::print_until_now() << " (SEC)";
            ofs << std::endl << std::endl;
            //       ofs << "\n ---------------------------------------------------------------------------------\n";
        }
    }
    else
    {
        //   ofs << " ---------------------------------------------------------------------------------\n";
        ofs << " DONE : " << description;
        ofs << " Time : " << ModuleBase::timer::print_until_now() << " (SEC)";
        ofs << std::endl << std::endl;
        //   ofs << "\n ---------------------------------------------------------------------------------\n";
    }
    //   	std::cout << "\n---------------------------------------------------------------------------------\n";
    std::cout << " DONE(" << std::setw(10) << ModuleBase::timer::print_until_now() << " SEC) : " << description
              << std::endl;
    //   	std::cout << "\n---------------------------------------------------------------------------------\n";
    return;
}


bool SCAN_BEGIN(std::ifstream &ifs, 
                const std::string &TargetName, 
                const bool restart, 
                const bool ifwarn)
{
    std::string SearchName;
    bool find = false;
    if (restart)
    {
        ifs.clear();
        ifs.seekg(0);
    }
    ifs.rdstate();
    while (ifs.good())
    {
        ifs >> SearchName;
        if (SearchName == TargetName)
        {
            find = true;
            break;
        }
    }
    if (!find && ifwarn)
    {
        GlobalV::ofs_warning << " In SCAN_BEGIN, can't find: " << TargetName << " block." << std::endl;
    }
    return find;
}


bool SCAN_LINE_BEGIN(std::ifstream &ifs, 
                const std::string &TargetName, 
                const bool restart, 
                const bool ifwarn)
{
    bool find = false;
    if (restart)
    {
        ifs.clear();
        ifs.seekg(0);
    }
    ifs.rdstate();

    std::string line;
    while (std::getline(ifs,line))
    {
        //! obtain the first character, should not be #
        size_t first_char_pos = line.find_first_not_of(" \t");
        if (first_char_pos != std::string::npos && line[first_char_pos] == '#') 
        {
            continue;
        } 

        //! search in each line
        std::istringstream iss(line);
        std::string SearchName;
        while (iss >> SearchName)
		{
			if (SearchName == TargetName)
			{
				find = true;
				//std::cout << " search name = " << SearchName << std::endl;
				return find;
			}
		}
	}

    if (!find && ifwarn)
    {
        GlobalV::ofs_warning << " In SCAN_LINE_BEGIN, can't find: " << TargetName << " block." << std::endl;
    }
    return find;
}

void SCAN_END(std::ifstream &ifs, const std::string &TargetName, const bool ifwarn)
{
    std::string SearchName;
    ifs >> SearchName;
    if (SearchName != TargetName && ifwarn)
    {
        GlobalV::ofs_warning << " In SCAN_END, can't find: " << TargetName << " block." << std::endl;
    }
    return;
}

void BLOCK_HERE(const std::string &description)
{
    //	return;
    std::cout << "\n********************************************";
    std::cout << "\n Here is a Block, 1: go on 0: quit";
    std::cout << "\n " << description;
    std::cout << "\n********************************************" << std::endl;
    bool go_on = false;
    if (GlobalV::MY_RANK == 0)
    {
        std::cin >> go_on;
    }

#ifdef __MPI
    int swap = go_on;
    if (GlobalV::MY_RANK == 0)
        swap = go_on;
    MPI_Bcast(&swap, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (GlobalV::MY_RANK != 0)
        go_on = static_cast<bool>(swap);
#endif
    if (go_on)
    {
        return;
    }
    else
    {
        ModuleBase::QUIT();
    }
}

void OUT_TIME(const std::string &name, time_t &start, time_t &end)
{
    double mini = difftime(end, start) / 60.0;
    if (mini > 0.1)
    {
	if(GlobalV::ofs_warning)
	{
        	GlobalV::ofs_warning << std::setprecision(2);
        	GlobalV::ofs_warning << " -------------------------------------------------------" << std::endl;
        	GlobalV::ofs_warning << " NAME < " << name << " > = " << std::endl;
        	GlobalV::ofs_warning << " -> " << ctime(&start) << " -> " << ctime(&end);
        	GlobalV::ofs_warning << " TIME = " << mini << " [Minutes]" << std::endl;
        	GlobalV::ofs_warning << " -------------------------------------------------------" << std::endl;
        	GlobalV::ofs_warning << std::setprecision(6);
	}
    }
}

size_t MemAvailable()
{
    size_t mem_sum = 0;
    int i = 0;
    std::ifstream ifs("/proc/meminfo");
    while (ifs.good())
    {
        std::string label, size, kB;
        ifs >> label >> size >> kB;
        if (label == "MemAvailable:")
            return std::stol(size);
        else if (label == "MemFree:" || label == "Buffers:" || label == "Cached:")
        {
            mem_sum += std::stol(size);
            ++i;
        }
        if (i == 3) return mem_sum;
    }
    throw std::runtime_error("read /proc/meminfo error in " + TO_STRING(__FILE__) + " line " + TO_STRING(__LINE__));
}

} // namespace GlobalFunc
} // namespace ModuleBase
