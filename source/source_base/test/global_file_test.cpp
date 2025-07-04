#include "gtest/gtest.h"
#include "gmock/gmock.h"
#define private public
#include "module_parameter/parameter.h"
#undef private
#include "../global_file.h"
#include "../global_variable.h"
#include <fstream>
#include <unistd.h>

#ifdef __MPI
#include "mpi.h"
#endif

/************************************************
 *  unit test of functions in global_file.cpp
 ***********************************************/

/**
 * - Tested Function
 * - mkdiratom
 *   - generate atom dir for each type of atom
 * - openlog
 *   - Open the out file with the name *.log
 */

class GlobalFile : public testing::Test
{

};

TEST_F(GlobalFile,mkdirout)
{
	    std::string output;
	    testing::internal::CaptureStdout();
	    ModuleBase::Global_File::make_dir_out("Si","m",false,0,true,true);
	    output = testing::internal::GetCapturedStdout();
	    EXPECT_THAT(output,testing::HasSubstr("MAKE THE DIR"));
	    GlobalV::ofs_warning.close();
	    EXPECT_TRUE(GlobalV::ofs_running.is_open());
	    GlobalV::ofs_running.close();
	    std::string dd = "OUT.Si/running_m_1.log";
        remove(dd.c_str());

	    testing::internal::CaptureStdout();
	    ModuleBase::Global_File::make_dir_out("Si","md",false,0,true,false);
	    output = testing::internal::GetCapturedStdout();
	    EXPECT_THAT(output,testing::HasSubstr("MAKE THE STRU DIR"));
	    EXPECT_TRUE(GlobalV::ofs_running.is_open());
	    GlobalV::ofs_running.close();
	    GlobalV::ofs_warning.close();
	    std::string bb = "OUT.Si/running_md.log";
        remove(bb.c_str());

	    testing::internal::CaptureStdout();
	    ModuleBase::Global_File::make_dir_out("Si","md",true,0,true,true);
	    output = testing::internal::GetCapturedStdout();
	    EXPECT_THAT(output,testing::HasSubstr("MAKE THE MATRIX DIR"));
	    EXPECT_TRUE(GlobalV::ofs_running.is_open());
	    GlobalV::ofs_running.close();
	    std::string cc = "OUT.Si/running_md_1.log";
        remove(cc.c_str());
        std::string aa = "OUT.Si/warning.log";
        remove(aa.c_str());
        rmdir(PARAM.sys.global_stru_dir.c_str());
        rmdir(PARAM.sys.global_matrix_dir.c_str());
        rmdir(PARAM.sys.global_out_dir.c_str());
}

TEST_F(GlobalFile,mkdiratom)
{
        PARAM.sys.global_out_dir = "./";
        ModuleBase::Global_File::make_dir_atom("Si");
        int a = access("./Si/",0);
        EXPECT_EQ(a , 0);
        std::string ss = "./Si/";
        rmdir(ss.c_str());
}

TEST_F(GlobalFile,openlog)
{
        std::ofstream ofs;
        ModuleBase::Global_File::open_log(ofs,"Si.log","md",true);
        EXPECT_TRUE(ofs.is_open());
        ofs.close();
        ModuleBase::Global_File::open_log(ofs,"Si.log","md",false);
        EXPECT_TRUE(ofs.is_open());
        ofs.close();
        std::string sss = "Si.log"; 
        remove(sss.c_str());
}

TEST_F(GlobalFile,closelog)
{
		std::ofstream ofs;
		std::string sss = "Si.log";
		ofs.open(sss.c_str());
		ModuleBase::Global_File::close_log(ofs, sss);
		EXPECT_FALSE(ofs.is_open());
		if (ofs.is_open())
		{
			ofs.close();
		}
		remove(sss.c_str());
}

TEST_F(GlobalFile,closealllog)
{
		/* 
		For source_io/input.cpp:line3578 close_log() is a void function,
		All its contents is calling close_all_log() in source_base/global_file.cpp
		For Input::close_log() what is left to test are the validities of parameters
		GlobalV::MY_RANK and this->out_alllog.
		*/
		/* Test out_alllog == true case */
		std::string header = "running_";
		std::string tailCpuRank0 = "_cpu0.log";
		std::string tail = ".log";
		std::string f1 = header + PARAM.input.calculation + tailCpuRank0;
		
		if (GlobalV::ofs_running.is_open())
		{
			GlobalV::ofs_running.close();
		}
		if (GlobalV::ofs_warning.is_open())
		{
			GlobalV::ofs_warning.close();
		}
		GlobalV::ofs_running.open(f1.c_str());
		GlobalV::ofs_warning.open("warning.log");
		ModuleBase::Global_File::close_all_log(0,true,PARAM.input.calculation);
		EXPECT_FALSE(GlobalV::ofs_running.is_open());
		if (GlobalV::ofs_running.is_open())
		{
			GlobalV::ofs_running.close();
		}
		EXPECT_FALSE(GlobalV::ofs_warning.is_open());
		if (GlobalV::ofs_warning.is_open())
		{
			GlobalV::ofs_warning.close();
		}
		remove(f1.c_str());
		//remove("warning.log");
		/* Test out_alllog == false case */
		GlobalV::ofs_running.open("running.log");
		GlobalV::ofs_warning.open("warning.log");
		ModuleBase::Global_File::close_all_log(0,false,PARAM.input.calculation);
		EXPECT_FALSE(GlobalV::ofs_running.is_open());
		if (GlobalV::ofs_running.is_open())
		{
			GlobalV::ofs_running.close();
		}
		EXPECT_FALSE(GlobalV::ofs_warning.is_open());
		if (GlobalV::ofs_warning.is_open())
		{
			GlobalV::ofs_warning.close();
		}
		remove("running.log");
		remove("warning.log");
}