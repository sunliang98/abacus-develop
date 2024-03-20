#include "../basic_funcs.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

/************************************************
 *  unit test of basic_funcs
 ***********************************************/

/**
 * - Tested Functions:
 *  - maxval_abs_2d(): return the maximum absolute value of an array of Vector3
 *  - maxloc_abs_2d(): return the location of the maximum absolute value of an array of Vector3
 *  - sum_2d(): return the sum of an array of Vector3
 *  - scalar_multiply_2d(): multiply an array of Vector3 by a scalar
 *  - add_scalar_multiply_2d(): add an array of Vector3 to another array of Vector3 multiplied by a scalar
 *      result[i] = array_1[i] + scalar * array_2[i];
 *  - subtract_2d(): subtract an array of Vector3 from another array of Vector3
 *      result[i] = array_1[i] - array_2[i];
 *  - fill_scalar_2d(): fill an array of Vector3 with a scalar
 *  - where_fill_scalar_2d(): fill an array of Vector3 with a scalar if the corresponding element is equal to mask
 *  - where_fill_scalar_else_2d(): fill an array of Vector3 with a scalar if the corresponding element is equal to mask, other places are filled with another array of Vector3 "rest"
 * - print_2d(): print an array of Vector3
 */

class BasicFuncsTest : public testing::Test
{
  protected:
    std::vector<ModuleBase::Vector3<double>> array;
    void SetUp()
    {
        array.push_back(ModuleBase::Vector3<double>(1.0, 2.0, 3.0));
        array.push_back(ModuleBase::Vector3<double>(4.0, 5.0, 6.0));
        array.push_back(ModuleBase::Vector3<double>(7.0, 8.0, 9.0));
    }
    std::string output;
};

TEST_F(BasicFuncsTest, MaxvalAbs2d)
{
    EXPECT_DOUBLE_EQ(maxval_abs_2d(array), 9.0);
}

TEST_F(BasicFuncsTest, MaxlocAbs2d)
{
    std::pair<int,int> maxloc;
    maxloc = maxloc_abs_2d(array);
    EXPECT_EQ(maxloc.first, 2);
    EXPECT_EQ(maxloc.second, 2);
}

TEST_F(BasicFuncsTest, Sum2dDoubleArray)
{
    EXPECT_DOUBLE_EQ(sum_2d(array), 45.0);
}

TEST_F(BasicFuncsTest, Sum2dIntArray)
{
    std::vector<ModuleBase::Vector3<int>> arrayInt;
    arrayInt.push_back(ModuleBase::Vector3<int>(1, 2, 3));
    arrayInt.push_back(ModuleBase::Vector3<int>(4, 5, 6));
    arrayInt.push_back(ModuleBase::Vector3<int>(7, 8, 9));
    int sum = sum_2d(arrayInt);
    double sum2 = sum_2d(arrayInt);
    EXPECT_EQ(sum, 45);
    EXPECT_DOUBLE_EQ(sum2, 45.0);
}

TEST_F(BasicFuncsTest, ScalarMul2d)
{
    std::vector<ModuleBase::Vector3<double>> result;
    scalar_multiply_2d(array, 2.0, result);
    EXPECT_DOUBLE_EQ(result[0][0], 2.0);
    EXPECT_DOUBLE_EQ(result[0][1], 4.0);
    EXPECT_DOUBLE_EQ(result[0][2], 6.0);
    EXPECT_DOUBLE_EQ(result[1][0], 8.0);
    EXPECT_DOUBLE_EQ(result[1][1], 10.0);
    EXPECT_DOUBLE_EQ(result[1][2], 12.0);
    EXPECT_DOUBLE_EQ(result[2][0], 14.0);
    EXPECT_DOUBLE_EQ(result[2][1], 16.0);
    EXPECT_DOUBLE_EQ(result[2][2], 18.0);
}

TEST_F(BasicFuncsTest, AddScalarMul2d)
{
    std::vector<ModuleBase::Vector3<double>> array_2, result;
    array_2.push_back(ModuleBase::Vector3<double>(1.0, 2.0, 3.0));
    array_2.push_back(ModuleBase::Vector3<double>(4.0, 5.0, 6.0));
    array_2.push_back(ModuleBase::Vector3<double>(7.0, 8.0, 9.0));
    add_scalar_multiply_2d(array, array_2, 2.0, result);
    EXPECT_DOUBLE_EQ(result[0][0], 3.0);
    EXPECT_DOUBLE_EQ(result[0][1], 6.0);
    EXPECT_DOUBLE_EQ(result[0][2], 9.0);
    EXPECT_DOUBLE_EQ(result[1][0], 12.0);
    EXPECT_DOUBLE_EQ(result[1][1], 15.0);
    EXPECT_DOUBLE_EQ(result[1][2], 18.0);
    EXPECT_DOUBLE_EQ(result[2][0], 21.0);
    EXPECT_DOUBLE_EQ(result[2][1], 24.0);
    EXPECT_DOUBLE_EQ(result[2][2], 27.0);
}

TEST_F(BasicFuncsTest, Subtract2d)
{
    std::vector<ModuleBase::Vector3<double>> array_2, result;
    array_2.push_back(ModuleBase::Vector3<double>(1.0, 2.0, 3.0));
    array_2.push_back(ModuleBase::Vector3<double>(4.0, 5.0, 6.0));
    array_2.push_back(ModuleBase::Vector3<double>(7.0, 8.0, 9.0));
    subtract_2d(array, array_2, result);
    EXPECT_DOUBLE_EQ(result[0][0], 0.0);
    EXPECT_DOUBLE_EQ(result[0][1], 0.0);
    EXPECT_DOUBLE_EQ(result[0][2], 0.0);
    EXPECT_DOUBLE_EQ(result[1][0], 0.0);
    EXPECT_DOUBLE_EQ(result[1][1], 0.0);
    EXPECT_DOUBLE_EQ(result[1][2], 0.0);
    EXPECT_DOUBLE_EQ(result[2][0], 0.0);
    EXPECT_DOUBLE_EQ(result[2][1], 0.0);
    EXPECT_DOUBLE_EQ(result[2][2], 0.0);
}

TEST_F(BasicFuncsTest, FillScalar2d)
{
    std::vector<ModuleBase::Vector3<double>> result;
    result.resize(3);
    fill_scalar_2d(2.0, result);
    EXPECT_DOUBLE_EQ(result[0][0], 2.0);
    EXPECT_DOUBLE_EQ(result[0][1], 2.0);
    EXPECT_DOUBLE_EQ(result[0][2], 2.0);
    EXPECT_DOUBLE_EQ(result[1][0], 2.0);
    EXPECT_DOUBLE_EQ(result[1][1], 2.0);
    EXPECT_DOUBLE_EQ(result[1][2], 2.0);
    EXPECT_DOUBLE_EQ(result[2][0], 2.0);
    EXPECT_DOUBLE_EQ(result[2][1], 2.0);
    EXPECT_DOUBLE_EQ(result[2][2], 2.0);
}

TEST_F(BasicFuncsTest, WhereFillScalar2d)
{
    std::vector<ModuleBase::Vector3<int>> array_mask;
    array_mask.push_back(ModuleBase::Vector3<int>(1,0,1));
    array_mask.push_back(ModuleBase::Vector3<int>(0,1,0));
    array_mask.push_back(ModuleBase::Vector3<int>(1,0,1));
    std::vector<ModuleBase::Vector3<double>> result;
    where_fill_scalar_2d(array_mask, 1, 2.0, result);
    EXPECT_DOUBLE_EQ(result[0][0], 2.0);
    EXPECT_DOUBLE_EQ(result[0][1], 0.0);
    EXPECT_DOUBLE_EQ(result[0][2], 2.0);
    EXPECT_DOUBLE_EQ(result[1][0], 0.0);
    EXPECT_DOUBLE_EQ(result[1][1], 2.0);
    EXPECT_DOUBLE_EQ(result[1][2], 0.0);
    EXPECT_DOUBLE_EQ(result[2][0], 2.0);
    EXPECT_DOUBLE_EQ(result[2][1], 0.0);
    EXPECT_DOUBLE_EQ(result[2][2], 2.0);
}

TEST_F(BasicFuncsTest, WhereFillScalarElse2d)
{
    std::vector<ModuleBase::Vector3<int>> array_mask;
    array_mask.push_back(ModuleBase::Vector3<int>(1,0,1));
    array_mask.push_back(ModuleBase::Vector3<int>(0,1,0));
    array_mask.push_back(ModuleBase::Vector3<int>(1,0,1));
    std::vector<ModuleBase::Vector3<double>> result;
    std::vector<ModuleBase::Vector3<double>> rest;
    rest.push_back(ModuleBase::Vector3<double>(1.0, 2.0, 3.0));
    rest.push_back(ModuleBase::Vector3<double>(4.0, 5.0, 6.0));
    rest.push_back(ModuleBase::Vector3<double>(7.0, 8.0, 9.0));
    where_fill_scalar_else_2d(array_mask, 1, 2.0, rest, result);
    EXPECT_DOUBLE_EQ(result[0][0], 2.0);
    EXPECT_DOUBLE_EQ(result[0][1], 2.0);
    EXPECT_DOUBLE_EQ(result[0][2], 2.0);
    EXPECT_DOUBLE_EQ(result[1][0], 4.0);
    EXPECT_DOUBLE_EQ(result[1][1], 2.0);
    EXPECT_DOUBLE_EQ(result[1][2], 6.0);
    EXPECT_DOUBLE_EQ(result[2][0], 2.0);
    EXPECT_DOUBLE_EQ(result[2][1], 8.0);
    EXPECT_DOUBLE_EQ(result[2][2], 2.0);
}

TEST_F(BasicFuncsTest, Prin2d)
{
    std::string info = "initial spin";
    testing::internal::CaptureStdout();
    print_2d(info, array, 4);
    output = testing::internal::GetCapturedStdout();
    EXPECT_THAT(output,testing::HasSubstr("initial spin"));
    EXPECT_THAT(output,testing::HasSubstr("ATOM 1         1.0000000000e+00    2.0000000000e+00    3.0000000000e+00"));
    EXPECT_THAT(output,testing::HasSubstr("ATOM 2         4.0000000000e+00    5.0000000000e+00    6.0000000000e+00"));
    EXPECT_THAT(output,testing::HasSubstr("ATOM 3         7.0000000000e+00    8.0000000000e+00    9.0000000000e+00"));
    testing::internal::CaptureStdout();
    print_2d(info, array, 2);
    output = testing::internal::GetCapturedStdout();
    EXPECT_THAT(output,testing::HasSubstr("initial spin"));
    EXPECT_THAT(output,testing::HasSubstr("ATOM 1         3.0000000000e+00"));
    EXPECT_THAT(output,testing::HasSubstr("ATOM 2         6.0000000000e+00"));
    EXPECT_THAT(output,testing::HasSubstr("ATOM 3         9.0000000000e+00"));
}