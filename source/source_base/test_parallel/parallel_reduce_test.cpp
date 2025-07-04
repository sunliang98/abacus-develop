#ifdef __MPI
#include "source_base/parallel_reduce.h"

#include "source_base/parallel_global.h"
#include "mpi.h"

#include "gtest/gtest.h"
#include <assert.h>
#include <random>
#include <time.h>

/*************************************************************
 *  unit test of functions in parallel_reduce.cpp
 *************************************************************/

/**
 * The tested functions are mainly wrappers of MPI_Allreduce
 * and MPI_Allgather in ABACUS, as defined in source_base/
 * parallel_reduce.h.
 *
 * The logic to test MPI_Allreduce wrapper functions is to
 * calculate the sum of the total array in two ways, one by
 * using MPI_Allreduce with only 1 number, another one by
 * using MPI_Allreduce with n numbers. The total array is
 * deemed as the sum of local arrays with the same length.
 *   1. ReduceIntAll:
 *       Tests two variations of reduce_all()
 *   2. ReduceDoubleAll:
 *       Tests two variations of reduce_all()
 *   3. ReduceComplexAll:
 *       Tests two variations of reduce_complex_all()
 *   4. GatherIntAll:
 *       Tests gather_int_all() and gather_min_int_all()
 *   5. GatherDoubleAll:
 *       Tests gather_min_double_all() and gather_max_double_all()
 *   6. ReduceIntDiag:
 *       Tests reduce_int_diag()
 *   7. ReduceDoubleDiag:
 *       Tests reduce_double_diag()
 *   8. ReduceIntGrid:
 *       Tests reduce_int_grid()
 *   9. ReduceDoubleGrid:
 *       Tests reduce_double_grid()
 *   10.ReduceDoublePool:
 *       Tests two variations of reduce_pool()
 *       and two variations of reduce_double_allpool()
 *   11.ReduceComplexPool:
 *       Tests two variations of reduce_pool()
 *   12.GatherDoublePool:
 *       Tests gather_min_double_pool() and gather_max_double_pool()
 *
 *
 */

class MPIContext
{
  public:
    MPIContext()
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &_size);
    }

    int GetRank() const
    {
        return _rank;
    }
    int GetSize() const
    {
        return _size;
    }

    int drank;
    int dsize;
    int dcolor;

    int grank;
    int gsize;

    int kpar;
    int nproc_in_pool;
    int my_pool;
    int rank_in_pool;

  private:
    int _rank;
    int _size;
};

const int MIN_FOR_RAND = 1;
const int MAX_FOR_RAND = 99999;

// generate an array of random numbers
template <typename T>
T* get_rand_array(int num, int my_rank)
{
    T* rand_array = new T[num]();
    assert(num > 0);
    std::default_random_engine e(time(NULL) * (my_rank + 1));
    std::uniform_int_distribution<unsigned> u(MIN_FOR_RAND, MAX_FOR_RAND);
    for (int i = 0; i < num; i++)
    {
        rand_array[i] = static_cast<int>(u(e)) % 100;
    }
    return rand_array;
}

class ParaReduce : public testing::Test
{
  protected:
    int num_per_process = 100;
    MPIContext mpiContext;
    int my_rank = 0;
    int nproc = 0;
    void SetUp() override
    {
        my_rank = mpiContext.GetRank();
        nproc = mpiContext.GetSize();
    }
};

TEST_F(ParaReduce, ReduceIntAll)
{
    // generate a random array
    int* rand_array = NULL;
    rand_array = get_rand_array<int>(num_per_process, my_rank);

    // calculate local sum
    int local_sum = 0;
    for (int i = 0; i < num_per_process; i++)
    {
        local_sum += rand_array[i];
    }

    // first way of calculating global sum
    int global_sum_first = local_sum;
    Parallel_Reduce::reduce_all(global_sum_first);
    // second way of calculating global sum
    Parallel_Reduce::reduce_all(rand_array, num_per_process);
    int global_sum_second = 0;
    for (int i = 0; i < num_per_process; i++)
    {
        global_sum_second += rand_array[i];
    }
    // compare two sums
    /// printf("rank %d sum1 = %d, sum2 = %d\n",my_rank,
    ///	global_sum_first, global_sum_second);
    EXPECT_EQ(global_sum_first, global_sum_second);
    delete[] rand_array;
}

TEST_F(ParaReduce, ReduceDoubleAll)
{
    // generate a random array
    double* rand_array = NULL;
    rand_array = get_rand_array<double>(num_per_process, my_rank);

    // calculate local sum
    double local_sum = 0.0;
    for (int i = 0; i < num_per_process; i++)
    {
        local_sum += rand_array[i];
    }

    // first way of calculating global sum
    double global_sum_first = local_sum;
    Parallel_Reduce::reduce_all(global_sum_first);
    // second way of calculating global sum
    Parallel_Reduce::reduce_all(rand_array, num_per_process);
    double global_sum_second = 0;
    for (int i = 0; i < num_per_process; i++)
    {
        global_sum_second += rand_array[i];
    }
    // compare two sums
    /// printf("rank %d sum1 = %f, sum2 = %f\n",my_rank,
    ///	global_sum_first, global_sum_second);
    EXPECT_NEAR(global_sum_first, global_sum_second, 1e-14);
    delete[] rand_array;
}

TEST_F(ParaReduce, ReduceComplexAll)
{
    // allocate local complex vector
    std::complex<double>* rand_array = nullptr;
    rand_array = new std::complex<double>[num_per_process];
    // set its elements to random complex numbers
    std::default_random_engine e(time(NULL) * (my_rank + 1));
    std::uniform_int_distribution<unsigned> u(MIN_FOR_RAND, MAX_FOR_RAND);
    // and calculate local sum
    std::complex<double> local_sum = std::complex<double>{0.0, 0.0};
    for (int i = 0; i < num_per_process; i++)
    {
        double realpart = pow(-1.0, u(e) % 2) * static_cast<double>(u(e)) / MAX_FOR_RAND;
        double imagpart = pow(-1.0, u(e) % 2) * static_cast<double>(u(e)) / MAX_FOR_RAND;
        rand_array[i] = std::complex<double>{realpart, imagpart};
        local_sum += rand_array[i];
        /// printf("pre rank %d rand_array[%d] = (%f,%f) \n",my_rank,i,
        /// rand_array[i].real(), rand_array[i].imag());
    }
    // first way of calculating global sum
    std::complex<double> global_sum_first = local_sum;
    Parallel_Reduce::reduce_all(global_sum_first);

    // second way of calculating global sum
    Parallel_Reduce::reduce_all(rand_array, num_per_process);
    std::complex<double> global_sum_second = std::complex<double>{0.0, 0.0};
    for (int i = 0; i < num_per_process; i++)
    {
        global_sum_second += rand_array[i];
        /// printf("pos rank %d rand_array[%d] = (%f,%f) \n",my_rank,i,
        /// rand_array[i].real(), rand_array[i].imag());
    }
    // compare two sums
    /// printf("rank %d sum1 = (%f,%f) sum2 = (%f,%f)\n",my_rank,
    /// global_sum_first.real(), global_sum_first.imag(),
    /// global_sum_second.real(), global_sum_second.imag());
    EXPECT_NEAR(global_sum_first.real(), global_sum_second.real(), 1e-13);
    EXPECT_NEAR(global_sum_first.imag(), global_sum_second.imag(), 1e-13);

    delete[] rand_array;
}

TEST_F(ParaReduce, GatherIntAll)
{
    std::default_random_engine e(time(NULL) * (my_rank + 1));
    std::uniform_int_distribution<unsigned> u(MIN_FOR_RAND, MAX_FOR_RAND);
    int local_number = static_cast<int>(u(e)) % 100;
    // printf("pre rank %d local_number = %d \n ",my_rank,local_number);
    int* array = new int[nproc]();
    // use MPI_Allgather to gather together numbers
    Parallel_Reduce::gather_int_all(local_number, array);
    EXPECT_EQ(local_number, array[my_rank]);
    // get minimum integer among all processes
    int min_number = local_number;
    Parallel_Reduce::gather_min_int_all(nproc, min_number);
    for (int i = 0; i < nproc; i++)
    {
        EXPECT_LE(min_number, array[i]);
        /// printf("post rank %d array[%d] = %d, min = %d \n",
        ///	my_rank,i,array[i],min_number);
    }
    delete[] array;
}

TEST_F(ParaReduce, GatherDoubleAll)
{
    std::default_random_engine e(time(NULL) * (my_rank + 1));
    std::uniform_int_distribution<unsigned> u(MIN_FOR_RAND, MAX_FOR_RAND);
    double local_number = static_cast<int>(u(e)) % 100;
    // printf("pre rank %d local_number = %d \n ",my_rank,local_number);
    double* array = new double[nproc]();
    // use MPI_Allgather to gather together numbers
    MPI_Allgather(&local_number, 1, MPI_DOUBLE, array, 1, MPI_DOUBLE, MPI_COMM_WORLD);

    EXPECT_EQ(local_number, array[my_rank]);
    // get minimum integer among all processes
    double min_number = local_number;
    Parallel_Reduce::gather_min_double_all(nproc, min_number);
    // get maximum integer among all processes
    double max_number = local_number;
    Parallel_Reduce::gather_max_double_all(nproc, max_number);
    for (int i = 0; i < nproc; i++)
    {
        EXPECT_LE(min_number, array[i]);
        EXPECT_GE(max_number, array[i]);
        /// printf("post rank %d array[%d] = %f, min = %f, max = %f \n",
        ///	my_rank,i,array[i],min_number,max_number);
    }
    delete[] array;
}

TEST_F(ParaReduce, ReduceIntDiag)
{
    /// num_per_process = 2;
    // NPROC is set to 4 in parallel_global_test.sh
    if (nproc == 4)
    {
        Parallel_Global::split_diag_world(2, nproc, my_rank, mpiContext.drank, mpiContext.dsize, mpiContext.dcolor);
        // generate a random array
        int* rand_array = NULL;
        rand_array = get_rand_array<int>(num_per_process, my_rank);

        // calculate local sum
        int local_sum = 0;
        for (int i = 0; i < num_per_process; i++)
        {
            local_sum += rand_array[i];
            /// printf(" pre world_rank %d, drank %d rand_array[%d] = %d\n",
            ///	my_rank,mpiContext.dsize,i, rand_array[i]);
        }

        // first way of calculating diag sum
        int diag_sum_first = local_sum;
        Parallel_Reduce::reduce_int_diag(diag_sum_first);
        // second way of calculating global sum
        int* swap = new int[num_per_process]();
        MPI_Allreduce(rand_array, swap, num_per_process, MPI_INT, MPI_SUM, DIAG_WORLD);
        int diag_sum_second = 0;
        for (int i = 0; i < num_per_process; i++)
        {
            diag_sum_second += swap[i];
            /// printf(" post world_rank %d, drank %d swap[%d] = %d\n",
            ///	my_rank,mpiContext.dsize,i, swap[i]);
        }
        // compare two sums
        /// printf("world_rank %d, drank %d sum1 = %d, sum2 = %d\n",
        ///	my_rank,mpiContext.dsize,diag_sum_first, diag_sum_second);
        EXPECT_EQ(diag_sum_first, diag_sum_second);
        delete[] rand_array;
        delete[] swap;
        MPI_Comm_free(&DIAG_WORLD);
    }
}

TEST_F(ParaReduce, ReduceDoubleDiag)
{
    /// num_per_process = 1;
    // NPROC is set to 4 in parallel_global_test.sh
    if (nproc == 4)
    {
        Parallel_Global::split_diag_world(2, nproc, my_rank, mpiContext.drank, mpiContext.dsize, mpiContext.dcolor);
        // generate a random array
        double* rand_array = NULL;
        rand_array = get_rand_array<double>(num_per_process, my_rank);

        // calculate local sum
        double local_sum = 0.0;
        for (int i = 0; i < num_per_process; i++)
        {
            local_sum += rand_array[i];
            /// printf(" pre world_rank %d, drank %d rand_array[%d] = %f\n",
            ///	my_rank,mpiContext.dsize,i, rand_array[i]);
        }

        // first way of calculating diag sum
        double diag_sum_first = 0.0;
        MPI_Allreduce(&local_sum, &diag_sum_first, 1, MPI_DOUBLE, MPI_SUM, DIAG_WORLD);

        // second way of calculating global sum
        Parallel_Reduce::reduce_double_diag(rand_array, num_per_process);
        double diag_sum_second = 0.0;
        for (int i = 0; i < num_per_process; i++)
        {
            diag_sum_second += rand_array[i];
            /// printf(" post world_rank %d, drank %d rand_array[%d] = %f\n",
            ///	my_rank,mpiContext.dsize,i, rand_array[i]);
        }
        // compare two sums
        /// printf("world_rank %d, drank %d sum1 = %f, sum2 = %f\n",
        ///	my_rank,mpiContext.dsize,diag_sum_first, diag_sum_second);
        EXPECT_NEAR(diag_sum_first, diag_sum_second, 1e-13);
        delete[] rand_array;
        MPI_Comm_free(&DIAG_WORLD);
    }
}

TEST_F(ParaReduce, ReduceIntGrid)
{
    /// num_per_process = 2;
    // NPROC is set to 4 in parallel_global_test.sh
    if (nproc == 4)
    {
        Parallel_Global::split_grid_world(2, nproc, my_rank, mpiContext.grank, mpiContext.gsize);
        // generate a random array
        int* rand_array = NULL;
        rand_array = get_rand_array<int>(num_per_process, my_rank);

        // calculate local sum
        int local_sum = 0;
        for (int i = 0; i < num_per_process; i++)
        {
            local_sum += rand_array[i];
            /// printf(" pre world_rank %d, drank %d rand_array[%d] = %d\n",
            ///	my_rank,mpiContext.dsize,i, rand_array[i]);
        }

        // first way of calculating diag sum
        int grid_sum_first = 0;
        MPI_Allreduce(&local_sum, &grid_sum_first, 1, MPI_INT, MPI_SUM, GRID_WORLD);

        // second way of calculating global sum
        Parallel_Reduce::reduce_int_grid(rand_array, num_per_process);
        int grid_sum_second = 0;
        for (int i = 0; i < num_per_process; i++)
        {
            grid_sum_second += rand_array[i];
            /// printf(" post world_rank %d, drank %d rand_array[%d] = %d\n",
            ///	my_rank,mpiContext.dsize,i, rand_array[i]);
        }
        // compare two sums
        /// printf("world_rank %d, drank %d sum1 = %d, sum2 = %d\n",
        ///	my_rank,mpiContext.dsize,grid_sum_first, grid_sum_second);
        EXPECT_EQ(grid_sum_first, grid_sum_second);
        delete[] rand_array;
        MPI_Comm_free(&GRID_WORLD);
    }
}

TEST_F(ParaReduce, ReduceDoubleGrid)
{
    /// num_per_process = 1;
    // NPROC is set to 4 in parallel_global_test.sh
    if (nproc == 4)
    {
        Parallel_Global::split_grid_world(2, nproc, my_rank, mpiContext.grank, mpiContext.gsize);
        // generate a random array
        double* rand_array = NULL;
        rand_array = get_rand_array<double>(num_per_process, my_rank);

        // calculate local sum
        double local_sum = 0.0;
        for (int i = 0; i < num_per_process; i++)
        {
            local_sum += rand_array[i];
            /// printf(" pre world_rank %d, drank %d rand_array[%d] = %f\n",
            ///	my_rank,mpiContext.dsize,i, rand_array[i]);
        }

        // first way of calculating diag sum
        double grid_sum_first = 0.0;
        MPI_Allreduce(&local_sum, &grid_sum_first, 1, MPI_DOUBLE, MPI_SUM, GRID_WORLD);

        // second way of calculating global sum
        Parallel_Reduce::reduce_double_grid(rand_array, num_per_process);
        double grid_sum_second = 0.0;
        for (int i = 0; i < num_per_process; i++)
        {
            grid_sum_second += rand_array[i];
            /// printf(" post world_rank %d, drank %d rand_array[%d] = %f\n",
            ///	my_rank,mpiContext.dsize,i, rand_array[i]);
        }
        // compare two sums
        /// printf("world_rank %d, drank %d sum1 = %f, sum2 = %f\n",
        ///	my_rank,mpiContext.dsize,grid_sum_first, grid_sum_second);
        EXPECT_NEAR(grid_sum_first, grid_sum_second, 1e-13);
        delete[] rand_array;
        MPI_Comm_free(&GRID_WORLD);
    }
}

TEST_F(ParaReduce, ReduceDoublePool)
{
    /// num_per_process = 1;
    // NPROC is set to 4 in parallel_global_test.sh
    if (nproc == 4)
    {
        mpiContext.kpar = 2;
        Parallel_Global::divide_mpi_groups(nproc,
                                           mpiContext.kpar,
                                           my_rank,
                                           mpiContext.nproc_in_pool,
                                           mpiContext.my_pool,
                                           mpiContext.rank_in_pool);
        MPI_Comm_split(MPI_COMM_WORLD, mpiContext.my_pool, mpiContext.rank_in_pool, &POOL_WORLD);
        /// printf("word_rank/world_size = %d/%d, pool_rank/pool_size = %d/%d \n",
        ///		my_rank,nproc,
        ///		mpiContext.rank_in_pool,mpiContext.nproc_in_pool);

        // generate a random array
        double* rand_array = NULL;
        rand_array = get_rand_array<double>(num_per_process, my_rank);

        // calculate local sum
        double local_sum = 0.0;
        for (int i = 0; i < num_per_process; i++)
        {
            local_sum += rand_array[i];
        }

        // first way of calculating pool sum
        double pool_sum_first = local_sum;
        Parallel_Reduce::reduce_pool(pool_sum_first);
        // second way of calculating pool sum
        Parallel_Reduce::reduce_pool(rand_array, num_per_process);
        double pool_sum_second = 0.0;
        for (int i = 0; i < num_per_process; i++)
        {
            pool_sum_second += rand_array[i];
        }
        // compare two pool sums
        /// printf("pool rank %d sum1 = %f, sum2 = %f\n",my_rank,
        ///	pool_sum_first, pool_sum_second);
        EXPECT_NEAR(pool_sum_first, pool_sum_second, 1e-14);

        // first way of calculating global sum
        double global_sum_first = pool_sum_first;
        Parallel_Reduce::reduce_double_allpool(mpiContext.kpar, mpiContext.nproc_in_pool, global_sum_first);
        // second way of calculating pool sum
        Parallel_Reduce::reduce_double_allpool(mpiContext.kpar, mpiContext.nproc_in_pool, rand_array, num_per_process);
        double global_sum_second = 0.0;
        for (int i = 0; i < num_per_process; i++)
        {
            global_sum_second += rand_array[i];
        }
        // compare two global sums
        /// printf("global rank %d sum1 = %f, sum2 = %f\n",my_rank,
        ///	global_sum_first, global_sum_second);
        EXPECT_NEAR(global_sum_first, global_sum_second, 1e-14);

        delete[] rand_array;
        MPI_Comm_free(&POOL_WORLD);
    }
}

TEST_F(ParaReduce, ReduceComplexPool)
{
    /// num_per_process = 1;
    // NPROC is set to 4 in parallel_global_test.sh
    if (nproc == 4)
    {
        mpiContext.kpar = 2;
        Parallel_Global::divide_mpi_groups(nproc,
                                           mpiContext.kpar,
                                           my_rank,
                                           mpiContext.nproc_in_pool,
                                           mpiContext.my_pool,
                                           mpiContext.rank_in_pool);
        MPI_Comm_split(MPI_COMM_WORLD, mpiContext.my_pool, mpiContext.rank_in_pool, &POOL_WORLD);
        /// printf("word_rank/world_size = %d/%d, pool_rank/pool_size = %d/%d \n",
        ///		my_rank,nproc,
        ///		mpiContext.rank_in_pool,mpiContext.nproc_in_pool);
        // allocate local complex vector
        std::complex<double>* rand_array = nullptr;
        rand_array = new std::complex<double>[num_per_process];
        // set its elements to random complex numbers
        std::default_random_engine e(time(NULL) * (my_rank + 1));
        std::uniform_int_distribution<unsigned> u(MIN_FOR_RAND, MAX_FOR_RAND);
        // and calculate local sum
        std::complex<double> local_sum = std::complex<double>{0.0, 0.0};
        for (int i = 0; i < num_per_process; i++)
        {
            double realpart = pow(-1.0, u(e) % 2) * static_cast<double>(u(e)) / MAX_FOR_RAND;
            double imagpart = pow(-1.0, u(e) % 2) * static_cast<double>(u(e)) / MAX_FOR_RAND;
            rand_array[i] = std::complex<double>{realpart, imagpart};
            local_sum += rand_array[i];
            /// printf("pre rank %d rand_array[%d] = (%f,%f) \n",my_rank,i,
            /// rand_array[i].real(), rand_array[i].imag());
        }
        // first way of calculating pool sum
        std::complex<double> pool_sum_first = local_sum;
        Parallel_Reduce::reduce_pool(pool_sum_first);

        // second way of calculating pool sum
        Parallel_Reduce::reduce_pool(rand_array, num_per_process);
        std::complex<double> pool_sum_second = std::complex<double>{0.0, 0.0};
        for (int i = 0; i < num_per_process; i++)
        {
            pool_sum_second += rand_array[i];
            /// printf("pos rank %d rand_array[%d] = (%f,%f) \n",my_rank,i,
            /// rand_array[i].real(), rand_array[i].imag());
        }
        // compare two sums
        /// printf("rank %d sum1 = (%f,%f) sum2 = (%f,%f)\n",my_rank,
        /// pool_sum_first.real(), pool_sum_first.imag(),
        /// pool_sum_second.real(), pool_sum_second.imag());
        EXPECT_NEAR(pool_sum_first.real(), pool_sum_second.real(), 1e-13);
        EXPECT_NEAR(pool_sum_first.imag(), pool_sum_second.imag(), 1e-13);

        delete[] rand_array;
        MPI_Comm_free(&POOL_WORLD);
    }
}

TEST_F(ParaReduce, GatherDoublePool)
{
    /// num_per_process = 1;
    // NPROC is set to 4 in parallel_global_test.sh
    if (nproc == 4)
    {
        mpiContext.kpar = 2;
        Parallel_Global::divide_mpi_groups(nproc,
                                           mpiContext.kpar,
                                           my_rank,
                                           mpiContext.nproc_in_pool,
                                           mpiContext.my_pool,
                                           mpiContext.rank_in_pool);
        MPI_Comm_split(MPI_COMM_WORLD, mpiContext.my_pool, mpiContext.rank_in_pool, &POOL_WORLD);
        std::default_random_engine e(time(NULL) * (my_rank + 1));
        std::uniform_int_distribution<unsigned> u(MIN_FOR_RAND, MAX_FOR_RAND);
        double local_number = static_cast<int>(u(e)) % 100;
        // printf("pre rank %d local_number = %d \n ",my_rank,local_number);
        double* array = new double[mpiContext.nproc_in_pool]();
        // use MPI_Allgather to gather together numbers
        MPI_Allgather(&local_number, 1, MPI_DOUBLE, array, 1, MPI_DOUBLE, POOL_WORLD);

        EXPECT_EQ(local_number, array[mpiContext.rank_in_pool]);
        // get minimum integer among all processes
        double min_number = local_number;
        Parallel_Reduce::gather_min_double_pool(mpiContext.nproc_in_pool, min_number);
        // get maximum integer among all processes
        double max_number = local_number;
        Parallel_Reduce::gather_max_double_pool(mpiContext.nproc_in_pool, max_number);
        for (int i = 0; i < mpiContext.nproc_in_pool; i++)
        {
            EXPECT_LE(min_number, array[i]);
            EXPECT_GE(max_number, array[i]);
            /// printf("post rank %d, pool rank %d, array[%d] = %f, min = %f, max = %f \n",
            ///  my_rank,mpiContext.rank_in_pool,i,array[i],min_number,max_number);
        }
        delete[] array;
        MPI_Comm_free(&POOL_WORLD);
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
#endif
