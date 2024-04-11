#include "parallel_2d.h"
#include "module_base/memory.h"

Parallel_2D::Parallel_2D()
{}
Parallel_2D::~Parallel_2D()
{
    delete[] global2local_row_;
    delete[] global2local_col_;
}

void Parallel_2D::set_proc_dim(const int& dsize, bool mode /*= 0*/)
{
    if (mode) //dim0 >= dim1
    {
        this->dim1 = (int)sqrt((double)dsize);
        while (dsize % this->dim1 != 0)
        {
            this->dim1 = this->dim1 - 1;
        }
        assert(this->dim1 > 0);
        this->dim0 = dsize / this->dim1;
    }
    else    //dim0 <= dim1
    {
        this->dim0 = (int)sqrt((double)dsize);
        while (dsize % this->dim0 != 0)
        {
            this->dim0 = this->dim0 - 1;
        }
        assert(this->dim0 > 0);
        this->dim1 = dsize / this->dim0;
    }
}

bool Parallel_2D::in_this_processor(const int& iw1_all, const int& iw2_all) const
{
    if (global2local_row(iw1_all) == -1)
        return false;
    else if (global2local_col(iw2_all) == -1)
        return false;
    return true;
}

void Parallel_2D::set_global2local(const int& M_A, const int& N_A,
    const bool& div_2d, std::ofstream& ofs_running)
{
    ModuleBase::TITLE("Parallel_2D", "set_global2local");

    this->init_global2local(M_A, N_A, ofs_running);
    if (!div_2d) // xiaohui add 2013-09-02
    {
        for (int i = 0; i < M_A; i++) this->global2local_row_[i] = i;
        for (int i = 0; i < N_A; i++) this->global2local_col_[i] = i;
        this->nrow = M_A;
        this->ncol = N_A;
        this->nloc = this->nrow * this->ncol;
    }
    else
    {
        // ofs_running << " nrow=" << nrow << std::endl;
        for (int irow = 0; irow < this->nrow; irow++)
        {
            int global_row = this->local2global_row(irow);
            this->global2local_row_[global_row] = irow;
        }

        // ofs_running << " ncol=" << ncol << std::endl;
        for (int icol = 0; icol < this->ncol; icol++)
        {
            int global_col = this->local2global_col(icol);
            this->global2local_col_[global_col] = icol;
        }
    }

    return;
}

void Parallel_2D::init_global2local(const int& M_A, const int& N_A, std::ofstream& ofs_running)
{
    ModuleBase::TITLE("Parallel_2D", "init_global2local");
    assert(M_A > 0);
    assert(N_A > 0);

    delete[] this->global2local_row_;
    delete[] this->global2local_col_;

    ModuleBase::GlobalFunc::OUT(ofs_running, "global2local_row dimension", M_A);
    ModuleBase::GlobalFunc::OUT(ofs_running, "global2local_col dimension", N_A);

    this->global2local_row_ = new int[M_A];
    this->global2local_col_ = new int[N_A];

    for (int i = 0; i < M_A; i++) this->global2local_row_[i] = -1;
    for (int i = 0; i < N_A; i++) this->global2local_col_[i] = -1;

    ModuleBase::Memory::record("trace_row_col", sizeof(int) * M_A);
    ModuleBase::Memory::record("trace_row_col", sizeof(int) * N_A);
    return;
}

#ifdef __MPI
extern "C"
{
#include "module_base/blacs_connector.h"
#include "module_base/scalapack_connector.h"
}

void Parallel_2D::mpi_create_cart(const MPI_Comm& diag_world)
{
    ModuleBase::TITLE("Parallel_2D", "mpi_create_cart");
#ifdef __DEBUG
    assert(this->comm_2D != MPI_COMM_NULL);
    assert(this->dim0 > 0 && this->dim1 > 0);
#endif
    // the matrix is divided as ( dim0 * dim1 )
    int period[2] = { 1,1 };
    int dim[2] = { this->dim0, this->dim1 };
    int reorder = 0;
    MPI_Cart_create(diag_world, 2, dim, period, reorder, &this->comm_2D);
    return;
}

void Parallel_2D::set_desc(const int& gr, const int& gc, const int& lld, bool first_time)
{
    ModuleBase::TITLE("Parallel_2D", "set_desc");
#ifdef __DEBUG
    assert(this->comm_2D != MPI_COMM_NULL);
    assert(gr > 0 && gc > 0 && lld > 0);
    assert(this->nb > 0 && this->dim0 > 0 && this->dim1 > 0);
#endif
    if (first_time)
    {
        int myprow=0, mypcol=0;
        int* usermap = new int[this->dim0 * this->dim1];
        for (int i = 0; i < this->dim0; ++i)
        {
            for (int j = 0; j < this->dim1; ++j)
            {
                int pcoord[2] = { i, j };
                MPI_Cart_rank(comm_2D, pcoord, &usermap[i + j * this->dim0]);
            }
        }
        MPI_Fint comm_2D_f = MPI_Comm_c2f(comm_2D);
        Cblacs_get(comm_2D_f, 0, &this->blacs_ctxt);
        Cblacs_gridmap(&this->blacs_ctxt, usermap, this->dim0, this->dim0, this->dim1);
        Cblacs_gridinfo(this->blacs_ctxt, &this->dim0, &this->dim1, &myprow, &mypcol);
        delete[] usermap;
    }
    int ISRC = 0;
    int info = 0;
    descinit_(desc, &gr, &gc, &this->nb, &this->nb, &ISRC, &ISRC, &this->blacs_ctxt, &lld, &info);
}

int Parallel_2D::set_local2global(
    const int& M_A,
    const int& N_A,
    std::ofstream& ofs_running,
    std::ofstream& ofs_warning)
{
    ModuleBase::TITLE("Parallel_2D", "set_local2global");
#ifdef __DEBUG
    assert(M_A > 0 && N_A > 0);
    assert(this->nb > 0);
#endif

    int dim[2];
    int period[2];
    int j, end_id, block;
    int row_b, col_b;

    // (0) every processor get it's id on the 2D comm
    // : ( coord[0], coord[1] )
    MPI_Cart_get(this->comm_2D, 2, dim, period, coord);
    assert(dim[0] == this->dim0);
    assert(dim[1] == this->dim1);

    // (1.1) how many blocks at least
    // eg. M_A = 6400, nb = 64;
    // so block = 10;
    block = M_A / nb;

    // (1.2) If data remain, add 1.
    if (block * nb < M_A)
    {
        block++;
    }

    if (this->testpb)ModuleBase::GlobalFunc::OUT(ofs_running, "Total Row Blocks Number", block);

    // mohan add 2010-09-12
    if (dim[0] > block)
    {
        ofs_warning << " cpu 2D distribution : " << dim[0] << "*" << dim[1] << std::endl;
        ofs_warning << " but, the number of row blocks is " << block << std::endl;
        if (nb > 1)
        {
            return 1;
        }
        else
        {
            ModuleBase::WARNING_QUIT("Parallel_2D::set_local2global", "some processor has no row blocks, try a smaller 'nb2d' parameter.");
        }
    }

    // (2.1) row_b : how many blocks for this processor. (at least)
    row_b = block / dim[0];

    // (2.2) row_b : how many blocks in this processor.
    // if there are blocks remain, some processors add 1.
    if (coord[0] < block % dim[0])
    {
        row_b++;
    }

    if (this->testpb)ModuleBase::GlobalFunc::OUT(ofs_running, "Local Row Block Number", row_b);

    // (3) end_id indicates the last block belong to
    // which processor.
    if (block % dim[0] == 0)
    {
        end_id = dim[0] - 1;
    }
    else
    {
        end_id = block % dim[0] - 1;
    }

    if (this->testpb)ModuleBase::GlobalFunc::OUT(ofs_running, "Ending Row Block in processor", end_id);

    // (4) this->nrow : how many rows in this processors :
    // the one owns the last block is different.
    if (coord[0] == end_id)
    {
        this->nrow = (row_b - 1) * nb + (M_A - (block - 1) * nb);
    }
    else
    {
        this->nrow = row_b * nb;
    }

    if (this->testpb)ModuleBase::GlobalFunc::OUT(ofs_running, "Local rows (including nb)", this->nrow);

    // (5) local2global_row, it's a global index :
    // save explicitly : every row in this processor
    // belongs to which row in the global matrix.
    this->local2global_row_.resize(this->nrow);
    j = 0;
    for (int i = 0; i < row_b; i++)
    {
        for (int k = 0; k < nb && (coord[0] * nb + i * nb * dim[0] + k < M_A); k++, j++)
        {
            this->local2global_row_[j] = coord[0] * nb + i * nb * dim[0] + k;
        }
    }

    // the same procedures for columns.
    block = N_A / nb;
    if (block * nb < N_A)
    {
        block++;
    }
    if (this->testpb)ModuleBase::GlobalFunc::OUT(ofs_running, "Total Col Blocks Number", block);

    if (dim[1] > block)
    {
        ofs_warning << " cpu 2D distribution : " << dim[0] << "*" << dim[1] << std::endl;
        ofs_warning << " but, the number of column blocks is " << block << std::endl;
        if (nb > 1)
        {
            return 1;
        }
        else
        {
            ModuleBase::WARNING_QUIT("Parallel_2D::set_local2global", "some processor has no column blocks.");
        }
    }

    col_b = block / dim[1];
    if (coord[1] < block % dim[1])
    {
        col_b++;
    }

    if (this->testpb)ModuleBase::GlobalFunc::OUT(ofs_running, "Local Col Block Number", col_b);

    if (block % dim[1] == 0)
    {
        end_id = dim[1] - 1;
    }
    else
    {
        end_id = block % dim[1] - 1;
    }

    if (this->testpb)ModuleBase::GlobalFunc::OUT(ofs_running, "Ending Col Block in processor", end_id);

    if (coord[1] == end_id)
    {
        this->ncol = (col_b - 1) * nb + (N_A - (block - 1) * nb);
    }
    else
    {
        this->ncol = col_b * nb;
    }

    if (this->testpb)ModuleBase::GlobalFunc::OUT(ofs_running, "Local columns (including nb)", this->ncol);

    //set nloc
    this->nloc = this->nrow * this->ncol;

    this->local2global_col_.resize(this->ncol);

    j = 0;
    for (int i = 0; i < col_b; i++)
    {
        for (int k = 0; k < nb && (coord[1] * nb + i * nb * dim[1] + k < N_A); k++, j++)
        {
            this->local2global_col_[j] = coord[1] * nb + i * nb * dim[1] + k;
        }
    }
    return 0;
}
#else
void Parallel_2D::set_serial(const int& M_A, const int& N_A)
{
    ModuleBase::TITLE("Parallel_2D", "set_serial");
    this->nrow = M_A;
    this->ncol = N_A;
    this->nloc = this->nrow * this->ncol;
    this->local2global_row_.resize(this->nrow);
    this->local2global_col_.resize(this->ncol);
    for (int i = 0; i < this->nrow; i++) this->local2global_row_[i] = i;
    for (int i = 0; i < this->ncol; i++) this->local2global_col_[i] = i;
}
#endif