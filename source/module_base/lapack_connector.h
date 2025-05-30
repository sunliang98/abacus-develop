#ifndef LAPACKCONNECTOR_HPP
#define LAPACKCONNECTOR_HPP

#include <new>
#include <stdexcept>
#include <iostream>
#include <cassert>
#include "matrix.h"
#include "complexmatrix.h"
#include "global_function.h"

//Naming convention of lapack subroutines : ammxxx, where
//"a" specifies the data type:
//  - d stands for double
//  - z stands for complex double
//"mm" specifies the type of matrix, for example:
//  - he stands for hermitian
//  - sy stands for symmetric
//"xxx" specifies the type of problem, for example:
//  - gv stands for generalized eigenvalue

extern "C"
{
// solve the generalized eigenproblem Ax=eBx, where A is Hermitian and complex couble
    // zhegv_ & zhegvd_ returns all eigenvalues while zhegvx_ returns selected ones
    void dsygvd_(const int* itype, const char* jobz, const char* uplo, const int* n,
        double* a, const int* lda,
        const double* b, const int* ldb, double* w,
        double* work, int* lwork,
        int* iwork, int* liwork, int* info);

    void chegvd_(const int* itype, const char* jobz, const char* uplo, const int* n,
             std::complex<float>* a, const int* lda,
             const std::complex<float>* b, const int* ldb, float* w,
             std::complex<float>* work, int* lwork, float* rwork, int* lrwork,
             int* iwork, int* liwork, int* info);

    void zhegvd_(const int* itype, const char* jobz, const char* uplo, const int* n,
                 std::complex<double>* a, const int* lda, 
                 const std::complex<double>* b, const int* ldb, double* w,
                 std::complex<double>* work, int* lwork, double* rwork, int* lrwork,
                 int* iwork, int* liwork, int* info);

    void dsyevx_(const char* jobz, const char* range, const char* uplo, const int* n,
        double* a, const int* lda,
        const double* vl, const double* vu, const int* il, const int* iu, const double* abstol,
        const int* m, double* w, double* z, const int* ldz,
        double* work, const int* lwork, double* rwork, int* iwork, int* ifail, int* info);

    void cheevx_(const char* jobz, const char* range, const char* uplo, const int* n,
             std::complex<float> *a, const int* lda,
             const float* vl, const float* vu, const int* il, const int* iu, const float* abstol,
             const int* m, float* w, std::complex<float> *z, const int *ldz,
             std::complex<float> *work, const int* lwork, float* rwork, int* iwork, int* ifail, int* info);

    void zheevx_(const char* jobz, const char* range, const char* uplo, const int* n, 
                 std::complex<double> *a, const int* lda,
                 const double* vl, const double* vu, const int* il, const int* iu, const double* abstol, 
                 const int* m, double* w, std::complex<double> *z, const int *ldz, 
                 std::complex<double> *work, const int* lwork, double* rwork, int* iwork, int* ifail, int* info);


    void dsygvx_(const int* itype, const char* jobz, const char* range, const char* uplo,
        const int* n, double* A, const int* lda, double* B, const int* ldb,
        const double* vl, const double* vu, const int* il, const int* iu,
        const double* abstol, const int* m, double* w, double* Z, const int* ldz,
        double* work, const int* lwork, int* iwork, int* ifail, int* info);

    void chegvx_(const int* itype,const char* jobz,const char* range,const char* uplo,
             const int* n,std::complex<float> *a,const int* lda,std::complex<float> *b,
             const int* ldb,const float* vl,const float* vu,const int* il,
             const int* iu,const float* abstol,const int* m,float* w,
             std::complex<float> *z,const int *ldz,std::complex<float> *work,const int* lwork,
             float* rwork,int* iwork,int* ifail,int* info);

    void zhegvx_(const int* itype,const char* jobz,const char* range,const char* uplo,
                 const int* n,std::complex<double> *a,const int* lda,std::complex<double> *b,
                 const int* ldb,const double* vl,const double* vu,const int* il,
                 const int* iu,const double* abstol,const int* m,double* w,
                 std::complex<double> *z,const int *ldz,std::complex<double> *work,const int* lwork,
                 double* rwork,int* iwork,int* ifail,int* info);

    void zhegv_(const int* itype,const char* jobz,const char* uplo,const int* n,
                std::complex<double>* a,const int* lda,std::complex<double>* b,const int* ldb,
                double* w,std::complex<double>* work,int* lwork,double* rwork,int* info);
    void chegv_(const int* itype,const char* jobz,const char* uplo,const int* n,
                std::complex<float>* a,const int* lda,std::complex<float>* b,const int* ldb,
                float* w,std::complex<float>* work,int* lwork,float* rwork,int* info);
	void dsygv_(const int* itype, const char* jobz,const char* uplo, const int* n,
				double* a,const int* lda,double* b,const int* ldb,
	 			double* w,double* work,int* lwork,int* info);

    // solve the eigenproblem Ax=ex, where A is Hermitian and complex couble
    // zheev_ returns all eigenvalues while zheevx_ returns selected ones
    void zheev_(const char* jobz,const char* uplo,const int* n,std::complex<double> *a,
                const int* lda,double* w,std::complex<double >* work,const int* lwork,
                double* rwork,int* info);
    void cheev_(const char* jobz,const char* uplo,const int* n,std::complex<float> *a,
                const int* lda,float* w,std::complex<float >* work,const int* lwork,
                float* rwork,int* info);
	void dsyev_(const char* jobz,const char* uplo,const int* n,double *a,
                const int* lda,double* w,double* work,const int* lwork, int* info);

    // solve the eigenproblem Ax=ex, where A is a general matrix
    void dgeev_(const char* jobvl, const char* jobvr, const int* n, double* a, const int* lda,
        double* wr, double* wi, double* vl, const int* ldvl, double* vr, const int* ldvr,
        double* work, const int* lwork, int* info);
    void zgeev_(const char* jobvl, const char* jobvr, const int* n, std::complex<double>* a, const int* lda,
        std::complex<double>* w, std::complex<double>* vl, const int* ldvl, std::complex<double>* vr, const int* ldvr,
        std::complex<double>* work, const int* lwork, double* rwork, int* info);
    // liuyu add 2023-10-03
    // dgetri and dgetrf computes the inverse of a n*n real matrix
    void dgetri_(const int* n, double* a, const int* lda, const int* ipiv, double* work, const int* lwork, int* info);
    void dgetrf_(const int* m, const int* n, double* a, const int* lda, int* ipiv, int* info);

    // dsytrf_ computes the Bunch-Kaufman factorization of a double precision
    // symmetric matrix, while dsytri takes its output to perform martrix inversion
    void dsytrf_(const char* uplo, const int* n, double * a, const int* lda,
                 int *ipiv,double *work, const int* lwork ,int *info);
    void dsytri_(const char* uplo,const int* n,double *a, const int *lda,
                 int *ipiv, double * work,int *info);
    // Peize Lin add dsptrf and dsptri 2016-06-21, to compute inverse real symmetry indefinit matrix.
    // dpotrf computes the Cholesky factorization of a real symmetric positive definite matrix
    // while dpotri taks its output to perform matrix inversion
    void spotrf_(const char*const uplo, const int*const n, float*const A, const int*const lda, int*const info);
    void dpotrf_(const char*const uplo, const int*const n, double*const A, const int*const lda, int*const info);
    void cpotrf_(const char*const uplo, const int*const n, std::complex<float>*const A, const int*const lda, int*const info);
    void zpotrf_(const char*const uplo, const int*const n, std::complex<double>*const A, const int*const lda, int*const info);
    void spotri_(const char*const uplo, const int*const n, float*const A, const int*const lda, int*const info);
    void dpotri_(const char*const uplo, const int*const n, double*const A, const int*const lda, int*const info);
    void cpotri_(const char*const uplo, const int*const n, std::complex<float>*const A, const int*const lda, int*const info);
    void zpotri_(const char*const uplo, const int*const n, std::complex<double>*const A, const int*const lda, int*const info);

    // zgetrf computes the LU factorization of a general matrix
    // while zgetri takes its output to perform matrix inversion
    void zgetrf_(const int* m, const int *n, std::complex<double> *A, const int *lda, int *ipiv, int* info);
    void zgetri_(const int* n, std::complex<double>* A, const int* lda, const int* ipiv, std::complex<double>* work, const int* lwork, int* info);

    // if trans=='N':	C = alpha * A * A.H + beta * C
	// if trans=='C':	C = alpha * A.H * A + beta * C
	void zherk_(const char *uplo, const char *trans, const int *n, const int *k,
		const double *alpha, const std::complex<double> *A, const int *lda,
		const double *beta, std::complex<double> *C, const int *ldc);
    void cherk_(const char* uplo, const char* trans, const int* n, const int* k,
        const float* alpha, const std::complex<float>* A, const int* lda,
        const float* beta, std::complex<float>* C, const int* ldc);

	// computes all eigenvalues of a symmetric tridiagonal matrix
	// using the Pal-Walker-Kahan variant of the QL or QR algorithm.
	void dsterf_(int *n, double *d, double *e, int *info);
    // computes the eigenvectors of a real symmetric tridiagonal
    // matrix T corresponding to specified eigenvalues
	void dstein_(int *n, double* d, double *e, int *m, double *w,
		int* block, int* isplit, double* z, int *lda, double *work,
		int* iwork, int* ifail, int *info);
    // computes the eigenvectors of a complex symmetric tridiagonal
    // matrix T corresponding to specified eigenvalues
 	void zstein_(int *n, double* d, double *e, int *m, double *w,
        int* block, int* isplit, std::complex<double>* z, int *lda, double *work,
        int* iwork, int* ifail, int *info);

	// computes the Cholesky factorization of a symmetric
	// positive definite matrix A.
	void dpotf2_(char *uplo, int *n, double *a, int *lda, int *info);
	void zpotf2_(char *uplo,int *n,std::complex<double> *a, int *lda, int *info);

    // reduces a symmetric definite generalized eigenproblem to standard form
    // using the factorization results obtained from spotrf
	void dsygs2_(int *itype, char *uplo, int *n, double *a, int *lda, double *b, int *ldb, int *info);
	void zhegs2_(int *itype, char *uplo, int *n, std::complex<double> *a, int *lda, std::complex<double> *b, int *ldb, int *info);

    // copies a into b
	void dlacpy_(char *uplo, int *m, int *n, double* a, int *lda, double *b, int *ldb);
	void zlacpy_(char *uplo, int *m, int *n, std::complex<double>* a, int *lda, std::complex<double> *b, int *ldb);

    // generates a real elementary reflector H of order n, such that
    //   H * ( alpha ) = ( beta ),   H is unitary.
    //       (   x   )   (   0  )
	void dlarfg_(int *n, double *alpha, double *x, int *incx, double *tau);
	void zlarfg_(int *n, std::complex<double> *alpha, std::complex<double> *x, int *incx, std::complex<double> *tau);

    // solve a tridiagonal linear system
    void dgtsv_(int* N, int* NRHS, double* DL, double* D, double* DU, double* B, int* LDB, int* INFO);

    // solve Ax = b 
    void dsysv_(const char* uplo, const int* n, const int* m, double * a, const int* lda,
                 int *ipiv, double * b, const int* ldb, double *work, const int* lwork ,int *info);
}

#ifdef GATHER_INFO
#define zhegvx_ zhegvx_i
void zhegvx_i(const int* itype,
              const char* jobz,
              const char* range,
              const char* uplo,
              const int* n,
              std::complex<double>* a,
              const int* lda,
              std::complex<double>* b,
              const int* ldb,
              const double* vl,
              const double* vu,
              const int* il,
              const int* iu,
              const double* abstol,
              const int* m,
              double* w,
              std::complex<double>* z,
              const int* ldz,
              std::complex<double>* work,
              const int* lwork,
              double* rwork,
              int* iwork,
              int* ifail,
              int* info);
#endif // GATHER_INFO

// Class LapackConnector provide the connector to fortran lapack routine.
// The entire function in this class are static and inline function.
// Usage example:	LapackConnector::functionname(parameter list).
class LapackConnector
{
private:
    // Transpose the std::complex matrix to the fortran-form real-std::complex array.
    static inline
    std::complex<double>* transpose(const ModuleBase::ComplexMatrix& a, const int n, const int lda)
    {
        std::complex<double>* aux = new std::complex<double>[lda*n];
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < lda; ++j)
            {
                aux[i*lda+j] = a(j,i);		// aux[i*lda+j] means aux[i][j] in semantic, not in syntax!
            }
        }
        return aux;
    }

    static inline
    std::complex<float>* transpose(const std::complex<float>* a, const int n, const int lda, const int nbase_x)
    {
        std::complex<float>* aux = new std::complex<float>[lda*n];
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < lda; ++j)
            {
                aux[j * n + i] = a[i * nbase_x + j];
            }
        }
        return aux;
    }

    static inline
    std::complex<double>* transpose(const std::complex<double>* a, const int n, const int lda, const int nbase_x)
    {
        std::complex<double>* aux = new std::complex<double>[lda*n];
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < lda; ++j)
            {
                aux[j * n + i] = a[i * nbase_x + j];
            }
        }
        return aux;
    }

    // Transpose the fortran-form real-std::complex array to the std::complex matrix.
    static inline
    void transpose(const std::complex<double>* aux, ModuleBase::ComplexMatrix& a, const int n, const int lda)
    {
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < lda; ++j)
            {
                a(j, i) = aux[i*lda+j];		// aux[i*lda+j] means aux[i][j] in semantic, not in syntax!
            }
        }
    }

    // Transpose the fortran-form real-std::complex array to the std::complex matrix.
    static inline
    void transpose(const std::complex<float>* aux, std::complex<float>* a, const int n, const int lda, const int nbase_x)
    {
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < lda; ++j)
            {
                a[j * nbase_x + i] = aux[i * lda + j];		// aux[i*lda+j] means aux[i][j] in semantic, not in syntax!
            }
        }
    }

    // Transpose the fortran-form real-std::complex array to the std::complex matrix.
    static inline
    void transpose(const std::complex<double>* aux, std::complex<double>* a, const int n, const int lda, const int nbase_x)
    {
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < lda; ++j)
            {
                a[j * nbase_x + i] = aux[i * lda + j];		// aux[i*lda+j] means aux[i][j] in semantic, not in syntax!
            }
        }
    }

	// Peize Lin add 2015-12-27
	static inline
	char change_uplo(const char &uplo)
	{
		switch(uplo)
		{
			case 'U': return 'L';
			case 'L': return 'U';
			default: throw std::invalid_argument("uplo must be 'U' or 'L'");
		}
	}

	// Peize Lin add 2019-04-14
	static inline
	char change_trans_NC(const char &trans)
	{
		switch(trans)
		{
			case 'N': return 'C';
			case 'C': return 'N';
			default: throw std::invalid_argument("trans must be 'N' or 'C'");
		}
	}

public:
    // wrap function of fortran lapack routine zheev.
    static inline
    void zheev( const char jobz,
                const char uplo,
                const int n,
                ModuleBase::ComplexMatrix& a,
                const int lda,
                double* w,
                std::complex< double >* work,
                const int lwork,
                double* rwork,
                int *info	)
    {	// Transpose the std::complex matrix to the fortran-form real-std::complex array.
        std::complex<double> *aux = LapackConnector::transpose(a, n, lda);
        // call the fortran routine
        zheev_(&jobz, &uplo, &n, aux, &lda, w, work, &lwork, rwork, info);
        // Transpose the fortran-form real-std::complex array to the std::complex matrix.
        LapackConnector::transpose(aux, a, n, lda);
        // free the memory.
        delete[] aux;
    }

    static inline
    void zgetrf(int m, int n, ModuleBase::ComplexMatrix &a, const int lda, int *ipiv, int *info)
    {
        std::complex<double> *aux = LapackConnector::transpose(a, n, lda);
        zgetrf_( &m, &n, aux, &lda, ipiv, info);
        LapackConnector::transpose(aux, a, n, lda);
        delete[] aux;
                return;
    }
    static inline
    void zgetri(int n, ModuleBase::ComplexMatrix &a,  int lda, int *ipiv, std::complex<double> * work, int lwork, int *info)
    {
        std::complex<double> *aux = LapackConnector::transpose(a, n, lda);
        zgetri_( &n, aux, &lda, ipiv, work, &lwork, info);
        LapackConnector::transpose(aux, a, n, lda);
        delete[] aux;
        return;
    }

	// Peize Lin add 2016-07-09
	static inline
	void potrf( const char &uplo, const int &n, float*const A, const int &lda, int &info )
	{
		const char uplo_changed = change_uplo(uplo);
		spotrf_( &uplo_changed, &n, A, &lda, &info );
	}	
	static inline
	void potrf( const char &uplo, const int &n, double*const A, const int &lda, int &info )
	{
		const char uplo_changed = change_uplo(uplo);
		dpotrf_( &uplo_changed, &n, A, &lda, &info );
	}	
	static inline
	void potrf( const char &uplo, const int &n, std::complex<float>*const A, const int &lda, int &info )
	{
		const char uplo_changed = change_uplo(uplo);
		cpotrf_( &uplo_changed, &n, A, &lda, &info );
	}	
	static inline
	void potrf( const char &uplo, const int &n, std::complex<double>*const A, const int &lda, int &info )
	{
		const char uplo_changed = change_uplo(uplo);
		zpotrf_( &uplo_changed, &n, A, &lda, &info );
	}	

	
	// Peize Lin add 2016-07-09
	static inline
	void potri( const char &uplo, const int &n, float*const A, const int &lda, int &info )
	{
		const char uplo_changed = change_uplo(uplo);
		spotri_( &uplo_changed, &n, A, &lda, &info);		
	}	
	static inline
	void potri( const char &uplo, const int &n, double*const A, const int &lda, int &info )
	{
		const char uplo_changed = change_uplo(uplo);
		dpotri_( &uplo_changed, &n, A, &lda, &info);		
	}
	static inline
	void potri( const char &uplo, const int &n, std::complex<float>*const A, const int &lda, int &info )
	{
		const char uplo_changed = change_uplo(uplo);
		cpotri_( &uplo_changed, &n, A, &lda, &info);		
	}
	static inline
	void potri( const char &uplo, const int &n, std::complex<double>*const A, const int &lda, int &info )
	{
		const char uplo_changed = change_uplo(uplo);
		zpotri_( &uplo_changed, &n, A, &lda, &info);		
	}

	// Peize Lin add 2016-07-09
	static inline
	void potrf( const char &uplo, const int &n, ModuleBase::matrix &A, const int &lda, int &info )
	{
		potrf( uplo, n, A.c, lda, info );
	}	
	static inline
	void potrf( const char &uplo, const int &n, ModuleBase::ComplexMatrix &A, const int &lda, int &info )
	{
		potrf( uplo, n, A.c, lda, info );
	}	
	
	// Peize Lin add 2016-07-09
	static inline
	void potri( const char &uplo, const int &n, ModuleBase::matrix &A, const int &lda, int &info )
	{
		potri( uplo, n, A.c, lda, info);		
	}	
	static inline
	void potri( const char &uplo, const int &n, ModuleBase::ComplexMatrix &A, const int &lda, int &info )
	{
		potri( uplo, n, A.c, lda, info);		
	}	
	
	// Peize Lin add 2019-04-14
	// if trans=='N':	C = a * A * A.H + b * C
	// if trans=='C':	C = a * A.H * A + b * C
	static inline
        void herk(const char uplo, const char trans, const int n, const int k,
		const double alpha, const std::complex<double> *A, const int lda,
		const double beta, std::complex<double> *C, const int ldc)
	{
		const char uplo_changed = change_uplo(uplo);
		const char trans_changed = change_trans_NC(trans);
		zherk_(&uplo_changed, &trans_changed, &n, &k, &alpha, A, &lda, &beta, C, &ldc);
	}
    static inline
        void herk(const char uplo, const char trans, const int n, const int k,
            const float alpha, const std::complex<float>* A, const int lda,
            const float beta, std::complex<float>* C, const int ldc)
    {
        const char uplo_changed = change_uplo(uplo);
        const char trans_changed = change_trans_NC(trans);
        cherk_(&uplo_changed, &trans_changed, &n, &k, &alpha, A, &lda, &beta, C, &ldc);
    }
};
#endif  // LAPACKCONNECTOR_HPP
