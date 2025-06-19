#ifndef BASE_THIRD_PARTY_CUSOLVER_H_
#define BASE_THIRD_PARTY_CUSOLVER_H_

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <base/macros/cuda.h>

namespace container {
namespace cuSolverConnector {

template <typename T>
static inline
void trtri (cusolverDnHandle_t& cusolver_handle, const char& uplo, const char& diag, const int& n, T* A, const int& lda)
{
    size_t d_lwork = 0, h_lwork = 0;
    using Type = typename GetTypeThrust<T>::type;
    cusolverErrcheck(cusolverDnXtrtri_bufferSize(cusolver_handle, cublas_fill_mode(uplo), cublas_diag_type(diag), n, GetTypeCuda<T>::cuda_data_type, reinterpret_cast<Type*>(A), lda, &d_lwork, &h_lwork));
    void* d_work = nullptr, *h_work = nullptr;
    cudaErrcheck(cudaMalloc((void**)&d_work, d_lwork));
    if (h_lwork) {
        h_work = malloc(h_lwork);
        if (h_work == nullptr) {
            throw std::bad_alloc();
        }
    }
    int h_info = 0;
    int* d_info = nullptr;
    cudaErrcheck(cudaMalloc((void**)&d_info, sizeof(int)));
    // Perform Cholesky decomposition
    cusolverErrcheck(cusolverDnXtrtri(cusolver_handle, cublas_fill_mode(uplo), cublas_diag_type(diag), n, GetTypeCuda<T>::cuda_data_type, reinterpret_cast<Type*>(A), n, d_work, d_lwork, h_work, h_lwork, d_info));
    cudaErrcheck(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        throw std::runtime_error("trtri: failed to invert matrix");
    }
    free(h_work);
    cudaErrcheck(cudaFree(d_work));
    cudaErrcheck(cudaFree(d_info));
}

static inline
void potri (cusolverDnHandle_t& cusolver_handle, const char& uplo, const char& diag, const int& n, float * A, const int& lda)
{
    int lwork;
    cusolverErrcheck(cusolverDnSpotri_bufferSize(cusolver_handle, cublas_fill_mode(uplo), n, A, n, &lwork));
    float* work;
    cudaErrcheck(cudaMalloc((void**)&work, lwork * sizeof(float)));
    // Perform Cholesky decomposition
    cusolverErrcheck(cusolverDnSpotri(cusolver_handle, cublas_fill_mode(uplo), n, A, n, work, lwork, nullptr));
    cudaErrcheck(cudaFree(work));
}
static inline
void potri (cusolverDnHandle_t& cusolver_handle, const char& uplo, const char& diag, const int& n, double * A, const int& lda)
{
    int lwork;
    cusolverErrcheck(cusolverDnDpotri_bufferSize(cusolver_handle, cublas_fill_mode(uplo), n, A, n, &lwork));
    double* work;
    cudaErrcheck(cudaMalloc((void**)&work, lwork * sizeof(double)));
    // Perform Cholesky decomposition
    cusolverErrcheck(cusolverDnDpotri(cusolver_handle, cublas_fill_mode(uplo), n, A, n, work, lwork, nullptr));
    cudaErrcheck(cudaFree(work));
}
static inline
void potri (cusolverDnHandle_t& cusolver_handle, const char& uplo, const char& diag, const int& n, std::complex<float> * A, const int& lda)
{
    int lwork;
    cusolverErrcheck(cusolverDnCpotri_bufferSize(cusolver_handle, cublas_fill_mode(uplo), n, reinterpret_cast<cuComplex *>(A), n, &lwork));
    cuComplex* work;
    cudaErrcheck(cudaMalloc((void**)&work, lwork * sizeof(cuComplex)));
    // Perform Cholesky decomposition
    cusolverErrcheck(cusolverDnCpotri(cusolver_handle, cublas_fill_mode(uplo), n, reinterpret_cast<cuComplex *>(A), n, work, lwork, nullptr));
    cudaErrcheck(cudaFree(work));
}
static inline
void potri (cusolverDnHandle_t& cusolver_handle, const char& uplo, const char& diag, const int& n, std::complex<double> * A, const int& lda)
{
    int lwork;
    cusolverErrcheck(cusolverDnZpotri_bufferSize(cusolver_handle, cublas_fill_mode(uplo), n, reinterpret_cast<cuDoubleComplex *>(A), n, &lwork));
    cuDoubleComplex* work;
    cudaErrcheck(cudaMalloc((void**)&work, lwork * sizeof(cuDoubleComplex)));
    // Perform Cholesky decomposition
    cusolverErrcheck(cusolverDnZpotri(cusolver_handle, cublas_fill_mode(uplo), n, reinterpret_cast<cuDoubleComplex *>(A), n, work, lwork, nullptr));
    cudaErrcheck(cudaFree(work));
}


static inline
void potrf (cusolverDnHandle_t& cusolver_handle, const char& uplo, const int& n, float * A, const int& lda)
{
    int lwork;
    cusolverErrcheck(cusolverDnSpotrf_bufferSize(cusolver_handle, cublas_fill_mode(uplo), n, A, n, &lwork));
    float* work;
    cudaErrcheck(cudaMalloc((void**)&work, lwork * sizeof(float)));
    // Perform Cholesky decomposition
    cusolverErrcheck(cusolverDnSpotrf(cusolver_handle, cublas_fill_mode(uplo), n, A, n, work, lwork, nullptr));
    cudaErrcheck(cudaFree(work));
}
static inline
void potrf (cusolverDnHandle_t& cusolver_handle, const char& uplo, const int& n, double * A, const int& lda)
{
    int lwork;
    cusolverErrcheck(cusolverDnDpotrf_bufferSize(cusolver_handle, cublas_fill_mode(uplo), n, A, n, &lwork));
    double* work;
    cudaErrcheck(cudaMalloc((void**)&work, lwork * sizeof(double)));
    // Perform Cholesky decomposition
    cusolverErrcheck(cusolverDnDpotrf(cusolver_handle, cublas_fill_mode(uplo), n, A, n, work, lwork, nullptr));
    cudaErrcheck(cudaFree(work));
}
static inline
void potrf (cusolverDnHandle_t& cusolver_handle, const char& uplo, const int& n, std::complex<float> * A, const int& lda)
{
    int lwork;
    cusolverErrcheck(cusolverDnCpotrf_bufferSize(cusolver_handle, cublas_fill_mode(uplo), n, reinterpret_cast<cuComplex*>(A), n, &lwork));
    cuComplex* work;
    cudaErrcheck(cudaMalloc((void**)&work, lwork * sizeof(cuComplex)));
    // Perform Cholesky decomposition
    cusolverErrcheck(cusolverDnCpotrf(cusolver_handle, cublas_fill_mode(uplo), n, reinterpret_cast<cuComplex*>(A), n, work, lwork, nullptr));
    cudaErrcheck(cudaFree(work));
}
static inline
void potrf (cusolverDnHandle_t& cusolver_handle, const char& uplo, const int& n, std::complex<double> * A, const int& lda)
{
    int lwork;
    cusolverErrcheck(cusolverDnZpotrf_bufferSize(cusolver_handle, cublas_fill_mode(uplo), n, reinterpret_cast<cuDoubleComplex*>(A), n, &lwork));
    cuDoubleComplex* work;
    cudaErrcheck(cudaMalloc((void**)&work, lwork * sizeof(cuDoubleComplex)));
    // Perform Cholesky decomposition
    cusolverErrcheck(cusolverDnZpotrf(cusolver_handle, cublas_fill_mode(uplo), n, reinterpret_cast<cuDoubleComplex*>(A), n, work, lwork, nullptr));
    cudaErrcheck(cudaFree(work));
}


static inline
void dnevd (cusolverDnHandle_t& cusolver_handle, const char& jobz, const char& uplo, const int& n, float* A, const int& lda, float * W)
{
    // prepare some values for cusolverDnSsyevd_bufferSize
    int lwork  = 0; 
    int h_info = 0; 
    int*   d_info = nullptr;
    float* d_work = nullptr;
    cudaErrcheck(cudaMalloc((void**)&d_info, sizeof(int)));

    // calculate the sizes needed for pre-allocated buffer.
    cusolverErrcheck(cusolverDnSsyevd_bufferSize(cusolver_handle, cublas_eig_mode(jobz), cublas_fill_mode(uplo), 
                                n, A, lda, W, &lwork));
    // allocate memory
    cudaErrcheck(cudaMalloc((void**)&d_work, sizeof(float) * lwork));
    // compute eigenvalues and eigenvectors.
    cusolverErrcheck(cusolverDnSsyevd(cusolver_handle, cublas_eig_mode(jobz), cublas_fill_mode(uplo),
                                n, A, lda, W, d_work, lwork, d_info));

    cudaErrcheck(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        throw std::runtime_error("dnevd: failed to invert matrix");
    }
    cudaErrcheck(cudaFree(d_info));
    cudaErrcheck(cudaFree(d_work));
}
static inline
void dnevd (cusolverDnHandle_t& cusolver_handle, const char& jobz, const char& uplo, const int& n, double* A, const int& lda, double * W)
{
    // prepare some values for cusolverDnDsyevd_bufferSize
    int lwork  = 0; 
    int h_info = 0; 
    int*    d_info = nullptr;
    double* d_work = nullptr;
    cudaErrcheck(cudaMalloc((void**)&d_info, sizeof(int)));

    // calculate the sizes needed for pre-allocated buffer.
    cusolverErrcheck(cusolverDnDsyevd_bufferSize(cusolver_handle, cublas_eig_mode(jobz), cublas_fill_mode(uplo), 
                                n, A, lda, W, &lwork));
    // allocate memory
    cudaErrcheck(cudaMalloc((void**)&d_work, sizeof(double) * lwork));
    // compute eigenvalues and eigenvectors.
    cusolverErrcheck(cusolverDnDsyevd(cusolver_handle, cublas_eig_mode(jobz), cublas_fill_mode(uplo),
                                n, A, lda, W, d_work, lwork, d_info));

    cudaErrcheck(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        throw std::runtime_error("dnevd: failed to invert matrix");
    }
    cudaErrcheck(cudaFree(d_info));
    cudaErrcheck(cudaFree(d_work));
}
static inline
void dnevd (cusolverDnHandle_t& cusolver_handle, const char& jobz, const char& uplo, const int& n, std::complex<float>* A, const int& lda, float * W)
{
    // prepare some values for cusolverDnCheevd_bufferSize
    int lwork  = 0; 
    int h_info = 0; 
    int*    d_info = nullptr;
    cuComplex* d_work = nullptr;
    cudaErrcheck(cudaMalloc((void**)&d_info, sizeof(int)));

    // calculate the sizes needed for pre-allocated buffer.
    cusolverErrcheck(cusolverDnCheevd_bufferSize(cusolver_handle, cublas_eig_mode(jobz), cublas_fill_mode(uplo), 
                                n, reinterpret_cast<cuComplex*>(A), lda, W, &lwork));
    // allocate memory
    cudaErrcheck(cudaMalloc((void**)&d_work, sizeof(cuComplex) * lwork));
    // compute eigenvalues and eigenvectors.
    cusolverErrcheck(cusolverDnCheevd(cusolver_handle, cublas_eig_mode(jobz), cublas_fill_mode(uplo),
                                n, reinterpret_cast<cuComplex*>(A), lda, W, d_work, lwork, d_info));

    cudaErrcheck(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        throw std::runtime_error("dnevd: failed to invert matrix");
    }
    cudaErrcheck(cudaFree(d_info));
    cudaErrcheck(cudaFree(d_work));
}
static inline
void dnevd (cusolverDnHandle_t& cusolver_handle, const char& jobz, const char& uplo, const int& n, std::complex<double>* A, const int& lda, double* W)
{
    // prepare some values for cusolverDnZheevd_bufferSize
    int lwork  = 0; 
    int h_info = 0; 
    int*    d_info = nullptr;
    cuDoubleComplex* d_work = nullptr;
    cudaErrcheck(cudaMalloc((void**)&d_info, sizeof(int)));

    // calculate the sizes needed for pre-allocated buffer.
    cusolverErrcheck(cusolverDnZheevd_bufferSize(cusolver_handle, cublas_eig_mode(jobz), cublas_fill_mode(uplo), 
                                n, reinterpret_cast<cuDoubleComplex*>(A), lda, W, &lwork));
    // allocate memory
    cudaErrcheck(cudaMalloc((void**)&d_work, sizeof(cuDoubleComplex) * lwork));
    // compute eigenvalues and eigenvectors.
    cusolverErrcheck(cusolverDnZheevd(cusolver_handle, cublas_eig_mode(jobz), cublas_fill_mode(uplo),
                                n, reinterpret_cast<cuDoubleComplex*>(A), lda, W, d_work, lwork, d_info));

    cudaErrcheck(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        throw std::runtime_error("dnevd: failed to invert matrix");
    }
    cudaErrcheck(cudaFree(d_info));
    cudaErrcheck(cudaFree(d_work));
}

static inline
void dngvd (cusolverDnHandle_t& cusolver_handle, const int& itype, const char& jobz, const char& uplo, const int& n, float* A, const int& lda, float* B, const int& ldb, float * W)
{
    // prepare some values for cusolverDnSsygvd_bufferSize
    int lwork  = 0; 
    int h_info = 0; 
    int*   d_info = nullptr;
    float* d_work = nullptr;
    cudaErrcheck(cudaMalloc((void**)&d_info, sizeof(int)));

    // calculate the sizes needed for pre-allocated buffer.
    cusolverErrcheck(cusolverDnSsygvd_bufferSize(cusolver_handle, cublas_eig_type(itype), cublas_eig_mode(jobz), cublas_fill_mode(uplo), 
                                n, A, lda, B, ldb, W, &lwork));
    // allocate memory
    cudaErrcheck(cudaMalloc((void**)&d_work, sizeof(float) * lwork));
    // compute eigenvalues and eigenvectors.
    cusolverErrcheck(cusolverDnSsygvd(cusolver_handle, cublas_eig_type(itype), cublas_eig_mode(jobz), cublas_fill_mode(uplo), 
                                n, A, lda, B, ldb, W, d_work, lwork, d_info));

    cudaErrcheck(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        throw std::runtime_error("dnevd: failed to invert matrix");
    }
    cudaErrcheck(cudaFree(d_info));
    cudaErrcheck(cudaFree(d_work));
}
static inline
void dngvd (cusolverDnHandle_t& cusolver_handle, const int& itype, const char& jobz, const char& uplo, const int& n, double* A, const int& lda, double* B, const int& ldb, double * W)
{
    // prepare some values for cusolverDnDsygvd_bufferSize
    int lwork  = 0; 
    int h_info = 0; 
    int*   d_info = nullptr;
    double* d_work = nullptr;
    cudaErrcheck(cudaMalloc((void**)&d_info, sizeof(int)));

    // calculate the sizes needed for pre-allocated buffer.
    cusolverErrcheck(cusolverDnDsygvd_bufferSize(cusolver_handle, cublas_eig_type(itype), cublas_eig_mode(jobz), cublas_fill_mode(uplo), 
                                n, A, lda, B, ldb, W, &lwork));
    // allocate memory
    cudaErrcheck(cudaMalloc((void**)&d_work, sizeof(double) * lwork));
    // compute eigenvalues and eigenvectors.
    cusolverErrcheck(cusolverDnDsygvd(cusolver_handle, cublas_eig_type(itype), cublas_eig_mode(jobz), cublas_fill_mode(uplo), 
                                n, A, lda, B, ldb, W, d_work, lwork, d_info));

    cudaErrcheck(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        throw std::runtime_error("dnevd: failed to invert matrix");
    }
    cudaErrcheck(cudaFree(d_info));
    cudaErrcheck(cudaFree(d_work));
}
static inline
void dngvd (cusolverDnHandle_t& cusolver_handle, const int& itype, const char& jobz, const char& uplo, const int& n, std::complex<float>* A, const int& lda, std::complex<float>* B, const int& ldb, float* W)
{
    // prepare some values for cusolverDnChegvd_bufferSize
    int lwork  = 0; 
    int h_info = 0; 
    int*   d_info = nullptr;
    cuComplex* d_work = nullptr;
    cudaErrcheck(cudaMalloc((void**)&d_info, sizeof(int)));

    // calculate the sizes needed for pre-allocated buffer.
    cusolverErrcheck(cusolverDnChegvd_bufferSize(cusolver_handle, cublas_eig_type(itype), cublas_eig_mode(jobz), cublas_fill_mode(uplo), 
                                n, reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb, W, &lwork));
    // allocate memory
    cudaErrcheck(cudaMalloc((void**)&d_work, sizeof(cuComplex) * lwork));
    // compute eigenvalues and eigenvectors.
    cusolverErrcheck(cusolverDnChegvd(cusolver_handle, cublas_eig_type(itype), cublas_eig_mode(jobz), cublas_fill_mode(uplo), 
                                n, reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb, W, d_work, lwork, d_info));

    cudaErrcheck(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        throw std::runtime_error("dnevd: failed to invert matrix");
    }
    cudaErrcheck(cudaFree(d_info));
    cudaErrcheck(cudaFree(d_work));
}
static inline
void dngvd (cusolverDnHandle_t& cusolver_handle, const int& itype, const char& jobz, const char& uplo, const int& n, std::complex<double>* A, const int& lda, std::complex<double>* B, const int& ldb, double* W)
{
    // prepare some values for cusolverDnZhegvd_bufferSize
    int lwork  = 0; 
    int h_info = 0; 
    int*   d_info = nullptr;
    cuDoubleComplex* d_work = nullptr;
    cudaErrcheck(cudaMalloc((void**)&d_info, sizeof(int)));

    // calculate the sizes needed for pre-allocated buffer.
    cusolverErrcheck(cusolverDnZhegvd_bufferSize(cusolver_handle, cublas_eig_type(itype), cublas_eig_mode(jobz), cublas_fill_mode(uplo), 
                                n, reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb, W, &lwork));
    // allocate memory
    cudaErrcheck(cudaMalloc((void**)&d_work, sizeof(cuDoubleComplex) * lwork));
    // compute eigenvalues and eigenvectors.
    cusolverErrcheck(cusolverDnZhegvd(cusolver_handle, cublas_eig_type(itype), cublas_eig_mode(jobz), cublas_fill_mode(uplo), 
                                n, reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb, W, d_work, lwork, d_info));

    cudaErrcheck(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        throw std::runtime_error("dnevd: failed to invert matrix");
    }
    cudaErrcheck(cudaFree(d_info));
    cudaErrcheck(cudaFree(d_work));
}

static inline
void getrf(cusolverDnHandle_t& cusolver_handle, const int& m, const int& n, float* A, const int& lda, int* ipiv)
{
    // prepare some values for cusolverDnSgetrf_bufferSize
    int lwork = 0;
    int h_info = 0;
    int* d_info = nullptr;
    float* d_work = nullptr;
    cudaErrcheck(cudaMalloc((void**)&d_info, sizeof(int)));

    // calculate the sizes needed for pre-allocated buffer.
    cusolverErrcheck(cusolverDnSgetrf_bufferSize(cusolver_handle, m, n, A, lda, &lwork));

    // allocate memory
    cudaErrcheck(cudaMalloc((void**)&d_work, sizeof(float) * lwork));

    // Perform LU decomposition
    cusolverErrcheck(cusolverDnSgetrf(cusolver_handle, m, n, A, lda, d_work, ipiv, d_info));

    cudaErrcheck(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        throw std::runtime_error("getrf: failed to compute LU factorization");
    }

    cudaErrcheck(cudaFree(d_work));
    cudaErrcheck(cudaFree(d_info));
}
static inline
void getrf(cusolverDnHandle_t& cusolver_handle, const int& m, const int& n, double* A, const int& lda, int* ipiv)
{
    // prepare some values for cusolverDnDgetrf_bufferSize
    int lwork = 0;
    int h_info = 0;
    int* d_info = nullptr;
    double* d_work = nullptr;
    cudaErrcheck(cudaMalloc((void**)&d_info, sizeof(int)));

    // calculate the sizes needed for pre-allocated buffer.
    cusolverErrcheck(cusolverDnDgetrf_bufferSize(cusolver_handle, m, n, A, lda, &lwork));

    // allocate memory
    cudaErrcheck(cudaMalloc((void**)&d_work, sizeof(double) * lwork));

    // Perform LU decomposition
    cusolverErrcheck(cusolverDnDgetrf(cusolver_handle, m, n, A, lda, d_work, ipiv, d_info));

    cudaErrcheck(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        throw std::runtime_error("getrf: failed to compute LU factorization");
    }

    cudaErrcheck(cudaFree(d_work));
    cudaErrcheck(cudaFree(d_info));
}
static inline
void getrf(cusolverDnHandle_t& cusolver_handle, const int& m, const int& n, std::complex<float>* A, const int& lda, int* ipiv)
{
    // prepare some values for cusolverDnCgetrf_bufferSize
    int lwork = 0;
    int h_info = 0;
    int* d_info = nullptr;
    cuComplex* d_work = nullptr;
    cudaErrcheck(cudaMalloc((void**)&d_info, sizeof(int)));

    // calculate the sizes needed for pre-allocated buffer.
    cusolverErrcheck(cusolverDnCgetrf_bufferSize(cusolver_handle, m, n, reinterpret_cast<cuComplex*>(A), lda, &lwork));

    // allocate memory
    cudaErrcheck(cudaMalloc((void**)&d_work, sizeof(cuComplex) * lwork));

    // Perform LU decomposition
    cusolverErrcheck(cusolverDnCgetrf(cusolver_handle, m, n, reinterpret_cast<cuComplex*>(A), lda, d_work, ipiv, d_info));

    cudaErrcheck(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        throw std::runtime_error("getrf: failed to compute LU factorization");
    }

    cudaErrcheck(cudaFree(d_work));
    cudaErrcheck(cudaFree(d_info));
}
static inline
void getrf(cusolverDnHandle_t& cusolver_handle, const int& m, const int& n, std::complex<double>* A, const int& lda, int* ipiv)
{
    // prepare some values for cusolverDnZgetrf_bufferSize
    int lwork = 0;
    int h_info = 0;
    int* d_info = nullptr;
    cuDoubleComplex* d_work = nullptr;
    cudaErrcheck(cudaMalloc((void**)&d_info, sizeof(int)));

    // calculate the sizes needed for pre-allocated buffer.
    cusolverErrcheck(cusolverDnZgetrf_bufferSize(cusolver_handle, m, n, reinterpret_cast<cuDoubleComplex*>(A), lda, &lwork));

    // allocate memory
    cudaErrcheck(cudaMalloc((void**)&d_work, sizeof(cuDoubleComplex) * lwork));

    // Perform LU decomposition
    cusolverErrcheck(cusolverDnZgetrf(cusolver_handle, m, n, reinterpret_cast<cuDoubleComplex*>(A), lda, d_work, ipiv, d_info));

    cudaErrcheck(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        throw std::runtime_error("getrf: failed to compute LU factorization");
    }

    cudaErrcheck(cudaFree(d_work));
    cudaErrcheck(cudaFree(d_info));
}

static inline
void getrs(cusolverDnHandle_t& cusolver_handle, const char& trans, const int& n, const int& nrhs, float* A, const int& lda, const int* ipiv, float* B, const int& ldb)
{
    int h_info = 0;
    int* d_info = nullptr;
    cudaErrcheck(cudaMalloc((void**)&d_info, sizeof(int)));

    cusolverErrcheck(cusolverDnSgetrs(cusolver_handle, GetCublasOperation(trans), n, nrhs, A, lda, ipiv, B, ldb, d_info));

    cudaErrcheck(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        throw std::runtime_error("getrs: failed to solve the linear system");
    }

    cudaErrcheck(cudaFree(d_info));
}
static inline
void getrs(cusolverDnHandle_t& cusolver_handle, const char& trans, const int& n, const int& nrhs, double* A, const int& lda, const int* ipiv, double* B, const int& ldb)
{
    int h_info = 0;
    int* d_info = nullptr;
    cudaErrcheck(cudaMalloc((void**)&d_info, sizeof(int)));

    cusolverErrcheck(cusolverDnDgetrs(cusolver_handle, GetCublasOperation(trans), n, nrhs, A, lda, ipiv, B, ldb, d_info));

    cudaErrcheck(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        throw std::runtime_error("getrs: failed to solve the linear system");
    }

    cudaErrcheck(cudaFree(d_info));
}
static inline
void getrs(cusolverDnHandle_t& cusolver_handle, const char& trans, const int& n, const int& nrhs, std::complex<float>* A, const int& lda, const int* ipiv, std::complex<float>* B, const int& ldb)
{
    int h_info = 0;
    int* d_info = nullptr;
    cudaErrcheck(cudaMalloc((void**)&d_info, sizeof(int)));

    cusolverErrcheck(cusolverDnCgetrs(cusolver_handle, GetCublasOperation(trans), n, nrhs, reinterpret_cast<cuComplex*>(A), lda, ipiv, reinterpret_cast<cuComplex*>(B), ldb, d_info));

    cudaErrcheck(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        throw std::runtime_error("getrs: failed to solve the linear system");
    }

    cudaErrcheck(cudaFree(d_info));
}
static inline
void getrs(cusolverDnHandle_t& cusolver_handle, const char& trans, const int& n, const int& nrhs, std::complex<double>* A, const int& lda, const int* ipiv, std::complex<double>* B, const int& ldb)
{
    int h_info = 0;
    int* d_info = nullptr;
    cudaErrcheck(cudaMalloc((void**)&d_info, sizeof(int)));

    cusolverErrcheck(cusolverDnZgetrs(cusolver_handle, GetCublasOperation(trans), n, nrhs, reinterpret_cast<cuDoubleComplex*>(A), lda, ipiv, reinterpret_cast<cuDoubleComplex*>(B), ldb, d_info));

    cudaErrcheck(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        throw std::runtime_error("getrs: failed to solve the linear system");
    }

    cudaErrcheck(cudaFree(d_info));
}

} // namespace cuSolverConnector
} // namespace container

#endif // BASE_THIRD_PARTY_CUSOLVER_H_