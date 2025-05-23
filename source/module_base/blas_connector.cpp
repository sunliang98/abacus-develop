#include "blas_connector.h"
#include "macros.h"

#ifdef __DSP
#include "module_base/kernels/dsp/dsp_connector.h"
#include "module_base/global_variable.h"
#endif

#ifdef __CUDA
#include <base/macros/macros.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "module_base/kernels/math_kernel_op.h"
#include "module_base/module_device/memory_op.h"


namespace BlasUtils{

	static cublasHandle_t cublas_handle = nullptr;

	void createGpuBlasHandle(){
		if (cublas_handle == nullptr) {
			cublasErrcheck(cublasCreate(&cublas_handle));
		}
	}

	void destoryBLAShandle(){
		if (cublas_handle != nullptr) {
			cublasErrcheck(cublasDestroy(cublas_handle));
			cublas_handle = nullptr;
		}
	}


	cublasOperation_t judge_trans(bool is_complex, const char& trans, const char* name)
	{
		if (trans == 'N')
		{
			return CUBLAS_OP_N;
		}
		else if(trans == 'T')
		{
			return CUBLAS_OP_T;
		}
		else if(is_complex && trans == 'C')
		{
			return CUBLAS_OP_C;
		}
		return CUBLAS_OP_N;
	}

	cublasSideMode_t judge_side(const char& trans)
	{
		if (trans == 'L')
		{
			return CUBLAS_SIDE_LEFT;
		}
		else if (trans == 'R')
		{
			return CUBLAS_SIDE_RIGHT;
		}
		return CUBLAS_SIDE_LEFT;
	}

	cublasFillMode_t judge_fill(const char& trans)
	{
		if (trans == 'F')
		{
			return CUBLAS_FILL_MODE_FULL;
		}
		else if (trans == 'U')
		{
			return CUBLAS_FILL_MODE_UPPER;
		}
		else if (trans == 'D')
		{
			return CUBLAS_FILL_MODE_LOWER;
		}
		return CUBLAS_FILL_MODE_FULL;
	}

} // namespace BlasUtils

#endif

void BlasConnector::axpy( const int n, const float alpha, const float *X, const int incX, float *Y, const int incY, base_device::AbacusDevice_t device_type)
{
	if (device_type == base_device::AbacusDevice_t::CpuDevice) {
		saxpy_(&n, &alpha, X, &incX, Y, &incY);
	}
	else if (device_type == base_device::AbacusDevice_t::GpuDevice){
#ifdef __CUDA
		cublasErrcheck(cublasSaxpy(BlasUtils::cublas_handle, n, &alpha, X, incX, Y, incY));
#endif
	}
}

void BlasConnector::axpy( const int n, const double alpha, const double *X, const int incX, double *Y, const int incY, base_device::AbacusDevice_t device_type)
{
	if (device_type == base_device::AbacusDevice_t::CpuDevice) {
		daxpy_(&n, &alpha, X, &incX, Y, &incY);
	}
	else if (device_type == base_device::AbacusDevice_t::GpuDevice){
#ifdef __CUDA
		cublasErrcheck(cublasDaxpy(BlasUtils::cublas_handle, n, &alpha, X, incX, Y, incY));
#endif
	}
}

void BlasConnector::axpy( const int n, const std::complex<float> alpha, const std::complex<float> *X, const int incX, std::complex<float> *Y, const int incY, base_device::AbacusDevice_t device_type)
{
	if (device_type == base_device::AbacusDevice_t::CpuDevice) {
		caxpy_(&n, &alpha, X, &incX, Y, &incY);
	}
	else if (device_type == base_device::AbacusDevice_t::GpuDevice){
#ifdef __CUDA
		cublasErrcheck(cublasCaxpy(BlasUtils::cublas_handle, n, (float2*)&alpha, (float2*)X, incX, (float2*)Y, incY));
#endif
	}
}

void BlasConnector::axpy( const int n, const std::complex<double> alpha, const std::complex<double> *X, const int incX, std::complex<double> *Y, const int incY, base_device::AbacusDevice_t device_type)
{
	if (device_type == base_device::AbacusDevice_t::CpuDevice) {
		zaxpy_(&n, &alpha, X, &incX, Y, &incY);
	}
	else if (device_type == base_device::AbacusDevice_t::GpuDevice){
#ifdef __CUDA
		cublasErrcheck(cublasZaxpy(BlasUtils::cublas_handle, n, (double2*)&alpha, (double2*)X, incX, (double2*)Y, incY));
#endif
	}
}


// x=a*x
void BlasConnector::scal( const int n,  const float alpha, float *X, const int incX, base_device::AbacusDevice_t device_type)
{
	if (device_type == base_device::AbacusDevice_t::CpuDevice) {
		sscal_(&n, &alpha, X, &incX);
	}
	else if (device_type == base_device::AbacusDevice_t::GpuDevice) {
#ifdef __CUDA
		cublasErrcheck(cublasSscal(BlasUtils::cublas_handle, n, &alpha, X, incX));
#endif
	}
}

void BlasConnector::scal( const int n, const double alpha, double *X, const int incX, base_device::AbacusDevice_t device_type)
{
	if (device_type == base_device::AbacusDevice_t::CpuDevice) {
		dscal_(&n, &alpha, X, &incX);
	}
	else if (device_type == base_device::AbacusDevice_t::GpuDevice) {
#ifdef __CUDA
		cublasErrcheck(cublasDscal(BlasUtils::cublas_handle, n, &alpha, X, incX));
#endif
	}
}

void BlasConnector::scal( const int n, const std::complex<float> alpha, std::complex<float> *X, const int incX, base_device::AbacusDevice_t device_type)
{
	if (device_type == base_device::AbacusDevice_t::CpuDevice) {
		cscal_(&n, &alpha, X, &incX);
	}
	else if (device_type == base_device::AbacusDevice_t::GpuDevice) {
#ifdef __CUDA
		cublasErrcheck(cublasCscal(BlasUtils::cublas_handle, n, (float2*)&alpha, (float2*)X, incX));
#endif
	}
}

void BlasConnector::scal( const int n, const std::complex<double> alpha, std::complex<double> *X, const int incX, base_device::AbacusDevice_t device_type)
{
	if (device_type == base_device::AbacusDevice_t::CpuDevice) {
		zscal_(&n, &alpha, X, &incX);
	}
	else if (device_type == base_device::AbacusDevice_t::GpuDevice) {
#ifdef __CUDA
		cublasErrcheck(cublasZscal(BlasUtils::cublas_handle, n, (double2*)&alpha, (double2*)X, incX));
#endif
	}
}


// d=x*y
float BlasConnector::dot( const int n, const float *X, const int incX, const float *Y, const int incY, base_device::AbacusDevice_t device_type)
{
	if (device_type == base_device::AbacusDevice_t::CpuDevice) {
		return sdot_(&n, X, &incX, Y, &incY);
	}
	else if (device_type == base_device::AbacusDevice_t::GpuDevice){
#ifdef __CUDA
		float result = 0.0;
		cublasErrcheck(cublasSdot(BlasUtils::cublas_handle, n, X, incX, Y, incY, &result));
		return result;
#endif
	}
	return sdot_(&n, X, &incX, Y, &incY);
}

double BlasConnector::dot( const int n, const double *X, const int incX, const double *Y, const int incY, base_device::AbacusDevice_t device_type)
{
	if (device_type == base_device::AbacusDevice_t::CpuDevice) {
		return ddot_(&n, X, &incX, Y, &incY);
	}
	else if (device_type == base_device::AbacusDevice_t::GpuDevice){
#ifdef __CUDA
		double result = 0.0;
		cublasErrcheck(cublasDdot(BlasUtils::cublas_handle, n, X, incX, Y, incY, &result));
		return result;
#endif
	}
	return ddot_(&n, X, &incX, Y, &incY);
}

// C = a * A.? * B.? + b * C
// Row-Major part
void BlasConnector::gemm(const char transa, const char transb, const int m, const int n, const int k,
	const float alpha, const float *a, const int lda, const float *b, const int ldb,
	const float beta, float *c, const int ldc, base_device::AbacusDevice_t device_type)
{
	if (device_type == base_device::AbacusDevice_t::CpuDevice) {
		sgemm_(&transb, &transa, &n, &m, &k,
		&alpha, b, &ldb, a, &lda,
		&beta, c, &ldc);
	}
#ifdef __DSP
	else if (device_type == base_device::AbacusDevice_t::DspDevice){
		mtfunc::sgemm_mth_(&transb, &transa, &n, &m, &k,
		&alpha, b, &ldb, a, &lda,
		&beta, c, &ldc, GlobalV::MY_RANK);
	}
#endif
	else if (device_type == base_device::AbacusDevice_t::GpuDevice){
#ifdef __CUDA
		cublasOperation_t cutransA = BlasUtils::judge_trans(false, transa, "gemm_op");
		cublasOperation_t cutransB = BlasUtils::judge_trans(false, transb, "gemm_op");
		cublasErrcheck(cublasSgemm(BlasUtils::cublas_handle, cutransA, cutransB, n, m, k, &alpha, b, ldb, a, lda, &beta, c, ldc));
#endif
	}
}

void BlasConnector::gemm(const char transa,
                         const char transb,
                         const int m,
                         const int n,
                         const int k,
                         const double alpha,
                         const double* a,
                         const int lda,
                         const double* b,
                         const int ldb,
                         const double beta,
                         double* c,
                         const int ldc,
                         base_device::AbacusDevice_t device_type)
{
    if (device_type == base_device::AbacusDevice_t::CpuDevice)
    {
        dgemm_(&transb, &transa, &n, &m, &k, &alpha, b, &ldb, a, &lda, &beta, c, &ldc);
    }
#ifdef __DSP
    else if (device_type == base_device::AbacusDevice_t::DspDevice)
    {
        mtfunc::dgemm_mth_(&transb, &transa, &n, &m, &k, &alpha, b, &ldb, a, &lda, &beta, c, &ldc, GlobalV::MY_RANK);
    }
#endif
    else if (device_type == base_device::AbacusDevice_t::GpuDevice)
    {
#ifdef __CUDA
        cublasOperation_t cutransA = BlasUtils::judge_trans(false, transa, "gemm_op");
        cublasOperation_t cutransB = BlasUtils::judge_trans(false, transb, "gemm_op");
        cublasErrcheck(
            cublasDgemm(BlasUtils::cublas_handle, cutransA, cutransB, n, m, k, &alpha, b, ldb, a, lda, &beta, c, ldc));
#endif
    }
}

void BlasConnector::gemm(const char transa,
                         const char transb,
                         const int m,
                         const int n,
                         const int k,
                         const std::complex<float> alpha,
                         const std::complex<float>* a,
                         const int lda,
                         const std::complex<float>* b,
                         const int ldb,
                         const std::complex<float> beta,
                         std::complex<float>* c,
                         const int ldc,
                         base_device::AbacusDevice_t device_type)
{
    if (device_type == base_device::AbacusDevice_t::CpuDevice)
    {
        cgemm_(&transb, &transa, &n, &m, &k, &alpha, b, &ldb, a, &lda, &beta, c, &ldc);
    }
#ifdef __DSP
    else if (device_type == base_device::AbacusDevice_t::DspDevice)
    {
        mtfunc::cgemm_mth_(&transb, &transa, &n, &m, &k, &alpha, b, &ldb, a, &lda, &beta, c, &ldc, GlobalV::MY_RANK);
    }
#endif
    else if (device_type == base_device::AbacusDevice_t::GpuDevice)
    {
#ifdef __CUDA
        cublasOperation_t cutransA = BlasUtils::judge_trans(false, transa, "gemm_op");
        cublasOperation_t cutransB = BlasUtils::judge_trans(false, transb, "gemm_op");
        cublasErrcheck(cublasCgemm(BlasUtils::cublas_handle,
                                   cutransA,
                                   cutransB,
                                   n,
                                   m,
                                   k,
                                   (float2*)&alpha,
                                   (float2*)b,
                                   ldb,
                                   (float2*)a,
                                   lda,
                                   (float2*)&beta,
                                   (float2*)c,
                                   ldc));
#endif
    }
}

void BlasConnector::gemm(const char transa,
                         const char transb,
                         const int m,
                         const int n,
                         const int k,
                         const std::complex<double> alpha,
                         const std::complex<double>* a,
                         const int lda,
                         const std::complex<double>* b,
                         const int ldb,
                         const std::complex<double> beta,
                         std::complex<double>* c,
                         const int ldc,
                         base_device::AbacusDevice_t device_type)
{
    if (device_type == base_device::AbacusDevice_t::CpuDevice)
    {
        zgemm_(&transb, &transa, &n, &m, &k, &alpha, b, &ldb, a, &lda, &beta, c, &ldc);
    }
#ifdef __DSP
    else if (device_type == base_device::AbacusDevice_t::DspDevice)
    {
        mtfunc::zgemm_mth_(&transb, &transa, &n, &m, &k, &alpha, b, &ldb, a, &lda, &beta, c, &ldc, GlobalV::MY_RANK);
    }
#endif
    else if (device_type == base_device::AbacusDevice_t::GpuDevice)
    {
#ifdef __CUDA
        cublasOperation_t cutransA = BlasUtils::judge_trans(false, transa, "gemm_op");
        cublasOperation_t cutransB = BlasUtils::judge_trans(false, transb, "gemm_op");
        cublasErrcheck(cublasZgemm(BlasUtils::cublas_handle,
                                   cutransA,
                                   cutransB,
                                   n,
                                   m,
                                   k,
                                   (double2*)&alpha,
                                   (double2*)b,
                                   ldb,
                                   (double2*)a,
                                   lda,
                                   (double2*)&beta,
                                   (double2*)c,
                                   ldc));
#endif
    }
}

// Col-Major part
void BlasConnector::gemm_cm(const char transa, const char transb, const int m, const int n, const int k,
	const float alpha, const float *a, const int lda, const float *b, const int ldb,
	const float beta, float *c, const int ldc, base_device::AbacusDevice_t device_type)
{
	if (device_type == base_device::AbacusDevice_t::CpuDevice) {
		sgemm_(&transa, &transb, &m, &n, &k,
		&alpha, a, &lda, b, &ldb,
		&beta, c, &ldc);
	}
#ifdef __DSP
	else if (device_type == base_device::AbacusDevice_t::DspDevice){
		mtfunc::sgemm_mth_(&transb, &transa, &m, &n, &k,
		&alpha, a, &lda, b, &ldb,
		&beta, c, &ldc, GlobalV::MY_RANK);
	}
#endif
	else if (device_type == base_device::AbacusDevice_t::GpuDevice){
#ifdef __CUDA
		cublasOperation_t cutransA = BlasUtils::judge_trans(false, transa, "gemm_op");
		cublasOperation_t cutransB = BlasUtils::judge_trans(false, transb, "gemm_op");
		cublasErrcheck(cublasSgemm(BlasUtils::cublas_handle, cutransA, cutransB, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc));
#endif
	}
}

void BlasConnector::gemm_cm(const char transa,
                            const char transb,
                            const int m,
                            const int n,
                            const int k,
                            const double alpha,
                            const double* a,
                            const int lda,
                            const double* b,
                            const int ldb,
                            const double beta,
                            double* c,
                            const int ldc,
                            base_device::AbacusDevice_t device_type)
{
    if (device_type == base_device::AbacusDevice_t::CpuDevice)
    {
        dgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    }
#ifdef __DSP
    else if (device_type == base_device::AbacusDevice_t::DspDevice)
    {
        mtfunc::dgemm_mth_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc, GlobalV::MY_RANK);
    }
#endif
    else if (device_type == base_device::AbacusDevice_t::GpuDevice)
    {
#ifdef __CUDA
        cublasOperation_t cutransA = BlasUtils::judge_trans(false, transa, "gemm_op");
        cublasOperation_t cutransB = BlasUtils::judge_trans(false, transb, "gemm_op");
        cublasErrcheck(
            cublasDgemm(BlasUtils::cublas_handle, cutransA, cutransB, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc));
#endif
    }
}

void BlasConnector::gemm_cm(const char transa,
                            const char transb,
                            const int m,
                            const int n,
                            const int k,
                            const std::complex<float> alpha,
                            const std::complex<float>* a,
                            const int lda,
                            const std::complex<float>* b,
                            const int ldb,
                            const std::complex<float> beta,
                            std::complex<float>* c,
                            const int ldc,
                            base_device::AbacusDevice_t device_type)
{
    if (device_type == base_device::AbacusDevice_t::CpuDevice)
    {
        cgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    }
#ifdef __DSP
    else if (device_type == base_device::AbacusDevice_t::DspDevice)
    {
        mtfunc::cgemm_mth_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc, GlobalV::MY_RANK);
    }
#endif
    else if (device_type == base_device::AbacusDevice_t::GpuDevice)
    {
#ifdef __CUDA
        cublasOperation_t cutransA = BlasUtils::judge_trans(false, transa, "gemm_op");
        cublasOperation_t cutransB = BlasUtils::judge_trans(false, transb, "gemm_op");
        cublasErrcheck(cublasCgemm(BlasUtils::cublas_handle,
                                   cutransA,
                                   cutransB,
                                   m,
                                   n,
                                   k,
                                   (float2*)&alpha,
                                   (float2*)a,
                                   lda,
                                   (float2*)b,
                                   ldb,
                                   (float2*)&beta,
                                   (float2*)c,
                                   ldc));
#endif
    }
}

void BlasConnector::gemm_cm(const char transa,
                            const char transb,
                            const int m,
                            const int n,
                            const int k,
                            const std::complex<double> alpha,
                            const std::complex<double>* a,
                            const int lda,
                            const std::complex<double>* b,
                            const int ldb,
                            const std::complex<double> beta,
                            std::complex<double>* c,
                            const int ldc,
                            base_device::AbacusDevice_t device_type)
{
    if (device_type == base_device::AbacusDevice_t::CpuDevice)
    {
        zgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    }
#ifdef __DSP
    else if (device_type == base_device::AbacusDevice_t::DspDevice)
    {
        mtfunc::zgemm_mth_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc, GlobalV::MY_RANK);
    }
#endif
    else if (device_type == base_device::AbacusDevice_t::GpuDevice)
    {
#ifdef __CUDA
        cublasOperation_t cutransA = BlasUtils::judge_trans(false, transa, "gemm_op");
        cublasOperation_t cutransB = BlasUtils::judge_trans(false, transb, "gemm_op");
        cublasErrcheck(cublasZgemm(BlasUtils::cublas_handle,
                                   cutransA,
                                   cutransB,
                                   m,
                                   n,
                                   k,
                                   (double2*)&alpha,
                                   (double2*)a,
                                   lda,
                                   (double2*)b,
                                   ldb,
                                   (double2*)&beta,
                                   (double2*)c,
                                   ldc));
#endif
    }
}

// Symm and Hemm part. Only col-major is supported.

void BlasConnector::symm_cm(const char side, const char uplo, const int m, const int n,
	const float alpha, const float *a, const int lda, const float *b, const int ldb,
	const float beta, float *c, const int ldc, base_device::AbacusDevice_t device_type)
{
	if (device_type == base_device::AbacusDevice_t::CpuDevice) {
		ssymm_(&side, &uplo, &m, &n,
		&alpha, a, &lda, b, &ldb,
		&beta, c, &ldc);
	}
	else if (device_type == base_device::AbacusDevice_t::GpuDevice){
#ifdef __CUDA
		cublasSideMode_t sideMode = BlasUtils::judge_side(side);
		cublasFillMode_t fillMode = BlasUtils::judge_fill(uplo);
		cublasErrcheck(cublasSsymm(BlasUtils::cublas_handle, sideMode, fillMode, m, n, &alpha, a, lda, b, ldb, &beta, c, ldc));
#endif
	}
}

void BlasConnector::symm_cm(const char side, const char uplo, const int m, const int n,
	const double alpha, const double *a, const int lda, const double *b, const int ldb,
	const double beta, double *c, const int ldc, base_device::AbacusDevice_t device_type)
{
	if (device_type == base_device::AbacusDevice_t::CpuDevice) {
		dsymm_(&side, &uplo, &m, &n,
		&alpha, a, &lda, b, &ldb,
		&beta, c, &ldc);
	}
	else if (device_type == base_device::AbacusDevice_t::GpuDevice){
#ifdef __CUDA
		cublasSideMode_t sideMode = BlasUtils::judge_side(side);
		cublasFillMode_t fillMode = BlasUtils::judge_fill(uplo);
		cublasErrcheck(cublasDsymm(BlasUtils::cublas_handle, sideMode, fillMode, m, n, &alpha, a, lda, b, ldb, &beta, c, ldc));
#endif
	}
}

void BlasConnector::symm_cm(const char side, const char uplo, const int m, const int n,
    const std::complex<float> alpha, const std::complex<float> *a, const int lda, const std::complex<float> *b, const int ldb,
    const std::complex<float> beta, std::complex<float> *c, const int ldc, base_device::AbacusDevice_t device_type)
{
	if (device_type == base_device::AbacusDevice_t::CpuDevice) {
    	csymm_(&side, &uplo, &m, &n,
		&alpha, a, &lda, b, &ldb,
		&beta, c, &ldc);
	}
	else if (device_type == base_device::AbacusDevice_t::GpuDevice){
#ifdef __CUDA
		cublasSideMode_t sideMode = BlasUtils::judge_side(side);
		cublasFillMode_t fillMode = BlasUtils::judge_fill(uplo);
		cublasErrcheck(cublasCsymm(BlasUtils::cublas_handle, sideMode, fillMode, m, n, (float2*)&alpha, (float2*)a, lda, (float2*)b, ldb, (float2*)&beta, (float2*)c, ldc));
#endif
	}
}

void BlasConnector::symm_cm(const char side, const char uplo, const int m, const int n,
	const std::complex<double> alpha, const std::complex<double> *a, const int lda, const std::complex<double> *b, const int ldb,
	const std::complex<double> beta, std::complex<double> *c, const int ldc, base_device::AbacusDevice_t device_type)
{
	if (device_type == base_device::AbacusDevice_t::CpuDevice) {
		zsymm_(&side, &uplo, &m, &n,
		&alpha, a, &lda, b, &ldb,
		&beta, c, &ldc);
	}
	else if (device_type == base_device::AbacusDevice_t::GpuDevice){
#ifdef __CUDA
		cublasSideMode_t sideMode = BlasUtils::judge_side(side);
		cublasFillMode_t fillMode = BlasUtils::judge_fill(uplo);
		cublasErrcheck(cublasZsymm(BlasUtils::cublas_handle, sideMode, fillMode, m, n, (double2*)&alpha, (double2*)a, lda, (double2*)b, ldb, (double2*)&beta, (double2*)c, ldc));
#endif
	}
}

void BlasConnector::hemm_cm(char side, char uplo, int m, int n,
    std::complex<float> alpha, std::complex<float> *a, int lda, std::complex<float> *b, int ldb,
    std::complex<float> beta, std::complex<float> *c, int ldc, base_device::AbacusDevice_t device_type)
{
	if (device_type == base_device::AbacusDevice_t::CpuDevice) {
    	chemm_(&side, &uplo, &m, &n,
		&alpha, a, &lda, b, &ldb,
		&beta, c, &ldc);
	}
	else if (device_type == base_device::AbacusDevice_t::GpuDevice){
#ifdef __CUDA
		cublasSideMode_t sideMode = BlasUtils::judge_side(side);
		cublasFillMode_t fillMode = BlasUtils::judge_fill(uplo);
		cublasErrcheck(cublasChemm(BlasUtils::cublas_handle, sideMode, fillMode, m, n, (float2*)&alpha, (float2*)a, lda, (float2*)b, ldb, (float2*)&beta, (float2*)c, ldc));
#endif
	}
}

void BlasConnector::hemm_cm(char side, char uplo, int m, int n,
	std::complex<double> alpha, std::complex<double> *a, int lda, std::complex<double> *b, int ldb,
	std::complex<double> beta, std::complex<double> *c, int ldc, base_device::AbacusDevice_t device_type)
{
	if (device_type == base_device::AbacusDevice_t::CpuDevice) {
		zhemm_(&side, &uplo, &m, &n,
		&alpha, a, &lda, b, &ldb,
		&beta, c, &ldc);
	}
	else if (device_type == base_device::AbacusDevice_t::GpuDevice){
#ifdef __CUDA
		cublasSideMode_t sideMode = BlasUtils::judge_side(side);
		cublasFillMode_t fillMode = BlasUtils::judge_fill(uplo);
		cublasErrcheck(cublasZhemm(BlasUtils::cublas_handle, sideMode, fillMode, m, n, (double2*)&alpha, (double2*)a, lda, (double2*)b, ldb, (double2*)&beta, (double2*)c, ldc));
#endif
	}
}

void BlasConnector::gemv(const char trans, const int m, const int n,
    const float alpha, const float* A, const int lda, const float* X, const int incx,
    const float beta, float* Y, const int incy, base_device::AbacusDevice_t device_type)
{
	if (device_type == base_device::AbacusDevice_t::CpuDevice) {
    	sgemv_(&trans, &m, &n, &alpha, A, &lda, X, &incx, &beta, Y, &incy);
	}
	else if (device_type == base_device::AbacusDevice_t::GpuDevice){
#ifdef __CUDA
		cublasOperation_t cutransA = BlasUtils::judge_trans(false, trans, "gemv_op");
		cublasErrcheck(cublasSgemv(BlasUtils::cublas_handle, cutransA, m, n, &alpha, A, lda, X, incx, &beta, Y, incy));
#endif
	}
}

void BlasConnector::gemv(const char trans, const int m, const int n,
    const double alpha, const double* A, const int lda, const double* X, const int incx,
    const double beta, double* Y, const int incy, base_device::AbacusDevice_t device_type)
{
	if (device_type == base_device::AbacusDevice_t::CpuDevice) {
    	dgemv_(&trans, &m, &n, &alpha, A, &lda, X, &incx, &beta, Y, &incy);
	}
	else if (device_type == base_device::AbacusDevice_t::GpuDevice){
#ifdef __CUDA
		cublasOperation_t cutransA = BlasUtils::judge_trans(false, trans, "gemv_op");
		cublasErrcheck(cublasDgemv(BlasUtils::cublas_handle, cutransA, m, n, &alpha, A, lda, X, incx, &beta, Y, incy));
#endif
	}
}

void BlasConnector::gemv(const char trans, const int m, const int n,
    const std::complex<float> alpha, const std::complex<float> *A, const int lda, const std::complex<float> *X, const int incx,
    const std::complex<float> beta, std::complex<float> *Y, const int incy, base_device::AbacusDevice_t device_type)
{
	if (device_type == base_device::AbacusDevice_t::CpuDevice) {
    	cgemv_(&trans, &m, &n, &alpha, A, &lda, X, &incx, &beta, Y, &incy);
	}
	else if (device_type == base_device::AbacusDevice_t::GpuDevice){
#ifdef __CUDA
		cuFloatComplex alpha_cu = make_cuFloatComplex(alpha.real(), alpha.imag());
    	cuFloatComplex beta_cu = make_cuFloatComplex(beta.real(), beta.imag());
		cublasOperation_t cutransA = BlasUtils::judge_trans(true, trans, "gemv_op");
		cublasErrcheck(cublasCgemv(BlasUtils::cublas_handle, cutransA, m, n, &alpha_cu, (cuFloatComplex*)A, lda, (cuFloatComplex*)X, incx, &beta_cu, (cuFloatComplex*)Y, incy));
#endif
	}
}

void BlasConnector::gemv(const char trans, const int m, const int n,
    const std::complex<double> alpha, const std::complex<double> *A, const int lda, const std::complex<double> *X, const int incx,
    const std::complex<double> beta, std::complex<double> *Y, const int incy, base_device::AbacusDevice_t device_type)
{
	if (device_type == base_device::AbacusDevice_t::CpuDevice) {
    	zgemv_(&trans, &m, &n, &alpha, A, &lda, X, &incx, &beta, Y, &incy);
	}
	else if (device_type == base_device::AbacusDevice_t::GpuDevice){
#ifdef __CUDA
		cuDoubleComplex alpha_cu = make_cuDoubleComplex(alpha.real(), alpha.imag());
    	cuDoubleComplex beta_cu = make_cuDoubleComplex(beta.real(), beta.imag());
		cublasOperation_t cutransA = BlasUtils::judge_trans(true, trans, "gemv_op");
		cublasErrcheck(cublasZgemv(BlasUtils::cublas_handle, cutransA, m, n, &alpha_cu, (cuDoubleComplex*)A, lda, (cuDoubleComplex*)X, incx, &beta_cu, (cuDoubleComplex*)Y, incy));
#endif
	}
}

// out = ||x||_2
float BlasConnector::nrm2( const int n, const float *X, const int incX, base_device::AbacusDevice_t device_type )
{
	if (device_type == base_device::AbacusDevice_t::CpuDevice) {
		return snrm2_( &n, X, &incX );
	}
	else if (device_type == base_device::AbacusDevice_t::GpuDevice){
#ifdef __CUDA
		float result = 0.0;
		cublasErrcheck(cublasSnrm2(BlasUtils::cublas_handle, n, X, incX, &result));
		return result;
#endif
	}
	return snrm2_( &n, X, &incX );
}


double BlasConnector::nrm2( const int n, const double *X, const int incX, base_device::AbacusDevice_t device_type )
{
	if (device_type == base_device::AbacusDevice_t::CpuDevice) {
		return dnrm2_( &n, X, &incX );
	}
	else if (device_type == base_device::AbacusDevice_t::GpuDevice){
#ifdef __CUDA
		double result = 0.0;
		cublasErrcheck(cublasDnrm2(BlasUtils::cublas_handle, n, X, incX, &result));
		return result;
#endif
	}
	return dnrm2_( &n, X, &incX );
}


double BlasConnector::nrm2( const int n, const std::complex<double> *X, const int incX, base_device::AbacusDevice_t device_type )
{
	if (device_type == base_device::AbacusDevice_t::CpuDevice) {
		return dznrm2_( &n, X, &incX );
	}
	else if (device_type == base_device::AbacusDevice_t::GpuDevice){
#ifdef __CUDA
		double result = 0.0;
		cublasErrcheck(cublasDznrm2(BlasUtils::cublas_handle, n, (double2*)X, incX, &result));
		return result;
#endif
	}
	return dznrm2_( &n, X, &incX );
}

// copies a into b
void BlasConnector::copy(const long n, const double *a, const int incx, double *b, const int incy, base_device::AbacusDevice_t device_type)
{
	if (device_type == base_device::AbacusDevice_t::CpuDevice) {
		dcopy_(&n, a, &incx, b, &incy);
	}
}

void BlasConnector::copy(const long n, const std::complex<double> *a, const int incx, std::complex<double> *b, const int incy, base_device::AbacusDevice_t device_type)
{
	if (device_type == base_device::AbacusDevice_t::CpuDevice) {
		zcopy_(&n, a, &incx, b, &incy);
	}
}


template <typename T>
void vector_mul_vector(const int& dim, T* result, const T* vector1, const T* vector2, base_device::AbacusDevice_t device_type){
	using Real = typename GetTypeReal<T>::type;
	if (device_type == base_device::AbacusDevice_t::CpuDevice) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 4096 / sizeof(Real))
#endif
        for (int i = 0; i < dim; i++)
        {
            result[i] = vector1[i] * vector2[i];
        }
	}
	else if (device_type == base_device::AbacusDevice_t::GpuDevice){
#ifdef __CUDA
		ModuleBase::vector_mul_vector_op<T, base_device::DEVICE_GPU>()(dim, result, vector1, vector2);
#endif
	}
}


template <typename T>
void vector_div_vector(const int& dim, T* result, const T* vector1, const T* vector2, base_device::AbacusDevice_t device_type){
	using Real = typename GetTypeReal<T>::type;
	if (device_type == base_device::AbacusDevice_t::CpuDevice) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 4096 / sizeof(Real))
#endif
		for (int i = 0; i < dim; i++)
        {
            result[i] = vector1[i] / vector2[i];
        }
	}
	else if (device_type == base_device::AbacusDevice_t::GpuDevice){
#ifdef __CUDA
		ModuleBase::vector_div_vector_op<T, base_device::DEVICE_GPU>()(dim, result, vector1, vector2);
#endif
	}
}

void vector_add_vector(const int& dim, float *result, const float *vector1, const float constant1, const float *vector2, const float constant2, base_device::AbacusDevice_t device_type)
{
	if (device_type == base_device::CpuDevice){
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 8192 / sizeof(float))
#endif
        for (int i = 0; i < dim; i++)
        {
            result[i] = vector1[i] * constant1 + vector2[i] * constant2;
        }
	}
	else if (device_type == base_device::GpuDevice){
#ifdef __CUDA
		ModuleBase::vector_add_vector_op<float, base_device::DEVICE_GPU>()(dim, result, vector1, constant1, vector2, constant2);
#endif
	}
}

void vector_add_vector(const int& dim, double *result, const double *vector1, const double constant1, const double *vector2, const double constant2, base_device::AbacusDevice_t device_type)
{
	if (device_type == base_device::CpuDevice){
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 8192 / sizeof(double))
#endif
        for (int i = 0; i < dim; i++)
        {
            result[i] = vector1[i] * constant1 + vector2[i] * constant2;
        }
	}
	else if (device_type == base_device::GpuDevice){
#ifdef __CUDA
		ModuleBase::vector_add_vector_op<double, base_device::DEVICE_GPU>()(dim, result, vector1, constant1, vector2, constant2);
#endif
	}
}

void vector_add_vector(const int& dim, std::complex<float> *result, const std::complex<float> *vector1, const float constant1, const std::complex<float> *vector2, const float constant2, base_device::AbacusDevice_t device_type)
{
	if (device_type == base_device::CpuDevice){
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 8192 / sizeof(std::complex<float>))
#endif
        for (int i = 0; i < dim; i++)
        {
            result[i] = vector1[i] * constant1 + vector2[i] * constant2;
        }
	}
	else if (device_type == base_device::GpuDevice){
#ifdef __CUDA
		ModuleBase::vector_add_vector_op<std::complex<float>, base_device::DEVICE_GPU>()(dim, result, vector1, constant1, vector2, constant2);
#endif
	}
}

void vector_add_vector(const int& dim, std::complex<double> *result, const std::complex<double> *vector1, const double constant1, const std::complex<double> *vector2, const double constant2, base_device::AbacusDevice_t device_type)
{
	if (device_type == base_device::CpuDevice){
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 8192 / sizeof(std::complex<double>))
#endif
        for (int i = 0; i < dim; i++)
        {
            result[i] = vector1[i] * constant1 + vector2[i] * constant2;
        }
	}
	else if (device_type == base_device::GpuDevice){
#ifdef __CUDA
		ModuleBase::vector_add_vector_op<std::complex<double>, base_device::DEVICE_GPU>()(dim, result, vector1, constant1, vector2, constant2);
#endif
	}
}