#include <ATen/kernels/memory.h>
#include <base/macros/rocm.h>

#include <hip/hip_runtime.h>
#include <thrust/complex.h>

namespace container {
namespace kernels {

template <typename T>
__global__ void do_set_memory(
    T* out,
    const T var,
    const size_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size) {return;}
    out[idx] = var;
}

template <typename T_out, typename T_in>
__global__ void do_cast_memory(
    T_out* out,
    const T_in* in,
    const int size)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size) {return;}
    out[idx] = static_cast<T_out>(in[idx]);
}

template <typename T_out, typename T_in>
__global__ void do_cast_memory(
    std::complex<T_out>* out,
    const std::complex<T_in>* in,
    const int size)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size) {return;}
    auto* _out = reinterpret_cast<thrust::complex<T_out>*>(out);
    const auto* _in = reinterpret_cast<const thrust::complex<T_in>*>(in);
    _out[idx] = static_cast<thrust::complex<T_out>>(_in[idx]);
}

template <typename T, typename Device>
void resize_memory<T, Device>::operator()(
    T*& arr,
    const size_t& size,
    const char* record_in)
{
    if (arr != nullptr) {
        delete_memory<T, container::DEVICE_GPU>()(arr);
    }
    hipMalloc((void **)&arr, sizeof(T) * size);
}

template <typename T, typename Device>
void set_memory<T, Device>::operator()(
    T* arr,
    const T& var, 
    const size_t& size) 
{
    const int block = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    do_set_memory<<<block, THREADS_PER_BLOCK>>>(arr, var, size);
}

template <typename T>
struct synchronize_memory<T, DEVICE_CPU, DEVICE_GPU> {
    void operator()(
        T *arr_out,
        const T *arr_in,
        const size_t& size)
    {
        hipMemcpy(arr_out, arr_in, sizeof(T) * size, hipMemcpyDeviceToHost);
    }
};

template <typename T>
struct synchronize_memory<T, DEVICE_GPU, DEVICE_CPU> {
    void operator()(
        T *arr_out,
        const T *arr_in,
        const size_t& size)
    {
        hipMemcpy(arr_out, arr_in, sizeof(T) * size, hipMemcpyHostToDevice);
    }
};

template <typename T>
struct synchronize_memory<T, DEVICE_GPU, DEVICE_GPU> {
    void operator()(
        T *arr_out,
        const T *arr_in,
        const size_t& size)
    {
        hipMemcpy(arr_out, arr_in, sizeof(T) * size, hipMemcpyDeviceToDevice);
    }
};

template <typename T_out, typename T_in>
struct cast_memory<T_out, T_in, container::DEVICE_GPU, container::DEVICE_GPU> {
    void operator()(
        T_out* arr_out,
        const T_in* arr_in,
        const size_t& size)
    {
        const int block = static_cast<int>((size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        do_cast_memory<<<block, THREADS_PER_BLOCK>>>(arr_out, arr_in, size);
    }
};


template <typename T_out, typename T_in>
struct cast_memory<T_out, T_in, container::DEVICE_GPU, container::DEVICE_CPU> {
    void operator()(
        T_out* arr_out,
        const T_in* arr_in,
        const size_t& size)
    {
        T_in * arr = nullptr;
        hipMalloc((void **)&arr, sizeof(T_in) * size);
        hipMemcpy(arr, arr_in, sizeof(T_in) * size, hipMemcpyHostToDevice);
        const int block = static_cast<int>((size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        do_cast_memory<<<block, THREADS_PER_BLOCK>>>(arr_out, arr, size);
        hipFree(arr);
    }
};


template <typename T_out, typename T_in>
struct cast_memory<T_out, T_in, container::DEVICE_CPU, container::DEVICE_GPU> {
    void operator()(
        T_out* arr_out,
        const T_in* arr_in,
        const size_t& size)
    {
        auto * arr = (T_in*) malloc(sizeof(T_in) * size);
        hipMemcpy(arr, arr_in, sizeof(T_in) * size, hipMemcpyDeviceToHost);
        for (int ii = 0; ii < size; ii++) {
            arr_out[ii] = static_cast<T_out>(arr[ii]);
        }
        free(arr);
    }
};

template <typename T, typename Device>
void delete_memory<T, Device>::operator() (
    T* arr)
{
    hipFree(arr);
}

template struct resize_memory<int, container::DEVICE_GPU>;
template struct resize_memory<int64_t, container::DEVICE_GPU>;
template struct resize_memory<float, container::DEVICE_GPU>;
template struct resize_memory<double, container::DEVICE_GPU>;
template struct resize_memory<std::complex<float>, container::DEVICE_GPU>;
template struct resize_memory<std::complex<double>, container::DEVICE_GPU>;

template struct set_memory<int, container::DEVICE_GPU>;
template struct set_memory<int64_t , container::DEVICE_GPU>;
template struct set_memory<float, container::DEVICE_GPU>;
template struct set_memory<double, container::DEVICE_GPU>;
template struct set_memory<std::complex<float>, container::DEVICE_GPU>;
template struct set_memory<std::complex<double>, container::DEVICE_GPU>;

template struct synchronize_memory<int, container::DEVICE_CPU, container::DEVICE_GPU>;
template struct synchronize_memory<int, container::DEVICE_GPU, container::DEVICE_CPU>;
template struct synchronize_memory<int, container::DEVICE_GPU, container::DEVICE_GPU>;
template struct synchronize_memory<int64_t, container::DEVICE_CPU, container::DEVICE_GPU>;
template struct synchronize_memory<int64_t, container::DEVICE_GPU, container::DEVICE_CPU>;
template struct synchronize_memory<int64_t, container::DEVICE_GPU, container::DEVICE_GPU>;
template struct synchronize_memory<float, container::DEVICE_CPU, container::DEVICE_GPU>;
template struct synchronize_memory<float, container::DEVICE_GPU, container::DEVICE_CPU>;
template struct synchronize_memory<float, container::DEVICE_GPU, container::DEVICE_GPU>;
template struct synchronize_memory<double, container::DEVICE_CPU, container::DEVICE_GPU>;
template struct synchronize_memory<double, container::DEVICE_GPU, container::DEVICE_CPU>;
template struct synchronize_memory<double, container::DEVICE_GPU, container::DEVICE_GPU>;
template struct synchronize_memory<std::complex<float>, container::DEVICE_CPU, container::DEVICE_GPU>;
template struct synchronize_memory<std::complex<float>, container::DEVICE_GPU, container::DEVICE_CPU>;
template struct synchronize_memory<std::complex<float>, container::DEVICE_GPU, container::DEVICE_GPU>;
template struct synchronize_memory<std::complex<double>, container::DEVICE_CPU, container::DEVICE_GPU>;
template struct synchronize_memory<std::complex<double>, container::DEVICE_GPU, container::DEVICE_CPU>;
template struct synchronize_memory<std::complex<double>, container::DEVICE_GPU, container::DEVICE_GPU>;

template struct cast_memory<float, float, container::DEVICE_GPU, container::DEVICE_GPU>;
template struct cast_memory<double, double, container::DEVICE_GPU, container::DEVICE_GPU>;
template struct cast_memory<float, double, container::DEVICE_GPU, container::DEVICE_GPU>;
template struct cast_memory<double, float, container::DEVICE_GPU, container::DEVICE_GPU>;
template struct cast_memory<std::complex<float>, std::complex<float>, container::DEVICE_GPU, container::DEVICE_GPU>;
template struct cast_memory<std::complex<double>, std::complex<double>, container::DEVICE_GPU, container::DEVICE_GPU>;
template struct cast_memory<std::complex<float>, std::complex<double>, container::DEVICE_GPU, container::DEVICE_GPU>;
template struct cast_memory<std::complex<double>, std::complex<float>, container::DEVICE_GPU, container::DEVICE_GPU>;
template struct cast_memory<float, float, container::DEVICE_GPU, container::DEVICE_CPU>;
template struct cast_memory<double, double, container::DEVICE_GPU, container::DEVICE_CPU>;
template struct cast_memory<float, double, container::DEVICE_GPU, container::DEVICE_CPU>;
template struct cast_memory<double, float, container::DEVICE_GPU, container::DEVICE_CPU>;
template struct cast_memory<std::complex<float>, std::complex<float>, container::DEVICE_GPU, container::DEVICE_CPU>;
template struct cast_memory<std::complex<double>, std::complex<double>, container::DEVICE_GPU, container::DEVICE_CPU>;
template struct cast_memory<std::complex<float>, std::complex<double>, container::DEVICE_GPU, container::DEVICE_CPU>;
template struct cast_memory<std::complex<double>, std::complex<float>, container::DEVICE_GPU, container::DEVICE_CPU>;
template struct cast_memory<float, float, container::DEVICE_CPU, container::DEVICE_GPU>;
template struct cast_memory<double, double, container::DEVICE_CPU, container::DEVICE_GPU>;
template struct cast_memory<float, double, container::DEVICE_CPU, container::DEVICE_GPU>;
template struct cast_memory<double, float, container::DEVICE_CPU, container::DEVICE_GPU>;
template struct cast_memory<std::complex<float>, std::complex<float>, container::DEVICE_CPU, container::DEVICE_GPU>;
template struct cast_memory<std::complex<double>, std::complex<double>, container::DEVICE_CPU, container::DEVICE_GPU>;
template struct cast_memory<std::complex<float>, std::complex<double>, container::DEVICE_CPU, container::DEVICE_GPU>;
template struct cast_memory<std::complex<double>, std::complex<float>, container::DEVICE_CPU, container::DEVICE_GPU>;

template struct delete_memory<int, container::DEVICE_GPU>;
template struct delete_memory<int64_t, container::DEVICE_GPU>;
template struct delete_memory<float, container::DEVICE_GPU>;
template struct delete_memory<double, container::DEVICE_GPU>;
template struct delete_memory<std::complex<float>, container::DEVICE_GPU>;
template struct delete_memory<std::complex<double>, container::DEVICE_GPU>;

} // end of namespace kernels
} // end of namespace container