#include <ATen/core/tensor_buffer.h>

#include <base/core/cpu_allocator.h>
#include <base/macros/macros.h>

#if defined(__CUDA) || defined(__ROCM)
#include <base/core/gpu_allocator.h>
#endif

namespace container {

// Construct a new TensorBuffer object.
TensorBuffer::TensorBuffer(base::core::Allocator* alloc, void* data_ptr) : alloc_(alloc), data_(data_ptr), owns_memory_(true) {}

// Construct a new TensorBuffer object.
// Note, this is a reference TensorBuffer, does not own memory itself.
TensorBuffer::TensorBuffer(void* data_ptr) : alloc_(), data_(data_ptr), owns_memory_(false) {}

// Class members are initialized in the order of their declaration, 
// rather than the order they appear in the initialization list!
TensorBuffer::TensorBuffer(base::core::Allocator* alloc, size_t size) {
    alloc_ = alloc; 
    if (size > 0) {
        data_ = alloc_->allocate(size);
        owns_memory_ = true;
        allocated_bytes_ = size;
    }
}

// Move constructor.
TensorBuffer::TensorBuffer(TensorBuffer&& other) noexcept
        : alloc_(other.alloc_),
          data_(other.data_), 
          owns_memory_(other.owns_memory_),
          allocated_bytes_(other.allocated_bytes_)
{
    // Reset the other TensorBuffer.
    other.data_ = nullptr;
    other.owns_memory_ = false;
    other.allocated_bytes_ = 0;
}

// Destroy the TensorBuffer object.
TensorBuffer::~TensorBuffer() {
    if (this->OwnsMemory() && data_ != nullptr) {
        alloc_->free(data_);
    }
    if (alloc_ != nullptr) {
        delete alloc_;
    }
}

// Get the raw data pointer.
void* TensorBuffer::data() const { return data_; }

// Get the total number of bytes allocated for the buffer.
// This method returns the total number of bytes allocated for the buffer by the allocator
// associated with the TensorBuffer. If the buffer is not yet allocated, the function returns 0.
size_t TensorBuffer::GetAllocatedBytes() const {
    return allocated_bytes_;
}

// Get the root TensorBuffer object.
// If this TensorBuffer is a sub-buffer of another TensorBuffer, returns that
// TensorBuffer. Otherwise, returns this.
TensorBuffer* TensorBuffer::root_buffer() { return this; } // Implementation goes here.

// Get the Allocator object used in this class.
base::core::Allocator* TensorBuffer::allocator() const {
    return alloc_;
}

// Check whether this TensorBuffer owns the underlying memory.
bool TensorBuffer::OwnsMemory() const { return this->owns_memory_; }

// Get the type of device used by the TensorBuffer.
DeviceType TensorBuffer::GetDeviceType() const {
    if (alloc_ != nullptr) {
        return alloc_->GetDeviceType();
    }
    return DeviceType::UnKnown;
}

void TensorBuffer::resize(size_t size) {
    // Allocate a new buffer.
    void* new_data = this->alloc_->allocate(size);

    // Free the old buffer.
    if (this->OwnsMemory()) {
        this->alloc_->free(data_);
    }

    // Update the internal state.
    this->data_ = new_data;
    this->owns_memory_ = true;
}

TensorBuffer& TensorBuffer::operator=(const TensorBuffer& other) {
    if (this->OwnsMemory()) {
        this->alloc_->free(data_);
    }

    delete this->alloc_;
    if (other.GetDeviceType() == DeviceType::CpuDevice) {
        this->alloc_ = new base::core::CPUAllocator();
    }
    #if defined(__CUDA) || defined(__ROCM)
    else if (other.GetDeviceType() == DeviceType::GpuDevice) {
        this->alloc_ = new base::core::GPUAllocator();
    }
    #endif // __CUDA || __ROCM


    this->data_ = this->alloc_->allocate(other.GetAllocatedBytes());
    this->owns_memory_ = true;
    return *this;
}

TensorBuffer& TensorBuffer::operator=(TensorBuffer&& other) noexcept {
    if (this->OwnsMemory()) {
        this->alloc_->free(data_);
    }
    delete this->alloc_;
    this->alloc_ = other.alloc_;
    this->data_ = other.data_;
    this->owns_memory_ = other.owns_memory_;

    // Reset the other TensorBuffer.
    other.data_ = nullptr;
    other.owns_memory_ = false;
    return *this;
}

}  // namespace container
