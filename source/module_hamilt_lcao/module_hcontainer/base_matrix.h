#ifndef BASE_MATRIX_H
#define BASE_MATRIX_H

#include <iostream>
#include <mutex>
#include <cassert>

namespace hamilt
{
/**
 * class: BaseMatrix
 * used to store a matrix for atom-pair local Hamiltonian with specific R-index
 * T can be double or complex<double>
 * It has two ways to arrange data:
 * 1. allocate data itself
 * 2. only access data but be arranged by other class
 */
template <typename T>
class BaseMatrix
{
  public:
    // Constructor of class BaseMatrix
    BaseMatrix(const int& nrow_, const int& ncol_, T* data_existed = nullptr);
    // copy constructor
    BaseMatrix(const BaseMatrix<T>& matrix);
    // move constructor
    BaseMatrix(BaseMatrix<T>&& matrix);
    // Destructor of class BaseMatrix
    ~BaseMatrix();

    /**
     * @brief allocate memory for the matrix
     * if this->value_begin is not nullptr, it will be neglected
     * if this->value_begin is nullptr, it will allocate memory with size nrow_local * ncol_local
    */
    void allocate(T* data_array = nullptr, bool if_zero = false);

    /**
     * @brief set value in the matrix to zero
    */
    void set_zero();

    /**
     * @brief add an array to the matrix
     *
     * @param array array to be added
     */
    void add_array(T* array);

    /**
     * @brief add a single element to the matrix
     *
     * @param mu row index
     * @param nu column index
     * @param value value to be added
     */
    void add_element(int mu, int nu, const T& value)
    {
        #ifdef __DEBUG
        assert(this->value_begin != nullptr);
        #endif
            int index = mu * this->ncol_local + nu;
            value_begin[index] += value;
    };

    // for inside matrix
    /**
     * @brief get value from a whole matrix
     *
     * @param i_row row index
     * @param j_col column index
     * @return T&
     */
    T& get_value(const size_t& i_row, const size_t& j_col) const
    {
        #ifdef __DEBUG
        assert(this->value_begin != nullptr);
        #endif
            int index = i_row * this->ncol_local + j_col;
            return value_begin[index];
    };

    /**
     * @brief get pointer of value from a submatrix
     */
    T* get_pointer() const { return value_begin; };

    // operator= for copy assignment
    BaseMatrix& operator=(const BaseMatrix& other);

    // operator= for move assignment
    BaseMatrix& operator=(BaseMatrix&& other) noexcept;

    /**
     * @brief get total memory size of BaseMatrix
    */
    size_t get_memory_size() const;

    /**
     * @brief get col_size for this matrix
    */
    int get_col_size() const {return ncol_local;};
    /**
     * @brief get row_size for this matrix
    */
    int get_row_size() const {return nrow_local;};
    /**
     * @brief set col_size and row_size
    */
    void set_size(const int& col_size_in, const int& row_size_in);

    void add_array_ts(T* array)
    {
        std::lock_guard<std::mutex> lock(mtx);
        const int size = nrow_local * ncol_local;
        for (int i = 0; i < size; ++i)
        {
            value_begin[i] += array[i];
        }
    }

  private:
    bool allocated = false;

    // pointer for accessing data
    // two ways to arrange data:
    // 1. allocate data itself
    // 2. only access data but be arranged by RealSparseHamiltonian
    T* value_begin = nullptr;

    // int current_multiple = 0;

    // for thread safe
    mutable std::mutex mtx;
    // number of rows and columns
    int nrow_local = 0;
    int ncol_local = 0;
};

} // namespace hamilt

#endif
