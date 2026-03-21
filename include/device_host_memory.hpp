/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * You may obtain a copy of the license in the LICENSE file
 * located in the root directory of this source tree or at:
 * https://opensource.org/licenses/BSD-3-Clause
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */


/**
 * @file device_host_memory.hpp
 * 
 */


#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <mutex>
#include <iomanip>
#include <sstream>
#include <unordered_map>

#include "utils.hpp" // THROW_EXCEPTION

namespace gansu{


/**
 * @brief Base class for managing CUDA memory.
 *
 * This class provides a unified interface for managing CUDA device and host memory.
 * It also tracks memory statistics across all instances.
 *
 * @tparam T The type of elements stored in the memory.
 */
template <typename T>
class CudaMemoryManager {
protected:
    T* device_ptr_;   ///< Pointer to the device memory
    T* host_ptr_;     ///< Pointer to the host memory
    size_t size_;     ///< Number of elements in the memory
    size_t device_bytes_; ///< Number of bytes allocated on device for this instance
    size_t host_bytes_;   ///< Number of bytes allocated on host for this instance

    // Static members for memory statistics (shared across all instances)
    inline static size_t current_allocated_bytes_ = 0;  ///< Current total allocated device memory
    inline static size_t total_allocated_bytes_ = 0;    ///< Cumulative total allocated device memory
    inline static size_t peak_allocated_bytes_ = 0;     ///< Peak device memory usage
    inline static std::mutex memory_stats_mutex_;       ///< Mutex for thread-safe statistics

public:
    /**
     * @brief Constructs a CudaMemoryManager with the given size.
     *
     * This constructor initializes the memory manager without allocating memory.
     *
     * @param size The number of elements in the memory.
     */
    CudaMemoryManager(size_t size)
        : device_ptr_(nullptr), host_ptr_(nullptr), size_(size),
          device_bytes_(0), host_bytes_(0) {}

    /**
     * @brief Virtual destructor that ensures proper memory cleanup.
     *
     * Frees the allocated device and host memory if they exist and updates statistics.
     */
    virtual ~CudaMemoryManager() {
        if (device_ptr_) {
            cudaFree(device_ptr_);
            // Update memory statistics
            track_deallocation(device_bytes_);
        }
        if (host_ptr_) {
            cudaFreeHost(host_ptr_);
        }
    }

    /**
     * @brief Allocates memory on the device.
     * 
     * This method must be implemented by derived classes.
     */
    virtual void allocate() = 0;

    /**
     * @brief Gets the number of elements in the memory.
     * @return The number of elements.
     */
    size_t size() const { return size_; }

    /**
     * @brief Gets a pointer to the device memory.
     * @return Pointer to the device memory.
     */
    T* device_ptr() { return device_ptr_; }

    /**
     * @brief Gets a constant pointer to the device memory.
     * @return Constant pointer to the device memory.
     */
    const T* device_ptr() const { return device_ptr_; }

    /**
     * @brief Gets a pointer to the host memory.
     * @return Pointer to the host memory.
     */
    T* host_ptr() { return host_ptr_; }

    /**
     * @brief Gets a constant pointer to the host memory.
     * @return Constant pointer to the host memory.
     */
    const T* host_ptr() const { return host_ptr_; }

    /**
     * @brief Copies data from the host to the device memory.
     *
     * This method must be implemented by derived classes if needed.
     */
    virtual void toDevice() = 0;

    /**
     * @brief Copies data from the device to the host memory.
     *
     * This method must be implemented by derived classes if needed.
     */
    virtual void toHost() = 0;

    /**
     * @brief Reports memory statistics for all CudaMemoryManager instances.
     *
     * Prints current, total, and peak device memory usage.
     */
    static void report_memory_statistics() {
        std::lock_guard<std::mutex> lock(memory_stats_mutex_);
        std::cout << std::endl;
        std::cout << "[Device Memory Statistics]" << std::endl;
        std::cout << "Current allocated: " << format_bytes(current_allocated_bytes_) << std::endl;
        std::cout << "Total allocated: " << format_bytes(total_allocated_bytes_) << std::endl;
        std::cout << "Peak usage: " << format_bytes(peak_allocated_bytes_) << std::endl;
    }

    /**
     * @brief Resets memory statistics to zero.
     *
     * Useful for tracking memory usage for specific code sections.
     */
    static void reset_memory_statistics() {
        std::lock_guard<std::mutex> lock(memory_stats_mutex_);
        current_allocated_bytes_ = 0;
        total_allocated_bytes_ = 0;
        peak_allocated_bytes_ = 0;
    }

    /**
     * @brief Gets the current allocated device memory in bytes.
     * @return Current allocated bytes.
     */
    static size_t get_current_allocated_bytes() {
        std::lock_guard<std::mutex> lock(memory_stats_mutex_);
        return current_allocated_bytes_;
    }

    /**
     * @brief Gets the total allocated device memory in bytes.
     * @return Total allocated bytes (cumulative).
     */
    static size_t get_total_allocated_bytes() {
        std::lock_guard<std::mutex> lock(memory_stats_mutex_);
        return total_allocated_bytes_;
    }

    /**
     * @brief Gets the peak device memory usage in bytes.
     * @return Peak allocated bytes.
     */
    static size_t get_peak_allocated_bytes() {
        std::lock_guard<std::mutex> lock(memory_stats_mutex_);
        return peak_allocated_bytes_;
    }

    /**
     * @brief Updates memory statistics when memory is allocated.
     * @param bytes Number of bytes allocated.
     */
    static void track_allocation(size_t bytes) {
        std::lock_guard<std::mutex> lock(memory_stats_mutex_);
        current_allocated_bytes_ += bytes;
        total_allocated_bytes_ += bytes;
        if (current_allocated_bytes_ > peak_allocated_bytes_) {
            peak_allocated_bytes_ = current_allocated_bytes_;
        }
    }

    /**
     * @brief Updates memory statistics when memory is deallocated.
     * @param bytes Number of bytes deallocated.
     */
    static void track_deallocation(size_t bytes) {
        std::lock_guard<std::mutex> lock(memory_stats_mutex_);
        if (current_allocated_bytes_ >= bytes) {
            current_allocated_bytes_ -= bytes;
        }
    }

    /**
     * @brief Formats bytes into human-readable string (B, KB, MB, GB).
     * @param bytes Number of bytes to format.
     * @return Formatted string.
     */
    static std::string format_bytes(size_t bytes) {
        const char* units[] = {"B", "KB", "MB", "GB", "TB"};
        int unit_index = 0;
        double size = static_cast<double>(bytes);

        while (size >= 1024.0 && unit_index < 4) {
            size /= 1024.0;
            unit_index++;
        }

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << size << " " << units[unit_index];
        return oss.str();
    }

protected:
};

template <typename T>
class DeviceHostMemory : public CudaMemoryManager<T> {
public:
    DeviceHostMemory(size_t size, bool allocate_host_memory_in_advance = false)
        : CudaMemoryManager<T>(size),
          allocate_host_memory_in_advance(allocate_host_memory_in_advance) {
        allocate();
    }

    DeviceHostMemory(const std::vector<T>& vec)
        : CudaMemoryManager<T>(vec.size()),
          allocate_host_memory_in_advance(true) {
        allocate();
        std::copy(vec.begin(), vec.end(), this->host_ptr_);
    }

    void allocate() override {
        cudaError_t err;

        if (allocate_host_memory_in_advance) {
            this->host_bytes_ = this->size_ * sizeof(T);
            err = cudaMallocHost(&this->host_ptr_, this->host_bytes_);
            if (err != cudaSuccess) {
                std::string error_msg = "Failed to allocate host memory: " + std::string(cudaGetErrorString(err));
                THROW_EXCEPTION(error_msg);
            }
            memset(this->host_ptr_, 0, this->host_bytes_);
        }

        this->device_bytes_ = this->size_ * sizeof(T);
        err = cudaMalloc(&this->device_ptr_, this->device_bytes_);
        if (err != cudaSuccess) {
            // Get current memory statistics for error message
            size_t current_mem = this->get_current_allocated_bytes();
            std::ostringstream error_msg;
            error_msg << "cudaMalloc failed: " << cudaGetErrorString(err) << "\n"
                      << "  Attempted to allocate: " << this->format_bytes(this->device_bytes_) << "\n"
                      << "  Current allocated:     " << this->format_bytes(current_mem) << "\n"
                      << "  Total would be:        " << this->format_bytes(current_mem + this->device_bytes_);
            THROW_EXCEPTION(error_msg.str());
        }

        // Zero-initialize device memory (required by kernels that use atomicAdd)
        cudaMemset(this->device_ptr_, 0, this->device_bytes_);

        // Track successful allocation
        this->track_allocation(this->device_bytes_);
    }

    void toDevice() override {
        if (!this->device_ptr_) {
            allocate();
        }
        if (this->device_ptr_ && this->host_ptr_) {
            cudaMemcpy(this->device_ptr_, this->host_ptr_, this->size_ * sizeof(T), cudaMemcpyHostToDevice);
        }
    }

    void toHost() override {
        if (!this->host_ptr_) {
            this->host_bytes_ = this->size_ * sizeof(T);
            cudaError_t err = cudaMallocHost(&this->host_ptr_, this->host_bytes_);
            if (err != cudaSuccess) {
                std::ostringstream error_msg;
                error_msg << "cudaMallocHost failed: " << cudaGetErrorString(err) << "\n"
                          << "  Attempted to allocate: " << this->format_bytes(this->host_bytes_);
                THROW_EXCEPTION(error_msg.str());
            }
        }
        if (this->device_ptr_ && this->host_ptr_) {
            cudaMemcpy(this->host_ptr_, this->device_ptr_, this->size_ * sizeof(T), cudaMemcpyDeviceToHost);
        }
    }

    T& operator[](size_t index) {
        if (index >= this->size_) {
            THROW_EXCEPTION("Index out of bounds");
        }
        return this->host_ptr_[index];
    }

    const T& operator[](size_t index) const {
        if (index >= this->size_) {
            THROW_EXCEPTION("Index out of bounds");
        }
        return this->host_ptr_[index];
    }


private:
    bool allocate_host_memory_in_advance;
};



/**
 * @brief DeviceHostMatrix class for 2D array management using CUDA memory.
 *
 * This class manages a 2D array stored as a 1D contiguous array and utilizes
 * `DeviceHostMemory` for efficient memory management.
 *
 * @tparam T The type of elements stored in the matrix.
 * @details This class is a simple 2D matrix class that uses a 1D array to store
 */
template <typename T>
class DeviceHostMatrix {
private:
    size_t rows_; ///< Number of rows in the matrix
    size_t cols_; ///< Number of columns in the matrix
    DeviceHostMemory<T> memory_manager_; ///< Memory manager for underlying data

public:
    /**
     * @brief Constructs a Matrix with the given dimensions.
     *
     * The memory manager is responsible for allocating memory and managing data.
     *
     * @param rows Number of rows in the matrix.
     * @param cols Number of columns in the matrix.
     * @param allocate_host_memory_in_advance Allocate host memory in advance
     */
    DeviceHostMatrix(size_t rows, size_t cols, bool allocate_host_memory_in_advance=false)
        : rows_(rows), cols_(cols), memory_manager_(rows * cols, allocate_host_memory_in_advance) {
        if (rows == 0 || cols == 0) {
            THROW_EXCEPTION("Matrix dimensions must be greater than zero.");
        }
        // Note: memory_manager_.allocate() is called automatically in DeviceHostMemory constructor
    }

    /**
     * @brief Accesses an element of the matrix (host-side).
     *
     * Bounds checking is performed to ensure valid access.
     *
     * @param row Row index (0-based).
     * @param col Column index (0-based).
     * @return Reference to the element at the given position.
     * @throws std::out_of_range If the indices are out of bounds.
     */
    T& operator()(size_t row, size_t col) {
        if (row >= rows_ || col >= cols_) {
            THROW_EXCEPTION("Matrix indices are out of bounds.");
        }
        return memory_manager_.host_ptr()[row * cols_ + col];
    }

    /**
     * @brief Const version of the element access operator (host-side).
     *
     * @param row Row index (0-based).
     * @param col Column index (0-based).
     * @return Const reference to the element at the given position.
     */
    const T& operator()(size_t row, size_t col) const {
        if (row >= rows_ || col >= cols_) {
            THROW_EXCEPTION("Matrix indices are out of bounds.");
        }
        return memory_manager_.host_ptr()[row * cols_ + col];
    }

    

    /**
     * @brief Copies data from the host to the device memory.
     */
    void toDevice() {
        memory_manager_.toDevice();
    }

    /**
     * @brief Copies data from the device to the host memory.
     */
    void toHost() {
        memory_manager_.toHost();
    }

    /**
     * @brief Gets the number of rows in the matrix.
     * @return Number of rows.
     */
    size_t rows() const { return rows_; }

    /**
     * @brief Gets the number of columns in the matrix.
     * @return Number of columns.
     */
    size_t cols() const { return cols_; }

    /**
     * @brief Gets the device pointer to the matrix data.
     * @return Pointer to the device memory.
     */
    T* device_ptr() { return memory_manager_.device_ptr(); }

    /**
     * @brief Gets the const device pointer to the matrix data.
     * @return Const pointer to the device memory.
     */
    const T* device_ptr() const { return memory_manager_.device_ptr(); }

    /**
     * @brief Gets the host pointer to the matrix data.
     * @return Pointer to the host memory.
     */
    T* host_ptr() { return memory_manager_.host_ptr(); }

    /**
     * @brief Gets the const host pointer to the matrix data.
     * @return Const pointer to the host memory.
     */
    const T* host_ptr() const { return memory_manager_.host_ptr(); }

    /**
     * @brief Copy constructor (deleted).
     */
    DeviceHostMatrix(const DeviceHostMatrix&) = delete;

    /**
     * @brief Move constructor (deleted).
     */
    DeviceHostMatrix(DeviceHostMatrix&&) = delete;

    /**
     * @brief Copy assignment operator (deleted).
     */
    DeviceHostMatrix& operator=(const DeviceHostMatrix&) = delete;

    /**
     * @brief Move assignment operator (deleted).
     */
    DeviceHostMatrix& operator=(DeviceHostMatrix&&) = delete;

    /**
     * @brief Destructor.
     */
    ~DeviceHostMatrix() = default;


};

/**
 * @brief Sort the ERI indexes
 * @param i Index i
 * @param j Index j
 * @param k Index k
 * @param l Index l
 * @return Tuple of sorted indexes
 * @details This function sorts the ERI indexes (i,j,k,l) as unique indexes for (ij|kl).
 * @details \f$ i \le j,  k \le l, (i,j) \le (k, l) \f$ (dictionary order)
 */
 inline std::tuple<int, int, int, int> sort_eri_indexes(int i, int j, int k, int l){
    if(i > j) std::swap(i, j);
    if(k > l) std::swap(k, l);
    if(!(i<k or (i==k and j<=l))){
        std::swap(i, k);
        std::swap(j, l);
    }
    return std::make_tuple(i,j,k,l);
}

/**
 * @brief Get 1D index of 4D ERI indexes
 * @param i Index i
 * @param j Index j
 * @param k Index k
 * @param l Index l
 * @param num_basis Number of basis functions
 * @return 1D index
 * @details This function returns the 1D index of the 4D ERI indexes (ij|kl).
 */
inline int eri_1D_index(const int i, const int j, const int k, const int l, const int num_basis){
    auto get_index_2to1 = [](int const i, const int j, const int n){
        return j - static_cast<int>(i*(i-2*n+1)/2);
    };
    const auto [a,b,c,d] = sort_eri_indexes(i,j,k,l);
    const int bra = get_index_2to1(a, b, num_basis);
    const int ket = get_index_2to1(c, d, num_basis);
    return get_index_2to1(bra, ket, static_cast<int>(num_basis*(num_basis+1)/2));
}


/**
 * @brief DeviceHostERIMatrix class for ERI array management using CUDA memory.
 *
 * This class manages an ERI array stored as a 1D contiguous array and utilizes
 * `DeviceHostMemory` for efficient memory management.
 *
 * @tparam T The type of elements stored in the matrix.
 * @details This class is an ERI matrix class that uses a 1D array to store
 * @details Each element of the ERI matrix is a 4D indexed element of the electron repulsion integral (ij|kl) where i,j,k,l are the indices of the basis functions
 * @details The symmetry of the ERI matrix is exploited to reduce the number of elements stored in the matrix using (ij|kl) = (ji|kl) = (ij|lk) = (ji|lk) = (kl|ij) = (kl|ji) = (lk|ij) = (lk|ji)
 * @details Within the ERI matrix, the following conditions are satisfied: i <= j, k <= l, (i,j) <= (k,l)
            (i,j) <= (k,l) means that if i<k or (i==k and j<=l)
 */
 template <typename T>
 class DeviceHostERIMatrix {
 private:
    size_t num_basis_; ///< Number of basis functions in the ERI matrix
    size_t size_; ///< Number of elements in the ERI matrix
    DeviceHostMemory<T> memory_manager_; ///< Memory manager for underlying data
 
 public:
    /**
     * @brief Constructs a Matrix with the given dimensions.
     *
     * The memory manager is responsible for allocating memory and managing data.
     *
     * @param num_basis Number of basis functions
     */
    DeviceHostERIMatrix(size_t num_basis)
       : num_basis_(num_basis),
         size_(static_cast<int>(num_basis*(num_basis+1)*(num_basis*num_basis+num_basis+2)/8)),
         memory_manager_(size_, false) // do not allocate host memory in advance
    {
        // Note: memory_manager_.allocate() is called automatically in DeviceHostMemory constructor
    }

    /**
     * @brief compute the size of the ERI matrix
     * @return size of the ERI matrix
     */
    size_t size() const { return size_; }
 
     /**
      * @brief Accesses an element of the ERI matrix by 4D index (host-side).
      *
      * Bounds checking is performed to ensure valid access.
      *
      * @param i index of (ij|kl).
      * @param j index of (ij|kl).
      * @param k index of (ij|kl).
      * @param l index of (ij|kl).
      * @return Reference to the element at the given position.
      * @throws std::out_of_range If the indices are out of bounds.
      */
     T& operator()(size_t i, size_t j, size_t k, size_t l) {
        auto index = get_eri_1D_index(i, j, k, l);
        if(index >= size_){
            THROW_EXCEPTION("ERI indices are out of bounds.");
        }

        return memory_manager_.host_ptr()[index];
     }
     
    /**
     * @brief Accesses an element of the ERI matrix by 1D index (host-side).        
     * @param index 1D index of the ERI matrix
     * @return Reference to the element at the given position.
     * @throws std::out_of_range If the index is out of bounds.
     */
    T& operator[](size_t index) {
        if (index >= size_) {
            THROW_EXCEPTION("ERI index is out of bounds.");
        }
        return memory_manager_.host_ptr()[index];
    }

    /**
     * @brief Const version of the element access operator by 4D index(host-side).
     * @param i index of (ij|kl).
     * @param j index of (ij|kl).
     * @param k index of (ij|kl).
     * @param l index of (ij|kl).
     * @return Const reference to the element at the given position.
     * @throws std::out_of_range If the indices are out of bounds.
     */
    const T& operator()(size_t i, size_t j, size_t k, size_t l) const {
        auto index = get_eri_1D_index(i, j, k, l);
        if(index >= size_){
            THROW_EXCEPTION("ERI indices are out of bounds.");
        }
        return memory_manager_.host_ptr()[index];
    }

    /**
     * @brief Const version of the element access operator by 1D index (host-side).
     * @param index 1D index of the ERI matrix
     * @return Const reference to the element at the given position.
     * @throws std::out_of_range If the index is out of bounds.
     */
    const T& operator[](size_t index) const {
        if (index >= size_) {
            THROW_EXCEPTION("ERI index is out of bounds.");
        }
        return memory_manager_.host_ptr()[index];
    }

 
     /**
      * @brief Copies data from the host to the device memory.
      */
     void toDevice() {
         memory_manager_.toDevice();
     }
 
     /**
      * @brief Copies data from the device to the host memory.
      */
     void toHost() {
         memory_manager_.toHost();
     }
 
     /**
      * @brief Gets the device pointer to the matrix data.
      * @return Pointer to the device memory.
      */
     T* device_ptr() { return memory_manager_.device_ptr(); }
 
     /**
      * @brief Gets the const device pointer to the matrix data.
      * @return Const pointer to the device memory.
      */
     const T* device_ptr() const { return memory_manager_.device_ptr(); }
 
     /**
      * @brief Gets the host pointer to the matrix data.
      * @return Pointer to the host memory.
      */
     T* host_ptr() { return memory_manager_.host_ptr(); }
 
     /**
      * @brief Gets the const host pointer to the matrix data.
      * @return Const pointer to the host memory.
      */
     const T* host_ptr() const { return memory_manager_.host_ptr(); }
 
     /**
      * @brief Copy constructor (deleted).
      */
      DeviceHostERIMatrix(const DeviceHostERIMatrix&) = delete;
 
     /**
      * @brief Move constructor (deleted).
      */
      DeviceHostERIMatrix(DeviceHostERIMatrix&&) = delete;
 
     /**
      * @brief Copy assignment operator (deleted).
      */
      DeviceHostERIMatrix& operator=(const DeviceHostERIMatrix&) = delete;
 
     /**
      * @brief Move assignment operator (deleted).
      */
      DeviceHostERIMatrix& operator=(DeviceHostERIMatrix&&) = delete;
 
     /**
      * @brief Destructor.
      */
     ~DeviceHostERIMatrix() = default;
 
protected:
    /**
     * @brief Get 1D index of 4D ERI indexes
     * @param i Index i
     * @param j Index j
     * @param k Index k
     * @param l Index l
     * @return 1D index
     * @details This function returns the 1D index of the 4D ERI indexes (i,j,k,l).
     */
    int get_eri_1D_index(const int i, const int j, const int k, const int l) const{
        return eri_1D_index(i, j, k, l, num_basis_);
    }

 };


// ========== Memory Tracking Wrapper Functions ==========

/**
 * @brief Global map to track allocated memory sizes
 *
 * This map stores the size of each allocated memory block, keyed by the pointer address.
 * This is necessary because cudaFree doesn't provide size information.
 */
inline std::unordered_map<void*, size_t> g_allocated_memory_map;
inline std::mutex g_allocated_memory_map_mutex;

/**
 * @brief Wrapper for cudaMalloc with memory statistics tracking.
 *
 * This function allocates device memory and automatically tracks the allocation
 * in the global memory statistics.
 *
 * @tparam T The type of the pointer
 * @param ptr Pointer to the device pointer
 * @param size Number of bytes to allocate
 * @return cudaError_t Error code from cudaMalloc
 */
template<typename T>
inline cudaError_t tracked_cudaMalloc(T** ptr, size_t size) {
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(ptr), size);

    if (err == cudaSuccess) {
        // Zero-initialize device memory to prevent stale data contamination
        cudaMemset(*ptr, 0, size);

        // Track allocation in map
        {
            std::lock_guard<std::mutex> lock(g_allocated_memory_map_mutex);
            g_allocated_memory_map[*ptr] = size;
        }

        // Update memory statistics
        CudaMemoryManager<T>::track_allocation(size);
    } else {
        // Print detailed error message and throw exception to prevent use of null pointer
        size_t current_mem = CudaMemoryManager<T>::get_current_allocated_bytes();
        std::ostringstream oss;
        oss << "tracked_cudaMalloc failed: " << cudaGetErrorString(err) << "\n"
            << "  Attempted to allocate: " << CudaMemoryManager<T>::format_bytes(size) << "\n"
            << "  Current allocated:     " << CudaMemoryManager<T>::format_bytes(current_mem) << "\n"
            << "  Total would be:        " << CudaMemoryManager<T>::format_bytes(current_mem + size);
        throw std::runtime_error(oss.str());
    }

    return err;
}

/**
 * @brief Wrapper for cudaFree with memory statistics tracking.
 *
 * This function frees device memory and automatically updates the global
 * memory statistics.
 *
 * @param ptr Pointer to the device memory to free
 * @return cudaError_t Error code from cudaFree
 */
inline cudaError_t tracked_cudaFree(void* ptr) {
    if (ptr == nullptr) {
        return cudaSuccess;
    }

    // Get the size of the allocation
    size_t size = 0;
    {
        std::lock_guard<std::mutex> lock(g_allocated_memory_map_mutex);
        auto it = g_allocated_memory_map.find(ptr);
        if (it != g_allocated_memory_map.end()) {
            size = it->second;
            g_allocated_memory_map.erase(it);
        }
    }

    // Free the memory
    cudaError_t err = cudaFree(ptr);

    // Update statistics if free was successful and size was tracked
    if (err == cudaSuccess && size > 0) {
        CudaMemoryManager<double>::track_deallocation(size);
    }

    return err;
}

/**
 * @brief Wrapper for cudaMallocAsync with memory statistics tracking.
 *
 * This function allocates device memory asynchronously and automatically tracks
 * the allocation in the global memory statistics.
 *
 * @tparam T The type of the pointer
 * @param ptr Pointer to the device pointer
 * @param size Number of bytes to allocate
 * @param stream CUDA stream for asynchronous allocation
 * @return cudaError_t Error code from cudaMallocAsync
 */
template<typename T>
inline cudaError_t tracked_cudaMallocAsync(T** ptr, size_t size, cudaStream_t stream) {
    cudaError_t err = cudaMallocAsync(reinterpret_cast<void**>(ptr), size, stream);

    if (err == cudaSuccess) {
        // Track allocation in map
        {
            std::lock_guard<std::mutex> lock(g_allocated_memory_map_mutex);
            g_allocated_memory_map[*ptr] = size;
        }

        // Update memory statistics
        CudaMemoryManager<T>::track_allocation(size);
    } else {
        // Print detailed error message with current memory statistics
        size_t current_mem = CudaMemoryManager<T>::get_current_allocated_bytes();
        std::cerr << "tracked_cudaMallocAsync failed: " << cudaGetErrorString(err) << "\n"
                  << "  Attempted to allocate: " << CudaMemoryManager<T>::format_bytes(size) << "\n"
                  << "  Current allocated:     " << CudaMemoryManager<T>::format_bytes(current_mem) << "\n"
                  << "  Total would be:        " << CudaMemoryManager<T>::format_bytes(current_mem + size) << std::endl;
    }

    return err;
}

/**
 * @brief Wrapper for cudaFreeAsync with memory statistics tracking.
 *
 * This function frees device memory asynchronously and automatically updates
 * the global memory statistics.
 *
 * @param ptr Pointer to the device memory to free
 * @param stream CUDA stream for asynchronous deallocation
 * @return cudaError_t Error code from cudaFreeAsync
 */
inline cudaError_t tracked_cudaFreeAsync(void* ptr, cudaStream_t stream) {
    if (ptr == nullptr) {
        return cudaSuccess;
    }

    // Get the size of the allocation
    size_t size = 0;
    {
        std::lock_guard<std::mutex> lock(g_allocated_memory_map_mutex);
        auto it = g_allocated_memory_map.find(ptr);
        if (it != g_allocated_memory_map.end()) {
            size = it->second;
            g_allocated_memory_map.erase(it);
        }
    }

    // Free the memory asynchronously
    cudaError_t err = cudaFreeAsync(ptr, stream);

    // Update statistics if free was successful and size was tracked
    if (err == cudaSuccess && size > 0) {
        CudaMemoryManager<double>::track_deallocation(size);
    }

    return err;
}



} // namespace gansu