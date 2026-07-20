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

#ifdef GANSU_CPU_ONLY
#include "cuda_compat.hpp"
#else
#include <cuda_runtime.h>
#endif
#include <algorithm>
#include <array>
#include <iostream>
#include <map>
#include <stdexcept>
#include <cstring>
#include <mutex>
#include <iomanip>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "utils.hpp" // THROW_EXCEPTION

// Forward declaration for gpu_available() — avoids circular include with gpu_manager.hpp
namespace gansu::gpu { bool gpu_available(); }

/**
 * @brief Safe memcpy that works in both GPU and CPU modes.
 *
 * When GPU is not available, device_ptr is actually a host pointer (from calloc),
 * so CUDA's cudaMemcpy would fail. This transparently falls back to std::memcpy.
 *
 * All code should call cudaMemcpy as usual — the macro below redirects it here
 * so no source changes are needed in individual .cu files.
 */
namespace gansu::detail {
    // Store pointers to the real CUDA functions before macro override.
    // In CPU-only builds, these point to the stubs in cuda_compat.hpp.
    inline auto real_cudaMemcpy = ::cudaMemcpy;
    inline auto real_cudaMemcpyAsync = ::cudaMemcpyAsync;
    inline auto real_cudaMemset = ::cudaMemset;
    // cudaMalloc has both a void** overload (C runtime) and a template T**
    // overload (cudart C++ wrapper). Use a small lambda thunk that always
    // dispatches to ::cudaMalloc to keep the signature stable after macro
    // redirect. Same for cudaFree.
    inline auto real_cudaMalloc = [](void** p, size_t n) { return ::cudaMalloc(p, n); };
    inline auto real_cudaFree   = [](void* p) { return ::cudaFree(p); };
}

/**
 * @brief Safe memcpy/memset that work in both GPU and CPU fallback modes.
 *
 * When GPU is not available, device_ptr is actually a host pointer (from calloc),
 * so CUDA's cudaMemcpy would fail on those addresses. These wrappers transparently
 * fall back to std::memcpy/std::memset.
 *
 * The macros below redirect all existing cudaMemcpy/cudaMemset calls through these
 * wrappers, so no per-file source changes are needed.
 */
inline cudaError_t gansu_memcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind = cudaMemcpyDefault) {
    if (gansu::gpu::gpu_available()) {
        return gansu::detail::real_cudaMemcpy(dst, src, count, kind);
    } else {
        std::memcpy(dst, src, count);
        return cudaSuccess;
    }
}

inline cudaError_t gansu_memcpy_async(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
    if (gansu::gpu::gpu_available()) {
        return gansu::detail::real_cudaMemcpyAsync(dst, src, count, kind, stream);
    } else {
        std::memcpy(dst, src, count);
        return cudaSuccess;
    }
}

inline cudaError_t gansu_memset(void* ptr, int value, size_t count) {
    if (gansu::gpu::gpu_available()) {
        return gansu::detail::real_cudaMemset(ptr, value, count);
    } else {
        std::memset(ptr, value, count);
        return cudaSuccess;
    }
}

inline cudaError_t gansu_memGetInfo(size_t* free, size_t* total) {
    if (gansu::gpu::gpu_available()) {
        return gansu::detail::real_cudaMemcpy ? // just a non-null check to use real CUDA
            ::cudaMemGetInfo(free, total) : cudaSuccess;
    }
    // CPU mode: report large available memory (no GPU memory limit)
    if (free)  *free  = (size_t)64 * 1024 * 1024 * 1024; // 64 GB
    if (total) *total = (size_t)64 * 1024 * 1024 * 1024;
    return cudaSuccess;
}

// Override CUDA memory functions globally so all existing code is safe in CPU mode.
// real_cudaMemcpy etc. captured above still point to the originals.
#define cudaMemcpy gansu_memcpy
#define cudaMemcpyAsync gansu_memcpy_async
#define cudaMemset gansu_memset
#define cudaMemGetInfo gansu_memGetInfo

namespace gansu{


/**
 * @brief Global (type-independent) GPU memory tracker.
 *
 * CudaMemoryManager<T> tracks memory per type T, so allocations of different types
 * (e.g., unsigned long long vs double) are counted separately. This global tracker
 * aggregates all allocations regardless of type to report correct total/peak memory.
 *
 * Multi-GPU: `*_by_device_` maps store per-device statistics. Allocation/deallocation
 * is associated with a device id captured at tracked_cudaMalloc time (cudaGetDevice).
 * The flat `current_bytes_/peak_bytes_/total_bytes_` fields remain the cross-device
 * totals so callers that don't care about per-GPU breakdown keep working.
 */
struct GlobalGpuMemoryTracker {
    inline static size_t current_bytes_ = 0;
    inline static size_t total_bytes_ = 0;
    inline static size_t peak_bytes_ = 0;
    inline static std::unordered_map<int, size_t> current_by_device_;
    inline static std::unordered_map<int, size_t> total_by_device_;
    inline static std::unordered_map<int, size_t> peak_by_device_;
    inline static std::mutex mutex_;

    static void track_allocation(size_t bytes, int device = -1) {
        std::lock_guard<std::mutex> lock(mutex_);
        current_bytes_ += bytes;
        total_bytes_ += bytes;
        if (current_bytes_ > peak_bytes_) peak_bytes_ = current_bytes_;
        if (device >= 0) {
            auto& cur = current_by_device_[device];
            cur += bytes;
            total_by_device_[device] += bytes;
            auto& pk = peak_by_device_[device];
            if (cur > pk) pk = cur;
        }
    }
    static void track_deallocation(size_t bytes, int device = -1) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (current_bytes_ >= bytes) current_bytes_ -= bytes;
        if (device >= 0) {
            auto it = current_by_device_.find(device);
            if (it != current_by_device_.end()) {
                if (it->second >= bytes) it->second -= bytes;
                else it->second = 0;
            }
        }
    }
    static size_t get_current() { std::lock_guard<std::mutex> lock(mutex_); return current_bytes_; }
    static size_t get_peak() { std::lock_guard<std::mutex> lock(mutex_); return peak_bytes_; }
    static size_t get_total() { std::lock_guard<std::mutex> lock(mutex_); return total_bytes_; }

    /// Snapshot of (current, total, peak) keyed by device id. Only devices
    /// that saw at least one tracked allocation are present.
    static std::unordered_map<int, std::array<size_t, 3>> get_per_device_snapshot() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::unordered_map<int, std::array<size_t, 3>> out;
        for (const auto& kv : peak_by_device_) {
            const int dev = kv.first;
            size_t cur = 0, tot = 0;
            auto cit = current_by_device_.find(dev);
            if (cit != current_by_device_.end()) cur = cit->second;
            auto tit = total_by_device_.find(dev);
            if (tit != total_by_device_.end()) tot = tit->second;
            out[dev] = {cur, tot, kv.second};
        }
        return out;
    }
};

/**
 * @brief Per-pointer registry for CudaMemoryManager-backed device buffers
 *        (DeviceHostMemory / DeviceHostMatrix).
 *
 * These do NOT go through tracked_cudaMalloc (they call real_cudaMalloc + a
 * separate track_allocation), so they are absent from g_allocated_memory_map.
 * The big RI 3-center tensor (ERI::intermediate_matrix_B_) is one of these.
 * Registering them here lets the measure-first dump (dump_tracked_allocations)
 * attribute large per-device totals to specific buffers (e.g. a 4-way B replica).
 * Stores (size_bytes, device_id) keyed by device pointer. Behaviour-inert.
 */
inline std::unordered_map<void*, std::pair<size_t, int>> g_cmm_allocated_memory_map;
inline std::mutex g_cmm_allocated_memory_map_mutex;


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
    int    device_id_ = -1; ///< Owning GPU id captured at allocate(); -1 for CPU mode / unknown

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
        if (gpu::gpu_available()) {
            if (device_ptr_) {
                track_deallocation(device_bytes_, device_id_, device_ptr_);
                gansu::detail::real_cudaFree(device_ptr_);
            }
            if (host_ptr_) {
                cudaFreeHost(host_ptr_);
            }
        } else {
            // CPU mode: device_ptr_ == host_ptr_, free once
            if (host_ptr_) {
                track_deallocation(device_bytes_, device_id_, device_ptr_);
                std::free(host_ptr_);
            }
            // Don't free device_ptr_ separately (same pointer)
        }
        device_ptr_ = nullptr;
        host_ptr_ = nullptr;
    }

    /**
     * @brief Release all allocated memory early (before destructor).
     * Safe to call multiple times; subsequent calls are no-ops.
     */
    void release() {
        if (gpu::gpu_available()) {
            if (device_ptr_) { track_deallocation(device_bytes_, device_id_, device_ptr_); gansu::detail::real_cudaFree(device_ptr_); }
            if (host_ptr_) cudaFreeHost(host_ptr_);
        } else {
            if (host_ptr_) { track_deallocation(device_bytes_, device_id_, device_ptr_); std::free(host_ptr_); }
        }
        device_ptr_ = nullptr;
        host_ptr_ = nullptr;
        size_ = 0;
        device_bytes_ = 0;
        host_bytes_ = 0;
        device_id_ = -1;
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
        std::cout << std::endl;
        std::cout << "[Device Memory Statistics]" << std::endl;
        // Per-device breakdown (only devices that saw a tracked allocation).
        const auto per_dev = GlobalGpuMemoryTracker::get_per_device_snapshot();
        if (per_dev.size() > 1) {
            std::vector<int> devs;
            devs.reserve(per_dev.size());
            for (const auto& kv : per_dev) devs.push_back(kv.first);
            std::sort(devs.begin(), devs.end());
            for (int dev : devs) {
                const auto& s = per_dev.at(dev);   // {current, total, peak}
                std::cout << "  GPU " << dev
                          << ":  current=" << format_bytes(s[0])
                          << "  total=" << format_bytes(s[1])
                          << "  peak=" << format_bytes(s[2])
                          << std::endl;
            }
            std::cout << "  All GPUs (sum):" << std::endl;
        }
        std::cout << "Current allocated: " << format_bytes(GlobalGpuMemoryTracker::get_current()) << std::endl;
        std::cout << "Total allocated: " << format_bytes(GlobalGpuMemoryTracker::get_total()) << std::endl;
        std::cout << "Peak usage: " << format_bytes(GlobalGpuMemoryTracker::get_peak()) << std::endl;
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
        return GlobalGpuMemoryTracker::get_peak();
    }

    /**
     * @brief Updates memory statistics when memory is allocated.
     * @param bytes Number of bytes allocated.
     * @param device Owning device id (-1 for CPU mode / unknown).
     */
    static void track_allocation(size_t bytes, int device = -1, void* ptr = nullptr) {
        {
            std::lock_guard<std::mutex> lock(memory_stats_mutex_);
            current_allocated_bytes_ += bytes;
            total_allocated_bytes_ += bytes;
            if (current_allocated_bytes_ > peak_allocated_bytes_) {
                peak_allocated_bytes_ = current_allocated_bytes_;
            }
        }
        GlobalGpuMemoryTracker::track_allocation(bytes, device);
        // Register CMM-backed buffers (ptr supplied only by DeviceHostMemory::
        // allocate) so the diagnostic dump can attribute them. The raw
        // tracked_cudaMalloc path calls this with ptr==nullptr (already tracked
        // in g_allocated_memory_map) ⇒ no double counting.
        if (ptr) {
            std::lock_guard<std::mutex> lock(g_cmm_allocated_memory_map_mutex);
            g_cmm_allocated_memory_map[ptr] = {bytes, device};
        }
    }

    /**
     * @brief Updates memory statistics when memory is deallocated.
     * @param bytes Number of bytes deallocated.
     * @param device Owning device id (captured at alloc time, -1 if unknown).
     */
    static void track_deallocation(size_t bytes, int device = -1, void* ptr = nullptr) {
        {
            std::lock_guard<std::mutex> lock(memory_stats_mutex_);
            if (current_allocated_bytes_ >= bytes) {
                current_allocated_bytes_ -= bytes;
            }
        }
        GlobalGpuMemoryTracker::track_deallocation(bytes, device);
        if (ptr) {
            std::lock_guard<std::mutex> lock(g_cmm_allocated_memory_map_mutex);
            g_cmm_allocated_memory_map.erase(ptr);
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
        if (!gpu::gpu_available()) {
            // CPU mode: single allocation, device_ptr == host_ptr
            this->host_bytes_ = this->size_ * sizeof(T);
            this->device_bytes_ = this->host_bytes_;
            this->host_ptr_ = static_cast<T*>(std::calloc(this->size_, sizeof(T)));
            if (!this->host_ptr_) {
                THROW_EXCEPTION("Failed to allocate host memory (CPU mode)");
            }
            this->device_ptr_ = this->host_ptr_; // Same pointer
            this->device_id_ = -1;
            this->track_allocation(this->device_bytes_, -1, this->device_ptr_);
            return;
        }

        // GPU mode: separate host and device allocations
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
        err = gansu::detail::real_cudaMalloc(reinterpret_cast<void**>(&this->device_ptr_), this->device_bytes_);
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

        // Track successful allocation (capture owning device for per-GPU stats)
        cudaGetDevice(&this->device_id_);
        this->track_allocation(this->device_bytes_, this->device_id_, this->device_ptr_);
    }

    void toDevice() override {
        if (!gpu::gpu_available()) return; // CPU mode: no-op (same pointer)
        if (!this->device_ptr_) {
            allocate();
        }
        if (this->device_ptr_ && this->host_ptr_) {
            cudaMemcpy(this->device_ptr_, this->host_ptr_, this->size_ * sizeof(T), cudaMemcpyHostToDevice);
        }
    }

    void toHost() override {
        if (!gpu::gpu_available()) return; // CPU mode: no-op (same pointer)
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

    /// Release all memory early (before destructor).
    void release() { memory_manager_.release(); rows_ = 0; cols_ = 0; }

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
 * @brief Global map to track allocated memory sizes (and owning device).
 *
 * Stores (size, device_id) for each allocated pointer. Required because cudaFree
 * doesn't provide either; the device id is captured at allocation time via
 * cudaGetDevice() so per-device stats stay correct under multi-GPU OpenMP.
 */
inline std::unordered_map<void*, std::pair<size_t, int>> g_allocated_memory_map;
inline std::mutex g_allocated_memory_map_mutex;

/**
 * @brief Diagnostic dump of the live tracked device allocations (measure-first).
 *
 * Walks g_allocated_memory_map and prints, for a given checkpoint label:
 *   - global + per-device current live bytes,
 *   - the largest individual allocations (size + owning device),
 *   - allocations grouped by identical byte-size (count × size per device) so
 *     replicated buffers stand out at a glance (e.g. a 4-way B replica appears
 *     as "4 × 17.60 GB" rather than four separate lines).
 *
 * Behaviour-inert (read-only). Intended to be called from env-gated checkpoints
 * (e.g. GANSU_STEOM_MEM_DUMP) to find what occupies the device just before a
 * large allocation fails — e.g. the STEOM dense-geev buffers at total_dim≈22200.
 *
 * @param label   Checkpoint name printed in the header.
 * @param top_k   Number of largest individual allocations to list (default 25).
 */
inline void dump_tracked_allocations(const char* label, int top_k = 25) {
    auto fmt = [](size_t bytes) {
        const char* units[] = {"B", "KB", "MB", "GB", "TB"};
        int u = 0; double s = static_cast<double>(bytes);
        while (s >= 1024.0 && u < 4) { s /= 1024.0; ++u; }
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << s << " " << units[u];
        return oss.str();
    };

    // Snapshot both registries under their locks, then format without holding.
    // Each entry: (size, device, is_cmm). "raw" = tracked_cudaMalloc path,
    // "CMM" = DeviceHostMemory/Matrix (e.g. the RI B tensor).
    struct Alloc { size_t size; int dev; bool cmm; };
    std::vector<Alloc> allocs;
    size_t raw_n = 0, cmm_n = 0;
    {
        std::lock_guard<std::mutex> lock(g_allocated_memory_map_mutex);
        raw_n = g_allocated_memory_map.size();
        allocs.reserve(raw_n);
        for (const auto& kv : g_allocated_memory_map)
            allocs.push_back({kv.second.first, kv.second.second, false});
    }
    {
        std::lock_guard<std::mutex> lock(g_cmm_allocated_memory_map_mutex);
        cmm_n = g_cmm_allocated_memory_map.size();
        allocs.reserve(allocs.size() + cmm_n);
        for (const auto& kv : g_cmm_allocated_memory_map)
            allocs.push_back({kv.second.first, kv.second.second, true});
    }

    std::cout << "\n=== [tracked-mem dump] " << label << " ===" << std::endl;
    std::cout << "  live allocations: " << allocs.size()
              << " (raw=" << raw_n << ", CMM=" << cmm_n << ")"
              << "   global current: "
              << fmt(GlobalGpuMemoryTracker::get_current()) << std::endl;
    const auto per_dev = GlobalGpuMemoryTracker::get_per_device_snapshot();
    {
        std::vector<int> devs;
        for (const auto& kv : per_dev) devs.push_back(kv.first);
        std::sort(devs.begin(), devs.end());
        for (int d : devs)
            std::cout << "    GPU " << d << ": current=" << fmt(per_dev.at(d)[0])
                      << "  peak=" << fmt(per_dev.at(d)[2]) << std::endl;
    }

    // Largest individual allocations.
    std::sort(allocs.begin(), allocs.end(),
              [](const Alloc& a, const Alloc& b) { return a.size > b.size; });
    const int klist = std::min<int>(top_k, static_cast<int>(allocs.size()));
    std::cout << "  largest " << klist << " allocations:" << std::endl;
    for (int i = 0; i < klist; ++i)
        std::cout << "    [" << std::setw(2) << i << "] " << std::setw(10)
                  << fmt(allocs[i].size) << "  (GPU " << allocs[i].dev << ", "
                  << (allocs[i].cmm ? "CMM" : "raw") << ")" << std::endl;

    // Group by identical byte-size to expose replicated buffers. Only sizes that
    // appear more than once OR are ≥ 256 MB are reported (keeps it short).
    std::map<size_t, std::map<int, int>> by_size;  // size -> (device -> count)
    for (const auto& a : allocs) by_size[a.size][a.dev]++;
    std::cout << "  grouped (count × size, ≥256 MB or replicated):" << std::endl;
    // Iterate largest-size first.
    for (auto it = by_size.rbegin(); it != by_size.rend(); ++it) {
        const size_t sz = it->first;
        int total_count = 0;
        for (const auto& dc : it->second) total_count += dc.second;
        if (sz < (size_t)256 * 1024 * 1024 && total_count <= 1) continue;
        std::cout << "    " << std::setw(10) << fmt(sz) << "  × " << total_count
                  << " = " << fmt(sz * (size_t)total_count) << "   [";
        bool first = true;
        for (const auto& dc : it->second) {
            std::cout << (first ? "" : ", ") << "GPU" << dc.first << "×" << dc.second;
            first = false;
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "=== [tracked-mem dump end] " << label << " ===\n" << std::endl;
}

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
    if (!gpu::gpu_available()) {
        // CPU mode: use calloc (zero-initialized, matching cudaMemset behavior)
        *ptr = static_cast<T*>(std::calloc(1, size));
        if (!*ptr) {
            throw std::runtime_error("tracked_cudaMalloc: CPU calloc failed (size=" + std::to_string(size) + ")");
        }
        { std::lock_guard<std::mutex> lock(g_allocated_memory_map_mutex);
          g_allocated_memory_map[*ptr] = {size, -1}; }
        CudaMemoryManager<T>::track_allocation(size, -1);
        return cudaSuccess;
    }

    // Use real_cudaMalloc to avoid recursion via the cudaMalloc → tracked_cudaMalloc
    // macro redirect installed at the end of this header.
    cudaError_t err = gansu::detail::real_cudaMalloc(reinterpret_cast<void**>(ptr), size);

    if (err == cudaSuccess) {
        // Zero-initialize device memory to prevent stale data contamination
        cudaMemset(*ptr, 0, size);

        // Capture device id at alloc time so the matching free credits the
        // right per-device counter (cudaFree itself can be called from any
        // thread/device but the bytes belong to the alloc-time device).
        int dev = -1;
        cudaGetDevice(&dev);

        // Track allocation in map
        {
            std::lock_guard<std::mutex> lock(g_allocated_memory_map_mutex);
            g_allocated_memory_map[*ptr] = {size, dev};
        }

        // Update memory statistics
        CudaMemoryManager<T>::track_allocation(size, dev);
    } else {
        size_t current_mem = GlobalGpuMemoryTracker::get_current();
        // (OOM diagnostic) dump the live tracked allocations so the failing site's
        // memory context (which device, what else is resident) is visible. This is
        // the fatal crash path, so the extra output is strictly helpful.
        dump_tracked_allocations("OOM failure (tracked_cudaMalloc)");
        std::ostringstream oss;
        oss << "tracked_cudaMalloc failed: " << cudaGetErrorString(err) << "\n"
            << "  Attempted to allocate: " << CudaMemoryManager<T>::format_bytes(size) << "\n"
            << "  Current allocated (global): " << CudaMemoryManager<T>::format_bytes(current_mem) << "\n"
            << "  Total would be:             " << CudaMemoryManager<T>::format_bytes(current_mem + size);
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

    // Get the size and owning device of the allocation
    size_t size = 0;
    int    dev  = -1;
    {
        std::lock_guard<std::mutex> lock(g_allocated_memory_map_mutex);
        auto it = g_allocated_memory_map.find(ptr);
        if (it != g_allocated_memory_map.end()) {
            size = it->second.first;
            dev  = it->second.second;
            g_allocated_memory_map.erase(it);
        }
    }

    if (!gpu::gpu_available()) {
        // CPU mode: memory was allocated with calloc
        std::free(ptr);
    } else {
        gansu::detail::real_cudaFree(ptr);
    }

    if (size > 0) {
        GlobalGpuMemoryTracker::track_deallocation(size, dev);
    }

    return cudaSuccess;
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
    if (!gpu::gpu_available()) {
        *ptr = static_cast<T*>(std::calloc(1, size));
        if (!*ptr) throw std::runtime_error("tracked_cudaMallocAsync: CPU calloc failed");
        { std::lock_guard<std::mutex> lock(g_allocated_memory_map_mutex);
          g_allocated_memory_map[*ptr] = {size, -1}; }
        CudaMemoryManager<T>::track_allocation(size, -1);
        return cudaSuccess;
    }

    cudaError_t err = cudaMallocAsync(reinterpret_cast<void**>(ptr), size, stream);

    if (err == cudaSuccess) {
        int dev = -1;
        cudaGetDevice(&dev);
        {
            std::lock_guard<std::mutex> lock(g_allocated_memory_map_mutex);
            g_allocated_memory_map[*ptr] = {size, dev};
        }
        CudaMemoryManager<T>::track_allocation(size, dev);
    } else {
        size_t current_mem = GlobalGpuMemoryTracker::get_current();
        std::cerr << "tracked_cudaMallocAsync failed: " << cudaGetErrorString(err) << "\n"
                  << "  Attempted to allocate: " << CudaMemoryManager<T>::format_bytes(size) << "\n"
                  << "  Current allocated (global): " << CudaMemoryManager<T>::format_bytes(current_mem) << "\n"
                  << "  Total would be:             " << CudaMemoryManager<T>::format_bytes(current_mem + size) << std::endl;
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

    size_t size = 0;
    int    dev  = -1;
    {
        std::lock_guard<std::mutex> lock(g_allocated_memory_map_mutex);
        auto it = g_allocated_memory_map.find(ptr);
        if (it != g_allocated_memory_map.end()) {
            size = it->second.first;
            dev  = it->second.second;
            g_allocated_memory_map.erase(it);
        }
    }

    if (!gpu::gpu_available()) {
        std::free(ptr);
    } else {
        cudaFreeAsync(ptr, stream);
    }

    if (size > 0) {
        GlobalGpuMemoryTracker::track_deallocation(size, dev);
    }

    return cudaSuccess;
}



} // namespace gansu

// Globally redirect plain cudaMalloc/cudaFree to the tracked wrappers so that
// per-GPU memory statistics capture allocations from every translation unit
// (incl. files using raw cudaMalloc such as eri_ri_distributed.cu). The
// internal callers inside this header use gansu::detail::real_cudaMalloc and
// real_cudaFree directly so they bypass the macro and avoid recursion. These
// macros are purely textual and only match the exact tokens `cudaMalloc` /
// `cudaFree` — cudaMallocHost, cudaMallocAsync, cudaFreeHost, cudaFreeAsync
// are different identifiers and are unaffected.
#define cudaMalloc gansu::tracked_cudaMalloc
#define cudaFree gansu::tracked_cudaFree