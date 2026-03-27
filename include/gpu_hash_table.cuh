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



#pragma once

#include "types.hpp"
#include "utils_cuda.hpp"

namespace gansu::gpu {

constexpr unsigned long long EMPTY_KEY = 0xFFFFFFFFFFFFFFFFULL;

/**
 * @brief Encode 4 basis indices into a 64-bit key (16 bits per index)
 */
__device__ __host__ inline unsigned long long encode_eri_key(
    size_t i, size_t j, size_t k, size_t l)
{
    return (static_cast<unsigned long long>(i) << 48) |
           (static_cast<unsigned long long>(j) << 32) |
           (static_cast<unsigned long long>(k) << 16) |
           static_cast<unsigned long long>(l);
}

/**
 * @brief Canonicalize 4 indices using 8-fold ERI symmetry and encode as 64-bit key
 * @details Uses sort_eri_index to enforce i<=j, k<=l, (i,j)<=(k,l)
 */
__device__ inline unsigned long long canonical_eri_key(size_t i, size_t j, size_t k, size_t l)
{
    size_t4 sorted = sort_eri_index(i, j, k, l);
    return encode_eri_key(sorted.x, sorted.y, sorted.z, sorted.w);
}

/**
 * @brief Hash function based on Murmur3 finalizer
 * @param key 64-bit key
 * @param capacity_mask capacity - 1 (capacity must be power of 2)
 * @return hash slot index
 */
__device__ inline size_t hash_function(unsigned long long key, size_t capacity_mask)
{
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    return static_cast<size_t>(key) & capacity_mask;
}

/**
 * @brief Insert a key-value pair into the hash table (GPU device function)
 * @details Uses atomicCAS for key insertion and atomicAdd for value accumulation.
 *          Multiple primitive shell contributions to the same contracted ERI
 *          are accumulated via atomicAdd.
 */
__device__ inline void hash_insert(
    unsigned long long* keys, real_t* values,
    size_t capacity_mask, unsigned long long key, real_t value)
{
    size_t slot = hash_function(key, capacity_mask);
    while (true) {
        unsigned long long prev = atomicCAS(&keys[slot], EMPTY_KEY, key);
        if (prev == EMPTY_KEY || prev == key) {
            atomicAdd(&values[slot], value);
            return;
        }
        slot = (slot + 1) & capacity_mask; // linear probing
    }
}

/**
 * @brief Look up a value in the hash table (GPU device function)
 * @details Returns 0.0 if the key is not found (integral is negligible)
 */
__device__ inline real_t hash_lookup(
    const unsigned long long* keys, const real_t* values,
    size_t capacity_mask, unsigned long long key)
{
    size_t slot = hash_function(key, capacity_mask);
    while (true) {
        unsigned long long k = keys[slot];
        if (k == key) return values[slot];
        if (k == EMPTY_KEY) return 0.0;
        slot = (slot + 1) & capacity_mask;
    }
}

/**
 * @brief Decode a 64-bit key back to 4 basis indices
 */
__device__ inline void decode_eri_key(unsigned long long key,
    int& a, int& b, int& c, int& d)
{
    a = static_cast<int>((key >> 48) & 0xFFFF);
    b = static_cast<int>((key >> 32) & 0xFFFF);
    c = static_cast<int>((key >> 16) & 0xFFFF);
    d = static_cast<int>(key & 0xFFFF);
}

} // namespace gansu::gpu
