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

#include <chrono>
#include <iostream>
#include <string>
#include <unordered_map>
#include <mutex>
#include <vector>

namespace gansu {

#ifdef PROFILE_ENABLED

/**
 * @class GlobalProfiler
 * @brief A singleton class for profiling the execution time of functions and reporting results.
 */
class GlobalProfiler {
public:
    /**
     * @brief Initializes the profiler by recording the program start time.
     */
    static void initialize();

    /**
     * @brief Starts timing for a given function or section of code.
     * @param name The name of the function or section being profiled.
     */
    static void start(const std::string& name);

    /**
     * @brief Stops timing for a given function or section of code.
     * @param name The name of the function or section being profiled.
     */
    static void stop(const std::string& name);

    /**
     * @brief Outputs a summary of all recorded execution times.
     */
    static void report();

    /**
     * @brief Display elapsed time
     */
    static void displayElapsedTime(const std::string& label = "");

private:
    static std::chrono::high_resolution_clock::time_point programStartTime; ///< Program start time
    static std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> startTimes; ///< Start times for each function
    static std::unordered_map<std::string, long long> executionTimes; ///< Accumulated execution times for each function
    static std::unordered_map<std::string, int> callCounts;  // Map to store call counts for each function
    static std::vector<std::string> executionOrder; ///< Maintain order of execution
    static std::mutex mutex_; ///< Mutex for thread safety
};

/**
 * @class ScopedTimer
 * @brief Automatically records the execution time of a scope using RAII.
 */
class ScopedTimer {
public:
    /**
     * @brief Constructs a ScopedTimer and starts timing.
     * @param name The name of the function or section being profiled.
     */
    explicit ScopedTimer(const std::string& name);

    /**
     * @brief Destructs the ScopedTimer and stops timing.
     */
    ~ScopedTimer();

private:
    std::string functionName; ///< The name of the function or section being profiled
};

/**
 * @def PROFILE_FUNCTION
 * @brief Macro to easily profile a function's execution time.
 * 
 * Place this macro at the beginning of any function to automatically record
 * its execution time.
 */
//#define PROFILE_FUNCTION() ScopedTimer timer(__FUNCTION__)
// Macro for profiling functions
#define PROFILE_FUNCTION(...)                                              \
    ScopedTimer _scopedTimer((#__VA_ARGS__[0] != '\0') ? #__VA_ARGS__ : __func__)

#define PROFILE_ELAPSED_TIME(label) GlobalProfiler::displayElapsedTime(label)

#else // PROFILE_ENABLED not defined

// Empty implementations when profiling is disabled
class GlobalProfiler {
public:
    static void initialize() {}
    static void start(const std::string&) {}
    static void stop(const std::string&) {}
    static void report() {}
};

class ScopedTimer {
public:
    explicit ScopedTimer(const std::string&) {}
    ~ScopedTimer() {}
};

#define PROFILE_FUNCTION(...)

#define PROFILE_ELAPSED_TIME(label)

#endif // PROFILE_ENABLED




/**
 * @class Timer
 * @brief A simple utility class for measuring elapsed time.
 *
 * This class provides an easy way to measure elapsed time in various units
 * (milliseconds, microseconds, nanoseconds, and seconds). It is useful for 
 * profiling code and measuring execution times.
 */
class Timer {
public:
    /**
     * @brief Constructs a Timer and starts timing immediately.
     */
    Timer() : start_time_(std::chrono::high_resolution_clock::now()) {}

    /**
     * @brief Restarts the timer by resetting the start time to the current time.
     */
    void restart() {
        start_time_ = std::chrono::high_resolution_clock::now();
    }

    /**
     * @brief Gets the elapsed time in milliseconds since the timer started or was last restarted.
     * @return The elapsed time in milliseconds.
     */
    long long elapsed_milliseconds() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_time_
        ).count();
    }

    /**
     * @brief Gets the elapsed time in microseconds since the timer started or was last restarted.
     * @return The elapsed time in microseconds.
     */
    long long elapsed_microseconds() const {
        return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - start_time_
        ).count();
    }

    /**
     * @brief Gets the elapsed time in nanoseconds since the timer started or was last restarted.
     * @return The elapsed time in nanoseconds.
     */
    long long elapsed_nanoseconds() const {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now() - start_time_
        ).count();
    }

    /**
     * @brief Gets the elapsed time in seconds as a double since the timer started or was last restarted.
     * @return The elapsed time in seconds.
     */
    double elapsed_seconds() const {
        return std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - start_time_
        ).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_time_; ///< The starting point of the timer.
};

} // namespace gansu




