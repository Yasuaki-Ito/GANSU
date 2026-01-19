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


#include "profiler.hpp"
#include <iomanip>  // For std::setw and std::setfill
#include "console.hpp" // Font color
#include <chrono>

namespace gansu {

#ifdef PROFILE_ENABLED



std::chrono::high_resolution_clock::time_point GlobalProfiler::programStartTime;
std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> GlobalProfiler::startTimes;
std::unordered_map<std::string, long long> GlobalProfiler::executionTimes;
std::unordered_map<std::string, int> GlobalProfiler::callCounts;  // Store the number of calls for each function
std::mutex GlobalProfiler::mutex_;

void GlobalProfiler::initialize() {
    programStartTime = std::chrono::high_resolution_clock::now();
}

void GlobalProfiler::start(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto now = std::chrono::high_resolution_clock::now();
    startTimes[name] = now;
    callCounts[name]++;
    auto sinceStart = std::chrono::duration_cast<std::chrono::microseconds>(now - programStartTime).count();

    // Convert microseconds to minutes, seconds, and milliseconds
    auto minutes = sinceStart / 60000000;
    auto seconds = (sinceStart % 60000000) / 1000000;
    auto milliseconds = (sinceStart % 1000000) / 1000;

    std::cout << "[" << ConsoleColor::GREEN 
              << std::setw(2) << std::setfill('0') << minutes << ":" 
              << std::setw(2) << std::setfill('0') << seconds << "."
              << std::setw(3) << std::setfill('0') << milliseconds << ConsoleColor::RESET << "] START: " 
              << name << std::endl;
}

void GlobalProfiler::stop(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - startTimes[name]).count();
    executionTimes[name] += duration;
    auto sinceStart = std::chrono::duration_cast<std::chrono::microseconds>(now - programStartTime).count();

    // Convert microseconds to minutes, seconds, and milliseconds
    auto minutes = sinceStart / 60000000;
    auto seconds = (sinceStart % 60000000) / 1000000;
    auto milliseconds = (sinceStart % 1000000) / 1000;

    double duration_milliseconds = static_cast<double>(duration) / 1000;

    std::cout << "[" << ConsoleColor::GREEN 
              << std::setw(2) << std::setfill('0') << minutes << ":"
              << std::setw(2) << std::setfill('0') << seconds << "."
              << std::setw(3) << std::setfill('0') << milliseconds << ConsoleColor::RESET << "] END:   " 
              << name << " after " << std::setprecision(5) << duration_milliseconds << " ms" << std::endl;
}

void GlobalProfiler::report() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::cout << "\n[Timing Summary]\n";
    for (const auto& entry : executionTimes) {
        double duration_milliseconds = static_cast<double>(entry.second) / 1000;
        std::cout << entry.first << ": " 
                  << std::setprecision(5) << duration_milliseconds << " microseconds total, "
                  << "called " << callCounts[entry.first] << " times.\n";
    }
}

void GlobalProfiler::displayElapsedTime(const std::string& label) {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - programStartTime);
    long long milliseconds = duration.count();
    
    int seconds = milliseconds / 1000;
    int minutes = seconds / 60;
    seconds = seconds % 60;
    milliseconds = milliseconds % 1000;

    std::cout << "[" << ConsoleColor::RED 
                << std::setw(2) << std::setfill('0') << minutes << ":"
                << std::setw(2) << std::setfill('0') << seconds << "."
                << std::setw(3) << std::setfill('0') << milliseconds << ConsoleColor::RESET 
                << "] " << label << std::endl;
}

ScopedTimer::ScopedTimer(const std::string& name) : functionName(name) {
    GlobalProfiler::start(functionName);
}

ScopedTimer::~ScopedTimer() {
    cudaDeviceSynchronize();
    GlobalProfiler::stop(functionName);
}

#endif // PROFILE_ENABLED





} // namespace gansu