/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 */

/**
 * @file gansu_api.cu
 * @brief C API implementation — thin wrapper around existing GANSU classes.
 */

#include "gansu_api.h"
#include "builder.hpp"
#include "parameter_manager.hpp"
#include "gpu_manager.hpp"
#include "rhf.hpp"
#include "uhf.hpp"
#include "progress.hpp"

#include <map>
#include <memory>
#include <string>
#include <iostream>
#include <fstream>

using namespace gansu;

// Helper: get RHF reference (most accessors are in RHF, not HF base)
static RHF* as_rhf(HF* hf) { return dynamic_cast<RHF*>(hf); }
static UHF* as_uhf(HF* hf) { return dynamic_cast<UHF*>(hf); }

// Internal context held by each handle
struct GansuContext {
    std::map<std::string, std::string> user_params;
    std::unique_ptr<HF> hf;
    std::string excited_state_report_cache;
    bool solved = false;
    bool quiet = false;
    gansu_progress_fn progress_fn = nullptr;
    void* progress_userdata = nullptr;
    std::vector<real_t> initial_density;  // for PES density reuse
};

static bool g_initialized = false;

// ---- Lifecycle ----

extern "C" void gansu_init(int force_cpu) {
    if (force_cpu) {
        gpu::disable_gpu();
    } else {
        gpu::initialize_gpu();
    }
    g_initialized = true;
}

extern "C" void gansu_finalize(void) {
    g_initialized = false;
}

extern "C" gansu_handle_t gansu_create(void) {
    auto* ctx = new GansuContext();
    return static_cast<gansu_handle_t>(ctx);
}

extern "C" void gansu_destroy(gansu_handle_t h) {
    if (!h) return;
    auto* ctx = static_cast<GansuContext*>(h);
    delete ctx;
}

// ---- Configuration ----

extern "C" int gansu_set(gansu_handle_t h, const char* key, const char* value) {
    if (!h || !key || !value) return -1;
    auto* ctx = static_cast<GansuContext*>(h);
    try {
        std::string k(key), v(value);
        if (k == "quiet") { ctx->quiet = (v == "1" || v == "true"); return 0; }
        ctx->user_params[k] = v;
        return 0;
    } catch (...) {
        return -1;
    }
}

extern "C" int gansu_set_xyz(gansu_handle_t h, const char* path) {
    return gansu_set(h, "xyzfilename", path);
}

extern "C" int gansu_set_basis(gansu_handle_t h, const char* path) {
    return gansu_set(h, "gbsfilename", path);
}

extern "C" int gansu_set_method(gansu_handle_t h, const char* method) {
    return gansu_set(h, "method", method);
}

extern "C" int gansu_set_post_hf(gansu_handle_t h, const char* post_hf) {
    return gansu_set(h, "post_hf_method", post_hf);
}

// ---- Execution ----

extern "C" int gansu_run(gansu_handle_t h) {
    if (!h) return -1;
    auto* ctx = static_cast<GansuContext*>(h);

    // RAII stdout suppression
    std::streambuf* orig_cout = nullptr;
    std::ofstream devnull;
    if (ctx->quiet) {
        devnull.open("/dev/null");
        orig_cout = std::cout.rdbuf(devnull.rdbuf());
    }

    try {
        // Clean up previous run (release GPU memory)
        ctx->hf.reset();
        ctx->solved = false;

        // Install progress callback for this thread
        if (ctx->progress_fn) {
            gansu::set_progress_callback(ctx->progress_fn, ctx->progress_userdata);
        }

        ParameterManager params(false);
        for (const auto& kv : ctx->user_params) {
            if (kv.first != "parameter_file" || !kv.second.empty()) {
                params[kv.first] = kv.second;
            }
        }

        ctx->hf = HFBuilder::buildHF(params);
        if (!ctx->initial_density.empty()) {
            // RHF density is D_total = D_alpha + D_beta = 2*D_alpha.
            // InitialGuess_RHF_Density expects (D_alpha, D_beta) and sums them.
            // So pass D/2 for both alpha and beta.
            size_t n = ctx->initial_density.size();
            std::vector<real_t> half_density(n);
            for (size_t i = 0; i < n; i++) half_density[i] = ctx->initial_density[i] * 0.5;
            ctx->hf->solve(half_density.data(), half_density.data(), true);
        } else {
            ctx->hf->solve();
        }
        ctx->solved = true;

        gansu::clear_progress_callback();
        if (orig_cout) std::cout.rdbuf(orig_cout);
        return 0;
    } catch (const std::exception& e) {
        // Release all resources on error — no stale state.
        // hf.reset() calls destructors that free GPU memory.
        // After OOM, some frees may fail, so clear CUDA errors first.
        if (gpu::gpu_available()) {
            cudaGetLastError();  // clear pending error before destructor frees
        }
        try {
            ctx->hf.reset();
        } catch (...) {
            // If destructor throws (shouldn't, but safety net), abandon the pointer
            ctx->hf.release();
        }
        ctx->solved = false;
        ctx->initial_density.clear();
        ctx->progress_fn = nullptr;
        ctx->progress_userdata = nullptr;
        ctx->excited_state_report_cache.clear();

        // Clear CUDA error flag without resetting or syncing the device.
        // cudaDeviceReset() and cudaDeviceSynchronize() can segfault after OOM.
        if (gpu::gpu_available()) {
            cudaError_t cuda_err = cudaGetLastError();
            if (cuda_err != cudaSuccess) {
                std::cerr << "gansu_run: CUDA error cleared (" << cudaGetErrorString(cuda_err)
                          << ")" << std::endl;
            }
        }

        // Reset memory tracking counters — after hf.reset() freed everything,
        // counters should be near zero. Recalculate from the authoritative map.
        {
            size_t actual = 0;
            {
                std::lock_guard<std::mutex> lock(g_allocated_memory_map_mutex);
                for (const auto& kv : g_allocated_memory_map) actual += kv.second;
            }
            {
                std::lock_guard<std::mutex> lock(GlobalGpuMemoryTracker::mutex_);
                GlobalGpuMemoryTracker::current_bytes_ = actual;
            }
            CudaMemoryManager<real_t>::reset_memory_statistics();
        }

        gansu::clear_progress_callback();
        if (orig_cout) std::cout.rdbuf(orig_cout);
        std::cerr << "gansu_run error: " << e.what() << std::endl;
        return -1;
    }
}

// ---- Results ----

extern "C" double gansu_get_total_energy(gansu_handle_t h) {
    if (!h) return 0.0;
    auto* ctx = static_cast<GansuContext*>(h);
    if (!ctx->solved || !ctx->hf) return 0.0;
    return ctx->hf->get_total_energy();
}

extern "C" double gansu_get_post_hf_energy(gansu_handle_t h) {
    if (!h) return 0.0;
    auto* ctx = static_cast<GansuContext*>(h);
    if (!ctx->solved || !ctx->hf) return 0.0;
    return ctx->hf->get_post_hf_energy();
}

extern "C" double gansu_get_nuclear_repulsion_energy(gansu_handle_t h) {
    if (!h) return 0.0;
    auto* ctx = static_cast<GansuContext*>(h);
    if (!ctx->solved || !ctx->hf) return 0.0;
    return ctx->hf->get_nuclear_repulsion_energy();
}

extern "C" int gansu_get_num_basis(gansu_handle_t h) {
    if (!h) return 0;
    auto* ctx = static_cast<GansuContext*>(h);
    if (!ctx->hf) return 0;
    return ctx->hf->get_num_basis();
}

extern "C" int gansu_get_num_electrons(gansu_handle_t h) {
    if (!h) return 0;
    auto* ctx = static_cast<GansuContext*>(h);
    if (!ctx->hf) return 0;
    return ctx->hf->get_num_electrons();
}

extern "C" int gansu_get_num_atoms(gansu_handle_t h) {
    if (!h) return 0;
    auto* ctx = static_cast<GansuContext*>(h);
    if (!ctx->hf) return 0;
    return (int)ctx->hf->get_atoms().size();
}

extern "C" int gansu_get_orbital_energies(gansu_handle_t h, double* buf, int buf_size) {
    if (!h || !buf) return -1;
    auto* ctx = static_cast<GansuContext*>(h);
    if (!ctx->solved || !ctx->hf) return -1;
    int nao = ctx->hf->get_num_basis();
    if (buf_size < nao) return -1;

    // Download from device (orbital energies are in RHF/UHF, not HF base)
    RHF* rhf = as_rhf(ctx->hf.get());
    if (!rhf) return -1;  // UHF orbital energies require separate handling
    auto& eps = rhf->get_orbital_energies();
    eps.toHost();
    for (int i = 0; i < nao; i++)
        buf[i] = eps.host_ptr()[i];
    return nao;
}

extern "C" int gansu_get_mo_coefficients(gansu_handle_t h, double* buf, int buf_size) {
    if (!h || !buf) return -1;
    auto* ctx = static_cast<GansuContext*>(h);
    if (!ctx->solved || !ctx->hf) return -1;
    int nao = ctx->hf->get_num_basis();
    int n2 = nao * nao;
    if (buf_size < n2) return -1;

    RHF* rhf = as_rhf(ctx->hf.get());
    if (!rhf) return -1;
    auto& C = rhf->get_coefficient_matrix();
    C.toHost();
    for (int i = 0; i < n2; i++)
        buf[i] = C.host_ptr()[i];
    return n2;
}

extern "C" int gansu_get_ccsd_1rdm_mo(gansu_handle_t h, double* buf, int buf_size) {
    if (!h || !buf) return -1;
    auto* ctx = static_cast<GansuContext*>(h);
    if (!ctx->solved || !ctx->hf) return -1;
    const auto& D = ctx->hf->get_ccsd_1rdm_mo();
    int n = (int)D.size();
    if (n == 0 || buf_size < n) return -1;
    for (int i = 0; i < n; i++) buf[i] = D[i];
    return n;
}

extern "C" const char* gansu_get_excited_state_report(gansu_handle_t h) {
    if (!h) return "";
    auto* ctx = static_cast<GansuContext*>(h);
    if (!ctx->hf) return "";
    ctx->excited_state_report_cache = ctx->hf->get_excited_state_report();
    return ctx->excited_state_report_cache.c_str();
}

// ---- Atom coordinates ----

extern "C" int gansu_get_atomic_number(gansu_handle_t h, int i) {
    if (!h) return 0;
    auto* ctx = static_cast<GansuContext*>(h);
    if (!ctx->hf) return 0;
    auto& atoms = const_cast<DeviceHostMemory<Atom>&>(ctx->hf->get_atoms());
    if (i < 0 || i >= (int)atoms.size()) return 0;
    atoms.toHost();
    return atoms.host_ptr()[i].atomic_number;
}

extern "C" int gansu_get_atom_coords(gansu_handle_t h, int i, double* x, double* y, double* z) {
    if (!h || !x || !y || !z) return -1;
    auto* ctx = static_cast<GansuContext*>(h);
    if (!ctx->hf) return -1;
    auto& atoms = const_cast<DeviceHostMemory<Atom>&>(ctx->hf->get_atoms());
    if (i < 0 || i >= (int)atoms.size()) return -1;
    atoms.toHost();
    const auto& coord = atoms.host_ptr()[i].coordinate;
    *x = coord.x; *y = coord.y; *z = coord.z;
    return 0;
}

// ---- Analysis ----
// Mulliken/bond order are protected in HF base. Access via RHF downcast.

extern "C" int gansu_get_mulliken_charges(gansu_handle_t h, double* buf, int buf_size) {
    if (!h || !buf) return -1;
    auto* ctx = static_cast<GansuContext*>(h);
    if (!ctx->solved || !ctx->hf) return -1;
    RHF* rhf = as_rhf(ctx->hf.get());
    if (!rhf) return -1;
    int natom = (int)ctx->hf->get_atoms().size();
    if (buf_size < natom) return -1;
    try {
        auto charges = rhf->analyze_mulliken_population();
        for (int i = 0; i < natom && i < (int)charges.size(); i++)
            buf[i] = charges[i];
        return natom;
    } catch (...) { return -1; }
}

extern "C" int gansu_get_mayer_bond_order(gansu_handle_t h, double* buf, int buf_size) {
    if (!h || !buf) return -1;
    auto* ctx = static_cast<GansuContext*>(h);
    if (!ctx->solved || !ctx->hf) return -1;
    RHF* rhf = as_rhf(ctx->hf.get());
    if (!rhf) return -1;
    int natom = (int)ctx->hf->get_atoms().size();
    if (buf_size < natom * natom) return -1;
    try {
        auto bo = rhf->compute_mayer_bond_order();
        for (int i = 0; i < natom; i++)
            for (int j = 0; j < natom; j++)
                buf[i * natom + j] = bo[i][j];
        return natom * natom;
    } catch (...) { return -1; }
}

extern "C" int gansu_get_wiberg_bond_order(gansu_handle_t h, double* buf, int buf_size) {
    if (!h || !buf) return -1;
    auto* ctx = static_cast<GansuContext*>(h);
    if (!ctx->solved || !ctx->hf) return -1;
    RHF* rhf = as_rhf(ctx->hf.get());
    if (!rhf) return -1;
    int natom = (int)ctx->hf->get_atoms().size();
    if (buf_size < natom * natom) return -1;
    try {
        auto bo = rhf->compute_wiberg_bond_order();
        for (int i = 0; i < natom; i++)
            for (int j = 0; j < natom; j++)
                buf[i * natom + j] = bo[i][j];
        return natom * natom;
    } catch (...) { return -1; }
}

extern "C" int gansu_get_density_matrix(gansu_handle_t h, double* buf, int buf_size) {
    if (!h || !buf) return -1;
    auto* ctx = static_cast<GansuContext*>(h);
    if (!ctx->solved || !ctx->hf) return -1;
    RHF* rhf = as_rhf(ctx->hf.get());
    if (!rhf) return -1;
    int nao = ctx->hf->get_num_basis();
    int n2 = nao * nao;
    if (buf_size < n2) return -1;
    auto& D = rhf->get_density_matrix();
    D.toHost();
    for (int i = 0; i < n2; i++) buf[i] = D.host_ptr()[i];
    return n2;
}

extern "C" int gansu_set_initial_density(gansu_handle_t h, const double* buf, int buf_size) {
    if (!h) return -1;
    auto* ctx = static_cast<GansuContext*>(h);
    if (buf && buf_size > 0) {
        ctx->initial_density.assign(buf, buf + buf_size);
    } else {
        ctx->initial_density.clear();
    }
    return 0;
}

extern "C" void gansu_set_progress_callback(gansu_handle_t h, gansu_progress_fn fn, void* user_data) {
    if (!h) return;
    auto* ctx = static_cast<GansuContext*>(h);
    ctx->progress_fn = fn;
    ctx->progress_userdata = user_data;
}

extern "C" int gansu_get_overlap_matrix(gansu_handle_t h, double* buf, int buf_size) {
    if (!h || !buf) return -1;
    auto* ctx = static_cast<GansuContext*>(h);
    if (!ctx->solved || !ctx->hf) return -1;
    int nao = ctx->hf->get_num_basis();
    int n2 = nao * nao;
    if (buf_size < n2) return -1;
    auto& S = ctx->hf->get_overlap_matrix();
    S.toHost();
    for (int i = 0; i < n2; i++) buf[i] = S.host_ptr()[i];
    return n2;
}
