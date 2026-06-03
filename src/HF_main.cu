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
 * @file HF_main.cu
 * @brief Main function for the Hartree-Fock method.
 */

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>
#include <memory>
#include <filesystem>
#include <fstream>
#include <set>

#include "hf.hpp"
#include "rhf.hpp"
#include "parameter_manager.hpp"
#include "spherical_transform.hpp"
#include "builder.hpp"
#include "gpu_manager.hpp"
#ifdef GANSU_MULTI_GPU
#include "multi_gpu_manager.hpp"
#endif

namespace fs = std::filesystem;
using namespace gansu;

/**
 * @brief Resolve basis set name to full path.
 *
 * If the given path already exists as a file, return it as-is.
 * Otherwise, search for "<name>" and "<name>.gbs" in standard locations
 * relative to the executable: ../basis/, ./basis/, and the executable directory.
 */
static std::string resolve_basis_path(const std::string& name_or_path, const char* argv0) {
    // Already a valid file path
    if (fs::exists(name_or_path)) return name_or_path;

    // Candidate file names
    std::vector<std::string> names = {name_or_path};
    if (name_or_path.find(".gbs") == std::string::npos)
        names.push_back(name_or_path + ".gbs");

    // Search directories relative to executable
    fs::path exe_dir = fs::path(argv0).parent_path();
    if (exe_dir.empty()) exe_dir = ".";
    std::vector<fs::path> search_dirs = {
        exe_dir / ".." / "basis",      // build/ → ../basis/
        exe_dir / "basis",             // same dir
        fs::path("basis"),             // cwd/basis/
        fs::path("../basis"),          // cwd/../basis/
    };

    for (const auto& dir : search_dirs)
        for (const auto& n : names) {
            auto p = dir / n;
            if (fs::exists(p)) return p.string();
        }

    // Not found — return original (will produce a clear error from GANSU)
    return name_or_path;
}

/**
 * @brief Main function
 * @param argc Number of arguments
 * @param argv Arguments
 * @return 0 if the program ends successfully
 * @details This function reads the command line arguments and calls the RHF or UHF class.
 */
int main(int argc, char* argv[]){
  // Force line-buffered stdout (subprocess pipes default to full-buffered,
  // which delays progress output to the UI)
  setvbuf(stdout, NULL, _IOLBF, 0);

  try {
    // Check for special flags before parameter parsing
    bool force_cpu = false;
    bool list_basis = false;
    bool dump_overlap_diag = false;
    bool test_spherical_transform = false;
    std::string save_density_path, load_density_path;
    std::vector<char*> filtered_argv;
    filtered_argv.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg == "--cpu") {
            force_cpu = true;
        } else if (arg == "--list-basis") {
            list_basis = true;
        } else if (arg == "--test_spherical_transform") {
            // Standalone unit test for spherical_transform.hpp d-shell matrix.
            // Builds analytical S_cart (6x6, 1/3 axial off-diag) for unit-
            // normalized Cartesian d Gaussians, applies U_d (5x6) on both
            // sides, expects identity I_5 (within machine epsilon).
            // Skips the rest of the calculation.
            test_spherical_transform = true;
        } else if (arg == "--dump_overlap_diag") {
            // Print S[i,i] for every basis function after HF setup, labelled by
            // shell type (s/p/d/f/g/h/i) and (lx,ly,lz) for d+ shells, so we
            // can determine GANSU's Cartesian Gaussian normalization convention
            // (whether per-(lx,ly,lz) unit-normalized or shell-level common).
            // Used to validate spherical_transform.hpp before full integration.
            dump_overlap_diag = true;
        } else if (arg == "--save_density" && i + 1 < argc) {
            save_density_path = argv[++i];
        } else if (arg == "--load_density" && i + 1 < argc) {
            load_density_path = argv[++i];
        } else {
            filtered_argv.push_back(argv[i]);
        }
    }
    int filtered_argc = (int)filtered_argv.size();

    // --test_spherical_transform: standalone verification of U_d S_cart U_d^T = I_5.
    if (test_spherical_transform) {
        const auto U = gansu::spherical::get_cart_to_sph_matrix(2);  // d shell

        // Analytical S_cart for unit-normalized d Gaussians at one center.
        // (Confirmed by GANSU dump: all diagonal = 1; 1/3 axial off-diagonals
        //  come from <xx|yy> = 1/3 in physical-space Cartesian d basis.)
        std::vector<std::vector<double>> S_cart = {
            //  xx     yy     zz    xy    xz    yz
            { 1.0,   1.0/3, 1.0/3, 0.0, 0.0, 0.0 },  // xx
            { 1.0/3, 1.0,   1.0/3, 0.0, 0.0, 0.0 },  // yy
            { 1.0/3, 1.0/3, 1.0,   0.0, 0.0, 0.0 },  // zz
            { 0.0,   0.0,   0.0,   1.0, 0.0, 0.0 },  // xy
            { 0.0,   0.0,   0.0,   0.0, 1.0, 0.0 },  // xz
            { 0.0,   0.0,   0.0,   0.0, 0.0, 1.0 }   // yz
        };

        // Compute S_sph = U S_cart U^T.
        std::vector<std::vector<double>> S_sph(5, std::vector<double>(5, 0.0));
        for (int i = 0; i < 5; ++i)
            for (int j = 0; j < 5; ++j) {
                double sum = 0.0;
                for (int k = 0; k < 6; ++k)
                    for (int l = 0; l < 6; ++l)
                        sum += U[i][k] * S_cart[k][l] * U[j][l];
                S_sph[i][j] = sum;
            }

        const char* m_label[5] = {"d_0", "d_+1", "d_-1", "d_+2", "d_-2"};

        std::cout << "\n=== Spherical d-shell transform self-test ===" << std::endl;
        std::cout << "U_d (5x6, Molden order: d_0, d_+1, d_-1, d_+2, d_-2):" << std::endl;
        std::cout << "          xx        yy        zz        xy        xz        yz" << std::endl;
        for (int i = 0; i < 5; ++i) {
            std::cout << "  " << m_label[i] << "  ";
            for (int k = 0; k < 6; ++k)
                std::cout << std::setw(9) << std::fixed << std::setprecision(5) << U[i][k] << " ";
            std::cout << std::endl;
        }

        std::cout << "\nU S_cart U^T  (should be I_5 within machine epsilon):" << std::endl;
        double max_off_diag = 0.0;
        double max_diag_err = 0.0;
        for (int i = 0; i < 5; ++i) {
            std::cout << "  ";
            for (int j = 0; j < 5; ++j) {
                std::cout << std::setw(13) << std::fixed << std::setprecision(10) << S_sph[i][j] << " ";
                if (i == j)        max_diag_err  = std::max(max_diag_err,  std::abs(S_sph[i][j] - 1.0));
                else if (i != j)   max_off_diag  = std::max(max_off_diag,  std::abs(S_sph[i][j]));
            }
            std::cout << std::endl;
        }

        std::cout << "\nmax |diag - 1.0|  = " << std::scientific << max_diag_err << std::endl;
        std::cout << "max |off-diag|    = " << std::scientific << max_off_diag << std::endl;
        bool ok1 = (max_diag_err < 1e-12) && (max_off_diag < 1e-12);
        std::cout << "Test 1 (U_d matrix only): "
                  << (ok1 ? "PASS ✓" : "FAIL ✗") << std::endl;

        // ============================================================
        // Test 2: full transform_matrix_cart_to_sph routine with 1 s + 1 d
        // ============================================================
        // Cart layout: [s, d_xx, d_yy, d_zz, d_xy, d_xz, d_yz]  (7 BF)
        // Sph  layout: [s, d_0, d_+1, d_-1, d_+2, d_-2]         (6 BF)
        std::cout << "\n=== Test 2: transform_matrix_cart_to_sph (1 s + 1 d) ===" << std::endl;
        const int nbf_cart = 7, nbf_sph = 6;
        // S_cart 7x7 row-major: s-s = 1, s-d_{cross} = 0 (parity), s-d_{axial} != 0 (radial)
        // For unit-normalized basis at the same center, s-d_axial overlap depends on
        // the radial functions (exponents); for this synthetic test we set s-d_axial = 0
        // (orthogonal radial), which is what would happen between non-overlapping shells.
        // Diagonal of d block uses (1, 1/3, 1/3, 1, 1, 1) etc. (same as Test 1).
        std::vector<double> S_cart_full(nbf_cart * nbf_cart, 0.0);
        // s-s
        S_cart_full[0 * 7 + 0] = 1.0;
        // d block (rows/cols 1..6 in {xx, yy, zz, xy, xz, yz} order)
        const double Sdd[6][6] = {
            { 1.0,   1.0/3, 1.0/3, 0.0, 0.0, 0.0 },
            { 1.0/3, 1.0,   1.0/3, 0.0, 0.0, 0.0 },
            { 1.0/3, 1.0/3, 1.0,   0.0, 0.0, 0.0 },
            { 0.0,   0.0,   0.0,   1.0, 0.0, 0.0 },
            { 0.0,   0.0,   0.0,   0.0, 1.0, 0.0 },
            { 0.0,   0.0,   0.0,   0.0, 0.0, 1.0 }
        };
        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 6; ++j)
                S_cart_full[(i + 1) * 7 + (j + 1)] = Sdd[i][j];

        // Shell info: 2 shells (s with L=0 at offset 0, d with L=2 at offset 1)
        std::vector<int> shell_types        = { 0, 2 };
        std::vector<int> shell_offsets_cart = { 0, 1, 7 };   // [start_s=0, start_d=1, end=7]
        std::vector<int> shell_offsets_sph  = { 0, 1, 6 };   // [start_s=0, start_d=1, end=6]

        std::vector<double> S_sph_full(nbf_sph * nbf_sph, 0.0);
        gansu::spherical::transform_matrix_cart_to_sph(
            S_cart_full.data(), S_sph_full.data(),
            shell_types, shell_offsets_cart, shell_offsets_sph);

        std::cout << "S_sph (6x6, should be I_6):" << std::endl;
        double max_diag_err2 = 0.0, max_off_diag2 = 0.0;
        for (int i = 0; i < nbf_sph; ++i) {
            std::cout << "  ";
            for (int j = 0; j < nbf_sph; ++j) {
                std::cout << std::setw(13) << std::fixed << std::setprecision(10)
                          << S_sph_full[i * nbf_sph + j] << " ";
                if (i == j) max_diag_err2 = std::max(max_diag_err2, std::abs(S_sph_full[i*nbf_sph+j] - 1.0));
                else        max_off_diag2 = std::max(max_off_diag2, std::abs(S_sph_full[i*nbf_sph+j]));
            }
            std::cout << std::endl;
        }
        std::cout << "max |diag - 1.0|  = " << std::scientific << max_diag_err2 << std::endl;
        std::cout << "max |off-diag|    = " << std::scientific << max_off_diag2 << std::endl;
        bool ok2 = (max_diag_err2 < 1e-12) && (max_off_diag2 < 1e-12);
        std::cout << "Test 2 (transform_matrix routine): "
                  << (ok2 ? "PASS ✓" : "FAIL ✗") << std::endl;

        // ============================================================
        // Test 3: transform_eri_cart_to_sph on a synthetic d-only ERI
        // ============================================================
        // For unit-normalized Cart d at the same center (1 d shell only, nbf_cart=6),
        // an identity-like ERI doesn't have a simple analytical reference.  Instead,
        // we use a "double-transform" sanity check: build ERI_cart = δ_{pp'} δ_{qq'} ...
        // (identity tensor), then transform it.  The result should have a known
        // pattern in spherical basis (block-diagonal).  We verify only that the
        // transformation runs without numerical issues and produces a finite result.
        std::cout << "\n=== Test 3: transform_eri_cart_to_sph (1 d shell smoke test) ===" << std::endl;
        const int nbf_c3 = 6, nbf_s3 = 5;
        std::vector<int> st3 = { 2 };
        std::vector<int> so_c3 = { 0, 6 };
        std::vector<int> so_s3 = { 0, 5 };
        // Build a "physical" ERI_cart: ERI[p,q,r,s] = S_cart[p,q] * S_cart[r,s] / 4
        // (this is the Hartree-product-like structure with random sign content for
        //  testing — has the right symmetry and finite magnitude).
        std::vector<double> ERI_cart(nbf_c3 * nbf_c3 * nbf_c3 * nbf_c3, 0.0);
        for (int p = 0; p < 6; ++p)
            for (int q = 0; q < 6; ++q)
                for (int r = 0; r < 6; ++r)
                    for (int s = 0; s < 6; ++s)
                        ERI_cart[((p * 6 + q) * 6 + r) * 6 + s] = Sdd[p][q] * Sdd[r][s] / 4.0;
        std::vector<double> ERI_sph(nbf_s3 * nbf_s3 * nbf_s3 * nbf_s3, 0.0);
        gansu::spherical::transform_eri_cart_to_sph(
            ERI_cart.data(), ERI_sph.data(), st3, so_c3, so_s3);

        // Smoke check: ERI_sph should equal S_sph_dd ⊗ S_sph_dd / 4 (with d block of S)
        // since S_sph d block = I_5, expect ERI_sph[p,q,r,s] = δ_{pq} · δ_{rs} / 4
        double max_eri_err = 0.0;
        for (int p = 0; p < 5; ++p)
            for (int q = 0; q < 5; ++q)
                for (int r = 0; r < 5; ++r)
                    for (int s = 0; s < 5; ++s) {
                        const double expected = (p == q ? 1.0 : 0.0) * (r == s ? 1.0 : 0.0) / 4.0;
                        const double got = ERI_sph[((p * 5 + q) * 5 + r) * 5 + s];
                        max_eri_err = std::max(max_eri_err, std::abs(got - expected));
                    }
        std::cout << "max |ERI_sph - expected (δ_pq δ_rs / 4)| = "
                  << std::scientific << max_eri_err << std::endl;
        bool ok3 = (max_eri_err < 1e-12);
        std::cout << "Test 3 (transform_eri routine):     "
                  << (ok3 ? "PASS ✓" : "FAIL ✗") << std::endl;

        // ============================================================
        // Test 4: U_L · S_cart_L · U_L^T = I_{2L+1} for L ∈ {2, 3, 4}
        //         Uses analytical unit-norm Cartesian overlap:
        //           S(a,b) = (lax+lbx-1)!! · (lay+lby-1)!! · (laz+lbz-1)!!
        //                    / sqrt(f_a · f_b)
        //         (all factors with odd argument vanish → S(a,b) = 0)
        //         where f_a = (2la_x-1)!!(2la_y-1)!!(2la_z-1)!!.
        // ============================================================
        std::cout << "\n=== Test 4: U_L · S_cart_L · U_L^T = I_{2L+1} for L=2,3,4 ===" << std::endl;
        auto dfact_neg1_to_n = [](int n) -> double {
            // double factorial (2k-1)!! with convention (-1)!! = 1 (used when exp=0)
            if (n < 0) return 1.0;
            double r = 1.0;
            for (int k = n; k > 0; k -= 2) r *= (double)k;
            return r;
        };
        bool ok4 = true;
        for (int L : {2, 3, 4}) {
            const auto& AM = gansu::AngularMomentums[L];
            const int ncart = (L + 1) * (L + 2) / 2;
            const int nsph  = 2 * L + 1;
            // f_a per Cart BF
            std::vector<double> fcart(ncart);
            for (int a = 0; a < ncart; ++a) {
                fcart[a] = dfact_neg1_to_n(2 * AM[a][0] - 1)
                         * dfact_neg1_to_n(2 * AM[a][1] - 1)
                         * dfact_neg1_to_n(2 * AM[a][2] - 1);
            }
            // S_cart[a,b] = product over axes of (la_x+lb_x-1)!! / sqrt(f_a f_b),
            //              zero if any (la+lb) on any axis is odd.
            std::vector<std::vector<double>> Scart(ncart, std::vector<double>(ncart, 0.0));
            for (int a = 0; a < ncart; ++a)
                for (int b = 0; b < ncart; ++b) {
                    int sxyz[3];
                    bool all_even = true;
                    for (int ax = 0; ax < 3; ++ax) {
                        sxyz[ax] = AM[a][ax] + AM[b][ax];
                        if (sxyz[ax] % 2 != 0) { all_even = false; break; }
                    }
                    if (!all_even) continue;
                    double s = dfact_neg1_to_n(sxyz[0] - 1)
                             * dfact_neg1_to_n(sxyz[1] - 1)
                             * dfact_neg1_to_n(sxyz[2] - 1);
                    Scart[a][b] = s / std::sqrt(fcart[a] * fcart[b]);
                }
            const auto U = gansu::spherical::get_cart_to_sph_matrix(L);
            // S_sph = U S_cart U^T
            double diag_err = 0.0, off_err = 0.0;
            for (int i = 0; i < nsph; ++i)
                for (int j = 0; j < nsph; ++j) {
                    double sum = 0.0;
                    for (int k = 0; k < ncart; ++k)
                        for (int l = 0; l < ncart; ++l)
                            sum += U[i][k] * Scart[k][l] * U[j][l];
                    if (i == j) diag_err = std::max(diag_err, std::abs(sum - 1.0));
                    else        off_err  = std::max(off_err,  std::abs(sum));
                }
            const char shell_lbl[5] = {'s','p','d','f','g'};
            bool okL = (diag_err < 1e-12) && (off_err < 1e-12);
            ok4 = ok4 && okL;
            std::cout << "  L=" << L << " (" << shell_lbl[L] << "): max|diag-1|="
                      << std::scientific << std::setprecision(3) << diag_err
                      << ", max|off|=" << off_err
                      << "  " << (okL ? "PASS ✓" : "FAIL ✗") << std::endl;
        }
        std::cout << "Test 4 (L=2,3,4 unit-norm Sph overlap): "
                  << (ok4 ? "PASS ✓" : "FAIL ✗") << std::endl;

        std::cout << "\n=== Overall: "
                  << ((ok1 && ok2 && ok3 && ok4) ? "ALL TESTS PASS ✓" : "SOME TESTS FAILED ✗")
                  << " ===" << std::endl;
        return (ok1 && ok2 && ok3 && ok4) ? 0 : 1;
    }

    // List available basis sets and exit
    if (list_basis) {
        fs::path exe_dir = fs::path(argv[0]).parent_path();
        if (exe_dir.empty()) exe_dir = ".";
        std::vector<fs::path> search_dirs = {
            exe_dir / ".." / "basis",
            exe_dir / "basis",
            fs::path("basis"),
            fs::path("../basis"),
        };
        std::set<std::string> names;
        for (const auto& dir : search_dirs) {
            if (!fs::is_directory(dir)) continue;
            for (const auto& entry : fs::directory_iterator(dir)) {
                auto fn = entry.path().filename().string();
                if (fn.size() > 4 && fn.substr(fn.size()-4) == ".gbs")
                    names.insert(fn.substr(0, fn.size()-4));
            }
        }
        for (const auto& n : names) std::cout << n << std::endl;
        return 0;
    }

    if (force_cpu) {
        gpu::disable_gpu();
    } else {
        gpu::initialize_gpu(); // Detect GPU availability (CPU fallback if no GPU)
    }

    ParameterManager parameters;
    parameters.parse_command_line_args(filtered_argc, filtered_argv.data());

    // Multi-GPU initialization is deferred to ERI factory in rhf.cu

    // Resolve basis set short names: -g cc-pvdz → ../basis/cc-pvdz.gbs
    if (parameters.contains("gbsfilename")) {
        std::string gbs = parameters.get<std::string>("gbsfilename");
        if (!gbs.empty()) {
            parameters["gbsfilename"] = resolve_basis_path(gbs, argv[0]);
        }
    }

    std::unique_ptr<HF> hf = HFBuilder::buildHF(parameters);

    // Lambda for --dump_overlap_diag (called after hf->solve so the overlap
    // matrix is populated by compute_core_hamiltonian_matrix).
    auto dump_overlap = [&]() {
        auto& S_dh = hf->get_overlap_matrix();
        const int nbf   = hf->get_num_basis();
        // Directly cudaMemcpy from device — bypasses any toHost() corner cases.
        std::vector<real_t> S_dev((size_t)nbf * nbf, 0.0);
        const real_t* dS = S_dh.device_ptr();
        std::cout << "[overlap-diag] device_ptr=" << (void*)dS
                  << " host_ptr=" << (void*)S_dh.host_ptr() << std::endl;
        if (dS) {
            cudaError_t err = cudaMemcpy(S_dev.data(), dS,
                                         (size_t)nbf * nbf * sizeof(real_t),
                                         cudaMemcpyDeviceToHost);
            std::cout << "[overlap-diag] cudaMemcpy from device: "
                      << cudaGetErrorString(err) << std::endl;
        } else {
            std::cout << "[overlap-diag] ERROR: device_ptr is null — falling back to host_ptr"
                      << std::endl;
            S_dh.toHost();
            const real_t* hS = S_dh.host_ptr();
            if (hS) std::memcpy(S_dev.data(), hS, (size_t)nbf * nbf * sizeof(real_t));
        }
        const real_t* S = S_dev.data();
        auto& prims = const_cast<DeviceHostMemory<PrimitiveShell>&>(hf->get_primitive_shells());
        prims.toHost();
        const auto* p = prims.host_ptr();
        const size_t n_prim = prims.size();

        std::vector<int> bf_to_L(nbf, -1);
        std::vector<int> bf_to_pos(nbf, -1);
        std::vector<int> bf_to_atom(nbf, -1);
        std::vector<size_t> seen;
        for (size_t i = 0; i < n_prim; ++i) {
            const int L = p[i].shell_type;
            const size_t b0 = p[i].basis_index;
            if (std::find(seen.begin(), seen.end(), b0) != seen.end()) continue;
            seen.push_back(b0);
            const int n_cart = (L + 1) * (L + 2) / 2;
            for (int k = 0; k < n_cart && (size_t)(b0 + k) < (size_t)nbf; ++k) {
                bf_to_L[b0 + k]    = L;
                bf_to_pos[b0 + k]  = k;
                bf_to_atom[b0 + k] = p[i].atom_index;
            }
        }
        const char* L_name[] = {"s","p","d","f","g","h","i"};
        const char* d_lbl[6] = {"xx","yy","zz","xy","xz","yz"};
        const char* p_lbl[3] = {"x","y","z"};
        std::cout << "\n[overlap-diag] nbf = " << nbf
                  << " (Cartesian, GANSU native order)\n";
        std::cout << "[overlap-diag] " << std::setw(5) << "idx"
                  << " " << std::setw(4) << "atom"
                  << " " << std::setw(3) << "L"
                  << " " << std::setw(5) << "comp"
                  << "   " << "S[i,i]" << "\n";
        for (int i = 0; i < nbf; ++i) {
            const int L = bf_to_L[i];
            const int k = bf_to_pos[i];
            std::string comp = "?";
            if (L == 0) comp = "s";
            else if (L == 1 && k >= 0 && k < 3) comp = p_lbl[k];
            else if (L == 2 && k >= 0 && k < 6) comp = d_lbl[k];
            else if (L >= 0) comp = std::string("k=") + std::to_string(k);
            std::cout << "[overlap-diag] " << std::setw(5) << i
                      << " " << std::setw(4) << bf_to_atom[i]
                      << " " << std::setw(3)
                      << ((L >= 0 && L < 7) ? L_name[L] : "?")
                      << " " << std::setw(5) << comp
                      << "   " << std::fixed << std::setprecision(12)
                      << S[(size_t)i * nbf + i]
                      << std::defaultfloat << "\n";
            if (i >= 29) {
                std::cout << "[overlap-diag] (truncated after first 30 BFs)\n";
                break;
            }
        }
        std::cout << std::endl;
    };

    // Load density matrix from previous run (for PES chaining)
    if (!load_density_path.empty()) {
        std::ifstream din(load_density_path, std::ios::binary);
        if (din.good()) {
            int nao = hf->get_num_basis();
            size_t n2 = (size_t)nao * nao;
            std::vector<real_t> D(n2);
            din.read(reinterpret_cast<char*>(D.data()), n2 * sizeof(real_t));
            if (din.good()) {
                // RHF: D_total = D_alpha + D_beta. InitialGuess expects (D_alpha, D_beta).
                // Pass D/2 for both.
                std::vector<real_t> half_D(n2);
                for (size_t k = 0; k < n2; k++) half_D[k] = D[k] * 0.5;
                hf->solve(half_D.data(), half_D.data(), true);
                std::cout << "[PES] Loaded density from " << load_density_path << std::endl;
            } else {
                std::cerr << "[PES] Warning: failed to read density file, using default guess" << std::endl;
                hf->solve();
            }
        } else {
            hf->solve();
        }
    } else {
        hf->solve(); // Solve the HF equation (SCF procedure)
    }
    hf->report(); // Print the HF results

    // Run the overlap diagonal dump now that solve() has computed the overlap.
    if (dump_overlap_diag) {
        dump_overlap();
    }

    // Save density matrix for next PES point
    if (!save_density_path.empty()) {
        RHF* rhf = dynamic_cast<RHF*>(hf.get());
        if (rhf) {
            int nao = hf->get_num_basis();
            size_t n2 = (size_t)nao * nao;
            auto& D = rhf->get_density_matrix();
            D.toHost();
            std::ofstream dout(save_density_path, std::ios::binary);
            dout.write(reinterpret_cast<const char*>(D.host_ptr()), n2 * sizeof(real_t));
            std::cout << "[PES] Saved density to " << save_density_path << std::endl;
        }
    }

    // Export the SAD density matrix to a file
    if (parameters.contains("export_sad_cache")) {
        std::cout << "Exporting SAD cache to 'temp_sad_cache.dat'..." << std::endl;
        hf->generate_sad_cache("temp_sad_cache.dat");
        std::cout << "SAD cache exported successfully." << std::endl;
    }

    return 0;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}