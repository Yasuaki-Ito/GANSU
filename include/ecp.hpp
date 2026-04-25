/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 */

/**
 * @file ecp.hpp
 * @brief Effective Core Potential (ECP) data structures
 *
 * ECP replaces core electrons with an effective potential:
 *   V_ECP = V_local(r) + Σ_l V_l(r) |l><l|
 *
 * where V_local and V_l are Gaussian-type potentials:
 *   V(r) = Σ_k d_k r^(n_k-2) exp(-ζ_k r²)
 *
 * Gaussian94 format:
 *   ELEMENT-ECP  l_max  n_core_electrons
 *   local_label potential (e.g., "f potential" for l_max=3)
 *     N
 *     n1 ζ1 d1
 *     ...
 *   semi-local potentials (e.g., "s-f potential", "p-f potential", ...)
 *     N
 *     n1 ζ1 d1
 *     ...
 */

#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <iomanip>

namespace gansu {

/**
 * @brief A single ECP Gaussian primitive: d * r^(n-2) * exp(-ζ r²)
 */
struct ECPPrimitive {
    int power;             // n (power of r: r^(n-2) in the potential)
    double exponent;       // ζ
    double coefficient;    // d
};

/**
 * @brief One angular momentum component of an ECP
 *
 * For local component (l = l_max): V_local(r)
 * For semi-local component (l < l_max): V_l(r) - V_local(r)
 *   (i.e., the difference potential, applied via projector |l><l|)
 */
struct ECPComponent {
    int angular_momentum;  // l value (-1 for local component)
    std::vector<ECPPrimitive> primitives;

    size_t num_primitives() const { return primitives.size(); }
};

/**
 * @brief ECP for a single element
 *
 * Contains the local potential and semi-local projectors.
 * The actual potential applied is:
 *   V = V_local + Σ_{l=0}^{l_max-1} (V_l - V_local) |l><l|
 *     = V_local + Σ_l ΔV_l |l><l|
 */
class ElementECP {
public:
    ElementECP() : element_name_(""), l_max_(-1), n_core_electrons_(0) {}

    void set_element_name(const std::string& name) { element_name_ = name; }
    void set_l_max(int l_max) { l_max_ = l_max; }
    void set_n_core_electrons(int n) { n_core_electrons_ = n; }

    const std::string& get_element_name() const { return element_name_; }
    int get_l_max() const { return l_max_; }
    int get_n_core_electrons() const { return n_core_electrons_; }

    /// Add the local component (highest angular momentum channel)
    void set_local_component(const ECPComponent& comp) { local_ = comp; }

    /// Add a semi-local component for angular momentum l
    void add_semilocal_component(const ECPComponent& comp) {
        semilocal_.push_back(comp);
    }

    const ECPComponent& get_local() const { return local_; }
    const std::vector<ECPComponent>& get_semilocal() const { return semilocal_; }
    size_t num_semilocal() const { return semilocal_.size(); }

    bool empty() const { return l_max_ < 0; }

    friend std::ostream& operator<<(std::ostream& os, const ElementECP& ecp) {
        os << "ECP for " << ecp.element_name_
           << ": l_max=" << ecp.l_max_
           << ", n_core=" << ecp.n_core_electrons_ << std::endl;
        os << "  Local (" << ecp.local_.num_primitives() << " primitives):" << std::endl;
        for (const auto& p : ecp.local_.primitives) {
            os << "    n=" << p.power << " zeta=" << std::setprecision(8) << p.exponent
               << " d=" << p.coefficient << std::endl;
        }
        for (const auto& comp : ecp.semilocal_) {
            os << "  l=" << comp.angular_momentum
               << " (" << comp.num_primitives() << " primitives):" << std::endl;
            for (const auto& p : comp.primitives) {
                os << "    n=" << p.power << " zeta=" << std::setprecision(8) << p.exponent
                   << " d=" << p.coefficient << std::endl;
            }
        }
        return os;
    }

private:
    std::string element_name_;
    int l_max_;               // Maximum angular momentum of projectors
    int n_core_electrons_;    // Number of electrons replaced by ECP
    ECPComponent local_;      // Local (V_local) component
    std::vector<ECPComponent> semilocal_;  // Semi-local (ΔV_l) components, l=0..l_max-1
};

} // namespace gansu
