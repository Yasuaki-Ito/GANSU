/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 * Hash RI: 3-center ERIs stored in COO format for reuse across SCF iterations.
 */

#include "rhf.hpp"
#include "device_host_memory.hpp"
#include "int3c2e.hpp"
#include <iomanip>

namespace gansu {

// 3-center COO key: Q(21bit) | μ(21bit) | ν(21bit), μ ≤ ν
__device__ __host__ inline unsigned long long encode_3c_key(int Q, int mu, int nu) {
    if (mu > nu) { int t = mu; mu = nu; nu = t; }
    return ((unsigned long long)Q << 42) | ((unsigned long long)mu << 21) | (unsigned long long)nu;
}

__device__ inline void decode_3c_key(unsigned long long key, int& Q, int& mu, int& nu) {
    Q  = (int)((key >> 42) & 0x1FFFFF);
    mu = (int)((key >> 21) & 0x1FFFFF);
    nu = (int)(key & 0x1FFFFF);
}

// COO → dense 3-center expansion
__global__ void hash_3c_coo_to_dense_kernel(
    const unsigned long long* __restrict__ g_keys,
    const double* __restrict__ g_vals,
    const size_t num_entries,
    double* __restrict__ g_B,
    const int nao)
{
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_entries) return;
    int Q, mu, nu;
    decode_3c_key(g_keys[tid], Q, mu, nu);
    double val = g_vals[tid];
    const size_t nao2 = (size_t)nao * nao;
    g_B[(size_t)Q * nao2 + mu * nao + nu] = val;
    g_B[(size_t)Q * nao2 + nu * nao + mu] = val;
}

// ============================================================
ERI_RI_Hash_RHF::~ERI_RI_Hash_RHF() {
    if (d_3c_coo_keys_) tracked_cudaFree(d_3c_coo_keys_);
    if (d_3c_coo_values_) tracked_cudaFree(d_3c_coo_values_);
}

// ============================================================
//  Precomputation: parent sets up 2-center; we compute 3-center → COO
// ============================================================
void ERI_RI_Hash_RHF::precomputation() {
    ERI_RI_Direct::precomputation();

    const int nao = num_basis_;
    const int naux = num_auxiliary_basis_;
    const size_t nao2 = (size_t)nao * nao;
    const auto& shell_type_infos = hf_.get_shell_type_infos();
    const auto& shell_pair_type_infos = hf_.get_shell_pair_type_infos();

    // Compute 3-center into dense temporary
    real_t* d_3c = nullptr;
    tracked_cudaMalloc(&d_3c, (size_t)naux * nao2 * sizeof(real_t));
    cudaMemset(d_3c, 0, (size_t)naux * nao2 * sizeof(real_t));
    {
        const int tpb = 128;
        const int stc = shell_type_infos.size();
        const int atc = auxiliary_shell_type_infos_.size();
        for (int s0 = 0; s0 < stc; ++s0)
            for (int s1 = s0; s1 < stc; ++s1)
                for (int s2 = 0; s2 < atc; ++s2) {
                    ShellTypeInfo ss0 = shell_type_infos[s0], ss1 = shell_type_infos[s1];
                    ShellTypeInfo ss2 = auxiliary_shell_type_infos_[s2];
                    int64_t nt = ((s0==s1) ? (int64_t)ss0.count*(ss0.count+1)/2 : (int64_t)ss0.count*ss1.count) * (int64_t)ss2.count;
                    int nb = (int)((nt + tpb - 1) / tpb);
                    int pi = s0 * stc - s0*(s0+1)/2 + s1;
                    ShellTypeInfo s0n = ss0; s0n.start_index = 0;
                    ShellTypeInfo s1n = ss1; s1n.start_index = 0;
                    gpu::get_3center_kernel(s0,s1,s2)<<<nb,tpb>>>(
                        d_3c, hf_.get_primitive_shells().device_ptr(),
                        auxiliary_primitive_shells_.device_ptr(),
                        hf_.get_cgto_normalization_factors().device_ptr(),
                        auxiliary_cgto_normalization_factors_.device_ptr(),
                        s0n, s1n, ss2, nt, nao,
                        &primitive_shell_pair_indices.device_ptr()[shell_pair_type_infos[pi].start_index],
                        &schwarz_upper_bound_factors.device_ptr()[shell_pair_type_infos[pi].start_index],
                        auxiliary_schwarz_upper_bound_factors.device_ptr(),
                        hf_.get_schwarz_screening_threshold(), naux,
                        hf_.get_boys_grid().device_ptr());
                }
        cudaDeviceSynchronize();
    }

    // Extract COO (host-side for simplicity)
    {
        std::vector<real_t> h_3c((size_t)naux * nao2);
        cudaMemcpy(h_3c.data(), d_3c, h_3c.size()*sizeof(real_t), cudaMemcpyDeviceToHost);
        std::vector<unsigned long long> h_keys;
        std::vector<real_t> h_vals;
        for (int Q = 0; Q < naux; Q++)
            for (int mu = 0; mu < nao; mu++)
                for (int nu = mu; nu < nao; nu++) {
                    double v = h_3c[(size_t)Q*nao2 + mu*nao + nu];
                    if (std::fabs(v) > 1e-15) {
                        h_keys.push_back(encode_3c_key(Q, mu, nu));
                        h_vals.push_back(v);
                    }
                }
        num_3c_entries_ = h_keys.size();
        tracked_cudaMalloc(&d_3c_coo_keys_, num_3c_entries_ * sizeof(unsigned long long));
        tracked_cudaMalloc(&d_3c_coo_values_, num_3c_entries_ * sizeof(real_t));
        cudaMemcpy(d_3c_coo_keys_, h_keys.data(), num_3c_entries_*sizeof(unsigned long long), cudaMemcpyHostToDevice);
        cudaMemcpy(d_3c_coo_values_, h_vals.data(), num_3c_entries_*sizeof(real_t), cudaMemcpyHostToDevice);
    }
    tracked_cudaFree(d_3c);

    size_t total = (size_t)naux * nao * (nao+1) / 2;
    std::cout << "  [Hash RI] 3c COO: " << num_3c_entries_ << " / " << total
              << " (" << (num_3c_entries_*24)/(1024*1024) << " MB)" << std::endl;
}

// ============================================================
//  Helper: expand COO → dense B, apply L^{-1}
// ============================================================
static real_t* build_B_from_3c_coo(
    const unsigned long long* d_keys, const real_t* d_vals, size_t n_entries,
    const real_t* d_L, int nao, int naux)
{
    const size_t nao2 = (size_t)nao * nao;
    const double one = 1.0;
    real_t* d_B = nullptr;
    tracked_cudaMalloc(&d_B, (size_t)naux * nao2 * sizeof(real_t));
    cudaMemset(d_B, 0, (size_t)naux * nao2 * sizeof(real_t));
    {
        const int threads = 256;
        const int blocks = ((int)n_entries + threads - 1) / threads;
        hash_3c_coo_to_dense_kernel<<<blocks, threads>>>(d_keys, d_vals, n_entries, d_B, nao);
        cudaDeviceSynchronize();
    }
    cublasHandle_t h = gpu::GPUHandle::cublas();
    cublasDtrsm(h, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                nao2, naux, &one, d_L, naux, d_B, nao2);
    return d_B;
}

// ============================================================
//  Fock: COO → dense B → existing RI Fock function
// ============================================================
void ERI_RI_Hash_RHF::compute_fock_matrix() {
    const int nao = num_basis_;
    const int naux = num_auxiliary_basis_;
    const size_t nao2 = (size_t)nao * nao;

    real_t* d_B = build_B_from_3c_coo(
        d_3c_coo_keys_, d_3c_coo_values_, num_3c_entries_,
        two_center_eris.device_ptr(), nao, naux);

    // Use existing Stored RI Fock function with temporary B
    real_t *d_J, *d_K, *d_W, *d_T, *d_V;
    tracked_cudaMalloc(&d_J, nao2 * sizeof(real_t));
    tracked_cudaMalloc(&d_K, nao2 * sizeof(real_t));
    tracked_cudaMalloc(&d_W, naux * sizeof(real_t));
    tracked_cudaMalloc(&d_T, (size_t)naux * nao2 * sizeof(real_t));
    tracked_cudaMalloc(&d_V, (size_t)naux * nao2 * sizeof(real_t));

    gpu::computeFockMatrix_RI_RHF_with_density_matrix(
        rhf_.get_density_matrix().device_ptr(),
        rhf_.get_core_hamiltonian_matrix().device_ptr(),
        d_B, rhf_.get_fock_matrix().device_ptr(),
        nao, naux, d_J, d_K, d_W, d_T, d_V);

    tracked_cudaFree(d_J); tracked_cudaFree(d_K);
    tracked_cudaFree(d_W); tracked_cudaFree(d_T); tracked_cudaFree(d_V);
    tracked_cudaFree(d_B);
}

// ============================================================
//  MP2: COO → dense B → existing RI-MP2
// ============================================================
extern real_t compute_ri_mp2_from_B(
    real_t* d_B, int num_basis, int num_auxiliary_basis,
    int nocc, int nvir, real_t* d_C, real_t* d_eps);

real_t ERI_RI_Hash_RHF::compute_mp2_energy() {
    PROFILE_FUNCTION();
    const int nao = num_basis_;
    const int naux = num_auxiliary_basis_;
    const int nocc = rhf_.get_num_electrons() / 2;
    const int nvir = nao - nocc;

    std::cout << "  [Hash RI-MP2] Expanding COO to B..." << std::endl;

    real_t* d_B = build_B_from_3c_coo(
        d_3c_coo_keys_, d_3c_coo_values_, num_3c_entries_,
        two_center_eris.device_ptr(), nao, naux);

    real_t energy = compute_ri_mp2_from_B(d_B, nao, naux, nocc, nvir,
        rhf_.get_coefficient_matrix().device_ptr(),
        rhf_.get_orbital_energies().device_ptr());

    tracked_cudaFree(d_B);
    std::cout << "h_E: " << std::setprecision(12) << energy << std::endl;
    return energy;
}

} // namespace gansu
