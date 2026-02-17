#ifndef AO2MO_CUH
#define AO2MO_CUH

#include "eri_stored.hpp"

namespace gansu {



static inline cublasStatus_t dgemm_device_row_major(
    cublasHandle_t h,
    cublasOperation_t opA_rm, cublasOperation_t opB_rm,
    int m, int n, int k,
    const double* alpha,
    const double* A_rm, int lda_rm,
    const double* B_rm, int ldb_rm,
    const double* beta,
    double* C_rm, int ldc_rm
){
    // Row-major: C = opA(A) * opB(B)  (m×n)
    // Column-major view: C^T = opB(B)^T * opA(A)^T  (n×m)
    //
    // Memory trick:
    //   A_rm (row-major) is A_cm = A_rm^T (col-major)
    //   B_rm (row-major) is B_cm = B_rm^T (col-major)
    //   C_rm (row-major) is C_cm = C_rm^T (col-major)
    //
    // Then:
    //   C_cm = op(B_cm) * op(A_cm)
    // where op(B_cm) uses the SAME op flag as opB_rm (not flipped),
    // and op(A_cm) uses the SAME op flag as opA_rm.

    return cublasDgemm(
        h,
        /*transa=*/opB_rm,  /*transb=*/opA_rm,
        /*m=*/n, /*n=*/m, /*k=*/k,
        alpha,
        B_rm, ldb_rm,
        A_rm, lda_rm,
        beta,
        C_rm, ldc_rm
    );
}


static inline cublasStatus_t dgemm_strided_batched_device_row_major(
    cublasHandle_t h,
    cublasOperation_t opA_rm, cublasOperation_t opB_rm,
    int m, int n, int k,
    const double* alpha,
    const double* A_rm, int lda_rm, long long int strideA_rm,
    const double* B_rm, int ldb_rm, long long int strideB_rm,
    const double* beta,
    double* C_rm, int ldc_rm, long long int strideC_rm,
    int batchCount
){
    // Row-major:  C = opA(A) * opB(B)     (m×n)
    // Transpose:  C^T = opB(B)^T * opA(A)^T  (n×m)
    //
    // Memory trick:
    //   A_rm memory == A_cm = A_rm^T in column-major
    //   B_rm memory == B_cm = B_rm^T in column-major
    //   C_rm memory == C_cm = C_rm^T in column-major
    //
    // Therefore in column-major we compute:
    //   C_cm (n×m) = op(B_cm) * op(A_cm)
    // where op flags are the same as row-major op flags (not flipped).
    //
    // Leading dimensions and strides in elements carry over unchanged.

    return cublasDgemmStridedBatched(
        h,
        /*transa=*/opB_rm,  /*transb=*/opA_rm,
        /*m=*/n, /*n=*/m, /*k=*/k,
        alpha,
        /*A=*/B_rm, /*lda=*/ldb_rm, /*strideA=*/strideB_rm,
        /*B=*/A_rm, /*ldb=*/lda_rm, /*strideB=*/strideA_rm,
        beta,
        /*C=*/C_rm, /*ldc=*/ldc_rm, /*strideC=*/strideC_rm,
        batchCount
    );
}

static inline cublasStatus_t dgeam_transpose_row_major(
    cublasHandle_t h,
    int rows_rm, int cols_rm,          // A_rm is rows_rm x cols_rm (row-major)
    const double* A_rm,
    double* AT_rm                     // AT_rm is cols_rm x rows_rm (row-major)
){
    const double alpha = 1.0;
    const double beta  = 0.0;

    // row-major A_rm(rows,cols) == column-major A_cm(cols,rows)
    // want AT_rm(cols,rows) == column-major AT_cm(rows,cols)
    // AT_cm = A_cm^T

    return cublasDgeam(
        h,
        CUBLAS_OP_T, CUBLAS_OP_N,
        /*m=*/rows_rm,              // rows(AT_cm)
        /*n=*/cols_rm,              // cols(AT_cm)
        &alpha,
        A_rm, /*lda=*/cols_rm,      // lda = rows(A_cm) = cols_rm
        &beta,
        A_rm, /*ldb=*/cols_rm,
        AT_rm, /*ldc=*/rows_rm      // ldc = rows(AT_cm) = rows_rm
    );
}


// two normal dgemms and two stridedbatched dgemms
//*
inline void transform_eri_ao2mo_dgemm_full(
    double* d_eri_ao, double* d_eri_mo, 
    const double* d_coefficient_matrix, const int num_basis)
{
    const size_t num_basis_2 = num_basis * num_basis;
    const size_t num_basis_3 = num_basis_2 * num_basis;

    const double alpha = 1.0;
    const double beta = 0.0;
    cublasHandle_t cublasH = NULL;
    cublasCreate(&cublasH);

    dgemm_device_row_major(
        cublasH, 
        CUBLAS_OP_T, CUBLAS_OP_N,
        num_basis, num_basis_3, num_basis,
        &alpha, 
        d_coefficient_matrix, num_basis, 
        d_eri_ao, num_basis_3, 
        &beta, 
        d_eri_mo, num_basis_3
    );
    dgemm_strided_batched_device_row_major(
        cublasH,
        CUBLAS_OP_T, CUBLAS_OP_N,
        num_basis, num_basis_2, num_basis,
        &alpha,
        d_coefficient_matrix, num_basis, 0,
        d_eri_mo, num_basis_2, num_basis_3,
        &beta,
        d_eri_ao, num_basis_2, num_basis_3,
        num_basis
    );

    dgeam_transpose_row_major(
        cublasH, 
        num_basis_2, num_basis_2, 
        d_eri_ao, 
        d_eri_mo);

    dgemm_device_row_major(
        cublasH, 
        CUBLAS_OP_T, CUBLAS_OP_N,
        num_basis, num_basis_3, num_basis,
        &alpha, 
        d_coefficient_matrix, num_basis, 
        d_eri_mo, num_basis_3, 
        &beta, 
        d_eri_ao, num_basis_3
    );
    dgemm_strided_batched_device_row_major(
        cublasH,
        CUBLAS_OP_T, CUBLAS_OP_N,
        num_basis, num_basis_2, num_basis,
        &alpha,
        d_coefficient_matrix, num_basis, 0,
        d_eri_ao, num_basis_2, num_basis_3,
        &beta,
        d_eri_mo, num_basis_2, num_basis_3,
        num_basis
    );

    cudaDeviceSynchronize();
    cublasDestroy(cublasH);
}
/**/


__device__ inline size_t ovov2seq(
    const int i, const int a, const int j, const int b, 
    const int num_occupied, const int num_virtual) 
{
    return (((size_t(i) * num_virtual + a) * num_occupied + j) * num_virtual + b);
}



__device__ inline size_t ovov2seq_aabb(
    const int i, const int a, const int j, const int b, 
    const int num_occupied_al, const int num_virtual_al, 
    const int num_occupied_be, const int num_virtual_be) 
{
    return (((size_t(i) * num_virtual_al + a) * num_occupied_be + j) * num_virtual_be + b);
}



// for rmp2 and ump2 (same spin)
//*
inline void transform_eri_ao2mo_dgemm_ovov(
    double* d_eri_ao, double* d_eri_mo, 
    const double* d_coefficient_matrix, 
    const int num_occ, const int num_vir)
{
    const size_t num_basis = num_occ + num_vir;
    const size_t num_basis_2 = num_basis * num_basis;
    const size_t num_basis_3 = num_basis_2 * num_basis;

    const double alpha = 1.0;
    const double beta = 0.0;
    cublasHandle_t cublasH = NULL;
    cublasCreate(&cublasH);

    dgemm_device_row_major(
        cublasH, 
        CUBLAS_OP_T, CUBLAS_OP_N,
        num_occ, num_basis_3, num_basis,
        &alpha, 
        d_coefficient_matrix, num_basis, 
        d_eri_ao, num_basis_3, 
        &beta, 
        d_eri_mo, num_basis_3
    );
    dgeam_transpose_row_major(
        cublasH, 
        num_occ * num_basis, 
        num_basis * num_basis, 
        d_eri_mo, 
        d_eri_ao);
    dgemm_device_row_major(
        cublasH, 
        CUBLAS_OP_T, CUBLAS_OP_N,
        num_occ, num_basis_2 * num_occ, num_basis,
        &alpha, 
        d_coefficient_matrix, num_basis, 
        d_eri_ao, num_basis_2 * num_occ, 
        &beta, 
        d_eri_mo, num_basis_2 * num_occ
    );

    dgemm_strided_batched_device_row_major(
        cublasH,
        CUBLAS_OP_T, CUBLAS_OP_N,
        num_vir, num_occ * num_basis, num_basis,
        &alpha,
        d_coefficient_matrix + num_occ, num_basis, 0,
        d_eri_mo, num_occ * num_basis, num_basis_2 * num_occ,
        &beta,
        d_eri_ao, num_occ * num_basis, num_vir * num_occ * num_basis,
        num_occ
    );
    dgeam_transpose_row_major(
        cublasH, 
        num_occ * num_vir, 
        num_occ * num_basis, 
        d_eri_ao, 
        d_eri_mo);
    dgemm_strided_batched_device_row_major(
        cublasH,
        CUBLAS_OP_T, CUBLAS_OP_N,
        num_vir, num_occ * num_vir, num_basis,
        &alpha,
        d_coefficient_matrix + num_occ, num_basis, 0,
        d_eri_mo, num_occ * num_vir, num_basis * num_occ * num_vir,
        &beta,
        d_eri_ao, num_occ * num_vir, num_vir * num_occ * num_vir,
        num_occ
    );

    cudaDeviceSynchronize();
    cublasDestroy(cublasH);
}
/**/



// for ump2 (opposite spin)
//*
inline void transform_eri_ao2mo_dgemm_ovov_os(
    double* d_eri_ao, double* d_eri_mo, 
    const double* d_coefficient_matrix_al, 
    const double* d_coefficient_matrix_be, 
    const int num_occ_al, const int num_vir_al,
    const int num_occ_be, const int num_vir_be)
{
    const size_t num_basis = num_occ_al + num_vir_al;
    const size_t num_basis_2 = num_basis * num_basis;
    const size_t num_basis_3 = num_basis_2 * num_basis;

    const double alpha = 1.0;
    const double beta = 0.0;
    cublasHandle_t cublasH = NULL;
    cublasCreate(&cublasH);

    dgemm_device_row_major(
        cublasH, 
        CUBLAS_OP_T, CUBLAS_OP_N,
        num_occ_al, num_basis_3, num_basis,
        &alpha, 
        d_coefficient_matrix_al, num_basis, 
        d_eri_ao, num_basis_3, 
        &beta, 
        d_eri_mo, num_basis_3
    );
    dgeam_transpose_row_major(
        cublasH, 
        num_occ_al * num_basis, 
        num_basis * num_basis, 
        d_eri_mo, 
        d_eri_ao);
    dgemm_device_row_major(
        cublasH, 
        CUBLAS_OP_T, CUBLAS_OP_N,
        num_occ_be, num_basis_2 * num_occ_al, num_basis,
        &alpha, 
        d_coefficient_matrix_be, num_basis, 
        d_eri_ao, num_basis_2 * num_occ_al, 
        &beta, 
        d_eri_mo, num_basis_2 * num_occ_al
    );

    dgemm_strided_batched_device_row_major(
        cublasH,
        CUBLAS_OP_T, CUBLAS_OP_N,
        num_vir_be, num_occ_al * num_basis, num_basis,
        &alpha,
        d_coefficient_matrix_be + num_occ_be, num_basis, 0,
        d_eri_mo, num_occ_al * num_basis, num_basis_2 * num_occ_al,
        &beta,
        d_eri_ao, num_occ_al * num_basis, num_vir_be * num_occ_al * num_basis,
        num_occ_be
    );
    dgeam_transpose_row_major(
        cublasH, 
        num_occ_be * num_vir_be, 
        num_occ_al * num_basis, 
        d_eri_ao, 
        d_eri_mo);
    dgemm_strided_batched_device_row_major(
        cublasH,
        CUBLAS_OP_T, CUBLAS_OP_N,
        num_vir_al, num_occ_be * num_vir_be, num_basis,
        &alpha,
        d_coefficient_matrix_al + num_occ_al, num_basis, 0,
        d_eri_mo, num_occ_be * num_vir_be, num_basis * num_occ_be * num_vir_be,
        &beta,
        d_eri_ao, num_occ_be * num_vir_be, num_vir_al * num_occ_be * num_vir_be,
        num_occ_al
    );

    cudaDeviceSynchronize();
    cublasDestroy(cublasH);
}
/**/




























} // namespace gansu













/*
void transform_eri_ao2mo_dgemm_full(
    double* d_eri_ao, double* d_eri_mo, 
    const double* d_coefficient_matrix, const int num_basis)
{
    double* d_G1;
    double* d_G2;
    const int num_threads_per_block = 256;
    const size_t num_basis_sq = num_basis * num_basis;
    const size_t num_blocks = ((size_t)num_basis_sq * num_basis_sq + num_threads_per_block - 1) / num_threads_per_block;

    const double alpha = 1.0;
    const double beta = 0.0;
    cublasHandle_t cublasH = NULL;
    cublasCreate(&cublasH);

    // mu2i for (i, nu, la, si)
    for (int i = 0; i < num_basis; ++i) {
        d_G1 = d_eri_ao + num_basis_sq * num_basis * i;
        d_G2 = d_eri_mo + num_basis_sq * num_basis * i;
        cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, num_basis_sq, num_basis, num_basis, 
                    &alpha, d_G1, num_basis_sq, d_coefficient_matrix, num_basis, &beta, d_G2, num_basis_sq);
    }
    swap_bra_index<<<num_blocks, num_threads_per_block>>>(d_eri_mo, d_eri_ao, num_basis);
    cudaDeviceSynchronize();

    // nu2j for (i, j, k, si)
    for (int i = 0; i < num_basis; ++i) {
        d_G1 = d_eri_ao + num_basis_sq * num_basis * i;
        d_G2 = d_eri_mo + num_basis_sq * num_basis * i;
        cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, num_basis_sq, num_basis, num_basis, 
                    &alpha, d_G1, num_basis_sq, d_coefficient_matrix, num_basis, &beta, d_G2, num_basis_sq);
    }

    // la2k for (i, nu, k, si)
    cublasDgeam(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, num_basis_sq, num_basis_sq,
                &alpha, d_eri_mo, num_basis_sq, &beta, d_eri_mo, num_basis_sq, d_eri_ao, num_basis_sq);
    for (int i = 0; i < num_basis; ++i) {
        d_G1 = d_eri_ao + num_basis_sq * num_basis * i;
        d_G2 = d_eri_mo + num_basis_sq * num_basis * i;
        cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, num_basis_sq, num_basis, num_basis, 
                    &alpha, d_G1, num_basis_sq, d_coefficient_matrix, num_basis, &beta, d_G2, num_basis_sq);
    }
    swap_bra_index<<<num_blocks, num_threads_per_block>>>(d_eri_mo, d_eri_ao, num_basis);
    cudaDeviceSynchronize();

    // si2l for (i, j, k, l)
    for (int i = 0; i < num_basis; ++i) {
        d_G1 = d_eri_ao + num_basis_sq * num_basis * i;
        d_G2 = d_eri_mo + num_basis_sq * num_basis * i;
        cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, num_basis_sq, num_basis, num_basis, 
                    &alpha, d_G1, num_basis_sq, d_coefficient_matrix, num_basis, &beta, d_G2, num_basis_sq);
    }
    //cublasDgeam(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, num_basis_sq, num_basis_sq,
    //            &alpha, d_eri_mo, num_basis_sq, &beta, d_eri_mo, num_basis_sq, d_eri_ao, num_basis_sq);

    cublasDestroy(cublasH);
}
/**/






#endif // AO2MO_CUH
