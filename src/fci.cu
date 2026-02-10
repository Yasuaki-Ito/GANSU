#include <cuda.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <omp.h>
#include <cmath>
#include <utility>
#include <iostream>
#include <cusolverDn.h>
#include "gpu_manager.hpp"
#include "utils.hpp"
#include<sys/time.h>

#include "device_host_memory.hpp"
#include "fci.hpp"



void range(int* in, int size) {
#pragma omp parallel for schedule(dynamic)
    for (int i=0; i<size; i++){
        in[i] = i;
    }
}

__global__ void FCImake_hdiag_uhf_kernel(double *hdiag, int size, double *h1e, double *jdiag, double *kdiag,
                       int norb, int nstr, int nocc, int *occslist)
{
        int i=blockIdx.x * blockDim.x + threadIdx.x;
        int j, j0, k0, jk, jk0, jb, jb0;
        int ia, ib;
        double e1, e2;
        int *paocc, *pbocc;
        if (i<size) {
                ia = int(i/nstr);
                ib=i%nstr;
                paocc = occslist + ia * nocc;
                e1 = 0;
                e2 = 0;
                pbocc = occslist + ib * nocc;
                for (j0 = 0; j0 < nocc; j0++) {
                                j = paocc[j0];
                                jk0 = j * norb;
                                jb=pbocc[j0];
                                jb0=jb*norb;
                                e1 +=  h1e[j*norb+j]+h1e[jb*norb+jb];
                                for (k0 = 0; k0 < nocc; k0++) { // (alpha|alpha)
                                        jk = jk0 + paocc[k0];
                                        e2 += jdiag[jk] - kdiag[jk];
                                        jk = jk0 + pbocc[k0];
                                        e2 += jdiag[jk] * 2;
                                        jk=jb0+pbocc[k0];
                                        e2 += jdiag[jk] - kdiag[jk];
                                }
                }
                hdiag[ia*nstr+ib] = e1 + e2 * .5;
        }
}



int combinations(int n, int r) {
    if (r > n) return 0;
    if (r * 2 > n) r = n - r;
    int res = 1;
    for (int i = 1; i <= r; ++i) {
        res = res * (n - i + 1) / i;
    }
    return res;
}





void combinationUtil(int *occslst_flat, int *index, int *arr, int n, int r, int *data, int data_index, int start) {
    if (data_index == r) {
        // 現在の組み合わせを occslst_flat に保存
        for (int i = 0; i < r; i++) {
            occslst_flat[(*index)++] = data[i];
        }
        return;
    }

    for (int i = start; i < n; i++) {
        data[data_index] = arr[i];
        combinationUtil(occslst_flat, index, arr, n, r, data, data_index + 1, i + 1);
    }
}


void gen_occs_iter_ci_new(int *occslst_flat, int *index, int *orb_list, int nelec, int norb) {
 
    int *data = (int*)malloc(nelec * sizeof(int));
    combinationUtil(occslst_flat, index, orb_list, norb, nelec, data, 0, 0);

    free(data);
}


void computeEigenvaluesAndVectorsn(cusolverDnHandle_t cusolverH, int N, double* d_A, double* values, double* vectors, int* devInfo, double* d_W) {
    double *d_work = NULL;
    int lwork = 0;
    using gansu::tracked_cudaMalloc;
    using gansu::tracked_cudaFree;
    // Query workspace size
    cusolverDnDsyevd_bufferSize(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, N, d_A, N, d_W, &lwork);
    tracked_cudaMalloc(&d_work, sizeof(double) * lwork);

    cusolverDnDsyevd(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, N, d_A, N, d_W, d_work, lwork, devInfo);
    // Copy results back to host
    cudaMemcpy(vectors, d_A, sizeof(double) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(values, d_W, sizeof(double) * 1, cudaMemcpyDeviceToHost);
    tracked_cudaFree(d_work);
}




__global__ void f1e_kernel(double* f1e, double* eri, double* h1e, int norb, int d2, int d3, double norm_factor){
      int jk=blockIdx.x * blockDim.x + threadIdx.x;
      int j=jk/norb;
      int k=jk%norb;
      double sum = 0.0;
      for (int i = 0; i < norb; i++) {
                sum += eri[j * d3 + i * d2 + i * norb + k];
      }
      f1e[j * norb + k] = (h1e[j * norb + k] - 0.5 * sum)*norm_factor;
}
__global__ void adderi_kernel(double* f1e, double* eri, int norb, int d2, int d3){
       int ijk=blockIdx.x * blockDim.x + threadIdx.x;
       int i = ijk/d2;
       int j = (ijk-i*d2)/norb;
       int k = (ijk-i*d2)%norb;
       //printf("norb:%d, ijk:%d, i:%d, j:%d, k:%d\n", norb, ijk, i, j, k);
       atomicAdd(&eri[k * d3 + k * d2 + i * norb + j], f1e[i * norb + j]);
       atomicAdd(&eri[i * d3 + j * d2 + k * norb + k], f1e[i * norb + j]);
}

__global__ void nr1to4_kernel(double* eri1, double* eri4, int norb, int d1, int d2, int d3, size_t npair, double fac)
{
        int idx=blockIdx.x * blockDim.x + threadIdx.x;
        //for (int idx=0; idx<npair*npair; idx++){
        int i, j, k, l, ij, kl;
        ij = idx / npair;
        kl = idx % npair;
        i = (int)((sqrt(8.0 * ij + 1) - 1) / 2);
        j = ij - i * (i + 1) / 2;
        k = (int)((sqrt(8.0 * kl + 1) - 1) / 2);
        l = kl - k * (k + 1) / 2;
        eri4[ij*npair+kl] = eri1[i*d3+j*d2+k*d1+l]*fac;
        //}
}

void absorb_h1e(double* d_h1e,  double* d_erio, double* d_eri, int norb, int nelec, double fac) {
    using gansu::tracked_cudaMalloc;
    using gansu::tracked_cudaFree;
    double* f1e = (double*)malloc(norb * norb * sizeof(double));
    int d2 = norb * norb;
    int d3 = norb * norb * norb;
    double norm_factor = 1.0 / (nelec + 1e-100);
    double *d_f1e;
    tracked_cudaMalloc((void **)&d_f1e, d2* sizeof(double));
    f1e_kernel<<<d2, 1>>>(d_f1e, d_erio, d_h1e, norb, d2, d3, norm_factor);
    adderi_kernel<<<d3, 1>>>(d_f1e, d_erio, norb, d2, d3);
    
    int nnorb=norb*(norb+1)/2;
    nr1to4_kernel<<<nnorb*nnorb, 1>>>(d_erio, d_eri, norb, norb, d2, d3, nnorb, fac);

    free(f1e);
    tracked_cudaFree(d_f1e);
}



int compare(const void* a, const void* b) {
    return (*(int*)a - *(int*)b);
}

void propgate1e(int nelec, int norb, int nstring, int* str0, int* linktab, int* linktab_str0, int* orb_list, int* strdic_keys, int* strdic_vals, int nvir, int addr0) {
    int i, j, n;
    int* vir = (int*)malloc(nvir * sizeof(int));
    int* where_vir = (int*)malloc(nvir * sizeof(int));
    int** str1s = (int**)malloc(nvir * sizeof(int*));
    int parity_occ_orb = 1;
    for (i = 0, j = 0; i < norb; ++i) {
        int found = 0;
        for (n = 0; n < nelec; ++n) {
            if (str0[n] == i) {
                found = 1;
                break;
            }
        }
        if (!found) {vir[j++] = i;}
    }

    for (i = 0; i < nvir; ++i) {
        where_vir[i] = 0;
        for (j = 0; j < nelec; ++j) {
            if (str0[j] < vir[i]) {
                where_vir[i]++;
            }
        }
    }
    for (i = 0; i < nelec; ++i) {
        linktab[i * 3 + 0] = str0[i];
        linktab_str0[i] = str0[i];
        linktab[i * 3 + 1] = addr0;
        linktab[i * 3 + 2] = 1;
    }
   
    
    for (n = 0; n < nelec; ++n) {
        for (i = 0; i < nvir; ++i) {
            str1s[i] = (int*)malloc(nelec * sizeof(int));
            memcpy(str1s[i], str0, nelec * sizeof(int));
            str1s[i][n] = vir[i];
            qsort(str1s[i], nelec, sizeof(int), compare);
            //printf("str1s[%d][%d]:%d\n", i,n,str1s[i][n]);
        }

        for (i = 0; i < nvir; ++i) {
            int comp = (vir[i] > str0[n]) ? 1 : 0;
            int sum = where_vir[i] + comp + 1;
            int parity = sum % 2 == 0 ? -1 : 1;
            parity *= parity_occ_orb;
            int s_index = 1;
            //int s_index = strdic_vals[str1s[0]];
            for (int k = 0; k < nstring; ++k) {
                if (memcmp(str1s[i], &strdic_keys[k *nelec], nelec * sizeof(int)) == 0) {
                   s_index = strdic_vals[k];
                   break;
                }
            }
            linktab[(nelec + n * nvir + i) * 3 + 0] = vir[i];
            //linktab[(nelec + n * nvir + i) * 4 + 1] = str0[n];
	    linktab_str0[nelec + n * nvir +i]=str0[n];
            linktab[(nelec + n * nvir + i) * 3 + 1] = s_index;
            linktab[(nelec + n * nvir + i) * 3 + 2] = parity;
        }

        parity_occ_orb *= -1;
    }

    free(vir);
    free(where_vir);
    for (i = 0; i < nvir; ++i) {
        free(str1s[i]);
    }
    free(str1s);
}

__device__ inline void sort_small(int* arr, int n) {
    // insertion sort
    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}


__global__ void propgate1e_kernel(int nelec, int norb, int nstring,
    int* d_link_index,   int* occslst, int* d_link_nnorb, int nvir, int nlink,
    int* d_scratch  // size = nstring * (nvir * 2 + nvir * nelec) * sizeof(int)
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= nstring) return;

    
    const int per_thread_scratch_size = (nvir * 2 + nvir * nelec); // in int units
    int* scratch_base_for_this_thread = &d_scratch[id * per_thread_scratch_size];

    int* vir = scratch_base_for_this_thread;
    int* where_vir = scratch_base_for_this_thread + nvir;
    
    int* str1buf = scratch_base_for_this_thread + nvir * 2;

    int parity_occ_orb = 1;
    int* str0 = &occslst[id * nelec]; 
    int a, idx;
    int i_idx, j_idx, n_idx; 

    // -----------------
    // compute vir[]
    // -----------------
    int j_vir_idx = 0;
    for (i_idx = 0; i_idx < norb; ++i_idx) {
        bool found = false;
        for (n_idx = 0; n_idx < nelec; ++n_idx) {
            if (str0[n_idx] == i_idx) {
                found = true;
                break;
            }
        }
        if (!found) { vir[j_vir_idx++] = i_idx; }
    }

    // -----------------
    // compute where_vir[]
    // -----------------
    for (i_idx = 0; i_idx < nvir; ++i_idx) {
        where_vir[i_idx] = 0;
        for (j_idx = 0; j_idx < nelec; ++j_idx) {
            if (str0[j_idx] < vir[i_idx]) {
                where_vir[i_idx]++;
            }
        }
    }

    // -----------------
    // fill link_index (initial entries)
    // -----------------
    for (i_idx = 0; i_idx < nelec; ++i_idx) {
        
        a = str0[i_idx];
        a = a * (a + 1) / 2 + a;
        d_link_index[(id * nlink + i_idx) * 3 + 0] = a; //str0[i_idx];
        d_link_index[(id * nlink + i_idx) * 3 + 1] = id;
        d_link_index[(id * nlink + i_idx) * 3 + 2] = 1;
    }

    // -----------------
    // loop over electrons
    // -----------------
    for (n_idx = 0; n_idx < nelec; ++n_idx) { 

        for (i_idx = 0; i_idx < nvir; ++i_idx) { 
            
            int* s1 = str1buf + i_idx * nelec;

            for (int k_idx = 0; k_idx < nelec; ++k_idx) {
                s1[k_idx] = str0[k_idx];
            }
            s1[n_idx] = vir[i_idx];

            // qsort(str1s[i], nelec, sizeof(int), compare_int64); 
            sort_small(s1, nelec);
        }

        for (i_idx = 0; i_idx < nvir; ++i_idx) { 
            int* s1 = str1buf + i_idx * nelec; 

            int comp = (vir[i_idx] > str0[n_idx]) ? 1 : 0;
            int sum = where_vir[i_idx] + comp + 1;
            int parity = (sum % 2 == 0 ? -1 : 1) * parity_occ_orb;

            int s_index = -1; 
            // memcmp(str1s[i], &occslst[k *nelec], nelec * sizeof(int)) == 0) 
            for (int k_idx = 0; k_idx < nstring; ++k_idx) {
                bool eq = true;
                for (int x_idx = 0; x_idx < nelec; ++x_idx) {
                    if (s1[x_idx] != occslst[k_idx * nelec + x_idx]) {
                        eq = false;
                        break;
                    }
                }
                if (eq) {
                    s_index = k_idx;
                    break;
                }
            }

            int pos_offset = nelec + n_idx * nvir + i_idx;
            a = vir[i_idx];
            idx = str0[n_idx];
            a = max(a * (a + 1) / 2 + idx, idx * (idx + 1) / 2 + a);
            d_link_index[(id * nlink + pos_offset) * 3 + 0] = a;
            d_link_index[(id * nlink + pos_offset) * 3 + 1] = s_index;
            d_link_index[(id * nlink + pos_offset) * 3 + 2] = parity;
            

        }
        parity_occ_orb *= -1;
    }
    for (int rlink_j_idx = 0; rlink_j_idx < nlink; ++rlink_j_idx) {
        int id_flat_idx = id * nlink + rlink_j_idx;
        int ia_aj = d_link_index[id_flat_idx * 3];
        size_t id_st = (size_t)ia_aj * (size_t)nstring + (size_t)id;
        d_link_nnorb[2*id_st] = (int)d_link_index[id_flat_idx * 3 + 1];
        d_link_nnorb[2*id_st+1] = (int)d_link_index[id_flat_idx * 3 + 2];
    }

}

void gen_linkstr_index(int nelec, int norb, int nstring, int* d_occslst,  int* d_link_index, int* d_link_nnorb) {
    using gansu::tracked_cudaMalloc;
    using gansu::tracked_cudaFree;
    int nvir = norb - nelec;
    int nlink = nelec + nelec * nvir;
    int  *d_scratch;
    int threads = 128;
    
    int blocks = (nstring + threads - 1) / threads;
    const int per_thread_scratch_size = (nvir * 2 + nvir * nelec); // in int units
    int total_scratch_bytes = nstring * per_thread_scratch_size * sizeof(int);

    tracked_cudaMalloc((void **)&d_scratch, total_scratch_bytes);
    
    
    cudaMemset(d_scratch, 0, total_scratch_bytes);
    
    propgate1e_kernel<<<blocks, threads>>>(nelec, norb, nstring,
        d_link_index,  d_occslst,  d_link_nnorb, nvir, nlink, d_scratch);

    //cudaGetLastError(); 
    tracked_cudaFree(d_scratch);
}



 __global__ void qr_decomposition_kernel(double* ci0,  int np, double lindep, double norm) {
     int id=blockIdx.x * blockDim.x + threadIdx.x;
     if (id>=np)return;
     ci0[id] = ci0[id]/norm; 
     
}
__global__ void Dcopy_kernel(double *in, double *out, int heff_size, int space){
     int id=blockIdx.x * blockDim.x + threadIdx.x;
     //if (id>=space*space)return;
     int j=id/space;
      out[id] = in[j*heff_size+id%space];
      
}



void fill_heff_hermitian_gpu(cublasHandle_t handle, double* d_heff_tmp, double* d_heff,  double* d_ci0, double* d_ci1, double* d_ci1_list, int row1, int nrow, int heff_size, int np){
    //xs: ci0_list, ax: ci1_list, xt:ci0, axt:ci1
    int row0 = row1 - nrow;
    cublasDdot(handle, np, d_ci0, 1, d_ci1, 1, &d_heff[row0*heff_size+row0]);
    
    for (int i=0; i<row0; i++){
	   cublasDdot(handle, np, d_ci0, 1, d_ci1_list+i*np, 1, &d_heff[row0*heff_size+i]);
           cudaMemcpy(&d_heff[i*heff_size+row0], &d_heff[row0*heff_size+i], sizeof(double), cudaMemcpyDeviceToDevice);
    }
    
    Dcopy_kernel<<<row1*row1, 1>>>(d_heff, d_heff_tmp, heff_size, row1);
    
}



__global__ void Dscal_kernel(double *in, double *out, double k,  int np){
     int id=blockIdx.x * blockDim.x + threadIdx.x;
     if (id>=np)return;
     out[id]=k*in[id];
}
__global__ void Dscalplus_kernel(double *in, double *out, double k,  int np){
     int id=blockIdx.x * blockDim.x + threadIdx.x;
     if (id>=np)return;
     out[id]+=k*in[id];
}


void gen_x0_gpu(double *v, double *d_c_list, double *d_x0, int space, int np){
     //einsum_cx(v + (space - 1), c_list+(space - 1)*np, x0, np);
     int nthread=256;
     int nblock=(np+nthread-1)/nthread;
     Dscal_kernel<<<nblock, nthread>>>(d_c_list+(space - 1)*np, d_x0, v[space - 1], np);
     for (int i = space - 2; i >= 0; i--) {
	     Dscalplus_kernel<<<nblock, nthread>>>(d_c_list+i*np, d_x0, v[i], np);
     }
     //cudaMemcpy(x0, d_x0, sizeof(double) * np, cudaMemcpyDeviceToHost);
}



__global__ void dr_kernel(double k, int np, double *d_ci0, double *d_ci1, double *d_citmp){
       int id=blockIdx.x * blockDim.x + threadIdx.x;
       if (id>=np)return;
       d_citmp[id] = d_ci1[id] - k*d_ci0[id];
}




__global__ void precond_kernel(double* diag, double* dx, double e, double level_shift, int np){
    int i=blockIdx.x * blockDim.x + threadIdx.x;
    if (i>=np)return;
    double diag_val = diag[i] - (e - level_shift);
    double abs_diag_val = fabs(diag_val);
    dx[i] = dx[i] / (abs_diag_val < 1e-8 ? 1e-8 : diag_val);      
    
}


__global__ void Dscalminus_kernel(double *in, double *out, double k,  int np){
     int id=blockIdx.x * blockDim.x + threadIdx.x;
     if (id>=np) return;
     out[id]-=k*in[id];
}
__global__ void Ddiv_kernel(double *in, double *out, double k,  int np){
     int id=blockIdx.x * blockDim.x + threadIdx.x;
     if (id>=np) return;
     out[id]=in[id]/k;
}
void normalize_xt_gpu(cublasHandle_t handle, double* d_ci0, double* d_ci0_list, double lindep, double norm_min, int space, int np){
     int i;
     double tmp, norm;
     int nthread=256;
     int nblock=(np+nthread-1)/nthread;
     for (i=0; i<space; i++){
           cublasDdot(handle, np, d_ci0_list+i*np, 1, d_ci0, 1, &tmp);
	   Dscalminus_kernel<<<nblock, nthread>>>(d_ci0_list+i*np, d_ci0, tmp, np);
     }
     cublasDdot(handle, np, d_ci0, 1, d_ci0, 1, &tmp);
     norm = sqrt(tmp); //pow(tmp, 0.5);
     if (pow(norm, 2)>lindep){
         Ddiv_kernel<<<nblock, nthread>>>(d_ci0, d_ci0, norm, np);
     }

}



__global__ void cab_kernel(const double* __restrict__ d_ci0, 
                                                  double* __restrict__ d_t1, 
                                                  int bcount, int stra_idb, int strb_idb, 
                                                  int norb, int na, int nb, 
                                                  int nlinka, int nlinkb, 
                                                  int* d_link_nnorb
                                                )
{
        int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
        if (thread_id >= na * bcount) {
                return;
        }
        
        int nnorb = norb * (norb + 1) / 2;
        int stra_id = thread_id / bcount;
        int str0 = thread_id % bcount;
        
        // Pre-calculate base addresses
        int ci0_base_stra = stra_id * na;
        int t1_base = stra_id * bcount + str0;
        
        for (int j = 0; j < nnorb; j++) {
                int str1b = d_link_nnorb[2*(j*na+str0)];
                int8_t signb = d_link_nnorb[2*(j*na+str0)+1];
                int str1a = d_link_nnorb[2*(j*na+stra_id)];
                int8_t signa = d_link_nnorb[2*(j*na+stra_id)+1];
                
                double val_a = signa * d_ci0[str1a * na + str0];
                double val_b = signb * d_ci0[ci0_base_stra + str1b];
                
                d_t1[j * na * bcount + t1_base] = val_a + val_b;
        }
}




__global__ void sigab_kernel(double* __restrict__ d_ci1, 
                                                        const double* __restrict__ d_vt1, 
                                                        int bcount, int stra_idb, int strb_idb, 
                                                        int norb, int na, int nb, int nlinka, int nlinkb, 
                                                        const int* __restrict__ d_clink_index)
{
        int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
        if (thread_id >= na * bcount) {
                return;
        }
        
        unsigned int stra_id = thread_id / bcount;
        unsigned int str0 = thread_id % bcount;
        
        // Pre-calculate base pointers
        const int* tabb = d_clink_index + (str0 + strb_idb) * nlinkb * 3;
        const int* taba = d_clink_index + (stra_id + stra_idb) * nlinka * 3;
        
        // Pre-calculate frequently used offsets
        int vt1_base_stra = stra_id * bcount;
        int bcount_na = bcount * na;
        
        double val = 0.0;
        
        for (int j = 0; j < nlinkb; j++) {
                unsigned short iab = (unsigned short)tabb[j * 3 + 0];
                unsigned int str1b = (unsigned int)tabb[j * 3 + 1];
                int8_t signa = (int8_t)taba[j * 3 + 2];

                unsigned short iaa = (unsigned short)taba[j * 3 + 0];
                unsigned int str1a = (unsigned int)taba[j * 3 + 1];
                int8_t signb = (int8_t)tabb[j * 3 + 2];
                
                val += signb * d_vt1[vt1_base_stra + iab * bcount_na + str1b] + 
                           signa * d_vt1[str1a * bcount + iaa * bcount_na + str0];
        }
        
        d_ci1[(stra_id + stra_idb) * nb + str0] = val;
}





void contract_2e_spin1_gpu(cublasHandle_t handle, double* d_eri, double* d_ci0, double* d_ci1, 
                                                  double* d_t1, double* d_vt1, int norb, int na, int nb, int nlink, 
                                                  int reset_state, int* d_clink, int* d_link_nnorb, 
                                                  double* d_ci0_list, double* d_ci1_list, int space, int nroots, 
                                                  int heff_size, int np)
{
        if (reset_state == 1) {
                printf("reset state space:%d\n", space);
                return;
        }
        
        const int nnorb = norb * (norb + 1) / 2;
        const int na_chunk = na;
        const int nb_chunk = na;
        const double D0 = 0.0, D1 = 1.0;
        
        // Optimize thread block size based on GPU occupancy
        const int threadsPerBlock = min(256, na_chunk);
        const int bcountn = nb_chunk * na_chunk;
        const int newblocks_all = (bcountn + threadsPerBlock - 1) / threadsPerBlock;
        
        for (int strb_id = 0; strb_id < nb; strb_id += nb_chunk) {
                int current_nb = min(nb - strb_id, nb_chunk);
                
                for (int stra_id = 0; stra_id < na; stra_id += na_chunk) {
                        int current_na = min(na - stra_id, na_chunk);
                        
                        // Compute intermediate tensor t1
                        cab_kernel<<<newblocks_all, threadsPerBlock>>>(
                                d_ci0, d_t1, current_nb, stra_id, strb_id, norb, 
                                current_na, nb, nlink, nlink,  d_link_nnorb);
                        
                        // Contract with 2-electron integrals
                        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, bcountn, nnorb, nnorb, &D1,
                                           d_t1, bcountn, d_eri, nnorb, &D0, d_vt1, bcountn);
                        
                        // Compute final result
                        sigab_kernel<<<newblocks_all, threadsPerBlock>>>(
                                d_ci1, d_vt1, current_nb, stra_id, strb_id, norb, 
                                current_na, nb, nlink, nlink, d_clink);
                }
        }
}




void davidson(cublasHandle_t handle, cusolverDnHandle_t cusolverH, 
                          double* d_ci0_list, double* d_ci1_list, double* d_ci0, double* d_ci1, 
                          double* d_t1, double* d_vt1, double* eri, double* d_hdiag, 
                          int* link_index, int* d_link_nnorb,  
                          double* e, double* dr, double* d_heff, double* d_heff_tmp, 
                          double* v, double* v_last, int* dev_Info, double* d_W,
                          int nroots, int max_space, int np, int na, int norb, 
                          int nelec, int nlinka, int heff_size, double E_rhf)
{
        // Convergence parameters for Davidson iteration
        double tol = 1e-10, lindep = 1e-10, level_shift = 1e-3;
        double toloose = sqrt(tol) / 100;
        double dx_norm, de, e_last, dr_result;
        int conv_last;
        int nthread = 256;
        int nblock = (np + nthread - 1) / nthread;
        int space = 0, conv = 0, reset_state = 0, max_cycle = 100;
        
        // Main Davidson iteration loop
        for (int icyc = 0; icyc < max_cycle; icyc++) {
                
                // Step 1: Compute matrix-vector product H*v (2-electron interaction)
                contract_2e_spin1_gpu(handle, eri, d_ci0, d_ci1, d_t1, d_vt1, norb, na, na, 
                                                         nlinka, reset_state, link_index, d_link_nnorb,  
                                                         d_ci0_list, d_ci1_list, space, nroots, heff_size, np);
                
                // Step 2: Store current vectors in subspace
                cudaMemcpy(d_ci1_list + space * np, d_ci1, sizeof(double) * np, cudaMemcpyDeviceToDevice);
                cudaMemcpy(d_ci0_list + space * np, d_ci0, sizeof(double) * np, cudaMemcpyDeviceToDevice);
                
                space += 1;
                // Step 3: Build effective Hamiltonian matrix in the subspace
                fill_heff_hermitian_gpu(handle, d_heff_tmp, d_heff, d_ci0, d_ci1, d_ci1_list, 
                                                           space, nroots, heff_size, np);
                
                e_last = e[0];

                // Save previous iteration results for convergence check
                memcpy(v_last, v, space * nroots * sizeof(double));
                conv_last = conv;
                
                // Step 4: Solve eigenvalue problem for the effective Hamiltonian
                computeEigenvaluesAndVectorsn(cusolverH, space, d_heff_tmp, e, v, dev_Info, d_W);
                
                // Step 5: Generate improved trial vectors from eigenvectors
                gen_x0_gpu(v, d_ci0_list, d_ci0, space, np);
                gen_x0_gpu(v, d_ci1_list, d_ci1, space, np);
                
                reset_state = 0;
                
                // Energy change from last iteration
                de = (icyc == 0) ? e[0] : e[0] - e_last;

                // Step 6: Check if subspace is full
                if (space >= max_space) {
                        if (fabs(de) >= tol) {
                                // Not converged, restart with current best vector
                                space = 0;
                                reset_state = 1;
                                continue;
                        } else {
                                // Compute residual: r = H*v - e*v
                                dr_kernel<<<nblock, nthread>>>(e[0], np, d_ci0, d_ci1, dr);
                                cublasDdot(handle, np, dr, 1, dr, 1, &dr_result);
                                dx_norm = sqrt(fabs(dr_result));
                                conv = (fabs(de) < tol && dx_norm < toloose) ? 1 : 0;
                                printf("icyc:%d, dx_norm:%f, e:%f, de:%f\n", icyc, dx_norm, e[0], de);
                                if (conv == 1) {
                                        break;
                                } else {
                                        space = 0;
                                        reset_state = 1;
                                        continue;
                                }
                        }
                }
                
                // Step 7: Compute residual and check convergence
                dr_kernel<<<nblock, nthread>>>(e[0], np, d_ci0, d_ci1, d_ci0);
                cublasDdot(handle, np, d_ci0, 1, d_ci0, 1, &dr_result);
                dx_norm = sqrt(fabs(dr_result));
                conv = (fabs(de) < tol && dx_norm < toloose) ? 1 : 0;
                printf("icyc:%d, dx_norm:%f, FCI_E:%.15f, Correction_E:%.15f, de:%.15f\n", icyc, dx_norm, e[0]+E_rhf, e[0], de);
                
                // Check for stable convergence (converged in consecutive iterations)
                if (conv == 1 && conv_last == 0) {
                        printf("Converged: dx_norm:%f, FCI_E:%.15f, Correction_E:%.15f, de:%.15f\n", dx_norm, e[0]+E_rhf, e[0], de);
                        break;
                }
                
                // Step 8: Apply preconditioner if not converged and residual is significant
                if (conv == 0 && dx_norm * dx_norm > lindep) {
                        precond_kernel<<<nblock, nthread>>>(d_hdiag, d_ci0, e[0], level_shift, np);
                        cublasDdot(handle, np, d_ci0, 1, d_ci0, 1, &dr_result);
                        double tmpk = 1.0 / sqrt(dr_result);
                        cublasDscal(handle, np, &tmpk, d_ci0, 1);
                } else {
                        break;
                }
                
                // Step 9: Orthogonalize new vector against existing subspace
                normalize_xt_gpu(handle, d_ci0, d_ci0_list, lindep, 1.0, space, np);
        }
}

__global__ void jkcopy_kernel(double *d_Gmo, double *d_jdiag, double *d_kdiag, int norb, int norb_sq, int norb_t){
       int k=blockIdx.x * blockDim.x + threadIdx.x;
       if (k>=norb_sq) return;
       int i = k/norb;
       int j = k%norb;
       d_jdiag[i*norb + j] = d_Gmo[i*norb_t + i*norb_sq + j*norb + j];
       d_kdiag[i*norb + j] = d_Gmo[i*norb_t + j*norb_sq + j*norb + i];
}

double fci(double* d_Gmo1e, double* d_Gmo, int norb, int nelec, int na, long long np, double E_rhf)
{
        struct timeval tv_begin, tv_end;
        gettimeofday(&tv_begin, NULL);
        using gansu::tracked_cudaMalloc;
        using gansu::tracked_cudaFree;
        int neleca = nelec/2;
        
        int nroots = 1;
        int norb_sq = norb * norb;
        int norb_t = norb_sq * norb;
        int nnorb = norb * (norb + 1) / 2;
        
        int *occslst = (int*)malloc(na * neleca * sizeof(int));
        int *orb_list = (int*)malloc(norb * sizeof(int));
        double E_fci;
        double lindep = 1e-10;

        int index = 0;
        range(orb_list, norb);
        gen_occs_iter_ci_new(occslst, &index, orb_list, neleca, norb);

        // Allocate device memory and setup jdiag/kdiag
        double *d_jdiag, *d_kdiag,  *d_hdiag;

        tracked_cudaMalloc(&d_jdiag, norb_sq * sizeof(double));
        tracked_cudaMalloc(&d_kdiag, norb_sq * sizeof(double));
        tracked_cudaMalloc(&d_hdiag, np * sizeof(double)); 

        jkcopy_kernel<<<norb_sq, 1>>>(d_Gmo, d_jdiag, d_kdiag, norb, norb_sq, norb_t);

        // Compute hdiag
        int* d_occslst;
        tracked_cudaMalloc((void**)&d_occslst, na * neleca * sizeof(int));
        cudaMemcpy(d_occslst, occslst, na * neleca * sizeof(int), cudaMemcpyHostToDevice);
        int nthread = 256;
        int nblock = (np + nthread - 1) / nthread;
        FCImake_hdiag_uhf_kernel<<<nblock, nthread>>>(d_hdiag, np, d_Gmo1e, d_jdiag, d_kdiag, norb, na, neleca, d_occslst);

        // Generate link index
        int nlinka = neleca + neleca * (norb - neleca);
        int *d_clink, *d_link_nnorb;
        tracked_cudaMalloc(&d_clink, nlinka * na * 3 * sizeof(int));
        tracked_cudaMalloc(&d_link_nnorb, nnorb * na * 2 * sizeof(int));
        cudaMemset(d_clink, 0, nlinka * na * 3 * sizeof(int));
        cudaMemset(d_link_nnorb, 0, nnorb * na * 2 * sizeof(int));
        gen_linkstr_index(neleca, norb, na,  d_occslst,  d_clink, d_link_nnorb);

        int max_space = 12;
        int heff_size = max_space + nroots;
        double *e = (double*)calloc(heff_size, sizeof(double));
        double *v = (double*)malloc(heff_size * heff_size * sizeof(double));
        double *v_last = (double*)calloc(heff_size * nroots, sizeof(double));

        double *d_ci0_list, *d_ci1_list, *d_eri, *d_ci0, *d_ci1, *d_t1, *d_vt1, *dr, *d_heff, *d_heff_tmp;
        tracked_cudaMalloc(&d_ci0_list, max_space * np * sizeof(double));
        tracked_cudaMalloc(&d_ci1_list, max_space * np * sizeof(double));
        tracked_cudaMalloc((void**)&d_eri, nnorb * nnorb * sizeof(double));
        tracked_cudaMalloc((void**)&d_ci0, np * sizeof(double));
        tracked_cudaMalloc((void**)&d_ci1, np * sizeof(double));
        tracked_cudaMalloc((void**)&d_t1, np * nnorb * sizeof(double));
        tracked_cudaMalloc((void**)&d_vt1, np * nnorb * sizeof(double));
        tracked_cudaMalloc((void**)&dr, np * sizeof(double));
        tracked_cudaMalloc((void**)&d_heff, heff_size * heff_size * sizeof(double));
        tracked_cudaMalloc((void**)&d_heff_tmp, heff_size * heff_size * sizeof(double));

        int *devInfo;
        double *d_W;
        tracked_cudaMalloc((void**)&devInfo, sizeof(int));
        tracked_cudaMalloc((void**)&d_W, max_space * sizeof(double));

        // Initialize ci0
        cudaMemset(d_ci0, 0, np * sizeof(double));
        double firstv = 1 + 1e-5;
        double lastv = -1e-5;
        cudaMemcpy(&d_ci0[0], &firstv, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(&d_ci0[np-1], &lastv, sizeof(double), cudaMemcpyHostToDevice);

        absorb_h1e(d_Gmo1e, d_Gmo, d_eri, norb, nelec, 0.5);

        // Normalize ci0
        cublasHandle_t handle;
        cublasCreate(&handle);
        double innerprod;
        cublasDdot(handle, np, d_ci0, 1, d_ci0, 1, &innerprod);
        double norm = sqrt(innerprod);
        if (innerprod > lindep && norm > 1e-14) {
            qr_decomposition_kernel<<<nblock, nthread>>>(d_ci0, np, lindep, norm);
            cudaDeviceSynchronize();
        }

        cusolverDnHandle_t cusolverH;
        cusolverDnCreate(&cusolverH);
        struct timeval tv_davidson_begin, tv_davidson_end;
        gettimeofday(&tv_davidson_begin, NULL);
        
        davidson(handle, cusolverH, d_ci0_list, d_ci1_list, d_ci0, d_ci1, d_t1, d_vt1, 
            d_eri, d_hdiag, d_clink, d_link_nnorb, e, dr, d_heff, d_heff_tmp, 
            v, v_last, devInfo, d_W, nroots, max_space, np, na, norb, 
            neleca, nlinka, heff_size, E_rhf);
        
        gettimeofday(&tv_davidson_end, NULL);
        float t_davidson = (float)(tv_davidson_end.tv_usec - tv_davidson_begin.tv_usec) / 1e6 + 
                (float)(tv_davidson_end.tv_sec - tv_davidson_begin.tv_sec);
        printf("Davidson time: %.3f s\n", t_davidson);

        E_fci = e[0];

        // Cleanup
        cublasDestroy(handle);
        cusolverDnDestroy(cusolverH);

   
        free(v);
        free(v_last);
        free(e);
        free(occslst);
        free(orb_list);

        tracked_cudaFree(d_ci0_list);
        tracked_cudaFree(d_ci1_list);
        tracked_cudaFree(d_eri);
        tracked_cudaFree(d_ci0);
        tracked_cudaFree(d_ci1);
        tracked_cudaFree(d_t1);
        tracked_cudaFree(d_vt1);
        tracked_cudaFree(d_clink);
        tracked_cudaFree(dr);
        tracked_cudaFree(d_hdiag);
        tracked_cudaFree(d_heff);
        tracked_cudaFree(d_heff_tmp);
        tracked_cudaFree(devInfo);
        tracked_cudaFree(d_W);
        tracked_cudaFree(d_jdiag);
        tracked_cudaFree(d_kdiag);
        tracked_cudaFree(d_occslst);
        tracked_cudaFree(d_link_nnorb);

        gettimeofday(&tv_end, NULL);
        float t_fci = (float)(tv_end.tv_usec - tv_begin.tv_usec) / 1e6 + 
                    (float)(tv_end.tv_sec - tv_begin.tv_sec);
        printf("Total FCI time: %.3f s\n", t_fci);

        return E_fci;
}


