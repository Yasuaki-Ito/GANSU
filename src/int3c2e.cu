#include <cuda.h>

#include "int3c2e.hpp"
#include "boys.hpp"
#include "parameters.h"
#include "types.hpp"
#include "utils_cuda.hpp"

#include "int2e.hpp"
#include "Et_functions.hpp"



namespace gansu::gpu{
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
// 3-center integrals [ss|s]~[pp|d]
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------//


/* (ss|s) */
__global__ void calc_sss_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
							const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, 
							ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
							int64_t num_tasks, int num_basis, 
							const size_t2* d_primitive_shell_pair_indices,
							const double* g_upper_bound_factors, 
							const double* g_auxiliary_upper_bound_factors, 
							const double schwarz_screening_threshold, 
							int num_auxiliary_basis, 
							const double* g_boys_grid){
                                
	//uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	size_t idx = (size_t)blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
	    // screening (suzuki)
	    if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[1];

		getIncrementalBoys(0, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);

		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 0, 0) 
		                     * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) 
							 * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		
        bool is_prim_id_not_equal = a!=b;

		#include "./integral_RI/int3c2e/orig_sss.txt"
	}
}



/* (ss|p) */
__global__ void calc_ssp_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {
	//uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	size_t idx = (size_t)blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
	    // screening (suzuki)
	    if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[2];
		getIncrementalBoys(1, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 0, 1) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		
        bool is_prim_id_not_equal = a!=b;

		#include "./integral_RI/int3c2e/orig_ssp.txt"
	}
}



/* (ss|d) */
__global__ void calc_ssd_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {
	//uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	size_t idx = (size_t)blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
	    // screening (suzuki)
	    if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

        // printf("ssd| %d: %d %d %d\n",threadIdx.x, (int)primitive_index_a,(int)primitive_index_b,(int)primitive_index_c);

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[3];
		getIncrementalBoys(2, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 0, 2) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
        bool is_prim_id_not_equal = a!=b;

		#include "./integral_RI/int3c2e/orig_ssd.txt"
	}
}



/* (ss|f) */
__global__ void calc_ssf_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {
	//uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	size_t idx = (size_t)blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
	    // screening (suzuki)
	    if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

        // printf("ssf| %d: %d %d %d\n",threadIdx.x, (int)primitive_index_a,(int)primitive_index_b,(int)primitive_index_c);

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[4];
		getIncrementalBoys(3, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 0, 3) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		
        bool is_prim_id_not_equal = a!=b;

		#include "./integral_RI/int3c2e/orig_ssf.txt"
	}
}



/* (sp|s) */
__global__ void calc_sps_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {
	//uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	size_t idx = (size_t)blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
	    // screening (suzuki)
	    if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[2];
		getIncrementalBoys(1, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 1, 0) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		bool is_prim_id_not_equal = a!=b;
		#include "./integral_RI/int3c2e/orig_sps.txt"
	}
}



/* (sp|p) */
__global__ void calc_spp_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {
	//uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	size_t idx = (size_t)blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
	    // screening (suzuki)
	    if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[3];
		getIncrementalBoys(2, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 1, 1) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		bool is_prim_id_not_equal = a!=b;
		#include "./integral_RI/int3c2e/orig_spp.txt"
	}
}



/* (sp|d) */
__global__ void calc_spd_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {
	//uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	size_t idx = (size_t)blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
	    // screening (suzuki)
	    if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[4];
		getIncrementalBoys(3, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 1, 2) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		bool is_prim_id_not_equal = a!=b;
		
		#include "./integral_RI/int3c2e/orig_spd.txt"
	}
}



/* (sp|f) */
__global__ void calc_spf_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {
	//uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	size_t idx = (size_t)blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
	    // screening (suzuki)
	    if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[5];
		getIncrementalBoys(4, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 1, 3) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		bool is_prim_id_not_equal = a!=b;
		#include "./integral_RI/int3c2e/orig_spf.txt"
	}
}




/* (pp|s) */
__global__ void calc_pps_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {
	//uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	size_t idx = (size_t)blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
	    // screening (suzuki)
	    if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[3];
		getIncrementalBoys(2, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 1, 1, 0) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		
        bool is_prim_id_not_equal = a!=b;

		#include "./integral_RI/int3c2e/orig_pps.txt"
	}
}



/* (pp|p) */
__global__ void calc_ppp_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {
	//uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	size_t idx = (size_t)blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
	    // screening (suzuki)
	    if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[4];
		getIncrementalBoys(3, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 1, 1, 1) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		
        bool is_prim_id_not_equal = a!=b;

		#include "./integral_RI/int3c2e/orig_ppp.txt"
	}
}



/* (pp|d) */
__global__ void calc_ppd_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {
	//uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	size_t idx = (size_t)blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
	    // screening (suzuki)
	    if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[5];
		getIncrementalBoys(4, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 1, 1, 2) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		
        bool is_prim_id_not_equal = a!=b;

		#include "./integral_RI/int3c2e/orig_ppd.txt"
	}
}



/* (pp|f) */
__global__ void calc_ppf_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {
	//uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	size_t idx = (size_t)blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
	    // screening (suzuki)
	    if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[6];
		getIncrementalBoys(5, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 1, 1, 3) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		
        bool is_prim_id_not_equal = a!=b;

		#include "./integral_RI/int3c2e/orig_ppf.txt"
	}
}



#if defined(COMPUTE_D_BASIS)
/* (sd|s) */
__global__ void calc_sds_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {
	//uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	size_t idx = (size_t)blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
	    // screening (suzuki)
	    if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[3];
		getIncrementalBoys(2, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 2, 0) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		bool is_prim_id_not_equal = a!=b;
		#include "./integral_RI/int3c2e/orig_sds.txt"
	}
}


/* (sd|p) */
__global__ void calc_sdp_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {
	//uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	size_t idx = (size_t)blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
	    // screening (suzuki)
	    if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[4];
		getIncrementalBoys(3, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 2, 1) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		bool is_prim_id_not_equal = a!=b;
		#include "./integral_RI/int3c2e/orig_sdp.txt"
	}
}



/* (sd|d) */
__global__ void calc_sdd_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {
	//uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	size_t idx = (size_t)blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
	    // screening (suzuki)
	    if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[5];
		getIncrementalBoys(4, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 2, 2) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		bool is_prim_id_not_equal = a!=b;
		#include "./integral_RI/int3c2e/orig_sdd.txt"
	}
}



/* (sd|f) */
__global__ void calc_sdf_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {
	//uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	size_t idx = (size_t)blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
	    // screening (suzuki)
	    if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[6];
		getIncrementalBoys(5, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 2, 3) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		bool is_prim_id_not_equal = a!=b;
		#include "./integral_RI/int3c2e/orig_sdf.txt"
	}
}

/* (pd|s) */
__global__ void calc_pds_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {
	//uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	size_t idx = (size_t)blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
	    // screening (suzuki)
	    if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[4];
		getIncrementalBoys(3, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 1, 2, 0) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		bool is_prim_id_not_equal = a!=b;
		#include "./integral_RI/int3c2e/orig_pds.txt"
	}
}


/* (pd|p) */
__global__ void calc_pdp_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {
	//uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	size_t idx = (size_t)blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
	    // screening (suzuki)
	    if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[5];
		getIncrementalBoys(4, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 1, 2, 1) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		bool is_prim_id_not_equal = a!=b;
		#include "./integral_RI/int3c2e/orig_pdp.txt"
	}
}



/* (pd|d) */
__global__ void calc_pdd_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {
	//uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	size_t idx = (size_t)blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
	    // screening (suzuki)
	    if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[6];
		getIncrementalBoys(5, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 1, 2, 2) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		bool is_prim_id_not_equal = a!=b;
		#include "./integral_RI/int3c2e/orig_pdd.txt"
	}
}



/* (pd|f) */
__global__ void calc_pdf_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {
	//uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	size_t idx = (size_t)blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
	    // screening (suzuki)
	    if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[7];
		getIncrementalBoys(6, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 1, 2, 3) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		bool is_prim_id_not_equal = a!=b;
		#include "./integral_RI/int3c2e/orig_pdf.txt"
	}
}


/* (dd|s) */
__global__ void calc_dds_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {
	//uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	size_t idx = (size_t)blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
	    // screening (suzuki)
	    if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[5];
		getIncrementalBoys(4, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 2, 2, 0) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		
       bool is_prim_id_not_equal = a!=b;
		#include "./integral_RI/int3c2e/orig_dds.txt"
	}
}



/* (dd|p) */
__global__ void calc_ddp_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {
	//uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	size_t idx = (size_t)blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
	    // screening (suzuki)
	    if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[6];
		getIncrementalBoys(5, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 2, 2, 1) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		
       bool is_prim_id_not_equal = a!=b;
		#include "./integral_RI/int3c2e/orig_ddp.txt"
	}
}



/* (dd|d) */
__global__ void calc_ddd_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {
	//uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	size_t idx = (size_t)blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
	    // screening (suzuki)
	    if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[7];
		getIncrementalBoys(6, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 2, 2, 2) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		
       bool is_prim_id_not_equal = a!=b;
		#include "./integral_RI/int3c2e/orig_ddd.txt"
	}
}


/* (dd|f) */
__global__ void calc_ddf_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {
	//uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	size_t idx = (size_t)blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
	    // screening (suzuki)
	    if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[8];
		getIncrementalBoys(7, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 2, 2, 3) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		
        bool is_prim_id_not_equal = a!=b;
		#include "./integral_RI/int3c2e/orig_ddf.txt"
	}
}

#else
/* (dd|f) */
__global__ void calc_ddf_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {}
/* (dd|d) */
__global__ void calc_ddd_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {}
/* (dd|p) */
 __global__ void calc_ddp_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {}
/* (dd|s) */
 __global__ void calc_dds_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {}
/* (pd|f) */
 __global__ void calc_pdf_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {}
/* (pd|d) */
 __global__ void calc_pdd_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {}
/* (pd|p) */
 __global__ void calc_pdp_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {}
/* (pd|s) */
 __global__ void calc_pds_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {}
/* (sd|f) */
 __global__ void calc_sdf_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {}
/* (sd|d) */
 __global__ void calc_sdd_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {}
/* (sd|p) */
 __global__ void calc_sdp_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {}
/* (sd|s) */
 __global__ void calc_sds_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {}
#endif




#if defined(COMPUTE_D_BASIS) && defined(COMPUTE_G_AUX)
/* (sd|g) */
__global__ void calc_sdg_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];
		// screening (suzuki)
		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[7];
		getIncrementalBoys(6, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 2, 4) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		bool is_prim_id_not_equal = a!=b;
		#include "./integral_RI/int3c2e/orig_sdg.txt"
	}
}

/* (pd|g) */
__global__ void calc_pdg_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;
		// screening (suzuki)
		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[8];
		getIncrementalBoys(7, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 1, 2, 4) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		bool is_prim_id_not_equal = a!=b;
		#include "./integral_RI/int3c2e/orig_pdg.txt"
	}
}


/* (dd|g) */
__global__ void calc_ddg_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
		const size_t primitive_index_c = abc.y + shell_s2.start_index;
		// screening (suzuki)
		if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[9];
		getIncrementalBoys(8, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 2, 2, 4) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		bool is_prim_id_not_equal = a!=b;
		#include "./integral_RI/int3c2e/orig_ddg.txt"
	}
}

#else
__global__ void calc_sdg_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {}
/* (dd|g) */
__global__ void calc_ddg_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {}
/* (pd|g) */
__global__ void calc_pdg_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {}
#endif



#if defined(COMPUTE_G_AUX)
/* (ss|g) */
__global__ void calc_ssg_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
	    // screening (suzuki)
	    if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[5];
		getIncrementalBoys(4, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 0, 4) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		
        bool is_prim_id_not_equal = a!=b;

		#include "./integral_RI/int3c2e/orig_ssg.txt"
	}
}

/* (sp|g) */
__global__ void calc_spg_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
	    // screening (suzuki)
	    if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[6];
		getIncrementalBoys(5, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 0, 1, 4) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		bool is_prim_id_not_equal = a!=b;
		#include "./integral_RI/int3c2e/orig_spg.txt"
	}
}

/* (pp|g) */
__global__ void calc_ppg_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_tasks){
		const size_t2 abc = index1to2(idx, false, shell_s2.count);
		
		const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
		const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
        const size_t primitive_index_c = abc.y + shell_s2.start_index;
	    // screening (suzuki)
	    if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;

		const PrimitiveShell *a = &g_pshell[primitive_index_a];
		const PrimitiveShell *b = &g_pshell[primitive_index_b];
		const PrimitiveShell *c = &g_pshell_aux[primitive_index_c];

		double p = a->exponent + b->exponent;
		double sum_exp = p + c->exponent;
		double Rp[3] = {(a->exponent*a->coordinate.x + b->exponent*b->coordinate.x)/p, (a->exponent*a->coordinate.y + b->exponent*b->coordinate.y)/p, (a->exponent*a->coordinate.z + b->exponent*b->coordinate.z)/p};
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};
		double Rpa[3] = {(Rp[0] - a->coordinate.x), (Rp[1] - a->coordinate.y), (Rp[2] - a->coordinate.z)};
		double Rpc[3] = {(Rp[0] - c->coordinate.x), (Rp[1] - c->coordinate.y), (Rp[2] - c->coordinate.z)};
		double Boys[7];
		getIncrementalBoys(6, p*c->exponent/(p+c->exponent)*((Rpc[0])*(Rpc[0]) + (Rpc[1])*(Rpc[1]) + (Rpc[2])*(Rpc[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient* c->coefficient * calcNormsWOFact2_3center(a->exponent, b->exponent, c->exponent, 1, 1, 4) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER/(p*c->exponent*sqrt(p+c->exponent)) * exp(-(a->exponent*b->exponent)*((a->coordinate.x-b->coordinate.x)*(a->coordinate.x-b->coordinate.x) + (a->coordinate.y-b->coordinate.y)*(a->coordinate.y-b->coordinate.y) + (a->coordinate.z-b->coordinate.z)*(a->coordinate.z-b->coordinate.z))/p);
		
        bool is_prim_id_not_equal = a!=b;

		#include "./integral_RI/int3c2e/orig_ppg.txt"
	}
}

#else
/* (ss|g) */
__global__ void calc_ssg_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {}
/* (sp|g) */	
__global__ void calc_spg_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {}
/* (pp|g) */
__global__ void calc_ppg_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {}
#endif

















__global__ void MD_int3c2e_1T1SP(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, 
                                 const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, 
                                 ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, 
                                 int64_t num_tasks, int num_basis, 
								 const size_t2* d_primitive_shell_pair_indices,
								 const double* g_upper_bound_factors, 
								 const double* g_auxiliary_upper_bound_factors, 
								 const double schwarz_screening_threshold, 
								 int num_auxiliary_basis, 
                                 const double* g_boys_grid){
{
    // 通し番号indexの計算
    //const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;


    if (id >= num_tasks) return;

    const double size_Rmid=1377;

    //使い捨ての中間体R_mid
    double R_mid[3*1377];

    //解を格納する配列R
    double R[2925];

    //thread内で結果を保持するメモリ
    double thread_val=0.0;

    const size_t2 abc = index1to2(id, false, shell_s2.count);
    const size_t2 ab =  index1to2(abc.x, (shell_s0.start_index == shell_s1.start_index), shell_s1.count);


    // Obtain primitive shells [ab|c]
	const size_t primitive_index_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
	const size_t primitive_index_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
    const size_t primitive_index_c = abc.y + shell_s2.start_index;

	// screening (suzuki)
	if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[primitive_index_c] < schwarz_screening_threshold) return;


    const PrimitiveShell a = g_pshell[primitive_index_a];
    const PrimitiveShell b = g_pshell[primitive_index_b];
    const PrimitiveShell c = g_pshell_aux[primitive_index_c];

        
    // Obtain basis index (ij|k)
    const size_t size_a = a.basis_index;
    const size_t size_b = b.basis_index;
    const size_t size_c = c.basis_index;


    bool is_prim_id_not_equal = (primitive_index_a != primitive_index_b);

    
    //使用データを取得，レジスタに書き込み

    //指数部
    const double alpha = a.exponent;
    const double beta  = b.exponent;
    const double gamma = c.exponent;
    const double p = alpha+beta;
    const double xi = p*gamma / (p+gamma);

    //係数部
    const double coef_a = a.coefficient;
    const double coef_b = b.coefficient;
    const double coef_c = c.coefficient;

    //座標
    const double pos_A[3] = {a.coordinate.x, a.coordinate.y, a.coordinate.z};
    const double pos_B[3] = {b.coordinate.x, b.coordinate.y, b.coordinate.z};
    const double pos_C[3] = {c.coordinate.x, c.coordinate.y, c.coordinate.z};
    const double pos_P[3] = {(alpha*pos_A[0]+beta*pos_B[0])/(alpha+beta), (alpha*pos_A[1]+beta*pos_B[1])/(alpha+beta), (alpha*pos_A[2]+beta*pos_B[2])/(alpha+beta)};


    //角運動量の総和
    const int orbital_A = a.shell_type;
    const int orbital_B = b.shell_type;
    const int orbital_C = c.shell_type;


    //軌道間距離の二乗
    const double dist = ((pos_P[0]-pos_C[0])*(pos_P[0]-pos_C[0]) + (pos_P[1]-pos_C[1])*(pos_P[1]-pos_C[1]) + (pos_P[2]-pos_C[2])*(pos_P[2]-pos_C[2]));


    const int K = orbital_A + orbital_B + orbital_C;
    
    double Boys[25];
    getIncrementalBoys(K, xi*dist, g_boys_grid, Boys);

    //Boys関数の値を計算(Single)
    for(int i=0; i <= K; i++){
        Boys[i] *= (right2left_binary_woif((-2*xi), i));
    }

    //各ERIを計算
    //事前計算⇒実際のERI計算の順に実行
    //p軌道の場合lmn_aが0:px, 1:py, 2:pz軌道のように対応付け
    //d以上はconstant配列のloop_to_ang_RIを参照

    
    
    double Norm_A, Norm_B, Norm_C;
    double Norm;

    int t,u,v;
    int t_max;
    int u_max;
    int v_max;
    int tau_max;
    int nu_max;
    int phi_max;

    // int tid=0;


    // 方位量子数l,m,nの値をループ変数から導出
    for(int lmn_a=0; lmn_a<comb_max(orbital_A); lmn_a++){
        int l1=loop_to_ang_RI[orbital_A][lmn_a][0]; int m1=loop_to_ang_RI[orbital_A][lmn_a][1]; int n1=loop_to_ang_RI[orbital_A][lmn_a][2];
        Norm_A = calcNorm(alpha, l1, m1, n1);

        for(int lmn_b=0; lmn_b<comb_max(orbital_B); lmn_b++){                  
            int l2=loop_to_ang_RI[orbital_B][lmn_b][0]; int m2=loop_to_ang_RI[orbital_B][lmn_b][1]; int n2=loop_to_ang_RI[orbital_B][lmn_b][2];
            Norm_B = calcNorm(beta, l2, m2, n2);


            for(int lmn_c=0; lmn_c<comb_max(orbital_C); lmn_c++){
                int l3=loop_to_ang_RI[orbital_C][lmn_c][0]; int m3=loop_to_ang_RI[orbital_C][lmn_c][1]; int n3=loop_to_ang_RI[orbital_C][lmn_c][2];
                Norm_C = calcNorm(gamma, l3, m3, n3);


				Norm = Norm_A * Norm_B * Norm_C;
				// 前回のループの計算結果をクリア
				thread_val=0.0;
				// 事前計算部
				//初期値：Boysとして計算済
				//Step 0: Boys関数評価
				R[0]=Boys[0];
				for(int i=0; i <= K; i++){
					R_mid[i]=Boys[i];
				}
                    
				// ループ変数の設定
				t_max = l1+l2+1;
				u_max = m1+m2+1;
				v_max = n1+n2+1;
				tau_max = l3+1;
				nu_max = m3+1;
				phi_max = n3+1;

				for(int k=1; k <= K; k++){//Step 1~Kの計算
					// t+u+v=kとなる全ペアに対して適切な計算
					// 0~K-kまでそれぞれ必要⇒ループでやる
	
        
					for(int z=0; z<=(K+1)*comb_max(k); z++){
                        
						int i = z/comb_max(k);
	
						if(i <= K-k){
							t=tuv_list[(k*(k+1)*(k+2))/6 + z%comb_max(k)][0];
							u=tuv_list[(k*(k+1)*(k+2))/6 + z%comb_max(k)][1];
							v=tuv_list[(k*(k+1)*(k+2))/6 + z%comb_max(k)][2];
	
							if((t <= (t_max+tau_max-2)) && (u <= (u_max+nu_max-2)) && (v <= (v_max+phi_max-2))){
								if(t >= 1){
									R_mid[calc_Idx_Rmid(k,u,v,i,comb_max(k),size_Rmid)] = (pos_P[0] - pos_C[0])*R_mid[calc_Idx_Rmid(k-1,u,v,i+1,comb_max(k-1),size_Rmid)] + (t-1)*R_mid[calc_Idx_Rmid(k-2,u,v,i+1,comb_max(k-2),size_Rmid)];
								}
								else if(u >= 1){
									R_mid[calc_Idx_Rmid(k,u,v,i,comb_max(k),size_Rmid)] = (pos_P[1] - pos_C[1])*R_mid[calc_Idx_Rmid(k-1,u-1,v,i+1,comb_max(k-1),size_Rmid)] + (u-1)*R_mid[calc_Idx_Rmid(k-2,u-2,v,i+1,comb_max(k-2),size_Rmid)];
								}
								else{
									R_mid[calc_Idx_Rmid(k,u,v,i,comb_max(k),size_Rmid)] = (pos_P[2] - pos_C[2])*R_mid[calc_Idx_Rmid(k-1,u,v-1,i+1,comb_max(k-1),size_Rmid)] + (v-1)*R_mid[calc_Idx_Rmid(k-2,u,v-2,i+1,comb_max(k-2),size_Rmid)];
								}
							}
						}
					}//step kの全計算が終了
        

                        //必要な結果を配列Rに書き込み
                        for(int i=0; i<=comb_max(k); i++){
                            R[static_cast<int>(k*(k+1)*(k+2)/6) + i] = R_mid[(k%3)*static_cast<int>(size_Rmid) + i];
                        }

                    }
                    //事前計算完了
                    
					double my_val = 0.0;
                    // ERI計算部
					double Et, Eu, Ev, Etau, Enu, Ephi;
                    for(int t=0; t<l1+l2+1; t++){
                        Et =  MD_Et_NonRecursion(l1, l2, t, alpha, beta, (pos_A[0]-pos_B[0]));
                        for(int u=0; u<m1+m2+1; u++){
                            Eu = MD_Et_NonRecursion(m1, m2, u, alpha, beta, (pos_A[1]-pos_B[1]));
                            for(int v=0; v<n1+n2+1; v++){
                                Ev = MD_Et_NonRecursion(n1, n2, v, alpha, beta, (pos_A[2]-pos_B[2]));
                                for(int tau=0; tau<l3+1; tau++){
                                    Etau = MD_Et_NonRecursion(l3, 0, tau, gamma, 0.0, 0.0);
                                    for(int nu=0; nu<m3+1; nu++){
                                        Enu = MD_Et_NonRecursion(m3, 0, nu, gamma, 0.0, 0.0);
                                        for(int phi=0; phi<n3+1; phi++){
                                            Ephi = MD_Et_NonRecursion(n3, 0, phi, gamma, 0.0, 0.0);
                                            int k=t+u+v+tau+nu+phi;
                                            my_val +=  Et * Eu * Ev * Etau * Enu * Ephi * (1 - 2*((tau+nu+phi)&1)) * R[k*(k+1)*(k+2)/6 + calc_Idx_Rmid(k,u+nu,v+phi,0,0,0)];
                                            // my_val +=  Et * Eu * Ev * Etau * Enu * Ephi * (1 - 2*((tau+nu+phi)&1)) * R_GPU_Recursion(0, t+tau, u+nu, v+phi, P, Q, Boys);
                                        }
                                    }
                                }
                            }
                        }
                    }
					thread_val = my_val*2 * M_PI_2_5 /(p*gamma * sqrt((p+gamma)))  *coef_a*coef_b*coef_c;

                    // 書き込み部

                    // Global Memoryへ書き込み
                    // 汎用カーネルでは全要素判定(case1)
                    addToResult_3center(
                        Norm*thread_val,
                        g_result,
                        size_a+lmn_a, size_b+lmn_b, size_c+lmn_c,
                        num_basis, num_auxiliary_basis,
                        is_prim_id_not_equal, 
						d_cgto_nomalization_factors, d_auxiliary_cgto_nomalization_factors
                    );
                }
            }
        }
    }
    return;
}


} // namespace gansu::gpu