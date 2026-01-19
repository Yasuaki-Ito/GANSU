#include <cuda.h>

#include "int2c2e.hpp"
#include "boys.hpp"
#include "parameters.h"
#include "types.hpp"
#include "utils_cuda.hpp"

#include "int2e.hpp"
#include "Et_functions.hpp"



namespace gansu::gpu{

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
// 2-center integrals [s|s]~[d|d]
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------//



/* (s|s) */
__global__ void calc_ss_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, const double* g_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_shell_pairs){
		const size_t2 ab = index1to2(idx, true);
        const size_t primitive_index_a = ab.x+shell_s0.start_index;
        const size_t primitive_index_b = ab.y+shell_s1.start_index;
        // screening (suzuki)
	    if (g_upper_bound_factors[primitive_index_a] * g_upper_bound_factors[primitive_index_b] < schwarz_screening_threshold) return;

        const PrimitiveShell *a = &g_pshell_aux[primitive_index_a];
        const PrimitiveShell *b = &g_pshell_aux[primitive_index_b];

		// printf("[s|s] %d: %d %d\n", threadIdx.x, (int)primitive_index_a, (int)primitive_index_b);

		double sum_exponent = a->exponent + b->exponent;  
		
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};

		double Boys[1];
		getIncrementalBoys(0, a->exponent*b->exponent/sum_exponent * ((Rab[0])*(Rab[0]) + (Rab[1])*(Rab[1]) + (Rab[2])*(Rab[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient * calcNormsWOFact2_2center(a->exponent, b->exponent, 0, 0) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER / (a->exponent*b->exponent*sqrt(sum_exponent));
		bool is_prim_id_not_equal = (primitive_index_a != primitive_index_b);

		#include "./integral_RI/int2c2e/orig_ss.txt"
	}
}


/* (s|p) */
__global__ void calc_sp_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, const double* g_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_shell_pairs){
		size_t2 ab = index1to2(idx, false, shell_s1.count);
        const size_t primitive_index_a = ab.x+shell_s0.start_index;
        const size_t primitive_index_b = ab.y+shell_s1.start_index;
        // screening (suzuki)
	    if (g_upper_bound_factors[primitive_index_a] * g_upper_bound_factors[primitive_index_b] < schwarz_screening_threshold) return;

        const PrimitiveShell *a = &g_pshell_aux[primitive_index_a];
        const PrimitiveShell *b = &g_pshell_aux[primitive_index_b];

		// printf("[s|p] %d: %d %d\n", threadIdx.x, (int)primitive_index_a, (int)primitive_index_b);
		double sum_exponent = a->exponent + b->exponent;  
		
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};

		double Boys[2];
		getIncrementalBoys(1, a->exponent*b->exponent/sum_exponent * ((Rab[0])*(Rab[0]) + (Rab[1])*(Rab[1]) + (Rab[2])*(Rab[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient * calcNormsWOFact2_2center(a->exponent, b->exponent, 0, 1) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER / (a->exponent*b->exponent*sqrt(sum_exponent));
		bool is_prim_id_not_equal = (primitive_index_a != primitive_index_b);

		#include "./integral_RI/int2c2e/orig_sp.txt"
	}
}


/* (s|d) */
__global__ void calc_sd_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, const double* g_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_shell_pairs){
		size_t2 ab = index1to2(idx, false, shell_s1.count);
        const size_t primitive_index_a = ab.x+shell_s0.start_index;
        const size_t primitive_index_b = ab.y+shell_s1.start_index;
        // screening (suzuki)
	    if (g_upper_bound_factors[primitive_index_a] * g_upper_bound_factors[primitive_index_b] < schwarz_screening_threshold) return;

        const PrimitiveShell *a = &g_pshell_aux[primitive_index_a];
        const PrimitiveShell *b = &g_pshell_aux[primitive_index_b];

		// printf("[s|d] %d: %d %d\n", threadIdx.x, (int)primitive_index_a, (int)primitive_index_b);

		double sum_exponent = a->exponent + b->exponent;  
		
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};

		double Boys[3];
		getIncrementalBoys(2, a->exponent*b->exponent/sum_exponent * ((Rab[0])*(Rab[0]) + (Rab[1])*(Rab[1]) + (Rab[2])*(Rab[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient * calcNormsWOFact2_2center(a->exponent, b->exponent, 0, 2) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER / (a->exponent*b->exponent*sqrt(sum_exponent));
		bool is_prim_id_not_equal = (primitive_index_a != primitive_index_b);

		#include "./integral_RI/int2c2e/orig_sd.txt"
	}
}


/* (s|f) */
__global__ void calc_sf_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, const double* g_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_shell_pairs){
		size_t2 ab = index1to2(idx, false, shell_s1.count);
        const size_t primitive_index_a = ab.x+shell_s0.start_index;
        const size_t primitive_index_b = ab.y+shell_s1.start_index;
        // screening (suzuki)
	    if (g_upper_bound_factors[primitive_index_a] * g_upper_bound_factors[primitive_index_b] < schwarz_screening_threshold) return;

        const PrimitiveShell *a = &g_pshell_aux[primitive_index_a];
        const PrimitiveShell *b = &g_pshell_aux[primitive_index_b];

		double sum_exponent = a->exponent + b->exponent;  
		
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};

		double Boys[4];
		getIncrementalBoys(3, a->exponent*b->exponent/sum_exponent * ((Rab[0])*(Rab[0]) + (Rab[1])*(Rab[1]) + (Rab[2])*(Rab[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient * calcNormsWOFact2_2center(a->exponent, b->exponent, 0, 3) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER / (a->exponent*b->exponent*sqrt(sum_exponent));
		bool is_prim_id_not_equal = (primitive_index_a != primitive_index_b);

		#include "./integral_RI/int2c2e/orig_sf.txt"
	}
}


/* (p|p) */
__global__ void calc_pp_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, const double* g_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_shell_pairs){
		const size_t2 ab = index1to2(idx, true);
        const size_t primitive_index_a = ab.x+shell_s0.start_index;
        const size_t primitive_index_b = ab.y+shell_s1.start_index;
        // screening (suzuki)
	    if (g_upper_bound_factors[primitive_index_a] * g_upper_bound_factors[primitive_index_b] < schwarz_screening_threshold) return;

        const PrimitiveShell *a = &g_pshell_aux[primitive_index_a];
        const PrimitiveShell *b = &g_pshell_aux[primitive_index_b];

		// printf("[p|p] %d: %d %d\n", threadIdx.x, (int)primitive_index_a, (int)primitive_index_b);

		double sum_exponent = a->exponent + b->exponent;  
		
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};

		double Boys[3];
		getIncrementalBoys(2, a->exponent*b->exponent/sum_exponent * ((Rab[0])*(Rab[0]) + (Rab[1])*(Rab[1]) + (Rab[2])*(Rab[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient * calcNormsWOFact2_2center(a->exponent, b->exponent, 1, 1) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER / (a->exponent*b->exponent*sqrt(sum_exponent));
		bool is_prim_id_not_equal = (primitive_index_a != primitive_index_b);
		#include "./integral_RI/int2c2e/orig_pp.txt"
	}
}


/* (p|d) */
__global__ void calc_pd_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, const double* g_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_shell_pairs){
		size_t2 ab = index1to2(idx, false, shell_s1.count);
        const size_t primitive_index_a = ab.x+shell_s0.start_index;
        const size_t primitive_index_b = ab.y+shell_s1.start_index;
        // screening (suzuki)
	    if (g_upper_bound_factors[primitive_index_a] * g_upper_bound_factors[primitive_index_b] < schwarz_screening_threshold) return;

        const PrimitiveShell *a = &g_pshell_aux[primitive_index_a];
        const PrimitiveShell *b = &g_pshell_aux[primitive_index_b];

		// printf("[p|d] %d: %d %d\n", threadIdx.x, (int)primitive_index_a, (int)primitive_index_b);

		double sum_exponent = a->exponent + b->exponent;  
		
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};

		double Boys[4];
		getIncrementalBoys(3, a->exponent*b->exponent/sum_exponent * ((Rab[0])*(Rab[0]) + (Rab[1])*(Rab[1]) + (Rab[2])*(Rab[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient * calcNormsWOFact2_2center(a->exponent, b->exponent, 1, 2) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER / (a->exponent*b->exponent*sqrt(sum_exponent));
		bool is_prim_id_not_equal = (primitive_index_a != primitive_index_b);

		#include "./integral_RI/int2c2e/orig_pd.txt"
	}
}


/* (p|f) */
__global__ void calc_pf_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, const double* g_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_shell_pairs){
		size_t2 ab = index1to2(idx, false, shell_s1.count);
        const size_t primitive_index_a = ab.x+shell_s0.start_index;
        const size_t primitive_index_b = ab.y+shell_s1.start_index;
        // screening (suzuki)
	    if (g_upper_bound_factors[primitive_index_a] * g_upper_bound_factors[primitive_index_b] < schwarz_screening_threshold) return;

        const PrimitiveShell *a = &g_pshell_aux[primitive_index_a];
        const PrimitiveShell *b = &g_pshell_aux[primitive_index_b];

		double sum_exponent = a->exponent + b->exponent;  
		
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};

		double Boys[5];
		getIncrementalBoys(4, a->exponent*b->exponent/sum_exponent * ((Rab[0])*(Rab[0]) + (Rab[1])*(Rab[1]) + (Rab[2])*(Rab[2])), g_boys_grid, Boys);

		double coefAndNorm = a->coefficient * b->coefficient * calcNormsWOFact2_2center(a->exponent, b->exponent, 1, 3) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER / (a->exponent*b->exponent*sqrt(sum_exponent));
		bool is_prim_id_not_equal = (primitive_index_a != primitive_index_b);

		#include "./integral_RI/int2c2e/orig_pf.txt"
	}
}



/* (d|d) */
__global__ void calc_dd_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, const double* g_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_shell_pairs){
		const size_t2 ab = index1to2(idx, true);
        const size_t primitive_index_a = ab.x+shell_s0.start_index;
        const size_t primitive_index_b = ab.y+shell_s1.start_index;
        // screening (suzuki)
	    if (g_upper_bound_factors[primitive_index_a] * g_upper_bound_factors[primitive_index_b] < schwarz_screening_threshold) return;

        const PrimitiveShell *a = &g_pshell_aux[primitive_index_a];
        const PrimitiveShell *b = &g_pshell_aux[primitive_index_b];

		// printf("[d|d] %d: %d %d\n", threadIdx.x, (int)primitive_index_a, (int)primitive_index_b);

		double sum_exponent = a->exponent + b->exponent;  
		
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};

		double Boys[5];
		getIncrementalBoys(4, a->exponent*b->exponent/sum_exponent * ((Rab[0])*(Rab[0]) + (Rab[1])*(Rab[1]) + (Rab[2])*(Rab[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient * calcNormsWOFact2_2center(a->exponent, b->exponent, 2, 2) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER / (a->exponent*b->exponent*sqrt(sum_exponent));
		bool is_prim_id_not_equal = (primitive_index_a != primitive_index_b);

		#include "./integral_RI/int2c2e/orig_dd.txt"
	}
}


/* (d|f) */
__global__ void calc_df_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, const double* g_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_shell_pairs){
		size_t2 ab = index1to2(idx, false, shell_s1.count);
        const size_t primitive_index_a = ab.x+shell_s0.start_index;
        const size_t primitive_index_b = ab.y+shell_s1.start_index;
        // screening (suzuki)
	    if (g_upper_bound_factors[primitive_index_a] * g_upper_bound_factors[primitive_index_b] < schwarz_screening_threshold) return;

        const PrimitiveShell *a = &g_pshell_aux[primitive_index_a];
        const PrimitiveShell *b = &g_pshell_aux[primitive_index_b];

		double sum_exponent = a->exponent + b->exponent;  
		
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};

		double Boys[6];
		getIncrementalBoys(5, a->exponent*b->exponent/sum_exponent * ((Rab[0])*(Rab[0]) + (Rab[1])*(Rab[1]) + (Rab[2])*(Rab[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient * calcNormsWOFact2_2center(a->exponent, b->exponent, 2, 3) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER / (a->exponent*b->exponent*sqrt(sum_exponent));
		bool is_prim_id_not_equal = (primitive_index_a != primitive_index_b);

		#include "./integral_RI/int2c2e/orig_df.txt"
	}
}


/* (f|f) */
__global__ void calc_ff_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, const double* g_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_shell_pairs){
		const size_t2 ab = index1to2(idx, true);
        const size_t primitive_index_a = ab.x+shell_s0.start_index;
        const size_t primitive_index_b = ab.y+shell_s1.start_index;
        // screening (suzuki)
	    if (g_upper_bound_factors[primitive_index_a] * g_upper_bound_factors[primitive_index_b] < schwarz_screening_threshold) return;

        const PrimitiveShell *a = &g_pshell_aux[primitive_index_a];
        const PrimitiveShell *b = &g_pshell_aux[primitive_index_b];

		double sum_exponent = a->exponent + b->exponent;  
		
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};

		double Boys[7];
		getIncrementalBoys(6, a->exponent*b->exponent/sum_exponent * ((Rab[0])*(Rab[0]) + (Rab[1])*(Rab[1]) + (Rab[2])*(Rab[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient * calcNormsWOFact2_2center(a->exponent, b->exponent, 3, 3) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER / (a->exponent*b->exponent*sqrt(sum_exponent));
		bool is_prim_id_not_equal = (primitive_index_a != primitive_index_b);

		#include "./integral_RI/int2c2e/orig_ff.txt"
	}
}







#if defined(COMPUTE_G_AUX)
/* (s|g) */
__global__ void calc_sg_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, const double* g_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_shell_pairs){
		size_t2 ab = index1to2(idx, false, shell_s1.count);
        const size_t primitive_index_a = ab.x+shell_s0.start_index;
        const size_t primitive_index_b = ab.y+shell_s1.start_index;
        // screening (suzuki)
	    if (g_upper_bound_factors[primitive_index_a] * g_upper_bound_factors[primitive_index_b] < schwarz_screening_threshold) return;

        const PrimitiveShell *a = &g_pshell_aux[primitive_index_a];
        const PrimitiveShell *b = &g_pshell_aux[primitive_index_b];

		double sum_exponent = a->exponent + b->exponent;  
		
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};

		double Boys[5];
		getIncrementalBoys(4, a->exponent*b->exponent/sum_exponent * ((Rab[0])*(Rab[0]) + (Rab[1])*(Rab[1]) + (Rab[2])*(Rab[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient * calcNormsWOFact2_2center(a->exponent, b->exponent, 0, 4) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER / (a->exponent*b->exponent*sqrt(sum_exponent));
		bool is_prim_id_not_equal = (primitive_index_a != primitive_index_b);

		#include "./integral_RI/int2c2e/orig_sg.txt"
	}
}

/* (p|g) */
__global__ void calc_pg_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, const double* g_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_shell_pairs){
		size_t2 ab = index1to2(idx, false, shell_s1.count);
        const size_t primitive_index_a = ab.x+shell_s0.start_index;
        const size_t primitive_index_b = ab.y+shell_s1.start_index;
        // screening (suzuki)
	    if (g_upper_bound_factors[primitive_index_a] * g_upper_bound_factors[primitive_index_b] < schwarz_screening_threshold) return;

        const PrimitiveShell *a = &g_pshell_aux[primitive_index_a];
        const PrimitiveShell *b = &g_pshell_aux[primitive_index_b];

		double sum_exponent = a->exponent + b->exponent;  
		
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};

		double Boys[6];
		getIncrementalBoys(5, a->exponent*b->exponent/sum_exponent * ((Rab[0])*(Rab[0]) + (Rab[1])*(Rab[1]) + (Rab[2])*(Rab[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient * calcNormsWOFact2_2center(a->exponent, b->exponent, 1, 4) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER / (a->exponent*b->exponent*sqrt(sum_exponent));
		bool is_prim_id_not_equal = (primitive_index_a != primitive_index_b);

		#include "./integral_RI/int2c2e/orig_pg.txt"
	}
}

/* (d|g) */
__global__ void calc_dg_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, const double* g_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_shell_pairs){
		size_t2 ab = index1to2(idx, false, shell_s1.count);
        const size_t primitive_index_a = ab.x+shell_s0.start_index;
        const size_t primitive_index_b = ab.y+shell_s1.start_index;
        // screening (suzuki)
	    if (g_upper_bound_factors[primitive_index_a] * g_upper_bound_factors[primitive_index_b] < schwarz_screening_threshold) return;

        const PrimitiveShell *a = &g_pshell_aux[primitive_index_a];
        const PrimitiveShell *b = &g_pshell_aux[primitive_index_b];

		double sum_exponent = a->exponent + b->exponent;  
		
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};

		double Boys[7];
		getIncrementalBoys(6, a->exponent*b->exponent/sum_exponent * ((Rab[0])*(Rab[0]) + (Rab[1])*(Rab[1]) + (Rab[2])*(Rab[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient * calcNormsWOFact2_2center(a->exponent, b->exponent, 2, 4) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER / (a->exponent*b->exponent*sqrt(sum_exponent));
		bool is_prim_id_not_equal = (primitive_index_a != primitive_index_b);

		#include "./integral_RI/int2c2e/orig_dg.txt"
	}
}

/* (f|g) */
__global__ void calc_fg_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, const double* g_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_shell_pairs){
		size_t2 ab = index1to2(idx, false, shell_s1.count);
        const size_t primitive_index_a = ab.x+shell_s0.start_index;
        const size_t primitive_index_b = ab.y+shell_s1.start_index;
        // screening (suzuki)
	    if (g_upper_bound_factors[primitive_index_a] * g_upper_bound_factors[primitive_index_b] < schwarz_screening_threshold) return;

        const PrimitiveShell *a = &g_pshell_aux[primitive_index_a];
        const PrimitiveShell *b = &g_pshell_aux[primitive_index_b];

		double sum_exponent = a->exponent + b->exponent;  
		
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};

		double Boys[8];
		getIncrementalBoys(7, a->exponent*b->exponent/sum_exponent * ((Rab[0])*(Rab[0]) + (Rab[1])*(Rab[1]) + (Rab[2])*(Rab[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient * calcNormsWOFact2_2center(a->exponent, b->exponent, 3, 4) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER / (a->exponent*b->exponent*sqrt(sum_exponent));
		bool is_prim_id_not_equal = (primitive_index_a != primitive_index_b);

		#include "./integral_RI/int2c2e/orig_fg.txt"
	}
}



/* (g|g) */
__global__ void calc_gg_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, const double* g_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {
	uint64_t idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx < num_shell_pairs){
		const size_t2 ab = index1to2(idx, true);
        const size_t primitive_index_a = ab.x+shell_s0.start_index;
        const size_t primitive_index_b = ab.y+shell_s1.start_index;
        // screening (suzuki)
	    if (g_upper_bound_factors[primitive_index_a] * g_upper_bound_factors[primitive_index_b] < schwarz_screening_threshold) return;

        const PrimitiveShell *a = &g_pshell_aux[primitive_index_a];
        const PrimitiveShell *b = &g_pshell_aux[primitive_index_b];

		double sum_exponent = a->exponent + b->exponent;  
		
		double Rab[3] = {(a->coordinate.x - b->coordinate.x), (a->coordinate.y - b->coordinate.y), (a->coordinate.z - b->coordinate.z)};

		double Boys[9];
		getIncrementalBoys(8, a->exponent*b->exponent/sum_exponent * ((Rab[0])*(Rab[0]) + (Rab[1])*(Rab[1]) + (Rab[2])*(Rab[2])), g_boys_grid, Boys);
		double coefAndNorm = a->coefficient * b->coefficient * calcNormsWOFact2_2center(a->exponent, b->exponent, 4, 4) * TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER / (a->exponent*b->exponent*sqrt(sum_exponent));
		bool is_prim_id_not_equal = (primitive_index_a != primitive_index_b);

		#include "./integral_RI/int2c2e/orig_gg.txt"
	}
}
#else
	__global__ void calc_sg_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, const double* g_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {}
	__global__ void calc_pg_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, const double* g_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {}
	__global__ void calc_dg_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, const double* g_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {}
	__global__ void calc_fg_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, const double* g_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {}
	__global__ void calc_gg_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, const double* g_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid) {}
#endif





__global__ void MD_int2c2e_1T1SP(real_t* g_result, const PrimitiveShell* g_pshell_aux, 
	                             const real_t* d_auxiliary_cgto_normalization_factors, 
								 ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, 
								 int num_shell_pairs, 
								 const double* g_upper_bound_factors, 
								 const double schwarz_screening_threshold, 
								 int num_auxiliary_basis, 
								 const double* g_boys_grid){
{
    // 通し番号indexの計算
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;


    if (id >= num_shell_pairs) return;

    const double size_Rmid=1377;

    //使い捨ての中間体R_mid
    double R_mid[3*1377];

    //解を格納する配列R
    double R[2925];

    //thread内で結果を保持するメモリ
    double thread_val=0.0;

	size_t2 ab = index1to2(id, (shell_s0.start_index == shell_s1.start_index), shell_s1.count);
    // Obtain primitive shells [ab|c]
    const size_t primitive_index_a = ab.x + shell_s0.start_index;
    const size_t primitive_index_b = ab.y + shell_s1.start_index;

	// screening
	if (g_upper_bound_factors[primitive_index_a] * g_upper_bound_factors[primitive_index_b] < schwarz_screening_threshold) return;
 

    const PrimitiveShell a = g_pshell_aux[primitive_index_a];
    const PrimitiveShell b = g_pshell_aux[primitive_index_b];




        
    // Obtain basis index (ij|k)
    const size_t size_a = a.basis_index;
    const size_t size_b = b.basis_index;

    bool is_prim_id_not_equal = (primitive_index_a != primitive_index_b);

    
    //使用データを取得，レジスタに書き込み

    //指数部
    const double alpha = a.exponent;
    const double beta  = b.exponent;
    const double xi = alpha * beta / (alpha + beta);

    //係数部
    const double coef_a = a.coefficient;
    const double coef_b = b.coefficient;


    //座標
    const double pos_A[3] = {a.coordinate.x, a.coordinate.y, a.coordinate.z};
    const double pos_B[3] = {b.coordinate.x, b.coordinate.y, b.coordinate.z};


    //角運動量の総和
    const int orbital_A = a.shell_type;
    const int orbital_B = b.shell_type;


    //軌道間距離の二乗
    const double dist = ((pos_A[0]-pos_B[0])*(pos_A[0]-pos_B[0]) + (pos_A[1]-pos_B[1])*(pos_A[1]-pos_B[1]) + (pos_A[2]-pos_B[2])*(pos_A[2]-pos_B[2]));


    const int K = orbital_A + orbital_B;
    
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

    
    
    double Norm_A, Norm_B;
    double Norm;

    int t,u,v;
    int t_max;
    int u_max;
    int v_max;
    int tau_max;
    int nu_max;
    int phi_max;


    // 方位量子数l,m,nの値をループ変数から導出
    for(int lmn_a=0; lmn_a<comb_max(orbital_A); lmn_a++){
        int l1=loop_to_ang_RI[orbital_A][lmn_a][0]; int m1=loop_to_ang_RI[orbital_A][lmn_a][1]; int n1=loop_to_ang_RI[orbital_A][lmn_a][2];
        Norm_A = calcNorm(alpha, l1, m1, n1);


        for(int lmn_b=0; lmn_b<comb_max(orbital_B); lmn_b++){                  
            int l2=loop_to_ang_RI[orbital_B][lmn_b][0]; int m2=loop_to_ang_RI[orbital_B][lmn_b][1]; int n2=loop_to_ang_RI[orbital_B][lmn_b][2];
            Norm_B = calcNorm(beta, l2, m2, n2);


			Norm = Norm_A * Norm_B;
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
			t_max = l1+1;
			u_max = m1+1;
			v_max = n1+1;
			tau_max = l2+1;
			nu_max = m2+1;
			phi_max = n2+1;

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
								R_mid[calc_Idx_Rmid(k,u,v,i,comb_max(k),size_Rmid)] = (pos_A[0] - pos_B[0])*R_mid[calc_Idx_Rmid(k-1,u,v,i+1,comb_max(k-1),size_Rmid)] + (t-1)*R_mid[calc_Idx_Rmid(k-2,u,v,i+1,comb_max(k-2),size_Rmid)];
							}
							else if(u >= 1){
								R_mid[calc_Idx_Rmid(k,u,v,i,comb_max(k),size_Rmid)] = (pos_A[1] - pos_B[1])*R_mid[calc_Idx_Rmid(k-1,u-1,v,i+1,comb_max(k-1),size_Rmid)] + (u-1)*R_mid[calc_Idx_Rmid(k-2,u-2,v,i+1,comb_max(k-2),size_Rmid)];
							}
							else{
								R_mid[calc_Idx_Rmid(k,u,v,i,comb_max(k),size_Rmid)] = (pos_A[2] - pos_B[2])*R_mid[calc_Idx_Rmid(k-1,u,v-1,i+1,comb_max(k-1),size_Rmid)] + (v-1)*R_mid[calc_Idx_Rmid(k-2,u,v-2,i+1,comb_max(k-2),size_Rmid)];
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
                    

				// ERI計算部
				double my_val = 0.0;
				// ERI計算部
				double Et, Eu, Ev, Etau, Enu, Ephi;
				for(int t=0; t<l1+1; t++){
					Et =  MD_Et_NonRecursion(l1, 0, t, alpha, 0.0, 0.0);
					for(int u=0; u<m1+1; u++){
						Eu = MD_Et_NonRecursion(m1, 0, u, alpha, 0.0, 0.0);
						for(int v=0; v<n1+1; v++){
							Ev = MD_Et_NonRecursion(n1, 0, v, alpha, 0.0, 0.0);
							for(int tau=0; tau<l2+1; tau++){
								Etau = MD_Et_NonRecursion(l2, 0, tau, beta, 0.0, 0.0);
								for(int nu=0; nu<m2+1; nu++){
									Enu = MD_Et_NonRecursion(m2, 0, nu, beta, 0.0, 0.0);
									for(int phi=0; phi<n2+1; phi++){
										Ephi = MD_Et_NonRecursion(n2, 0, phi, beta, 0.0, 0.0);
										int k=t+u+v+tau+nu+phi;
										my_val +=  Et * Eu * Ev * Etau * Enu * Ephi * (1 - 2*((tau+nu+phi)&1)) * R[k*(k+1)*(k+2)/6 + calc_Idx_Rmid(k,u+nu,v+phi,0,0,0)];
										// my_val +=  Et * Eu * Ev * Etau * Enu * Ephi * (1 - 2*((tau+nu+phi)&1)) * R_GPU_Recursion(0, t+tau, u+nu, v+phi, P, Q, Boys);
									}
								}
							}
						}
					}
				}
				thread_val = my_val*2 * M_PI_2_5 /(alpha * beta * sqrt((alpha + beta)))  *coef_a*coef_b;
	    


				// 書き込み部

				// Global Memoryへ書き込み
				// 汎用カーネルでは全要素判定(case1)
				addToResult_2center(
					Norm*thread_val,
					g_result,
					size_a+lmn_a, size_b+lmn_b,
					num_auxiliary_basis,
					is_prim_id_not_equal, 
					d_auxiliary_cgto_normalization_factors
				);
                
            }
        }
    }
    return;
}

} // namespace gansu::gpu