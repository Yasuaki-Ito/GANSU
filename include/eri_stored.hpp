#pragma once
#include "eri.hpp"


namespace gansu {

static __device__ double block_reduce_sum(double x){
  extern __shared__ double sdata[];
  int tid = threadIdx.x;
  sdata[tid] = x;
  __syncthreads();

  for(int s = blockDim.x/2; s>0; s>>=1){
    if(tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  return sdata[0];
}

static __device__  size_t idx4_to_1(int num_basis, int mu, int nu, int la, int si){
  return ( ( (size_t(mu)*num_basis + nu)*num_basis + la)*num_basis + si );
}

static __device__ __forceinline__ real_t antisym_eri(const real_t* __restrict__ eri_mo,
                                    const int num_basis,
                                    const int p, const int q, const int r, const int s)
{
    assert(p >= 0 && p < num_basis*2);
    assert(q >= 0 && q < num_basis*2);
    assert(r >= 0 && r < num_basis*2);
    assert(s >= 0 && s < num_basis*2);

    // <pq||rs> = (pr|qs) - (ps|qr)
    real_t prqs = ((p%2)==(r%2) && ((q%2)==(s%2))) ? eri_mo[idx4_to_1(num_basis, p/2, r/2, q/2, s/2)] : 0.0;
    real_t psqr = ((p%2)==(s%2) && ((q%2)==(r%2))) ? eri_mo[idx4_to_1(num_basis, p/2, s/2, q/2, r/2)] : 0.0;
    return prqs - psqr;
}

























} // namespace gansu