/*
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

extern void fci(double *h_Gmo1e, double *h_Gmo, double *e,
               int32_t *occslst, int32_t na, int32_t norb,
               int32_t neleca, int max_space, int max_cycle, 
               int in_cpu, int tile, int debug_mode, 
               double tol, double E_rhf);

void fci_result(double *h_Gmo1e, double *h_Gmo, double *e,
                int32_t *occslst, int32_t na, int32_t norb,
                int32_t neleca, int max_space, int max_cycle,
                int in_cpu, int tile, int debug_mode, 
                double tol, double E_rhf)
{
   fci(h_Gmo1e, h_Gmo, e, occslst, na, norb, neleca, 
      max_space, max_cycle, in_cpu, tile, debug_mode,  tol, E_rhf);
}

