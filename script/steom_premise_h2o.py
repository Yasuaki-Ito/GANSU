import os, sys, tempfile
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v,'1')
sys.path.insert(0,'script')
import numpy as np
from scipy.linalg import expm
import steom_cfour_weff as CW
from pyscf_steom_feff_reference import build_g_canonical_full
from steom_fockspace_ref import (get_active_data, build_sector, solve_ip, solve_ea,
                                 build_S, hf_det, project_1h1p)
Ha=27.211386245988
data=get_active_data(xyz='xyz/H2O.xyz',basis='sto-3g',ncore=1)
nocc,nvir=data['nocc'],data['nvir']
dets,index,Hbar=build_sector(data,data['nelec']); E_N=Hbar[index[hf_det(data)],index[hf_det(data)]]
sIP=solve_ip(data,E_N); sEA=solve_ea(data)
S=build_S(data,dets,index,sIP,sEA)
Gs,_=project_1h1p(data,dets,index,expm(S)@Hbar@expm(-S))
eproj=np.sort(np.linalg.eigvals(Gs).real-E_N)*Ha
d=CW.load('xyz/H2O.xyz','sto-3g',1)
def gansu(**env):
    for k,v in env.items(): os.environ[k]=v
    G,*_=build_g_canonical_full(d['bar'],d['r2_ip'],d['r2_ea'],d['r1_ip'],d['r1_ea'],d['occ_idx'],d['vir_idx'],nocc,nvir)
    for k in env: os.environ.pop(k,None)
    return np.sort(np.linalg.eigvals(G).real)*Ha
eg=gansu(STEOM_EE_BASE='1'); esh=gansu()
print(f'H2O FC1 nocc={nocc} nvir={nvir}  (MIXED projection)')
for i in range(len(eproj)):
    print(f'  root{i}: proj={eproj[i]:8.3f}  GANSU-EE={eg[i]:8.3f} (d={eg[i]-eproj[i]:+.3f})  GANSU-shipped={esh[i]:8.3f}')
print(f'  ORCA(memory)=11.849  mixed-proj[0]={eproj[0]:.3f}  GANSU-EE[0]={eg[0]:.3f}')
