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
def pr(v):
    p=np.abs(v)**2; s=p.sum(); return float(1.0/np.sum((p/s)**2)) if s>0 else 0.0
atom='; '.join(f'H {2.0*(k%2)} {1.4*(k//2)} 0' for k in range(6))   # H6 rect ladder, nocc=nvir=3
data=get_active_data(atom=atom,basis='sto-3g',ncore=0)
nocc,nvir=data['nocc'],data['nvir']
dets,index,Hbar=build_sector(data,data['nelec']); E_N=Hbar[index[hf_det(data)],index[hf_det(data)]]
sIP=solve_ip(data,E_N); sEA=solve_ea(data)
S=build_S(data,dets,index,sIP,sEA)
Gs,_=project_1h1p(data,dets,index,expm(S)@Hbar@expm(-S))
w,vr=np.linalg.eig(Gs); o=np.argsort(w.real); eproj=(w[o].real-E_N)*Ha; Vp=vr[:,o]
xyzf=os.path.join(tempfile.gettempdir(),'h6.xyz'); lines=[a.strip() for a in atom.split(';')]
open(xyzf,'w').write(f'{len(lines)}\n\n'+'\n'.join(lines)+'\n')
d=CW.load(xyzf,'sto-3g',0,atom=atom,active=None)
def gansu(**env):
    for k,v in env.items(): os.environ[k]=v
    G,*_=build_g_canonical_full(d['bar'],d['r2_ip'],d['r2_ea'],d['r1_ip'],d['r1_ea'],d['occ_idx'],d['vir_idx'],nocc,nvir)
    for k in env: os.environ.pop(k,None)
    return np.sort(np.linalg.eigvals(G).real)*Ha
eg=gansu(STEOM_EE_BASE='1')
print(f'H6 rect ladder nocc={nocc} nvir={nvir}  (geometry for ORCA:)')
for a in lines: print('   ',a)
print('  root  proj    GANSU    d      PR')
for i in range(len(eproj)):
    print(f'   {i}: {eproj[i]:7.3f} {eg[i]:7.3f} {eg[i]-eproj[i]:+.3f}  {pr(Vp[:,i]):.2f}')
print(f'  sorted-RMS(GANSU-proj)={np.sqrt(np.mean((eg-eproj)**2)):.3f}')
