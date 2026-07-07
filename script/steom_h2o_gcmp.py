import os, sys, tempfile
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v,'1')
sys.path.insert(0,'script')
import numpy as np
from scipy.linalg import expm
import steom_cfour_weff as CW
import steom_ea_spinadapt as EA
from pyscf_steom_feff_reference import build_g_canonical_full
from steom_fockspace_ref import (get_active_data, build_sector, solve_ip, solve_ea,
                                 build_S, hf_det, project_1h1p, so_index, vir_so, occ_so)
import importlib.util
spec=importlib.util.spec_from_file_location('SR','script/steom_sroute_compare.py')
# reuse build_sip_recon/extract_sip from the saved module by exec of the funcs
import types
Ha=27.211386245988
# --- inline the IP recon funcs (same as steom_sroute_compare) ---
def extract_sip(sIP,data):
    nact,nocc,nvir=data['nact'],data['nocc'],data['nvir']; res={}
    for mm in range(nocc):
        mA=so_index(mm,0,nact); r=np.zeros((nocc,nocc,nvir))
        for i in range(nocc):
            for j in range(nocc):
                for b in range(nvir):
                    r[i,j,b]=-sIP[mA][so_index(i,0,nact),so_index(j,1,nact),so_index(b+nocc,1,nact)]
        res[mm]=r
    return res
def build_sip_recon(sip_sp,data):
    nact,nocc,nvir=data['nact'],data['nocc'],data['nvir']; nso=2*nact
    def O(i,s): return so_index(i,s,nact)
    def Vv(a,s): return so_index(a+nocc,s,nact)
    res={}
    for mm in range(nocc):
        rx=sip_sp[mm]; aa=rx.transpose(1,0,2)-rx
        for (root,sr) in [(O(mm,0),0),(O(mm,1),1)]:
            r2=np.zeros((nso,nso,nso))
            for i in range(nocc):
                for j in range(nocc):
                    for b in range(nvir):
                        if sr==0:
                            r2[O(i,0),O(j,0),Vv(b,0)]=aa[i,j,b]; r2[O(i,0),O(j,1),Vv(b,1)]=-rx[i,j,b]; r2[O(j,1),O(i,0),Vv(b,1)]=rx[i,j,b]
                        else:
                            r2[O(i,1),O(j,1),Vv(b,1)]=aa[i,j,b]; r2[O(i,1),O(j,0),Vv(b,0)]=-rx[i,j,b]; r2[O(j,0),O(i,1),Vv(b,0)]=rx[i,j,b]
            res[root]=r2
    return res

# H2O FC1
data=get_active_data(xyz='xyz/H2O.xyz',basis='sto-3g',ncore=1)
nocc,nvir=data['nocc'],data['nvir']
dets,index,Hbar=build_sector(data,data['nelec']); E_N=Hbar[index[hf_det(data)],index[hf_det(data)]]
sIP=solve_ip(data,E_N); sEA=solve_ea(data)
sip_sp=extract_sip(sIP,data); s_sp=EA.extract_spatial_amp(sEA,data)
sea_rec=EA.build_sea_recon(s_sp,data); sip_rec=build_sip_recon(sip_sp,data)
SEA={E:sea_rec[E] for E in vir_so(data)}; SIP={m:sip_rec[m] for m in occ_so(data)}
S=build_S(data,dets,index,SIP,SEA); Gs,Gt=project_1h1p(data,dets,index,expm(S)@Hbar@expm(-S))
GsD=Gs-E_N*np.eye(nocc*nvir)
d=CW.load('xyz/H2O.xyz','sto-3g',1)
G_g,*_=build_g_canonical_full(d['bar'],d['r2_ip'],d['r2_ea'],d['r1_ip'],d['r1_ea'],d['occ_idx'],d['vir_idx'],nocc,nvir)
# eigenvalues
def spec(M): return np.sort(np.linalg.eigvals(M).real)*Ha
dim=nocc*nvir; dm=np.eye(dim,dtype=bool)
diff=G_g-GsD
print(f'H2O FC1: nocc={nocc} nvir={nvir} dim={dim}')
print(f'  ||G_gansu - GsD|| = {np.linalg.norm(diff):.4f} (||GsD||={np.linalg.norm(GsD):.3f})')
print(f'    diag ||diff||={np.linalg.norm(diff[dm]):.4f}  offdiag ||diff||={np.linalg.norm(diff[~dm]):.4f} (||GsD_off||={np.linalg.norm(GsD[~dm]):.3f})')
print(f'  eig GANSU (lowest 3): {np.round(spec(G_g)[:3],3)}')
print(f'  eig GsD   (lowest 3): {np.round(spec(GsD)[:3],3)}')
print(f'  sorted-RMS = {np.sqrt(np.mean((spec(G_g)-spec(GsD))**2)):.3f}')
