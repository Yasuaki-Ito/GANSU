import os, sys, tempfile
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v,'1')
sys.path.insert(0,'script')
import numpy as np
from scipy.linalg import expm
import steom_cas_verify as V
import steom_cfour_weff as CW
import steom_ea_spinadapt as EA
from pyscf_steom_feff_reference import build_g_canonical_full
from steom_fockspace_ref import (get_active_data, build_sector, solve_ip, solve_ea,
                                 build_S, hf_det, project_1h1p, so_index, vir_so, occ_so)
Ha=27.211386245988

def extract_sip(sIP, data):
    # clean s_ip[m][i,j,b] = -sIP[m_alpha][I_alpha, J_beta, B_beta]  (abb block = -rx)
    nact,nocc,nvir=data['nact'],data['nocc'],data['nvir']
    s=np.zeros((nocc,nocc,nvir,nvir))  # placeholder wrong shape; fix below
    out=np.zeros((nocc,nocc,nvir))     # per occ root m
    res={}
    for mm in range(nocc):
        mA=so_index(mm,0,nact)
        r=np.zeros((nocc,nocc,nvir))
        for i in range(nocc):
            for j in range(nocc):
                for b in range(nvir):
                    r[i,j,b] = -sIP[mA][so_index(i,0,nact), so_index(j,1,nact), so_index(b+nocc,1,nact)]
        res[mm]=r
    return res  # dict m-> rx[i,j,b]

def verify_sip(sIP, sip_sp, data):
    nact,nocc,nvir=data['nact'],data['nocc'],data['nvir']
    err=nrm=0.0
    for mm in range(nocc):
        mA=so_index(mm,0,nact); rx=sip_sp[mm]
        for i in range(nocc):
            for j in range(nocc):
                for b in range(nvir):
                    ref=rx[j,i,b]-rx[i,j,b]  # aaa block
                    got=sIP[mA][so_index(i,0,nact),so_index(j,0,nact),so_index(b+nocc,0,nact)]
                    err+=(got-ref)**2; nrm+=ref**2
    print(f'  [IP convention] ||aaa-(rx^T-rx)||={np.sqrt(err):.3e} (||aaa||={np.sqrt(nrm):.3f})')

def build_sip_recon(sip_sp, data):
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
                        if sr==0:  # alpha root
                            r2[O(i,0),O(j,0),Vv(b,0)] = aa[i,j,b]
                            r2[O(i,0),O(j,1),Vv(b,1)] = -rx[i,j,b]
                            r2[O(j,1),O(i,0),Vv(b,1)] = +rx[i,j,b]
                        else:      # beta root (Kramers flip)
                            r2[O(i,1),O(j,1),Vv(b,1)] = aa[i,j,b]
                            r2[O(i,1),O(j,0),Vv(b,0)] = -rx[i,j,b]
                            r2[O(j,0),O(i,1),Vv(b,0)] = +rx[i,j,b]
            res[root]=r2
    return res

atom=V.polyene(6,0.0); active,_=V.detect_pi(atom,'sto-3g',3,3)
data=get_active_data(atom=atom,basis='sto-3g',active=active)
nocc,nvir=data['nocc'],data['nvir']
dets,index,Hbar=build_sector(data,data['nelec'])
E_N=Hbar[index[hf_det(data)],index[hf_det(data)]]
sIP=solve_ip(data,E_N); sEA=solve_ea(data)
sip_sp=extract_sip(sIP,data); verify_sip(sIP,sip_sp,data)
s_sp=EA.extract_spatial_amp(sEA,data)
# verify sip_recon reproduces oracle solve_ip's alpha & beta blocks fully
sip_rec=build_sip_recon(sip_sp,data)
# compare full transform spectra: oracle uses sIP/sEA (mixed), clean uses recon
sea_rec=EA.build_sea_recon(s_sp,data)
sEA_clean={E:sea_rec[E] for E in vir_so(data)}
sIP_clean={m:sip_rec[m] for m in occ_so(data)}
def spec(SIP,SEA):
    S=build_S(data,dets,index,SIP,SEA)
    Gs,_=project_1h1p(data,dets,index,expm(S)@Hbar@expm(-S))
    return np.sort(np.linalg.eigvals(Gs).real-E_N)*Ha
e_mixed=spec(sIP,sEA); e_clean=spec(sIP_clean,sEA_clean)
print('  oracle mixed vs clean gauge spectrum (must match, gauge-invariant):')
print('   ',np.round(e_mixed,3))
print('   ',np.round(e_clean,3))
print(f'   max|diff|={np.max(np.abs(e_mixed-e_clean)):.3e}')

# ---- clean-gauge oracle g_phhp / g_phph (F-free g_phhp) vs GANSU ----
def blocks(SIP,SEA):
    S=build_S(data,dets,index,SIP,SEA)
    G=expm(S)@Hbar@expm(-S)
    Gs,Gt=project_1h1p(data,dets,index,G)
    GsD=Gs-E_N*np.eye(nocc*nvir); GtD=Gt-E_N*np.eye(nocc*nvir)
    gph=np.zeros((nvir,nocc,nocc,nvir)); gpp=np.zeros((nvir,nocc,nvir,nocc))
    for i in range(nocc):
        for a in range(nvir):
            r=i*nvir+a
            for j in range(nocc):
                for b in range(nvir):
                    c=j*nvir+b
                    gph[b,j,i,a]=0.5*(GsD[r,c]-GtD[r,c])   # F-free g_phhp
                    gpp[a,j,b,i]=-GtD[r,c]                  # g_phph - F
    return gph,gpp
gph_or,gpp_or=blocks(sIP_clean,sEA_clean)   # clean gauge = GANSU gauge

xyzf=os.path.join(tempfile.gettempdir(),'g.xyz'); lines=[a.strip() for a in atom.split(';')]
open(xyzf,'w').write(f'{len(lines)}\n\n'+'\n'.join(lines)+'\n')
d=CW.load(xyzf,'sto-3g',0,atom=atom,active=active)
def gansu(**env):
    for k,v in env.items(): os.environ[k]=v
    G,gpp,gph,*_=build_g_canonical_full(d['bar'],d['r2_ip'],d['r2_ea'],d['r1_ip'],d['r1_ea'],
                                        d['occ_idx'],d['vir_idx'],nocc,nvir)
    for k in env: os.environ.pop(k,None)
    return gph,gpp
for lbl,env in [('shipped',{}),('EEbase',{'STEOM_EE_BASE':'1'}),('EEbase+EA',{'STEOM_EE_BASE':'1','STEOM_EA_ROUTE':'1'})]:
    gph_g,gpp_g=gansu(**env)
    dph=np.linalg.norm(gph_g-gph_or); dpp=np.linalg.norm(gpp_g-gpp_or)
    print(f'{lbl:11s}: ||g_phhp_gansu-g_phhp_oracle||={dph:.4f} (||or||={np.linalg.norm(gph_or):.3f})   '
          f'||g_phph diff||={dpp:.4f} (note: oracle=g_phph-F, gansu=g_phph, F offset expected)')

# ---- route-by-route g_phhp decomposition: oracle (clean gauge) vs GANSU ----
def gph_only(SIP,SEA):
    g,_=blocks(SIP,SEA); return g
zIP={m:np.zeros_like(sIP_clean[m]) for m in sIP_clean}
zEA={e:np.zeros_like(sEA_clean[e]) for e in sEA_clean}
g0=gph_only(zIP,zEA); gip=gph_only(sIP_clean,zEA); gea=gph_only(zIP,sEA_clean); gf=gph_only(sIP_clean,sEA_clean)
or_base=g0; or_IP=gip-g0; or_EA=gea-g0; or_cross=gf-gip-gea+g0
# GANSU decomposition via returned _g_phhp_decomp
def gansu_decomp(**env):
    for k,v in env.items(): os.environ[k]=v
    G,gpp,gph,uamei,ubmje,decomp=build_g_canonical_full(d['bar'],d['r2_ip'],d['r2_ea'],d['r1_ip'],d['r1_ea'],
                                                        d['occ_idx'],d['vir_idx'],nocc,nvir)
    for k in env: os.environ.pop(k,None)
    return decomp  # (base, +IP, +EA, +cross)
gb,gI,gE,gC=gansu_decomp(STEOM_EE_BASE='1',STEOM_EA_ROUTE='1')
print('  g_phhp route  ||oracle||  ||gansu||  ||diff||')
for nm,o,g in [('base',or_base,gb),('IP (u_bmjc)',or_IP,gI),('EA (u_bkje)',or_EA,gE),('cross(u_bmje)',or_cross,gC)]:
    print(f'    {nm:14s} {np.linalg.norm(o):8.4f}  {np.linalg.norm(g):8.4f}  {np.linalg.norm(o-g):8.4f}')

# ---- DECISIVE: inject correct IP route (or_IP - gansu_IP) and check spectrum ----

corr = or_IP - gI          # g_phhp[b,j,i,a] correction to fix the IP route
npf=os.path.join(tempfile.gettempdir(),'ipcorr.npy'); np.save(npf,corr)
def gspec(**env):
    for k,v in env.items(): os.environ[k]=v
    G,*_=build_g_canonical_full(d['bar'],d['r2_ip'],d['r2_ea'],d['r1_ip'],d['r1_ea'],
                                d['occ_idx'],d['vir_idx'],nocc,nvir)
    for k in env: os.environ.pop(k,None)
    w=np.linalg.eigvals(G).real; return np.sort(w)*Ha
e_ee=gspec(STEOM_EE_BASE='1')
e_ipfix=gspec(STEOM_EE_BASE='1', STEOM_EA_EXACT_NPY=npf)
print('  spectrum: clean-oracle | GANSU+EEbase | GANSU+EEbase+IPfix')
for i in range(len(e_clean)):
    print(f'   {i}: {e_clean[i]:8.3f}  {e_ee[i]:8.3f} ({e_ee[i]-e_clean[i]:+.3f})  {e_ipfix[i]:8.3f} ({e_ipfix[i]-e_clean[i]:+.3f})')
print(f'  RMS +EEbase={np.sqrt(np.mean((e_ee-e_clean)**2)):.3f}  +IPfix={np.sqrt(np.mean((e_ipfix-e_clean)**2)):.3f}')

# ================= g_phph route decomposition (off-diagonal, F-free) =================
def gpp_only(SIP,SEA):
    _,g=blocks(SIP,SEA); return g
p0=gpp_only(zIP,zEA); pip=gpp_only(sIP_clean,zEA); pea=gpp_only(zIP,sEA_clean); pf=gpp_only(sIP_clean,sEA_clean)
opp_base=p0; opp_IP=pip-p0; opp_EA=pea-p0; opp_cross=pf-pip-pea+p0
# off-diagonal mask (i!=j or a!=b) on [a,j,b,i]
mask=np.ones((nvir,nocc,nvir,nocc),bool)
for i in range(nocc):
    for a in range(nvir):
        mask[a,:,a,i]=True
        for j in range(nocc):
            for b in range(nvir):
                if i==j and a==b: mask[a,j,b,i]=False
def off(X): return np.linalg.norm(X[mask])
# GANSU g_phph total (F-free) with base fix
gph_g,gpp_g=gansu(STEOM_EE_BASE='1')   # returns (g_phhp, g_phph)
# oracle total F-free g_phph on off-diag = pf off-diag (F only on diag)
print('  g_phph OFF-DIAGONAL (F-free):')
print(f'    ||oracle g_phph(full) offdiag|| = {off(pf):.4f}')
print(f'    ||GANSU  g_phph(full) offdiag|| = {off(gpp_g):.4f}')
print(f'    ||diff|| offdiag = {off(pf-gpp_g):.4f}')
print('  oracle g_phph route norms (offdiag): base=%.4f IP=%.4f EA=%.4f cross=%.4f'%(
    off(opp_base),off(opp_IP),off(opp_EA),off(opp_cross)))
# also g_phhp offdiag diff for reference (using earlier gph_or, gph_g via EEbase)
def offh(X):
    m=np.ones((nvir,nocc,nocc,nvir),bool)
    for i in range(nocc):
        for a in range(nvir):
            for j in range(nocc):
                for b in range(nvir):
                    if i==j and a==b: m[b,j,i,a]=False
    return np.linalg.norm(X[m])
print(f'  g_phhp OFF-DIAG diff (EEbase) = {offh(gph_g-gph_or):.4f} (||oracle offdiag||={offh(gph_or):.4f})')

# ================= DECISIVE: fix BOTH g_phhp + g_phph(offdiag) together =================
def Ggansu(**env):
    for k,v in env.items(): os.environ[k]=v
    G,gpp,gph,*_=build_g_canonical_full(d['bar'],d['r2_ip'],d['r2_ea'],d['r1_ip'],d['r1_ea'],
                                        d['occ_idx'],d['vir_idx'],nocc,nvir)
    for k in env: os.environ.pop(k,None)
    return G,gph,gpp
G_g,gph_g2,gpp_g2=Ggansu(STEOM_EE_BASE='1')
# corrections to match clean-gauge oracle (same gauge as GANSU s)
dphhp = gph_or - gph_g2                 # [b,j,i,a] full correction
dphph = gpp_or - gpp_g2                 # [a,j,b,i]; keep OFF-diagonal only (diag has F entanglement)
dphph_off = dphph*mask                  # mask True where offdiag (F-free)
G_fix = G_g.copy()
for i in range(nocc):
    for a in range(nvir):
        r=i*nvir+a
        for j in range(nocc):
            for b in range(nvir):
                c=j*nvir+b
                G_fix[r,c] += 2.0*dphhp[b,j,i,a] - dphph_off[a,j,b,i]
e_fix=np.sort(np.linalg.eigvals(G_fix).real)*Ha
e_ee2=np.sort(np.linalg.eigvals(G_g).real)*Ha
print('  DECISIVE fix-both:  clean-oracle | GANSU+EE | +fix(phhp&phph-offdiag)')
for i in range(len(e_clean)):
    print(f'   {i}: {e_clean[i]:8.3f}  {e_ee2[i]:8.3f} ({e_ee2[i]-e_clean[i]:+.3f})  {e_fix[i]:8.3f} ({e_fix[i]-e_clean[i]:+.3f})')
print(f'  RMS +EE={np.sqrt(np.mean((e_ee2-e_clean)**2)):.3f}  +fix-both={np.sqrt(np.mean((e_fix-e_clean)**2)):.3f}')

# ============ CLEAN element-wise: GANSU full G vs oracle GsD (=2gph_or - gpp_or) ============
GsD=np.zeros((nocc*nvir,nocc*nvir))
for i in range(nocc):
    for a in range(nvir):
        r=i*nvir+a
        for j in range(nocc):
            for b in range(nvir):
                c=j*nvir+b
                GsD[r,c]=2.0*gph_or[b,j,i,a]-gpp_or[a,j,b,i]
# GANSU G (EEbase) is excitation-relative (E_N subtracted) — same as GsD
diff=G_g-GsD
dim=nocc*nvir
diagmask=np.eye(dim,dtype=bool)
print(f'  ||G_gansu - GsD_oracle|| = {np.linalg.norm(diff):.4f}  (||GsD||={np.linalg.norm(GsD):.3f})')
print(f'    diagonal   part: ||diff||={np.linalg.norm(diff[diagmask]):.4f}  (||GsD_diag||={np.linalg.norm(GsD[diagmask]):.3f})')
print(f'    offdiag    part: ||diff||={np.linalg.norm(diff[~diagmask]):.4f}  (||GsD_off||={np.linalg.norm(GsD[~diagmask]):.3f})')
# eigenvalue check: does GsD reproduce e_clean? (sanity)
ee_gsd=np.sort(np.linalg.eigvals(GsD).real)*Ha
print(f'    GsD eig vs e_clean max|diff|={np.max(np.abs(ee_gsd-e_clean)):.3e}  (sanity: should be ~0)')
# symmetry check: is G_gansu non-symmetric like GsD?
print(f'    ||G_gansu - G_gansu^T||={np.linalg.norm(G_g-G_g.T):.3f}  ||GsD-GsD^T||={np.linalg.norm(GsD-GsD.T):.3f}')

# ===== isolate: GANSU diag + oracle offdiag  vs  oracle diag + GANSU offdiag =====
G_off = G_g.copy(); G_off[~diagmask]=GsD[~diagmask]     # fix only off-diagonal
G_dia = G_g.copy(); G_dia[diagmask]=GsD[diagmask]       # fix only diagonal
e_off=np.sort(np.linalg.eigvals(G_off).real)*Ha
e_dia=np.sort(np.linalg.eigvals(G_dia).real)*Ha
print('  isolate:  clean | GANSU | fix-offdiag-only | fix-diag-only')
for i in range(len(e_clean)):
    print(f'   {i}: {e_clean[i]:8.3f} {e_ee2[i]:8.3f} {e_off[i]:8.3f} {e_dia[i]:8.3f}')
print(f'  RMS  GANSU={np.sqrt(np.mean((e_ee2-e_clean)**2)):.3f}  fix-offdiag={np.sqrt(np.mean((e_off-e_clean)**2)):.3f}  fix-diag={np.sqrt(np.mean((e_dia-e_clean)**2)):.3f}')
