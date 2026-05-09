#!/bin/bash
# Numerical gradient of RI-HF 2e energy only.
# 2e energy = E_total - E_1e - E_nuc
# = E_total - 0.5*Tr[D(H+F)] + 0.5*Tr[D*J] - 0.25*Tr[D*K] ... too complex
#
# Simpler: just compute total energy numerical gradient and compare with GANSU analytical.
# The 1e part is from existing kernels (verified for Stored HF).
# So: analytical_total = 1e_analytical + 2e_analytical
# If 1e is correct (same kernels as Stored), then 2e error = total error.

EXEC="./HF_main"
DELTA=0.00001  # Angstrom

echo "=== Numerical RI-HF gradient (O atom, z direction) ==="

# E(+dz)
tmpxyz=$(mktemp /tmp/mol_XXXXXX.xyz)
cat > "$tmpxyz" << 'EOF'
3
perturbed
O          0.000        0.000       0.12701
H          0.000        0.758      -0.509
H          0.000       -0.758      -0.509
EOF
Ep=$($EXEC -x "$tmpxyz" -g ../basis/sto-3g.gbs --eri_method ri -ag ../auxiliary_basis/cc-pvdz-rifit.gbs 2>&1 | grep "Total Energy:" | tail -1 | awk '{print $3}')
rm "$tmpxyz"

# E(-dz)
tmpxyz=$(mktemp /tmp/mol_XXXXXX.xyz)
cat > "$tmpxyz" << 'EOF'
3
perturbed
O          0.000        0.000       0.12699
H          0.000        0.758      -0.509
H          0.000       -0.758      -0.509
EOF
Em=$($EXEC -x "$tmpxyz" -g ../basis/sto-3g.gbs --eri_method ri -ag ../auxiliary_basis/cc-pvdz-rifit.gbs 2>&1 | grep "Total Energy:" | tail -1 | awk '{print $3}')
rm "$tmpxyz"

echo "E+ = $Ep"
echo "E- = $Em"

# dE/dR = (E+ - E-) / (2*delta_bohr)
python3 -c "
Ep=$Ep; Em=$Em; delta=$DELTA
bohr = 1.8897259886
dR = 2.0 * delta * bohr
grad = (Ep - Em) / dR
print(f'Numerical gradient O dz = {grad:.10e}')
print(f'  (delta = {delta} Ang = {delta*bohr:.6e} Bohr)')
"

# GANSU analytical
echo ""
echo "=== GANSU analytical ==="
$EXEC -x ../xyz/H2O.xyz -g ../basis/sto-3g.gbs --eri_method ri -ag ../auxiliary_basis/cc-pvdz-rifit.gbs -r gradient 2>&1 | grep "000O"
