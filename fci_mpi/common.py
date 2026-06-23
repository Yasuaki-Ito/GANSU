#!/usr/bin/env python

import pandas as pd

class RunResult:
    def __init__(self, energy, niters, exetime):
        self.energy = energy
        self.niters = niters
        self.exetime = exetime

def str2geometry(atoms_str, distance):
    atoms = atoms_str.split('-')

    if 'Hchain' in atoms_str:
        num = int(atoms_str.replace('Hchain',''))
        geometry = [['H', (0, 0, distance*i)] for i in range(num)]
    else:
        if len(atoms) < 2:
            print('invalid geometry')
            return -1

        geometry = []
        for i in range(len(atoms)):
            geometry.append([atoms[i], (0, 0, distance*i)])

    return geometry

def xyz2geometry(fname, distance):
    f = open(fname, 'r')
    natoms = int(f.readline())
    label = f.readline()

    geometry = []
    for i in range(natoms):
        atom = f.readline().split()
        elem = [atom[0], (float(atom[1]), float(atom[2]), float(atom[3]))]
        geometry.append(elem)
    f.close()
    return Molecule.absolute_distance(distance, geometry, (1,0))

def get_x(atom):
    return atom[1][0]

def set_x(atom, x):
    atom[1][0] = round(x, 4)

def get_z(atom):
    return atom[1][2]

def set_z(atom, z):
    atom[1][2] = round(z, 4)

def absolute_distance(distance, geometry, pair=(1,0)):
    i, j = pair

    g = [[a[0], list(a[1])] for a in geometry]

    r1 = np.array(g[i][1], dtype=float)
    r0 = np.array(g[j][1], dtype=float)

    vec = r1 - r0
    norm = np.linalg.norm(vec)

    if norm < 1e-12:
        raise ValueError("atoms are at same position")

    vec = vec / norm

    g[i][1] = list(r0 + distance * vec)

    return g


def mol2geometry(fname: str, distance: float or None):
    f = open(fname, 'r')
    f.readline(); f.readline(); f.readline()
    natoms = int(f.readline().split()[0])

    geometry = []
    is_chain = True
    for i in range(natoms):
        atom = f.readline().split()
        elem = [atom[3], [float(atom[0]), float(atom[1]), float(atom[2])]]
        if float(atom[0]) != 0.0 or float(atom[1]) != 0.0:
            is_chain = False
        geometry.append(elem)
    f.close()

    if distance is None:
        return geometry
    else:
        if is_chain:
            original_z = get_z(geometry[1])
            set_z(geometry[1], get_z(geometry[0]) + distance)
            for i in range(2, natoms):
                if get_z(geometry[i]) > original_z:
                    set_z(geometry[i], get_z(geometry[i]) + (get_z(geometry[1])-original_z))
            return geometry
        else:
            return absolute_distance(distance, geometry, (1,0))

def get_geometry_and_spin(molname: str, dist: float or None):
    mol_info = pd.read_json('share/molecule_info.json', orient='index')
    if molname in mol_info.index:
        if mol_info['file'][molname] is not None:
            geometry = mol2geometry('share/' + mol_info['file'][molname], dist)
        else:
            geometry = str2geometry(molname, dist)
        spin = int(mol_info['spin'][molname])
    else:
        geometry = str2geometry(molname, dist)
        spin = 0
        if 'Hchain' in molname:
            n = int(molname.replace('Hchain',''))
            spin = 1 if n%2 == 1 else 0        

    geometry = '; '.join([f'{a[0]} {a[1][0]} {a[1][1]} {a[1][2]}' for a in geometry])
    return geometry, spin

if __name__ == '__main__':
    #geometry = xyz2geometry('share/C2H2.xyz', 3.0)
    geometry = mol2geometry('share/C2H2.mol', 3.0)
    print(geometry)
