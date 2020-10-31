from ...multilinear.tensor import einsum, transform_all_indices
import psi4
import numpy as np
from scipy import linalg as spla

def construct_integrals(orb, wfn):
    """
    Build our needed integrals.

    Input
    -----
    orb: dict
        Maps from {"A", "B"} to the relevant spincase of the C matrix.
    wfn: psi4.core.wfn
        A wavefunction. Doesn't need to be FCI.

    Output
    ------
    h, g, f: np.ndarray
        Each is a dict from spincase to the array of interest. h for one-electron integrals,
        g for two-electron integrals, and f for the Fock matrix.
    """
    nalpha = wfn.nalpha()
    nbeta = wfn.nbeta()
    mints = wfn.mintshelper()
    H = mints.ao_potential().np + mints.ao_kinetic().np
    I = np.transpose(mints.ao_eri(), (1, 3, 0, 2))
    h = {string[0]: transform_all_indices(H, tuple(orb[x] for x in string)) for string in {"AA", "BB"}}
    g = {string[:2]: transform_all_indices(I, tuple(orb[x] for x in string)) for string in {"AAAA", "ABAB", "BBBB"}}
    g["AA"] -= g["AA"].swapaxes(2, 3)
    g["BB"] -= g["BB"].swapaxes(2, 3)
    f = dict()
    f["A"] = h["A"].diagonal() + einsum("pRpR -> p", g["AA"][:, :nalpha, :, :nalpha]) + einsum("pRpR -> p", g["AB"][:, :nbeta, :, :nbeta])
    f["B"] = h["B"].diagonal() + einsum("RpRp -> p", g["AB"][:nalpha, :, :nalpha, :]) + einsum("RpRp -> p", g["BB"][:nbeta, :, :nbeta, :])
    return h, g, f

def rotate_orbitals(amplitudes, orbitals, space):

    num_mo = orbitals["A"].shape[1]
    norm = 0

    # A
    X = np.zeros((num_mo,  num_mo))
    for x in range(space["A"]["O"]["start"], space["A"]["O"]["start"] + space["A"]["O"]["length"]):
        for y in range(space["A"]["V"]["start"], space["A"]["V"]["start"] + space["A"]["V"]["length"]):
            index = (( (x, ), tuple() ), ( (y, ), tuple() ))
            value = amplitudes[index]
            amplitudes[index] = 0
            norm += value ** 2
            X[x][y] = - value
            X[y][x] = value
    U = spla.expm(X)
    orbitals["A"] = orbitals["A"] @ U

    # B
    X = np.zeros((num_mo,  num_mo))
    for x in range(space["B"]["O"]["start"], space["B"]["O"]["start"] + space["B"]["O"]["length"]):
        for y in range(space["B"]["V"]["start"], space["B"]["V"]["start"] + space["B"]["V"]["length"]):
            index = (( tuple(), (x, ) ), ( tuple(), (y, ) ))
            value = amplitudes[index]
            amplitudes[index] = 0
            norm += value ** 2
            X[x][y] = - value
            X[y][x] = value
    U = spla.expm(X)
    orbitals["B"] = orbitals["B"] @ U

    return np.sqrt(norm)
