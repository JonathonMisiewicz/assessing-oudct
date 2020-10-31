from ..qc_codes import psi as program
import itertools
import numpy as np
from .. import math_util, chem
from scipy import linalg as spla
from .. import multilinear as mla

def subspace(molecule, solver, test=False, comp_grad=False, e_thresh=1e-14, r_thresh=1e-9, **kwargs):
    CHARGE = molecule["charge"]
    NUM_UNPAIRED = molecule["num_unpaired"]
    BASIS = molecule["basis"]
    GEOM = molecule["geom"]

    atoms = [i[0] for i in GEOM]

    # The following integrals are in the atomic orbital basis.
    h_ao = program.core_hamiltonian(BASIS, GEOM)
    r_ao = program.repulsion(BASIS, GEOM)

    h_aso = mla.to_spinorb(h_ao, electron_indices=((0, 1),))
    r_aso = mla.to_spinorb(r_ao, electron_indices=((0, 2), (1, 3)))
    g_aso = r_aso - np.transpose(r_aso, (0, 1, 3, 2))

    dim_basis = program.nbf(BASIS, GEOM)
    dim_s_basis = 2 * dim_basis

    try:
        orbitals = program.read_orbitals(BASIS, GEOM)
    except AssertionError:
        orbitals = program.unrestricted_orbitals(BASIS, GEOM, CHARGE, NUM_UNPAIRED, **kwargs)
    en_nuc = chem.nuc.energy(GEOM)

    intermed, orbitals = solver(en_nuc, h_ao, r_ao, orbitals, e_thresh=e_thresh, r_thresh=r_thresh)

    if comp_grad:
        nx = program.nuclear_potential_deriv(GEOM)
        oei = math_util.mo_oei_hermitian_block(orbitals, intermed)
        tei = math_util.mo_tei_hermitian_even(orbitals, intermed)
        gei = math_util.mo_gei_hermitian_even(orbitals, intermed)
        grad = np.zeros(nx.shape)
        for atom in range(len(GEOM)):
            hx_ao = program.core_hamiltonian_grad(BASIS, GEOM, atom)
            rx_ao = program.repulsion_grad(BASIS, GEOM, atom)
            sx_ao = program.overlap_grad(BASIS, GEOM, atom)
            for i, (h, r, s) in enumerate(zip(hx_ao, rx_ao, sx_ao)):
                hx_aso = mla.to_spinorb(h, electron_indices=((0,1), ))
                sx_aso = mla.to_spinorb(s, electron_indices=((0,1), ))
                rx_aso = mla.to_spinorb(r, electron_indices=((0,2), (1,3)))
                gx_aso = rx_aso - np.transpose(rx_aso, (0, 1, 3, 2))
                grad[atom][i] = math_util.gradient_contribution(gei, hx_aso, gx_aso, sx_aso, nx[atom][i], oei, tei)
        print(grad)
        intermed["gradient"] = grad

    if test:
        # The dipole moment is the negative derivative of energy with respect to
        # electric field strength. Compare the analytic dipole moment (using the 1RDMs)
        # and the numerical dipole moment (finite difference of energies.)
        # If our implementation is correct, the two should match.
        p_ao = program.dipole(BASIS, GEOM)
        p_aso = mla.to_spinorb(p_ao, electron_indices=((0, 1),))
        intermed["deriv"] = - math_util.hellmann_test(h_ao, p_ao, r_ao, orbitals, solver)
        # TODO: This list should probably be pulled from elsewhere.
        p = mla.request_asym(p_ao, orbitals, itertools.combinations_with_replacement(["c", "o", "v", "w"], 2))
        intermed["mu"] = 0
        for label, block in p.items():
            rdm_label = f"rdm_{label}"
            if rdm_label in intermed:
                prefactor = 1 if label[0] == label[1] else 2
                intermed["mu"] -= mla.tensor.einsum("pq x, pq -> x", block, intermed[rdm_label]) * prefactor

    return intermed, orbitals

