from . import hamiltonian
import numpy as np
from scipy import linalg as spla
from . import amplitude_setup, integrals
from ...qc_codes.psi import wfn_analysis
from ...math_util.convergence import DirectSumDiis
import itertools
from . import ov_product

def OUCC(fci_wfn):
    """
    Perform orbital-optimized UCC, exactly. Small basis sets only, unless you have a lot of resources.

    Input
    -----
    fci_wfn: psi4.core.wfn
        A Psi4 FCI wavefunction.

    Output
    ------
    T2_dict: dict
        Maps from {"AA", "AB", "BB"} to the relevant T2 amplitudes.
    orbitals: dict
        Maps from {"A", "B"} to the orbitals.
    energy: float
        The final energy
    """
    spaces = get_spaces(fci_wfn)
    amplitudes, det_list, ab_idx_map = amplitude_setup.amplitudes(spaces)
    # We'll use the natural orbitals as a guess.
    # For a two-electron system, these are the exact Brueckner/optimal UCC orbitals.
    orbitals = wfn_analysis.NSO_C_from_fci(fci_wfn)
    while True:
        h, g, f = integrals.construct_integrals(orbitals, fci_wfn)
        amplitudes, energy, ci_vec = solve_UCC(amplitudes, det_list, h, g, f, fci_wfn.molecule().nuclear_repulsion_energy())
        t1_norm = integrals.rotate_orbitals(amplitudes, orbitals, spaces)
        if t1_norm < 1e-8:
            print("Orbitals converged")
            break
        else:
            print(f"T1 Norm: {t1_norm:10.14f}")
    # Now construct OPDM A...
    pq0 = np.zeros(orbitals["A"].shape)
    pq1 = np.zeros(orbitals["A"].shape)
    pq2 = np.zeros(orbitals["A"].shape)
    pqinf = np.zeros(orbitals["A"].shape)
    OPDM_AA = np.zeros(orbitals["A"].shape)
    OPDM_AA0 = np.zeros(orbitals["A"].shape)
    OPDM_AA1 = np.zeros(orbitals["A"].shape)
    OPDM_AA2 = np.zeros(orbitals["A"].shape)
    num_mo = OPDM_AA.shape[0]
    for p, q in itertools.product(range(num_mo), repeat=2):
        pq0 = ov_product.pq_action(ci_vec[0], det_list, p, q, ab_idx_map)
        pq1 = ov_product.pq_action(ci_vec[1], det_list, p, q, ab_idx_map)
        pq2 = ov_product.pq_action(ci_vec[2], det_list, p, q, ab_idx_map)
        pqinf = ov_product.pq_action(ci_vec["inf"], det_list, p, q, ab_idx_map)
        OPDM_AA0[p, q] = np.dot(ci_vec[0], pq0)
        OPDM_AA1[p, q] = np.dot(ci_vec[0], pq1) + np.dot(ci_vec[1], pq0)
        OPDM_AA2[p, q] = np.dot(ci_vec[0], pq2) + np.dot(ci_vec[1], pq1) + np.dot(ci_vec[2], pq0)
        OPDM_AA[p, q] = np.dot(ci_vec["inf"], pqinf)
    print("Printing AA OPDM: 0 commutators")
    print(OPDM_AA0)
    print("Printing AA OPDM: 1 commutators")
    print(OPDM_AA1)
    print("Printing AA OPDM: 2 commutators")
    print(OPDM_AA2)
    print("Printing AA OPDM: all commutators")
    print(OPDM_AA)
    T2_dict = {key: amplitude_setup.extract_spincase(amplitudes, key, spaces) for key in {"AA", "AB", "BB"}}
    return T2_dict, orbitals, energy


def get_spaces(wfn):
    space_dict = {}
    for symbol, method in {("A", wfn.Ca_subset), ("B", wfn.Cb_subset)}:
        get_dim = lambda x: method("AO", x).np.shape[1]
        temp = {}
        start = 0
        for (space_symbol, space) in [("F", "FROZEN_OCC"), ("O", "ACTIVE_OCC"), ("V", "ACTIVE_VIR"), ("W", "FROZEN_VIR")]:
            length = get_dim(space)
            temp[space_symbol] = {"start": start, "length": length}
            start += length
        space_dict[symbol] = temp

    return space_dict


def solve_UCC(amplitudes, det_list, h, g, f, nuclear_repulsion):
    """ Solve the UCC equations for a given set of orbitals.

    Input
    -----
    amplitudes
    det_list: list of Determinants
    h: dict
        Map from spincases {"A", "B"}  to one-electron integrals
    g: dict
        Map from spincases {"AA", "AB", "BB"} to two-electron integrals
    f: dict
        Map from spincases {"A", "B"} to Fock elements

    Output
    ------
    amplitudes: dict
        Maps an amplitude-tuple to a float. An amplitude tuple stores ((ao, bo), (av, bv)).
        a vs b specifies alpha vs beta.
        o vs v specifies an occupied orbital excited FROM vs virtual orbtial excited TO.
    energy: float
    ci_vecs: dict
        Map from {0, 1, 2, "inf"} to a CI vector. For 0, 1, and 2, it's that n-commutator part of the CI vector.
        For "inf", the result is the final FCI vector.
    """
    ref = det_list[0]
    phi = np.zeros((len(det_list)))
    phi[0] = 1
    FCIH = hamiltonian.construct_H(det_list, h, g)
    HeffPhi = np.ones((len(det_list))) # Initialize this to a value that will satisfy the while condition.
    dsd = DirectSumDiis(3, 7)
    while np.linalg.norm(HeffPhi[1:]) > 1e-10:
        S = amplitude_setup.construct_lie_algebra_elt(amplitudes, det_list)
        HeffPhi = spla.expm(-S) @ (FCIH @ (spla.expm(S) @ phi))
        # To implement DIIS, we exploit that in Python 3.6, dicts are ordered by insertion, and that the order of amplitude
        # insertion is the same as the det_list order. We store the residual and amplitudes in an nd.array, pass that to
        # the DIIS code, then convert the new result back into our amplitudes dict.
        R = np.zeros(len(amplitudes))
        T = np.zeros(len(amplitudes))
        # Each element is the action of our effective Hamiltonian on Phi
        for det_idx, (r, det) in enumerate(zip(HeffPhi, det_list)):
            # Projection onto the reference is the energy
            if det_idx == 0:
                energy = r
                continue
            # Projection onto an excited reference should be zero, so we need to change the amplitude to enforce this.
            # We do this by the standard Jacobi step
            # r(t) + r'(t) dt = 0 => dt = - [r'(t)]^-1 r(t)
            # If we crudely approximate the similarity transformed Hamiltonian as a sum of diagonal operators with orbital
            # energies, r'^-1 is just an MP-esque denominator times a phase factor.
            # Orbital energies are given by h^p_p + g^pi_pi. (See construct_integrals.)
            # Amplitude (occ, vir) is the coefficient of ^v_o - ^o_v, so take r' as (εv - εo) times determinant phase.
            occ = ref.set_subtraction_list(det)
            vir = det.set_subtraction_list(ref)
            vir_epsilon = sum(sum(ints[p] for p in orbs) for orbs, ints in zip(vir, [f["A"], f["B"]]))
            occ_epsilon = sum(sum(ints[p] for p in orbs) for orbs, ints in zip(occ, [f["A"], f["B"]]))
            R[det_idx - 1] = r
            T[det_idx - 1] = amplitudes[occ, vir] - r / (vir_epsilon - occ_epsilon) * det.phase(ref, False)
            #amplitudes[occ, vir] -= r / (vir_epsilon - occ_epsilon) * det.phase(ref, False)
        new_amps = dsd.diis(tuple(R), tuple(T))
        for i, (key) in enumerate(amplitudes):
            amplitudes[key] = new_amps[i]
        print(energy + nuclear_repulsion)
    ci_vecs = {"inf": spla.expm(S) @ phi}
    ci_vecs[0] = phi
    ci_vecs[1] = S @ ci_vecs[0]
    ci_vecs[2] = S @ ci_vecs[1] / 2
    return amplitudes, energy + nuclear_repulsion, ci_vecs
