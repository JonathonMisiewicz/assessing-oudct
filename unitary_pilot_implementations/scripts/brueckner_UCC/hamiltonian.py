import numpy as np
import itertools

def construct_H(det_list, h, g):
    """
    Input
    -----
    det_list: list of Determinant
    h: np.ndarray
        One-electron integrals, in spatial MOs.
    g: np.ndarray
        Two-electron integrals, in spatial MOs.

    Output
    -------
    FCIH: np.ndarray
        The Hamiltonian in the basis of Slater Determinants
    """
    FCIH = np.zeros((len(det_list), len(det_list)))

    # Construct off-diagonals.
    for (det_idx1, det1), (det_idx2, det2) in itertools.combinations(enumerate(det_list), 2):
        excitation_level = (det1 - det2) / 2
        if excitation_level == 1:
            alpha_list1, beta_list1 = det1.set_subtraction_list(det2)
            if alpha_list1:
                m = alpha_list1[0]
                n = det2.set_subtraction_list(det1)[0][0]
                value = h["A"][m][n]
                for p, (i, j) in enumerate(zip(det1.alpha_list(), det2.alpha_list())):
                    if i and j:
                        value += g["AA"][m, p, n, p]
                for p, (i, j) in enumerate(zip(det1.beta_list(), det2.beta_list())):
                    if i and j:
                        value += g["AB"][m, p, n, p]
            else:
                m = beta_list1[0]
                n = det2.set_subtraction_list(det1)[1][0]
                value = h["B"][m, n]
                for p, (i, j) in enumerate(zip(det1.alpha_list(), det2.alpha_list())):
                    if i and j:
                        value += g["AB"][p, m, p, n]
                for p, (i, j) in enumerate(zip(det1.beta_list(), det2.beta_list())):
                    if i and j:
                        value += g["BB"][p, m, p, n]
        elif excitation_level == 2:
            set_subtract = det1.set_subtraction_list(det2)
            alpha_excitations = len(set_subtract[0])
            if alpha_excitations == 0:
                m, n = set_subtract[1]
                p, q = det2.set_subtraction_list(det1)[1]
                value = g["BB"][m, n, p, q]
            elif alpha_excitations == 1:
                m = set_subtract[0][0]
                n = set_subtract[1][0]
                set_subtract2 = det2.set_subtraction_list(det1)
                p = set_subtract2[0][0]
                q = set_subtract2[1][0]
                value = g["AB"][m, n, p, q]
            elif alpha_excitations == 2:
                m, n = set_subtract[0]
                p, q = det2.set_subtraction_list(det1)[0]
                value = g["AA"][m, n, p, q]
            else:
                raise Exception("You have two excitations... But " + str(alpha_excitations) + " alpha excitations?")
        else:
            continue
        FCIH[det_idx1][det_idx2] = FCIH[det_idx2][det_idx1] = value * det1.phase(det2)

    # Construct diagonals.
    for det_idx, det in enumerate(det_list):
        alpha_list = det.alpha_list()
        beta_list = det.beta_list()
        # One-electron alpha
        value = sum(h["A"][orb_idx, orb_idx] for orb_idx, occ in enumerate(alpha_list) if occ)
        # One-electron beta
        value += sum(h["B"][orb_idx, orb_idx] for orb_idx, occ in enumerate(beta_list) if occ)
        # Two-electron AAAA
        value += sum(g["AA"][orb1, orb2, orb1, orb2] for (orb1, occ1), (orb2, occ2) in itertools.combinations(enumerate(alpha_list), 2) if occ1 and occ2)
        # Two-electron ABAB
        value += sum(g["AB"][orb1, orb2, orb1, orb2] for (orb1, occ1), (orb2, occ2) in itertools.product(enumerate(alpha_list), enumerate(beta_list)) if occ1 and occ2)
        # Two-electron BBBB
        value += sum(g["BB"][orb1, orb2, orb1, orb2] for (orb1, occ1), (orb2, occ2) in itertools.combinations(enumerate(beta_list), 2) if occ1 and occ2)
        FCIH[det_idx][det_idx] = value

    return FCIH
