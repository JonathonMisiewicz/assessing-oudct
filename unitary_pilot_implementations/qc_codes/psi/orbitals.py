from collections import OrderedDict
import psi4.core
import numpy as np
from . import misc, integrals

def unrestricted_orbitals(basis, geom, charge=0, spin=0, niter=100,
                         e_thresh=1e-12, r_thresh=1e-9, fc=False, **kwargs):
    """urestricted alpha and beta Hartree-Fock orbital coefficients
    :param basis: basis set name
    :type basis: str
    :param labels: atomic symbols labeling the nuclei
    :type labels: tuple
    :param coords: nuclear coordinates in Bohr
    :type coords: numpy.ndarray
    :param charge: total molecular charge
    :type charge: int
    :param spin: number of unpaired electrons
    :type spin: int
    :param niter: maximum number of iterations
    :type niter: int
    :param e_thresh: energy convergence threshold
    :type e_thresh: float
    :param r_thresh: residual convergence threshold
    :type r_thresh: float
    :return: an array of two square matrices
    :rtype: numpy.ndarray
    """

    mol = _psi4_molecule(geom, charge, spin)
    options = {
        "e_convergence": e_thresh,
        "d_convergence": r_thresh,
        "maxiter": niter,
        "reference": "UHF",
        "scf_type": "pk",
        "basis": basis,
        "freeze_core": False
    }
    for key, value in options.items():
        psi4.core.set_global_option(key, value)
    kwargs["molecule"] = mol
    kwargs["return_wfn"] = True
    uhf_wfn = psi4.energy('hf', **kwargs)[1]
    return OrderedDict([
        ("c", (np.array(uhf_wfn.Ca_subset("AO", "FROZEN_OCC")),
              np.array(uhf_wfn.Cb_subset("AO", "FROZEN_OCC")))),
        ("o", (np.array(uhf_wfn.Ca_subset("AO", "ACTIVE_OCC")),
              np.array(uhf_wfn.Cb_subset("AO", "ACTIVE_OCC")))),
        ("v", (np.array(uhf_wfn.Ca_subset("AO", "ACTIVE_VIR")),
              np.array(uhf_wfn.Cb_subset("AO", "ACTIVE_VIR")))),
        ("w", (np.array(uhf_wfn.Ca_subset("AO", "FROZEN_VIR")),
              np.array(uhf_wfn.Cb_subset("AO", "FROZEN_VIR"))))
        ])

def _psi4_molecule(geom, charge, spin, symmetry=True):
    """build a Psi4 Molecule object
    :param basis: basis set name
    :type basis: str
    :param labels: atomic symbols labeling the nuclei
    :type labels: tuple
    :param coords: nuclear coordinates in Bohr
    :type coords: numpy.ndarray
    :rtype: psi4.core.Molecule
    """

    coord_str = misc._coordinate_string(geom)
    if not symmetry: coord_str += "\nsymmetry c1"
    mol = psi4.core.Molecule.from_string(coord_str)
    mol.set_molecular_charge(charge)
    mol.set_multiplicity(spin + 1)
    return mol

def read_orbitals(basis, geom):
    try:
        cca = np.load("cca.npy")
        ccb = np.load("ccb.npy")
        cia = np.load("cia.npy")
        cib = np.load("cib.npy")
        cva = np.load("cva.npy")
        cvb = np.load("cvb.npy")
        cwa = np.load("cwa.npy")
        cwb = np.load("cwb.npy")
    except IOError:
        raise AssertionError
    else:
        S = integrals.overlap(basis, geom)
        return spaces_orthonormalized(((cca, ccb), (cia, cib), (cva, cvb), (cwa, cwb)), S)

def spaces_orthonormalized(C, S):
    cc, ci, cv, cw = C
    alphas = np.hstack((cc[0], ci[0], cv[0], cw[0]))
    betas = np.hstack((cc[1], ci[1], cv[1], cw[1]))
    ON_alpha = orthonormalize(alphas, S)
    ON_beta = orthonormalize(betas, S)
    cA = cc[0].shape[1]
    iA = ci[0].shape[1]
    vA = cv[0].shape[1]
    cB = cc[1].shape[1]
    iB = ci[1].shape[1]
    vB = cv[1].shape[1]
    cc_a, ci_a, cv_a, cw_a = np.hsplit(ON_alpha, (cA, cA + iA, cA + iA + vA))
    cc_b, ci_b, cv_b, cw_b = np.hsplit(ON_beta, (cB, cB + iB, cB + iB + vB))
    return OrderedDict([
        ("c", (cc_a, cc_b)),
        ("o", (ci_a, ci_b)),
        ("v", (cv_a, cv_b)),
        ("w", (cw_a, cw_b))
        ])

def orthonormalize(C, S):
    # We SYMMETRIC ORTHOGONALIZE rather than CANONICAL ORTHOGONALIZE.
    # Metric matrix for the old MOs
    try:
        S_C = C.T @ S @ C
    except ValueError:
        raise Exception("Dimension mismatch between orbitals and overlap matrices. Try removing orbital files.")
    # NEXT TASK: The ^-0.5
    vals, vecs = np.linalg.eigh(S_C)
    A = np.linalg.inv(np.diag(np.sqrt(vals)))
    X = vecs @ A @ vecs.T
    return C @ X
