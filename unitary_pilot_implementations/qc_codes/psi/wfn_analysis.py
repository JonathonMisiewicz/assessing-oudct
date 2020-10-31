import numpy as np
from scipy import linalg as spla

def OPDM_from_fci(fci_wfn):
    """
    Parameters: CIWavefunction from Psi4
    Returns: {"A": np.ndarray, "B": np.ndarray}
             Each ndarray is the relevant spin-block of the OPDM in the CI compute basis.
             Orbitals are in Pitzer order for the symmetry used. Matrices of different irreps
             are concatenated.
    """
    return {spin_case: mat_2d(fci_wfn.get_opdm(-1, -1, spin_case, True)) for spin_case in {"A", "B"}}

def mat_2d(mat):
    return spla.block_diag(*mat.nph)

def TPDM_from_fci(fci_wfn):
    """
    Parameters: CIWavefunction from Psi4
    Returns: {"AA": np.ndarray, "AB": np.ndarray, "BB": np.ndarray}
             Each ndarray is the relevant spin-block of the TPDM in the CI compute basis.
    """
    # The Psi4 convention of ordering axes is that pqrs means p^ r^ s q. Mine is that pqrs means p^ q^ s r.
    # Those unconvinced about the Psi4 convention are referred to ciwave.h:get_tpdm in the psi4 detci source.
    return {spin_case: np.swapaxes(fci_wfn.get_tpdm(spin_case, False), 1, 2) for spin_case in {"AA", "AB", "BB"}}

def NSO_from_fci(fci_wfn):
    """
    Parameters: CIWavefunction from Psi4
    Returns: {"A": np.ndarray, "B": np.ndarray}
             Each ndarray is the relevant spin-block of natural spinorbitals in the CI compute basis.
             First index (rows) is compute orbital. Second index (columns) is NSOs.
             NSOs are ordered by occupation number descending, most occupied to least occupied.
    """
    OPDM = OPDM_from_fci(fci_wfn)
    return {spin_case: np.linalg.eigh(opdm)[1][:, ::-1] for spin_case, opdm in OPDM.items()}

def Ca_aso(fci_wfn):
    return vecs(fci_wfn.aotoso()) @ mat_2d(fci_wfn.Ca())

def Cb_aso(fci_wfn):
    return vecs(fci_wfn.aotoso()) @ mat_2d(fci_wfn.Cb())

def NSO_C_from_fci(fci_wfn):
    """
    Parameters; CIWavefunction from Psi4
    Returns: {"A": np.ndarray, "B": np.ndarray}
             Each ndarray is the relevant spin-block of natural spinorbitals in the AO basis.
             First index (row) is AO. Second index (columns) is NSOs.
             NSO ordering is inherited from NSO_from_fci.
    """
    NSO = NSO_from_fci(fci_wfn)
    return {"A": vecs(fci_wfn.aotoso()) @ mat_2d(fci_wfn.Ca()) @ NSO["A"],
            "B": vecs(fci_wfn.aotoso()) @ mat_2d(fci_wfn.Cb()) @ NSO["B"]}

def NSO_from_fci_symm(fci_wfn):
    OPDM = {spin_case: fci_wfn.get_opdm(-1, -1, spin_case, True) for spin_case in {"A", "B"}} # TODO: Refactor to reduce code duplication with OPDM_from_fci once I'm sure this works.
    return {spin_case: [np.linalg.eigh(opdm_h)[1][:, ::-1] for opdm_h in opdm[spin_case]] for spin_case, opdm in OPDM.items()}

def NSO_C_from_fci_symm(fci_wfn):
    NSO = NSO_from_fci(fci_wfn)
    return {"A": [C_h @ NSO_h for C_h, NSO_h in zip(fci_wfn.Ca(), NSO["A"])],
            "B": [C_h @ NSO_h for C_h, NSO_h in zip(fci_wfn.Cb(), NSO["B"])]}

def vecs(mat):
    return np.hstack(mat.nph)

def cumulant_from_fci(fci_wfn):
    OPDM = OPDM_from_fci(fci_wfn)
    TPDM = TPDM_from_fci(fci_wfn)
    cumulant = {}
    cumulant["AA"] = TPDM["AA"] - np.einsum("pr, qs -> pqrs", OPDM["A"], OPDM["A"]) + np.einsum("ps, qr -> pqrs", OPDM["A"], OPDM["A"])
    cumulant["BB"] = TPDM["BB"] - np.einsum("pr, qs -> pqrs", OPDM["B"], OPDM["B"]) + np.einsum("ps, qr -> pqrs", OPDM["B"], OPDM["B"])
    cumulant["AB"] = TPDM["AB"] - np.einsum("pr, qs -> pqrs", OPDM["A"], OPDM["B"])
    return cumulant

