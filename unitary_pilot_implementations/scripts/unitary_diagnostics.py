import psi4
from ..qc_codes.psi import wfn_analysis, orbitals
from ..multilinear import tensor
from . import scripts
import numpy as np
from .brueckner_UCC import main as bUCC
import math
import qcelemental

def test_amplitudes(mol_data, solver, opdm_ov_norm=True):
    """
    This is the function responsible for all the H2 analysis!
    I've included some other diagnostics here which didn't get mentioned in the paper, mainly involving comparisons to the cumulant.
    These are included for interest, but won't be fully discussed until the paper I mention in the Conclusion, where I present a "new ansatz."
    """
    print("Performing approximate unitary computation.")
    intermed, odc_orbitals = scripts.runner.subspace(mol_data, test=False, comp_grad=False)
    print("Approximate unitary computation complete. Performing additional processing.")
    # Construct the alpha and beta coefficient matrices for the natural orbitals.
    approx_orbs = {
            "A": np.hstack((odc_orbitals["o"][0], odc_orbitals["v"][0])),
            "B": np.hstack((odc_orbitals["o"][1], odc_orbitals["v"][1])),
            }
    # Next, construct spin-blocked cumulants (entire and OOVV) from our spinorbital cumulants.
    # Because our integrals our stored as all alpha and then all beta, our amplitudes are as well.
    num_alpha_occ_nso = odc_orbitals["o"][0].shape[1]
    num_alpha_vir_nso = odc_orbitals["v"][0].shape[1]
    num_beta_occ_nso = odc_orbitals["o"][1].shape[1]
    num_beta_vir_nso = odc_orbitals["v"][1].shape[1]
    OA = slice(None, num_alpha_occ_nso)
    VA = slice(None, num_alpha_vir_nso)
    OB = slice(num_beta_occ_nso, None)
    VB = slice(num_beta_vir_nso, None)
    λOOVV_AA_app_app = intermed["c_oovv"][OA, OA, VA, VA]
    λOOVV_AB_app_app = intermed["c_oovv"][OA, OB, VA, VB]
    λOOVV_BB_app_app = intermed["c_oovv"][OB, OB, VB, VB]
    λOOOO_AA_app_app = intermed["c_oooo"][OA, OA, OA, OA]
    λOOOO_AB_app_app = intermed["c_oooo"][OA, OB, OA, OB]
    λOOOO_BB_app_app = intermed["c_oooo"][OB, OB, OB, OB]
    λVVVV_AA_app_app = intermed["c_vvvv"][VA, VA, VA, VA]
    λVVVV_AB_app_app = intermed["c_vvvv"][VA, VB, VA, VB]
    λVVVV_BB_app_app = intermed["c_vvvv"][VB, VB, VB, VB]
    λOVOV_AA_app_app = intermed["c_ovov"][OA, VA, OA, VA]
    λOVOV_BB_app_app = intermed["c_ovov"][OB, VB, OB, VB]
    λOVOV_AB_app_app = intermed["c_ovov"][OA, VB, OA, VB]
    λVOOV_AB_app_app = - np.transpose(intermed["c_ovov"][OB, VA, OA, VB], (1, 0, 2, 3))
    λVOVO_AB_app_app = np.transpose(intermed["c_ovov"][OB, VA, OB, VA], (1, 0, 3, 2))
    T2_AA_app = intermed["t2"][OA, OA, VA, VA]
    T2_AB_app = intermed["t2"][OA, OB, VA, VB]
    T2_BB_app = intermed["t2"][OB, OB, VB, VB]

    OA = slice(None, num_alpha_occ_nso)
    VA = slice(num_alpha_occ_nso, None)
    OB = slice(None, num_beta_occ_nso)
    VB = slice(num_beta_occ_nso, None)
    num_nso = approx_orbs["A"].shape[1]

    # Construct the AAAA block of the parameters and the full cumulant.
    λAA_app_app = np.zeros((num_nso,) * 4)
    AA_param_app_app = np.copy(λAA_app_app)
    λAA_app_app[OA, OA, VA, VA] = λOOVV_AA_app_app
    AA_param_app_app[OA, OA, VA, VA] = T2_AA_app
    λAA_app_app[VA, VA, OA, OA] = np.transpose(λOOVV_AA_app_app, (2, 3, 0, 1))
    λAA_app_app[OA, OA, OA, OA] = λOOOO_AA_app_app
    λAA_app_app[VA, VA, VA, VA] = λVVVV_AA_app_app
    λAA_app_app[OA, VA, OA, VA] = λOVOV_AA_app_app
    λAA_app_app[VA, OA, OA, VA] = - np.transpose(λOVOV_AA_app_app, (1, 0, 2, 3))
    λAA_app_app[OA, VA, VA, OA] = - np.transpose(λOVOV_AA_app_app, (0, 1, 3, 2))
    λAA_app_app[VA, OA, VA, OA] = np.transpose(λOVOV_AA_app_app, (1, 0, 3, 2))

    # Construct the ABAB block of the parameters and the full cumulant.
    λAB_app_app = np.zeros((num_nso,) * 4)
    AB_param_app_app = np.copy(λAB_app_app)
    λAB_app_app[OA, OB, VA, VB] = λOOVV_AB_app_app
    AB_param_app_app[OA, OB, VA, VB] = T2_AB_app
    λAB_app_app[VA, VB, OA, OB] = np.transpose(λOOVV_AB_app_app, (2, 3, 0, 1))
    λAB_app_app[OA, OB, OA, OB] = λOOOO_AB_app_app
    λAB_app_app[VA, VB, VA, VB] = λVVVV_AB_app_app
    λAB_app_app[OA, VB, OA, VB] = λOVOV_AB_app_app
    λAB_app_app[VA, OB, OA, VB] = λVOOV_AB_app_app
    λAB_app_app[OA, VB, VA, OB] = np.transpose(λVOOV_AB_app_app, (2, 3, 0, 1))
    λAB_app_app[VA, OB, VA, OB] = λVOVO_AB_app_app

    # Construct the BBBB block of the parameters and the full cumulant.
    λBB_app_app = np.zeros((num_nso,) * 4)
    BB_param_app_app = np.copy(λBB_app_app)
    λBB_app_app[OB, OB, VB, VB] = λOOVV_BB_app_app
    BB_param_app_app[OB, OB, VB, VB] = T2_BB_app
    λBB_app_app[VB, VB, OB, OB] = np.transpose(λOOVV_BB_app_app, (2, 3, 0, 1))
    λBB_app_app[OB, OB, OB, OB] = λOOOO_BB_app_app
    λBB_app_app[VB, VB, VB, VB] = λVVVV_BB_app_app
    λBB_app_app[OA, VA, OA, VA] = λOVOV_AA_app_app
    λBB_app_app[VA, OA, OA, VA] = - np.transpose(λOVOV_BB_app_app, (1, 0, 2, 3))
    λBB_app_app[OB, VB, VB, OB] = - np.transpose(λOVOV_BB_app_app, (0, 1, 3, 2))
    λBB_app_app[VB, OB, VB, OB] = np.transpose(λOVOV_BB_app_app, (1, 0, 3, 2))
    print("Approximate unitary post-processing complete.")

    molecule = orbitals._psi4_molecule(mol_data["geom"], mol_data["charge"], mol_data["num_unpaired"], True)
    print(molecule.nuclear_repulsion_energy())
    print("Performing FCI computation.")
    psi4.set_options({"opdm": True, "tpdm": True, "e_convergence": 12, "r_convergence": 8, "reference": "RHF"}) # Needed for OPDM/TPDM extraction. 
    fci_energy, fci_wfn = psi4.energy("fci/" + mol_data["basis"], molecule = molecule, return_wfn = True)
    print("FCI computation complete.")

    print("Performing OUCC computation.")
    T2_fci, OUCC_orbs, oucc_en = bUCC.OUCC(fci_wfn)
    assert math.isclose(fci_energy, oucc_en, abs_tol=1e-10)
    print("OUCC computation complete.")
    print("OUCC and FCI agree about the energy. Sanity check passed!")

    if opdm_ov_norm:
        opdm = wfn_analysis.OPDM_from_fci(fci_wfn)
        Ca = wfn_analysis.Ca_aso(fci_wfn)
        alpha_transform = tensor.compute_basis_transformation(Ca, OUCC_orbs["A"])
        transformed = tensor.transform_all_indices(opdm["A"], (alpha_transform, alpha_transform))
        print(f"OV norm: {np.linalg.norm(transformed[:fci_wfn.nalpha(), fci_wfn.nalpha():]):20.10f}")

    print("Operating on the FCI RDMs.")
    # Compute natural orbitals, from most virtual to most occupied.
    nat_orbs = wfn_analysis.NSO_from_fci(fci_wfn)
    nat_orbs_C = wfn_analysis.NSO_C_from_fci(fci_wfn)
    print("Orbitals extracted.")

    # Construct the exact cumulant in the NSO space.
    cumulant_fci_fci = wfn_analysis.cumulant_from_fci(fci_wfn)
    λ2_AA_fci_nso = tensor.transform_all_indices(cumulant_fci_fci["AA"], (nat_orbs["A"], nat_orbs["A"], nat_orbs["A"], nat_orbs["A"]))
    λ2_BB_fci_nso = tensor.transform_all_indices(cumulant_fci_fci["BB"], (nat_orbs["B"], nat_orbs["B"], nat_orbs["B"], nat_orbs["B"]))
    λ2_AB_fci_nso = tensor.transform_all_indices(cumulant_fci_fci["AB"], (nat_orbs["A"], nat_orbs["B"], nat_orbs["A"], nat_orbs["B"]))

    # Construct the OOVV block of the exact cumulant in the NSO space.
    λ2_AA_OOVV_fci_nso = λ2_AA_fci_nso[OA, OA, VA, VA]
    λ2_BB_OOVV_fci_nso = λ2_BB_fci_nso[OB, OB, VB, VB]
    λ2_AB_OOVV_fci_nso = λ2_AB_fci_nso[OA, OB, VA, VB]

    # Compute blocks of the transformation from exact NSOs to approximate unitary orbitals. We can do this by breaking it into blocks.
    transA = tensor.compute_basis_transformation(nat_orbs_C["A"], approx_orbs["A"])
    transB = tensor.compute_basis_transformation(nat_orbs_C["B"], approx_orbs["B"])

    # Cast the FCI cumulant from the NSO basis to the approx. unitary basis.
    λ2_AA_fci_app = tensor.transform_all_indices(λ2_AA_fci_nso, (transA, transA, transA, transA))
    λ2_BB_fci_app = tensor.transform_all_indices(λ2_BB_fci_nso, (transB, transB, transB, transB))
    λ2_AB_fci_app = tensor.transform_all_indices(λ2_AB_fci_nso, (transA, transB, transA, transB))

    # Cast the FCI OOVV (NSO orbital) cumulant from the NSO basis to the approx. unitary basis.
    λ2_AA_OOVV_fci_app = tensor.transform_all_indices(λ2_AA_OOVV_fci_nso, (transA[OA], transA[OA], transA[VA], transA[VA]))
    λ2_BB_OOVV_fci_app = tensor.transform_all_indices(λ2_BB_OOVV_fci_nso, (transB[OB], transB[OB], transB[VB], transB[VB]))
    λ2_AB_OOVV_fci_app = tensor.transform_all_indices(λ2_AB_OOVV_fci_nso, (transA[OA], transB[OB], transA[VA], transB[VB]))

    # Compute blocks of the transformation from exact OUCC orbitals to approximate DCT orbitals.
    transA = tensor.compute_basis_transformation(OUCC_orbs["A"], approx_orbs["A"])
    transB = tensor.compute_basis_transformation(OUCC_orbs["B"], approx_orbs["B"])

    # Transform the exact amplitudes from the exact OUCC basis to the approximate unitary basis.
    T2_AA_fci_app = tensor.transform_all_indices(T2_fci["AA"], (transA[OA], transA[OA], transA[VA], transA[VA]))
    T2_BB_fci_app = tensor.transform_all_indices(T2_fci["BB"], (transB[OB], transB[OB], transB[VB], transB[VB]))
    T2_AB_fci_app = tensor.transform_all_indices(T2_fci["AB"], (transA[OA], transA[OA], transB[VB], transB[VB]))

    cumulant_amp_error = compute_metric(λ2_AA_OOVV_fci_app, λ2_AB_OOVV_fci_app, λ2_BB_OOVV_fci_app, AA_param_app_app, AB_param_app_app, BB_param_app_app)
    ucc_amp_error = compute_metric(T2_AA_fci_app, T2_AB_fci_app, T2_BB_fci_app, AA_param_app_app, AB_param_app_app, BB_param_app_app)
    cumulant_error = compute_metric(λ2_AA_fci_app, λ2_AB_fci_app, λ2_BB_fci_app, λAA_app_app, λAB_app_app, λBB_app_app)
    cumulant_norm = compute_metric(λ2_AA_OOVV_fci_app, λ2_AB_OOVV_fci_app, λ2_BB_OOVV_fci_app, 0, 0, 0)
    ucc_norm = compute_metric(T2_AA_fci_app, T2_AB_fci_app, T2_BB_fci_app, 0, 0, 0)
    cross_measure = compute_metric(λ2_AA_OOVV_fci_app, λ2_AB_OOVV_fci_app, λ2_BB_OOVV_fci_app, T2_AA_fci_app, T2_AB_fci_app, T2_BB_fci_app)
    print(f"λOOVV vs Params {cumulant_amp_error:17.10f}")
    print(f"UCC T vs Params {ucc_amp_error:17.10f}")
    print(f"λ vs λ {cumulant_error:26.10f}")
    print(f"λOOVV Norm {cumulant_norm:22.10f}")
    print(f"UCC T Norm {ucc_norm:22.10f}")
    print(f"λOOVV-T Metric {cross_measure:18.10f}")

    print(f"Approx. Energy {intermed['energy']:18.10f}")
    print(f"Exact Energy {fci_energy:20.10f}")
    print(f"Energy Error {intermed['energy']-fci_energy:20.10f}")

    opdm_trace = intermed["rdm_oo"].trace() + intermed["rdm_vv"].trace()
    tpdm_trace = np.einsum("IJ IJ ->", intermed["rdm_oooo"]) + np.einsum("AB AB ->", intermed["rdm_vvvv"]) + 2 * np.einsum("IA IA ->", intermed["rdm_ovov"])
    nelec = num_alpha_occ_nso + num_beta_occ_nso
    opdm_defect = opdm_trace - nelec
    tpdm_defect = tpdm_trace - nelec * (nelec - 1)
    print(f"OPDM Trace Defect {opdm_defect:15.10f}")
    print(f"TPDM Trace Defect {tpdm_defect:15.10f}")

def compute_metric(AA1, AB1, BB1, AA2, AB2, BB2):
    # Why the factor of 4? ABAB, ABBA, BAAB, and BABA
    return np.sqrt(np.sum((AA1 - AA2) ** 2) + 4 * np.sum((AB1 - AB2) ** 2) + np.sum((BB1 - BB2) ** 2))

test_amplitudes({ 
    "charge": 0,
    "num_unpaired": 0,
    "geom": [
         ('H', (0.000000000000, 0, 0)),
         ('H', (0.000000000000, 0, 1.28 / qcelemental.constants.bohr2angstroms))
         ],  
    "basis": "cc-pvdz"
    },
    script.odc12.simultaneous # Swap me out to change the electronic structure method you study!
)
