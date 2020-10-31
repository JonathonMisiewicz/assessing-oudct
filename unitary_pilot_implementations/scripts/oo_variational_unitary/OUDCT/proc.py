from unitary_pilot_implementations import multilinear as mla
from unitary_pilot_implementations.multilinear.tensor import einsum
import numpy as np
from scipy import linalg as spla
from .. import unitary_common
from functools import partial

def dct_simultaneous(cumulant, d, opdm, intermed):
    cumulant(intermed)
    d(intermed)
    opdm(intermed)
    assemble_fock_block_diagonal(intermed)
    block_diagonal_ft(intermed)
    rdm_construct(intermed)

# We're assuming ov vanishes.
def rdm_construct(inter):
    rdm_cc = np.eye(inter["f_cc"].shape[0])
    inter["rdm_oovv"] = inter["c_oovv"]
    # To the cumulant must be added the antisymmetrized product of 1RDMs
    inter["rdm_oooo"] = inter["c_oooo"] + mla.antisymmetrize_axes_plus(
        einsum("IK, JL -> IJKL", inter["rdm_oo"], inter["rdm_oo"]), ((0,), (1,)))
    inter["rdm_vvvv"] = inter["c_vvvv"] + mla.antisymmetrize_axes_plus(
        einsum("AC, BD -> ABCD", inter["rdm_vv"], inter["rdm_vv"]), ((0,), (1,)))
    inter["rdm_ovov"] = inter["c_ovov"] + einsum("IJ, AB -> IAJB", inter["rdm_oo"], inter["rdm_vv"])
    inter["rdm_cccc"] = mla.antisymmetrize_axes_plus(einsum("IK, JL -> IJKL", rdm_cc, rdm_cc), ((0,), (1,)))
    inter["rdm_coco"] = einsum("IK, JL -> IJKL", rdm_cc, inter["rdm_oo"])
    inter["rdm_cvcv"] = einsum("IK, JL -> IJKL", rdm_cc, inter["rdm_vv"])
    if "c_ooov" in inter:
        inter["rdm_ooov"] = inter["c_ooov"]
        inter["rdm_ovvv"] = inter["c_ovvv"]

def assemble_fock_block_diagonal(inter):
    inter["f_cc"] = inter["h_cc"] + einsum("CI DJ, JI -> CD", inter["g_coco"], inter["rdm_oo"]) + (
            einsum("CA DB, BA -> CD", inter["g_cvcv"], inter["rdm_vv"])) + einsum("CE DE -> CD", inter["g_cccc"])
    inter["f_oo"] = inter["h_oo"] + einsum("IJ KL, KI -> JL", inter["g_oooo"], inter["rdm_oo"]) + (
            einsum("IA JB, BA -> IJ", inter["g_ovov"], inter["rdm_vv"])) + einsum("CI CJ -> IJ", inter["g_coco"])
    inter["f_vv"] = inter["h_vv"] + einsum("AB CD, DB -> AC", inter["g_vvvv"], inter["rdm_vv"]) + (
            einsum("IA JB, JI -> AB", inter["g_ovov"], inter["rdm_oo"])) + einsum("CA CB -> AB", inter["g_cvcv"])
    inter["f_ww"] = inter["h_ww"] + einsum("IW JX, JI -> WX", inter["g_owow"], inter["rdm_oo"]) + (
            einsum("AW BX, BA -> WX", inter["g_vwvw"], inter["rdm_vv"])) + einsum("CW CX -> WX", inter["g_cwcw"])

def block_diagonal_ft(inter):
    """
    Special simplification of dct_fock_transformer available for block-diagonal 1RDM.
    """
    inter["ft_oo"] = dct_fock_transformer(inter["f_oo"], inter["rdm_oo"])
    inter["ft_vv"] = dct_fock_transformer(inter["f_vv"], inter["rdm_vv"])

def dct_fock_transformer(f, RDM1):
    """ 
    Computes the intermediate needed in the variational DCT cumulant update equation, corresponding
    to the 1RDM and product of 1RDMs.

    Input
    -----
    f: np.ndarray
        The generalized Fock matrix, h_pq + gbar_pqrs gamma_rs
    RDM1: np.ndarray
        The 1RDM/

    Output
    ------
    np.ndarray
        The target intermediate.
    """
    evals, evecs = np.linalg.eigh(RDM1)
    denom = mla.full_broadcaster((evals, evals)) - 1
    inter_f = mla.tensor.contra_transform(f, evecs) / denom
    return mla.tensor.contra_transform(inter_f, evecs.T)

def canonical_block_diagonal_d(inter):
    inter["d_oo"] = inter["c_oooo"].trace(axis1=1, axis2=3) + inter["c_ovov"].trace(axis1=1, axis2=3)
    inter["d_vv"] = inter["c_vvvv"].trace(axis1=1, axis2=3) + inter["c_ovov"].trace(axis1=0, axis2=2)

def block_1rdm(inter):
    I_o = np.eye(*inter["d_oo"].shape)
    inter["rdm_oo"] = 1/2 * I_o + np.real(spla.sqrtm(inter["d_oo"] + 1/4 * I_o))
    I_v = np.eye(*inter["d_vv"].shape) 
    inter["rdm_vv"] = 1/2 * I_v - np.real(spla.sqrtm(inter["d_vv"] + 1/4 * I_v))

def simultaneous_block_canonical(cumulant, intermed):
    return dct_simultaneous(cumulant, canonical_block_diagonal_d, block_1rdm, intermed)

simultaneous_step = partial(unitary_common.simultaneous_step, diagonal_dict = {
        "c": lambda x: np.diagonal(x["f_cc"]),
        "o": lambda x: np.diagonal(x["ft_oo"]),
        "v": lambda x: np.diagonal(x["ft_vv"]),
        "w": lambda x: - np.diagonal(x["f_ww"])
        })
