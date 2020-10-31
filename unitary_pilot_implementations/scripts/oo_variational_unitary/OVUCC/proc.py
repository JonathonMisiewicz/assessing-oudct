from unitary_pilot_implementations import multilinear as mla
from unitary_pilot_implementations.multilinear.tensor import einsum
import numpy as np
from .. import unitary_common
from functools import partial

def simultaneous(connected, opdm, opdm_product, intermed):
    zero_rdms(intermed)
    connected(intermed)
    opdm(intermed)
    opdm_product(intermed)
    assemble_fock_block_diagonal(intermed)
    rdm_construct(intermed)

def zero_rdms(inter):
    inter["rdm_cccc"] = 0
    inter["rdm_coco"] = 0
    inter["rdm_cvcv"] = 0
    inter["rdm_oovv"] = 0
    inter["rdm_oooo"] = 0
    inter["rdm_vvvv"] = 0
    inter["rdm_ovov"] = 0
    inter["rdm_oo"] = 0
    inter["rdm_vv"] = 0
    inter["rdm_cc"] = 0

def rdm_construct(inter):
    kappa = np.eye(inter["rdm_oo"].shape[0])
    inter["rdm_oovv"] += inter["c_oovv"]
    inter["rdm_oooo"] += inter["c_oooo"]
    inter["rdm_oooo"] += mla.antisymmetrize_axes_plus(einsum("pr, qs -> pqrs", kappa, kappa), ((2,), (3,)))
    inter["rdm_oooo"] += mla.antisymmetrize_axes_plus(einsum("pr, qs -> pqrs", kappa, inter["rdm_oo"]), ((0,), (1,)), ((2,), (3,)))
    inter["rdm_vvvv"] += inter["c_vvvv"]
    inter["rdm_ovov"] += inter["c_ovov"] + einsum("p r, q s -> pqrs", kappa, inter["rdm_vv"])
    inter["rdm_oo"] += kappa
    rdm_cc = np.eye(inter["h_cc"].shape[0])
    inter["rdm_cc"] = rdm_cc
    inter["rdm_cccc"] = mla.antisymmetrize_axes_plus(einsum("IK, JL -> IJKL", rdm_cc, rdm_cc), ((0,), (1,)))
    inter["rdm_coco"] = einsum("IK, JL -> IJKL", rdm_cc, inter["rdm_oo"])
    inter["rdm_cvcv"] = einsum("IK, JL -> IJKL", rdm_cc, inter["rdm_vv"])

def assemble_fock_block_diagonal(inter):
    inter["f_cc"] = inter["h_cc"] + einsum("Ii Ji -> IJ", inter["g_coco"]) + einsum("Ii Ji -> IJ", inter["g_cccc"])
    inter["f_oo"] = inter["h_oo"] + einsum("Ii Ji -> IJ", inter["g_oooo"]) + einsum("iI iJ -> IJ", inter["g_coco"])
    inter["f_vv"] = inter["h_vv"] + einsum("iA iB -> AB", inter["g_ovov"]) + einsum("iA iB -> AB", inter["g_cvcv"])
    inter["f_ww"] = inter["h_ww"] + einsum("iA iB -> AB", inter["g_owow"]) + einsum("iA iB -> AB", inter["g_cwcw"])

simultaneous_step = partial(unitary_common.simultaneous_step, diagonal_dict = {
        "c": lambda x: np.diagonal(x["f_cc"]),
        "o": lambda x: np.diagonal(x["f_oo"]),
        "v": lambda x: - np.diagonal(x["f_vv"]),
        "w": lambda x: - np.diagonal(x["f_ww"])
})
