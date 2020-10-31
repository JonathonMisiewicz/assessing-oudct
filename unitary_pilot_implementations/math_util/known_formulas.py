import numpy as np
from scipy import linalg as spla
from ..multilinear.tensor import einsum
from .. import multilinear as mla

def return_generalized_fock(h, g, RDM1, RDM2):
    """ Return the generalized Fock matrix used in the orbital gradient.
        The orbital gradient is just the antisymmetry of this matrix. """
    return (np.einsum("q a,i q->i a", h, RDM1)
            + 0.5 * np.einsum("rs aq,iq rs->i a", g, RDM2))


# This function assumes hermiticity and vanishing of the ov blocks
# This function assumes: rdm_cc is delta
def even_block_orbital_gradient(i):
    """
    This function computes the orbital gradient, within the following assumptions:
    * Hermitian RDMs
    * OV block of the OPDM is zero
    * Any RDM block involving a W (frozen virtual) index is 0
    * The only nonzero OPDM block involving a core orbital is a kronecker delta
    * The only core TPDM blocks have the same number of upper and lower indices
    ...That said, frozen core and virtual blocks aren't even in this paper.

    The orbial gradient is stored in the intermediate dictionary.

    Input
    -----
    i: dict from string to np.ndarray
        All needed intermediates. RDMs and integrals.
    """
    def make_unit(mat):
        return np.eye(mat.shape[0])
    ### CO - Should be good
    # OV term missing
    grad = 2 * einsum("cI, Ii -> ci", i["h_co"], make_unit(i["rdm_oo"]) - i["rdm_oo"])

    # COCV term missing
    grad += 2 * einsum("CJiI, CJcI -> ci", i["g_cooo"], i["rdm_coco"])
    grad += 2 * einsum("CBiA, CBcA -> ci", i["g_cvov"], i["rdm_cvcv"])
    grad -= einsum("CEDi, CEcD -> ci", i["g_ccco"], i["rdm_cccc"])

    grad -= einsum("cKIJ, iKIJ -> ci", i["g_cooo"], i["rdm_oooo"])
    grad -= einsum("cIAB, iIAB -> ci", i["g_covv"], i["rdm_oovv"])
    grad -= 2 * einsum("cAIB, iAIB -> ci", i["g_cvov"], i["rdm_ovov"])
    grad -= 2 * einsum("CcDI, CiDI -> ci", i["g_ccco"], i["rdm_coco"])
    if "rdm_ovvv" in i:
        grad -= 2 * einsum("cIJA, iIJA -> ci", i["g_coov"], i["rdm_ooov"])
        grad -= einsum("cAIJ, IJiA -> ci", i["g_cvoo"], i["rdm_ooov"])
        grad -= einsum("cABC, iABC -> ci", i["g_cvvv"], i["rdm_ovvv"])

    i["r1co"] = grad
    
    ### CV - I'm fairly sure something is wrong here.
    # OV term missing
    grad = 2 * einsum("cA, Aa -> ca", i["h_cv"], make_unit(i["rdm_vv"]) - i["rdm_vv"])

    grad += 2 * einsum("CBaA, CBcA -> ca", i["g_cvvv"], i["rdm_cvcv"])
    grad -= 2 * einsum("CJIa, CJcI -> ca", i["g_coov"], i["rdm_coco"])
    grad += einsum("DECa, DECc -> ca", i["g_cccv"], i["rdm_cccc"])

    grad -= einsum("cAIJ, IJaA -> ca", i["g_cvoo"], i["rdm_oovv"])
    grad -= einsum("cABC, aABC -> ca", i["g_cvvv"], i["rdm_vvvv"])
    grad += 2 * einsum("cIJA, IaJA -> ca", i["g_coov"], i["rdm_ovov"])
    grad -= 2 * einsum("CcDA, CaDA -> ca", i["g_cccv"], i["rdm_cvcv"])
    if "rdm_ooov" in i:
        grad -= 2 * einsum("cBIA, IAaB -> ca", i["g_cvov"], i["rdm_ovvv"])
        grad += einsum("cIAB, IaAB -> ca", i["g_covv"], i["rdm_ovvv"])
        grad += einsum("cKIJ, IJKa -> ca", i["g_cooo"], i["rdm_ooov"])
    
    i["r1cv"] = grad

    ### CW
    grad = 2 * i["h_cw"]
    # COCV term missing
    grad -= 2 * einsum("CJIa, cICJ -> ca", i["g_coow"], i["rdm_coco"])
    grad -= 2 * einsum("CBAw, cACB -> cw", i["g_cvvw"], i["rdm_cvcv"])
    grad -= einsum("DECa, cCDE -> ca", i["g_cccw"], i["rdm_cccc"])
    i["r1cw"] = grad

    ### OV
    # 1RDM part
    # OV term missing
    grad = 2 * einsum("Ia, Ii -> ia", i["h_ov"], i["rdm_oo"])
    grad -= 2 * einsum("iA, Aa -> ia", i["h_ov"], i["rdm_vv"])
    # 2RDM
    grad += 2 * einsum("DI Ca, Ci DI -> ia", i["g_cocv"], i["rdm_coco"])
    grad += -1 * einsum("JK Ia, iI JK -> ia", i["g_ooov"], i["rdm_oooo"])
    grad += -1 * einsum("Ia AB, iI AB -> ia", i["g_ovvv"], i["rdm_oovv"])
    grad +=  2 * einsum("IB aA, iA IB -> ia", i["g_ovvv"], i["rdm_ovov"])

    grad += -2 * einsum("Ci DA, Ca DA -> ia", i["g_cocv"], i["rdm_cvcv"])
    grad += -1 * einsum("iA BC, aA BC -> ia", i["g_ovvv"], i["rdm_vvvv"])
    grad += -1 * einsum("IJ iA, IJ aA -> ia", i["g_ooov"], i["rdm_oovv"])
    grad += -2 * einsum("Ii JA, Ia JA -> ia", i["g_ooov"], i["rdm_ovov"])

    if "rdm_ooov" in i:
        grad += 2 * einsum("IaJA, IiJA-> ia", i["g_ovov"], i["rdm_ooov"])
        grad += einsum("IJaA, IJiA -> ia", i["g_oovv"], i["rdm_ooov"])
        grad += einsum("aABC, iABC-> ia", i["g_vvvv"], i["rdm_ovvv"])

        grad -= einsum("IiAB, IaAB -> ia", i["g_oovv"], i["rdm_ovvv"])
        grad -= 2 * einsum("IAiB, IAaB -> ia", i["g_ovov"], i["rdm_ovvv"])
        grad -= einsum("IJKi, IJKa -> ia", i["g_oooo"], i["rdm_ooov"])

    i["r1ov"] = grad

    ### OW
    # OV term missing
    grad = 2 * einsum("Ia, Ii -> ia", i["h_ow"], i["rdm_oo"])
    grad += 2 * einsum("DI Ca, Ci DI -> ia", i["g_cocw"], i["rdm_coco"])
    grad += -1 * einsum("JK Ia, iI JK -> ia", i["g_ooow"], i["rdm_oooo"])
    grad += -1 * einsum("Ia AB, iI AB -> ia", i["g_owvv"], i["rdm_oovv"])
    grad += -2 * einsum("IB Aa, iA IB -> ia", i["g_ovvw"], i["rdm_ovov"])
    if "rdm_ooov" in i:
        grad -= einsum("IJAw, IJiA -> iw", i["g_oovw"], i["rdm_ooov"])
        grad += 2 * einsum("IAJw, JiIA -> iw", i["g_ovow"], i["rdm_ooov"])
        grad -= einsum("BCAw, iABC -> iw", i["g_vvvw"], i["rdm_ovvv"])
    i["r1ow"] = grad

    ### VW
    # VO term missing
    grad = 2 * einsum("Ia, Ii -> ia", i["h_vw"], i["rdm_vv"])
    grad += 2 * einsum("DA Cw, Ca DA -> aw", i["g_cvcw"], i["rdm_cvcv"])
    grad += einsum("BC Aw, BC Aa -> aw", i["g_vvvw"], i["rdm_vvvv"])
    grad += einsum("IJ Aw, IJ Aa -> aw", i["g_oovw"], i["rdm_oovv"])
    grad += einsum("IA Jw, IA Ja -> aw", i["g_ovow"], i["rdm_ovov"])
    if "rdm_ooov" in i:
        grad += einsum("IJKw, IJKa -> aw", i["g_ooow"], i["rdm_ooov"])
        grad += 2 * einsum("IABw, IABa -> aw", i["g_ovvw"], i["rdm_ovvv"])
        grad += einsum("IwAB, IaAB -> aw", i["g_owvv"], i["rdm_ovvv"])
    i["r1vw"] = grad


def gradient_contribution(gen_fock, hx, gx, sx, nx, RDM1, RDM2):
    """ Compute a single matrix element of the gradient using the standard formulas. """
    one_term = einsum("q p,p q", hx, RDM1)
    two_term = 1 / 4 * einsum("rs pq,pq rs", gx, RDM2)
    lag_term = - einsum("q p,p q", gen_fock, sx)
    return 1 * one_term + 1 * two_term + 1 * lag_term + 1 * nx


def mo_oei_hermitian_block(orbitals, intermed):
    """ Construct the OPDM in the AO basis. All blocks assumed hermitian.
        Blocks not treated are assumed zero.

    Input
    -----
    orbitals: dict
        Maps from letter to a tuple of alpha, beta np.ndarray with MO coefficients
    intermed: dict
        Maps from string to np.ndarray. Here, we use it to get 1RDM blocks.

    Output
    ------
    np.ndarray
    """
    cc = orbitals["c"]
    ci = orbitals["o"]
    cv = orbitals["v"]
    cw = orbitals["w"]
    ncor = cc[0].shape[1] + cc[1].shape[1]
    opdm = mla.mso_to_aso(np.eye(ncor), (cc, cc))
    opdm += mla.mso_to_aso(intermed["rdm_oo"], (ci, ci))
    opdm += mla.mso_to_aso(intermed["rdm_vv"], (cv, cv))
    return opdm 

def mo_tei_hermitian_even(orbitals, intermed):
    """ Construct the TPDM in the AO basis. All blocks assumed hermitian, antisymmetric.
        Blocks not treated are assumed zero.
        Frozen spaces will need to be treated in future.

    Input
    -----
    orbitals: dict
        Maps from letter to a tuple of alpha, beta np.ndarray with MO coefficients
    intermed: dict
        Maps from string to np.ndarray. Here, we use it to get 1RDM blocks.

    Output
    ------
    np.ndarray
    """
    cc = orbitals["c"]
    ci = orbitals["o"]
    cv = orbitals["v"]
    cw = orbitals["w"]
    # oovv; vvoo
    tei = 2 * mla.mso_to_aso(intermed["rdm_oovv"], (ci, ci, cv, cv))
    # ovov; voov; ovvo; vovo
    tei += 4 * mla.mso_to_aso(intermed["rdm_ovov"], (ci, cv, ci, cv))
    if "rdm_cvcv" in intermed:
        tei += 4 * mla.mso_to_aso(intermed["rdm_cvcv"], (cc, cv, cc, cv))
        tei += mla.mso_to_aso(intermed["rdm_cccc"], (cc, cc, cc, cc))
        tei += 4 * mla.mso_to_aso(intermed["rdm_coco"], (cc, ci, cc, ci))
    tei += mla.mso_to_aso(intermed["rdm_oooo"], (ci, ci, ci, ci))
    if "rdm_vvvv" in intermed: # False for OMP2
        tei += mla.mso_to_aso(intermed["rdm_vvvv"], (cv, cv, cv, cv))
    if "rdm_ovvv" in intermed: # False for OMP2
        tei += 4 * mla.mso_to_aso(intermed["rdm_ovvv"], (ci, cv, cv, cv))
        tei += 4 * mla.mso_to_aso(intermed["rdm_ooov"], (ci, ci, ci, cv))
    return tei

def mo_gei_hermitian_even(orbitals, intermed):
    """ Construct the generalized Fock matrix in the AO basis. Also known as the
        energy-weighted density matrix.
        All blocks assumed hermitian, antisymmetric.
        Blocks not treated are assumed zero.
        Frozen spaces will need to be treated in future.

    Input
    -----
    orbitals: dict
        Maps from letter to a tuple of alpha, beta np.ndarray with MO coefficients
    intermed: dict
        Maps from string to np.ndarray. Here, we use it to get 1RDM blocks.

    Output
    ------
    np.ndarray
    """
    cc = orbitals["c"]
    ci = orbitals["o"]
    cv = orbitals["v"]
    cw = orbitals["w"]
    nina = ci[0].shape[1] + ci[1].shape[1]
    nvir = cv[0].shape[1] + cv[1].shape[1]
    ncor = cc[0].shape[1] + cc[1].shape[1]
    nfrz = cw[0].shape[1] + cw[1].shape[1]
    assert nfrz + ncor == 0
    gei = np.zeros((nina + nvir + ncor + nfrz, nina + nvir + ncor + nfrz))
    # II block:
    term = einsum("i J, I i -> IJ", intermed["h_oo"], intermed["rdm_oo"])
    term += 0.5 * einsum("jk Ji, Ii jk -> IJ", intermed["g_oooo"], intermed["rdm_oooo"])
    term += 0.5 * einsum("Ji ab, Ii ab -> IJ", intermed["g_oovv"], intermed["rdm_oovv"])
    term += einsum("ib Ja, Ia ib -> IJ", intermed["g_ovov"], intermed["rdm_ovov"])
    if "rdm_ooov" in intermed:
        term += 0.5 * einsum("Ja bc, Ia bc -> IJ", intermed["g_ovvv"], intermed["rdm_ovvv"])
        term += einsum("Ji ja, Ii ja -> IJ", intermed["g_ooov"], intermed["rdm_ooov"])
        term += 0.5 * einsum("ij Ja, ij Ia -> IJ", intermed["g_ooov"], intermed["rdm_ooov"])
    gei = mla.mso_to_aso(term, (ci, ci))
    # IV block:
    term = einsum("i A, I i -> IA", intermed["h_ov"], intermed["rdm_oo"])
    term += -0.5 * einsum("jk iA, Ii jk -> IA", intermed["g_ooov"], intermed["rdm_oooo"])
    term += -0.5 * einsum("iA ab, Ii ab -> IA", intermed["g_ovvv"], intermed["rdm_oovv"])
    term += -einsum("ib aA, Ia ib -> IA", intermed["g_ovvv"], intermed["rdm_ovov"])
    if "rdm_ooov" in intermed:
        term += 0.5 * einsum("Aa bc, Ia bc -> IA", intermed["g_vvvv"], intermed["rdm_ovvv"])
        term += 0.5 * einsum("ij Aa, ij Ia -> IA", intermed["g_oovv"], intermed["rdm_ooov"])
        term += einsum("iA ja, iI ja -> IA", intermed["g_ovov"], intermed["rdm_ooov"])
    gei += mla.mso_to_aso(term, (ci, cv))
    # VI block:
    # N.B.: SHOULD be the same as the IV block, for optimized orbitals.
    term = einsum("I a, A a -> AI", intermed["h_ov"], intermed["rdm_vv"])
    term += 0.5 * einsum("Ia bc, Aa bc -> AI", intermed["g_ovvv"], intermed["rdm_vvvv"])
    term += 0.5 * einsum("ij Ia, ij Aa -> AI", intermed["g_ooov"], intermed["rdm_oovv"])
    term += einsum("iI ja, iA ja -> AI", intermed["g_ooov"], intermed["rdm_ovov"])
    if "rdm_ooov" in intermed:
        term += 0.5 * einsum("ij kI, ij kA -> AI", intermed["g_oooo"], intermed["rdm_ooov"])
        term += 0.5 * einsum("iI ab, iA ab -> AI", intermed["g_oovv"], intermed["rdm_ovvv"])
        term += einsum("ia Ib, ia Ab -> AI", intermed["g_ovov"], intermed["rdm_ovvv"])
    gei += mla.mso_to_aso(term, (cv, ci))
    # VV block:
    term = einsum("a B, A a -> AB", intermed["h_vv"], intermed["rdm_vv"])
    term += 0.5 * einsum("bc Ba, Aa bc -> AB", intermed["g_vvvv"], intermed["rdm_vvvv"])
    term += 0.5 * einsum("ij Ba, ij Aa -> AB", intermed["g_oovv"], intermed["rdm_oovv"])
    term += einsum("iB ja, iA ja -> AB", intermed["g_ovov"], intermed["rdm_ovov"])
    if "rdm_ovvv" in intermed:
        term += 0.5 * einsum("ij kB, ij kA -> AB", intermed["g_ooov"], intermed["rdm_ooov"])
        term += 0.5 * einsum("iB ab, iA ab -> AB", intermed["g_ovvv"], intermed["rdm_ovvv"])
        term += einsum("ia Bb, ia Ab -> AB", intermed["g_ovvv"], intermed["rdm_ovvv"])
    gei += mla.mso_to_aso(term, (cv, cv))

    return gei 

