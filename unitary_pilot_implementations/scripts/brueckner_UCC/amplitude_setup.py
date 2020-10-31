import itertools
from .determinant import Determinant
import numpy as np

def amplitudes(space):
    """ Given orbital space information, construct a corresponding list of amplitudes and determinants

    Input
    -----
    space: dict
        Maps from spin, then occupied/virtual status, and start/length, to the desired quantity
        for the desired space.

    Output
    ------
    amplitude_dict: dict from orbital specifier to its amplitudes
    ((alpha-indices-excited-from, alpha-indices-excited-to), (beta-indices-excited-from, beta-indices-ecited-to))
    det_list: list of Determinants
    ab_idx_map: Maps from an alpha string, beta string to an index.
    """
    alpha_max_excit = min(space["A"]["O"]["length"], space["A"]["V"]["length"])
    beta_max_excit = min(space["B"]["O"]["length"], space["B"]["V"]["length"])
    max_excitation_level = alpha_max_excit + beta_max_excit
    det_list = []
    amplitude_dict = dict()
    ab_idx_map = dict()
    for excitation_level in range(max_excitation_level + 1): 
        for alpha_excit in range(excitation_level + 1):
            # TODO: Define alpha_max_excit
            if alpha_excit > alpha_max_excit: break # We can't add more alpha excitations.
            beta_excit = excitation_level - alpha_excit
            if beta_excit > beta_max_excit: continue # Let's try to shift a beta excitation to an alpha...
            active_occ = ""
            alpha_occ = excitation_generator(space["A"]["O"], alpha_excit, True)
            alpha_vir = excitation_generator(space["A"]["V"], alpha_excit, False)
            beta_occ = excitation_generator(space["B"]["O"], beta_excit, True)
            beta_vir = excitation_generator(space["B"]["V"], beta_excit, False)
            for (A, a), (B, b), (C, c), (D, d) in itertools.product(alpha_occ, alpha_vir, beta_occ, beta_vir):
                alpha_str = "1" * space["A"]["F"]["length"] + A + B + "0" * space["A"]["W"]["length"]
                beta_str = "1" * space["A"]["F"]["length"] + C + D + "0" * space["A"]["W"]["length"]
                det = Determinant(alpha_str, beta_str)
                ab_idx_map[(det.alpha, det.beta)] = len(det_list)
                det_list.append(det)
                if excitation_level: amplitude_dict[((a, c), (b, d))] = 0
    return amplitude_dict, det_list, ab_idx_map

def excitation_generator(space, excite_level, annihilated):
    """
    Input
    -----
    space: {"length": natural, "start": natural}
        length is the number of orbitals in the space
        start is the number of orbtials in prior spaces
    excite_level: int, number of orbitals you will excite to/form
    annihilated: bool, does an excitation ANNIHILATE an orbital here, or create one?

    Output
    ------
    str, tuple(int)
        The string specifies the occupied bits.
        The tuple specifies the excited orbitals.
    """
    for excitation in itertools.combinations(range(space["length"]), excite_level):
        string = ""
        for i in range(space["length"]):
            string += "0" if ((i in excitation) == annihilated) else "1"
        # I'm going to return both the string and the excitation for indexing purposes
        yield (string, tuple(space["start"] + i for i in excitation))

def construct_lie_algebra_elt(amplitudes, det_list):
    """ Build T - T^ in the determinant basis.

    Input
    -----
    amplitudes: dict
        Maps an amplitude-tuple to a float. An amplitude tuple stores ((ao, bo), (av, bv)).
        a vs b specifies alpha vs beta.
    det_list: list of Determinant

    Output
    ------
    np.ndarray"""
    nalpha = sum(det_list[0].alpha_list())
    nbeta = sum(det_list[0].beta_list())
    is_alpha_occ = lambda x: x < nalpha
    is_alpha_vir = lambda x: not is_alpha_occ(x)
    is_beta_occ = lambda x: x < nbeta
    is_beta_vir = lambda x: not is_beta_occ(x)
    lie = np.zeros((len(det_list), len(det_list)))
    for (det_idx1, det1), (det_idx2, det2) in itertools.combinations(enumerate(det_list), 2):
        # Identify the orbitals occupied in only one determinant.
        det1_particles = det1.set_subtraction_list(det2)
        det2_particles = det2.set_subtraction_list(det1)
        # Extract the amplitude.
        if all(is_alpha_occ(p) for p in det1_particles[0]) and all(is_beta_occ(p) for p in det1_particles[1]) and (
                all(is_alpha_vir(p) for p in det2_particles[0])) and all(is_beta_vir(p) for p in det2_particles[1]):
            # Amplitude * Deexcitation phase * Permutational phase
            value = amplitudes[(det1_particles, det2_particles)] * -1 * det1.phase(det2, False)
            # Extract the amplitude!
        elif all(is_alpha_occ(p) for p in det2_particles[0]) and all(is_beta_occ(p) for p in det2_particles[1]) and (
                all(is_alpha_vir(p) for p in det1_particles[0])) and all(is_beta_vir(p) for p in det1_particles[1]):
            # Amplitude * Excitation phase * Permutational phase
            value = amplitudes[(det2_particles, det1_particles)] * 1 * det1.phase(det2, False)
            # Extract the amplitude.
        else:
            continue
        lie[det_idx1][det_idx2] = value
        lie[det_idx2][det_idx1] = -value
    return lie

def extract_spincase(amplitudes, spincase, space):
    """ Extract the given spincase of T2 amplitudes for amplitude analysis.

    Input
    -----
    amplitudes: dict
        Maps an amplitude-tuple to a float. An amplitude tuple stores ((ao, bo), (av, bv)).
        a vs b specifies alpha vs beta.
        o vs v specifies an occupied orbital excited FROM vs virtual orbtial excited TO.
    spincase: {"AA", "AB", BB"}
        Which spin are we interested in?
    space: dict
        Maps from spin, then occupied/virtual status, and start/length, to the desired quantity
        for the desired space.
        

    Output
    ------
    np.ndarray
    """
    occ_alpha = space["A"]["O"]["length"]
    vir_alpha = space["A"]["V"]["length"]
    occ_beta = space["B"]["O"]["length"]
    vir_beta = space["B"]["V"]["length"]
    if spincase == "AA":
        array = np.zeros((occ_alpha, occ_alpha, vir_alpha, vir_beta))
        for index in np.ndindex(array.shape):
            ao = tuple(sorted([x + space["A"]["O"]["start"] for x in index[:2]]))
            av = tuple(sorted([x + space["A"]["V"]["start"] for x in index[2:]]))
            bo = tuple()
            bv = tuple()
            amp_index = ((ao, bo), (av, bv))
            array[index] = amplitudes.get(amp_index, 0)
    elif spincase == "AB":
        array = np.zeros((occ_alpha, occ_beta, vir_alpha, vir_beta))
        for index in np.ndindex(array.shape):
            ao = (index[0] + space["A"]["O"]["start"],)
            av = (index[2] + space["A"]["V"]["start"],)
            bo = (index[1] + space["B"]["O"]["start"],)
            bv = (index[3] + space["B"]["V"]["start"],)
            amp_index = ((ao, bo), (av, bv))
            array[index] = amplitudes.get(amp_index, 0)
    elif spincase == "BB":
        array = np.zeros((occ_beta, occ_beta, vir_beta, vir_beta))
        for index in np.ndindex(array.shape):
            bo = tuple(sorted([x + space["B"]["O"]["start"] for x in index[:2]]))
            bv = tuple(sorted([x + space["B"]["V"]["start"] for x in index[2:]]))
            ao = tuple()
            av = tuple()
            amp_index = ((ao, bo), (av, bv))
            array[index] = amplitudes.get(amp_index, 0)
    else:
        raise Exception
    return array
