# Cumulant derived from UCCD2

from functools import partial
from unitary_pilot_implementations.scripts.oo_variational_unitary import unitary_common
from unitary_pilot_implementations.math_util import algorithms, known_formulas
from . import proc

def residual(intermed):
    intermed["r2"] = 0
    unitary_common.D2_cumulant_residual(intermed)
    unitary_common.D2_opdm_residual_dct(intermed)
    unitary_common.D3_cumulant_residual(intermed)
    unitary_common.D4_opdm_residual_dct(intermed)

def compute_cumulant(i):
    unitary_common.D2_cumulant(i)
    unitary_common.D3_cumulant(i)

def compute_d(i):
    unitary_common.D2_d(i)
    unitary_common.D4_d(i)

def intermed(i):
    return proc.dct_simultaneous(compute_cumulant, compute_d, proc.block_1rdm, i)

simultaneous = partial(algorithms.simultaneous,
                       compute_intermediates = intermed,
                       compute_energy = unitary_common.even_rdm_energy,
                       compute_orbital_residual = known_formulas.even_block_orbital_gradient,
                       compute_amplitude_residual = residual,
                       compute_step = proc.simultaneous_step,
                       initialize_intermediates = unitary_common.initialize_intermediates
                      ) 
