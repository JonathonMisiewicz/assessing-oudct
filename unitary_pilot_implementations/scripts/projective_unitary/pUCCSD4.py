from functools import partial
from unitary_pilot_implementations.math_util import algorithms
from . import proc

def energy(i):
    return proc.SD1_energy(i) + proc.SD2_energy(i) + proc.SD3_energy(i) + proc.SD4_energy(i)

def residual(i):
    proc.SD1_residual(i)
    proc.SD2_residual(i)
    proc.SD3_residual(i)
    proc.SD4_residual(i)

conventional = partial(algorithms.conventional,
                       compute_energy = energy,
                       compute_amplitude_residual = residual,
                       compute_step = proc.compute_step,
                       initialize_intermediates = proc.initialize_intermediates
                      )
