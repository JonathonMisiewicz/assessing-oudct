from functools import partial
from unitary_pilot_implementations.math_util import algorithms
from . import proc

conventional = partial(algorithms.conventional,
                       compute_energy = proc.SD1_energy,
                       compute_amplitude_residual = proc.SD1_residual,
                       compute_step = proc.compute_step,
                       initialize_intermediates = proc.initialize_intermediates
                      )
