import itertools
import numpy as np
from . import elements

def energy(geom):
    """
    Compute the nuclear repulsion energy. Independent of any external programs.

    Input
    -----
    geom: Iterable of (str, Iterable((float, float, float)))
        Each entry is the atomic label, then a tuple of the Cartesian coordinates.
        Must be in bohr.

    Output
    ------
    float
    """
    energy = 0
    for combination in itertools.combinations(geom, 2):
        labels, coords = zip(*combination)
        charges = tuple(map(elements.charge, labels))
        del_r = np.linalg.norm(np.subtract(*coords))
        energy += charges[0] * charges[1] / del_r
    return energy
