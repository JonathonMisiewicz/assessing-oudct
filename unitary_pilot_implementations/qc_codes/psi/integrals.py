import psi4.core
from . import misc
import numpy as np

# Public
def core_hamiltonian(basis, geom):
    """
    Return the integrals for the core Hamiltonian.
    (Electron kinetic energy and nuclear attraction.)

    :param basis: basis set name
    :type basis: str
    :param geom: list of atomic descriptors. Each consists of a tuple of
    the atom's symbol, followed by the atom's coords (as a tuple)
    :type labels: list[(str, (float, float, float)), ...]

    :return: a square matrix
    :rtype: np.ndarray
    """
    return kinetic(basis, geom) + electron_nuclear(basis, geom)

def kinetic(basis, geom):
    """
    Return the kinetic energy integrals.

    :param basis: basis set name
    :type basis: str
    :param geom: list of atomic descriptors. Each consists of a tuple of
    the atom's symbol, followed by the atom's coords (as a tuple)
    :type labels: list[(str, (float, float, float)), ...]

    :return: a square matrix
    :rtype: np.ndarray
    """
    mints = _psi4_mints_object(basis, geom)
    return np.array(mints.ao_kinetic())

def core_hamiltonian_grad(basis, geom, atom):
    """
    Return the gradient of the core Hamiltonian integrals for the specified atom.

    :param basis: basis set name
    :type basis: str
    :param geom: list of atomic descriptors. Each consists of a tuple of
    the atom's symbol, followed by the atom's coords (as a tuple)
    :type labels: list[(str, (float, float, float)), ...]
    :param atom: number of the atom to displace, 0-indexed
    :type atom: int

    :return: a list of (3 * natoms) square matrices
    :rtype: np.ndarray
    """
    mints = _psi4_mints_object(basis, geom)
    potential = mints.ao_oei_deriv1("POTENTIAL", atom)
    kinetic = mints.ao_oei_deriv1("KINETIC", atom)
    return [np.array(i) + np.array(j) for i, j in zip(potential, kinetic)]

def dipole(basis, geom):
    """
    Return the dipole integrals.

    :param basis: basis set name
    :type basis: str
    :param geom: list of atomic descriptors. Each consists of a tuple of
    the atom's symbol, followed by the atoms' coords
    :type labels: list[(str, (float, float, float)), ...] 
    """
    mints = _psi4_mints_object(basis, geom)
    return np.transpose(np.array(tuple(map(np.array, mints.ao_dipole()))))

def electron_nuclear(basis, geom):
    """
    Return the electron-nuclear attraction integrals.

    :param basis: basis set name
    :type basis: str
    :param geom: list of atomic descriptors. Each consists of a tuple of
    the atom's symbol, followed by the atom's coords (as a tuple)
    :type labels: list[(str, (float, float, float)), ...]

    :return: a square matrix
    :rtype: np.ndarray
    """
    mints = _psi4_mints_object(basis, geom)
    return np.transpose(np.array(mints.ao_potential()))

def overlap(basis, geom):
    mints = _psi4_mints_object(basis, geom)
    return mints.ao_overlap()

def repulsion(basis, geom):
    """
    Return the electron repulsion integrals in physics notation.

    :param basis: basis set name
    :type basis: str
    :param geom: list of atomic descriptors. Each consists of a tuple of
    the atom's symbol, followed by the atom's coords (as a tuple)
    :type labels: list[(str, (float, float, float)), ...]

    :return: a square matrix
    :rtype: np.ndarray
    """

    mints = _psi4_mints_object(basis, geom)
    return np.transpose(np.array(mints.ao_eri()), (1, 3, 0, 2))

def repulsion_grad(basis, geom, atom):
    """
    Return the gradient of the electron repulsion integrals for the specified atom.

    :param basis: basis set name
    :type basis: str
    :param geom: list of atomic descriptors. Each consists of a tuple of
    the atom's symbol, followed by the atom's coords (as a tuple)
    :type labels: list[(str, (float, float, float)), ...]
    :param atom: number of the atom to displace, 0-indexed
    :type atom: int

    :return: the gradient of the eri integrals as a list for x, y, z displacements
    :rtype: [np.ndarray, np.ndarray, np.ndarray]
    """
    mints = _psi4_mints_object(basis, geom)
    eri = mints.ao_tei_deriv1(atom)
    return [np.transpose(np.array(i), (1, 3, 0, 2)) for i in eri]

def overlap_grad(basis, geom, atom):
    """
    Return the gradient of the overlap integrals for the specified atom.

    :param basis: basis set name
    :type basis: str
    :param geom: list of atomic descriptors. Each consists of a tuple of
    the atom's symbol, followed by the atom's coords (as a tuple)
    :type labels: list[(str, (float, float, float)), ...]
    :param atom: number of the atom to displace, 0-indexed
    :type atom: int

    :return: the gradient of the overlap integrals as a list for x, y, z displacements
    :rtype: [np.ndarray, np.ndarray, np.ndarray]
    """
    mints = _psi4_mints_object(basis, geom)
    overlap = mints.ao_oei_deriv1("OVERLAP", atom)
    return [np.transpose(np.array(i)) for i in overlap]

def nuclear_potential_deriv(geom):
    coord_str = misc._coordinate_string(geom)
    mol = psi4.core.Molecule.from_string(coord_str)
    return np.array(mol.nuclear_repulsion_energy_deriv1())

# Hmm, this may be worth moving my mints maker to misc.
def nbf(basis, geom):
    coords = np.random.rand(3, len(geom))
    mints = _psi4_mints_object(basis, geom)
    return int(mints.nbf())

# Private
def _psi4_mints_object(basis, geom, return_basis=False):
    """
    Build a Psi4 MintsHelper object
    :param basis: basis set name
    :type basis: str
    :param labels: atomic symbols labeling the nuclei
    :type labels: tuple
    :param coords: nuclear coordinates in Bohr
    :type coords: numpy.ndarray
    :rtype: psi4.core.MintsHelper
    """
    coord_str = misc._coordinate_string(geom)
    mol = psi4.core.Molecule.from_string(coord_str)
    mol.reset_point_group("c1")
    mol.update_geometry()

    basis_obj = psi4.core.BasisSet.build(mol, 'BASIS', basis, quiet=True)
    mints_obj = psi4.core.MintsHelper(basis_obj)

    if return_basis:
        return mints_obj, basis_obj, mol
    else:
        return mints_obj
