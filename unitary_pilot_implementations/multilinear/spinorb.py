import numpy as np
import scipy.linalg as spla
from . import tensor as mla
from collections import OrderedDict

def orb_rot(intermed, start_orbitals):
    """
    Perform the orbital rotation exp(X) specified by the t1 orbital rotation parameters in intermed.

    Input
    -----
    intermed: dict
        Map from strings to orbital amplitudes as arrays.
    start_orbitals: OrderedDict
        Map from orbital space labels (c, o, v, w) to a tuple of the alpha and beta orbitals.

    Output
    ------
    OrderedDict
        Map from ordered space labels to a tuple of the new alpha and beta orbitals.
    """
    dim, dim_a = dict(), dict()
    for key, val in start_orbitals.items():
        dim_a[key] = val[0].shape[1]
        dim[key] = val[0].shape[1] + val[1].shape[1]
    # Assemble the blocks into the X matrix which we need to exponentiate
    X = []
    for i, space_i in enumerate(start_orbitals):
        X.append([])
        for j, space_j in enumerate(start_orbitals):
            if i > j and f"t1{space_j}{space_i}" in intermed: # Lower triangle
                block = intermed.get(f"t1{space_j}{space_i}", np.zeros(0)).T
            elif i < j and f"t1{space_i}{space_j}" in intermed: # Upper triangle
                block = intermed.get(f"t1{space_i}{space_j}", np.array(0)) * -1
            else:
                block = np.zeros((dim[space_i], dim[space_j]))
            X[-1].append(block)
    U = spla.expm(np.block(X))
    # Space 1 Alpha, Space 1 Beta, Space 2 Alpha, etc.
    reassembled_C = np.hstack(tuple(i for sub in start_orbitals.values() for i in sub))
    new_C = reassembled_C @ U
    # From our C matrix, construct the new orbitals.
    new_orbitals = OrderedDict()
    for key in start_orbitals:
        alpha_split = val[0]
        # Split off the orbitals of this subspace from the others.
        alpha, beta, new_C = np.hsplit(new_C, [dim_a[key], dim[key]])
        new_orbitals[key] = (alpha, beta)
    return new_orbitals


def antisym_subspace(tensor, integral_transformers):
    """ Given a tensor, return orbital blocks of its antisymmetrized versions.

    Input
    -----
    tensor: np.ndarray
        A tensor in the AO basis. Not antisymmetrized.
    integral_transformers: list of tuple of nd.array
        A list of tuples. Each tuple contains the alpha orbital coefficients, then the beta coefficients.

    Output
    ------
    np.ndarray
    """
    num_pairs = len(integral_transformers) // 2
    tensor_trans = spatial_subspace(tensor, integral_transformers)
    if num_pairs == 1:
        return tensor_trans 
    elif num_pairs == 2:
        integral_transformers = (integral_transformers[0], integral_transformers[1], integral_transformers[3], integral_transformers[2])
        second_tensor = spatial_subspace(tensor, integral_transformers)
        return tensor_trans - second_tensor.swapaxes(2, 3)
    else:
        print("You have an example of an integral that requires you to")
        print("antisymmetrize more than the simplest case.")
        print("Write a proper antisymmetrizing function to replace this.")
        raise Exception


def request_asym(integral, space_dict, strings):
    """ Return the antisymmetrized tensor, in the subspaces specified by strings.

    Input
    -----
    integral: np.ndarray
        A tensor of integrals.
    space_dict: dict
        A map from a letter specifying an orbital space to a tuple of the alpha and beta MO coefficients
    strings: iterable of list of string
       The list of all the orbital blocks to request. Each orbital block is a list, each element of which
       is the letter of an orbital space, in space_dict.

    Output
    ------
    dict from string to np.ndarray
    """
    return {"".join(string) : antisym_subspace(integral, [space_dict[i] for i in string]) for string in strings}


def mso_to_aso(tensor, subspaces):
    """ Given an MO basis tensor, return it transformed back to the AO basis, spatial orbitals.

    Input
    -----
    tensor: np.ndarray
        A real tensor
    subspaces: Iterable of tuple(np.ndarray, np.ndarray)
        Each inner tuple consists of an alpha spatial subspace, and the right a beta spatial subspace.
        AOs are rows and basis vectors (MOs) are columns.
        If there are n subspaces specified, it is assumed that the first n axes of tensor are bras,
        the next n are kets, and the remainder should not be transformed.

    Output
    ------
    np.ndarray"""
    if not tensor.size:
        # The tensor is empty. We need to return a trivial tensor of the correct shape.
        # The first 0 index selects the alpha tuple, and the second gets the row number.
        dims = [2 * subspace[0].shape[0] for subspace in subspaces]
        return np.zeros(dims)

    num_pairs = len(subspaces) // 2
    bras = subspaces[:num_pairs]
    kets = subspaces[num_pairs:]

    for i, (bra, ket) in enumerate(zip(bras, kets)):
        spin_bra = [np.hstack((bra[0], np.zeros(bra[1].shape))), np.hstack((np.zeros(bra[0].shape), bra[1]))]
        spin_ket = [np.hstack((ket[0], np.zeros(ket[1].shape))), np.hstack((np.zeros(ket[0].shape), ket[1]))]
        braspace = np.vstack(spin_bra).T
        ketspace = np.vstack(spin_ket).T

        # We are transforming MSO -> ASO. Otherwise, assume ASO -> MSO

        bra_axis = i
        ket_axis = i + num_pairs
        # First, perform the spatial transformation. Ignore spin.
        tensor = mla.one_index_transform(tensor, bra_axis, braspace)
        tensor = mla.one_index_transform(tensor, ket_axis, ketspace)

    return tensor

def spatial_subspace(tensor, subspaces):
    """ Given an AO basis tensor, return the blocks transformed to the given spinorbital subspace.
    For example, extract the VVVV block of the two-electron integrals.
    Alpha orbitals precede beta. It is assumed that each upper index MUST have the same
    spin as its lower index partner. This is why the transformations occur in pairs.

    Input
    -----
    tensor: np.ndarray
        A real tensor
    subspaces: Iterable of tuple(np.ndarray, np.ndarray)
        Each inner tuple consists of an alpha spatial subspace, and the right a beta spatial subspace.
        AOs are rows and basis vectors (MOs) are columns.
        If there are n subspaces specified, it is assumed that the first n axes of tensor are bras,
        the next n are kets, and the remainder should not be transformed.

    Output
    ------
    np.ndarray
    """

    ndim = len(tensor.shape)

    num_pairs = len(subspaces) // 2
    bras = subspaces[:num_pairs]
    kets = subspaces[num_pairs:]

    for i, (bra, ket) in enumerate(zip(bras, kets)):
        braspace = np.hstack(bra)
        ketspace = np.hstack(ket)

        bra_axis = i
        ket_axis = i + num_pairs
        # First, perform the spatial transformation. Ignore spin.
        tensor = mla.one_index_transform(tensor, bra_axis, braspace)
        tensor = mla.one_index_transform(tensor, ket_axis, ketspace)
        # Now, perform the spin transformation.
        # All we need to do is zero the blocks where the spins disagree.
        num_bra_a = bra[0].shape[1]
        num_ket_a = ket[0].shape[1]
        # Construct the indices for the opposite spin blocks.
        ab_block = [slice(None)] * ndim
        ba_block = ab_block[:]
        ab_block[bra_axis], ab_block[ket_axis] = slice(None, num_bra_a), slice(num_ket_a, None)
        ba_block[bra_axis], ba_block[ket_axis] = slice(num_bra_a, None), slice(None, num_ket_a)
        # Indices constructed. Now, set the block to zero to end the transform!
        tensor[tuple(ab_block)] = tensor[tuple(ba_block)] = 0
    return tensor


def to_spinorb(tensor, electron_indices):
    """Expand the given atomic orbital indices of the tensor to spin orbital indices.

    Input
    -----
    tensor: np.ndarray
        An array of integrals.
    electron_indices: iterable of (int, int)
        The pairs of indices to be transformed.

    Output
    ------
    np.ndarray
    """
    for index_pair in electron_indices:
        tensor = _one_electron_transform(tensor, index_pair)
    return tensor


def _one_electron_transform(tensor, index_pair):
    """Expand the two indices in the pair from AO basis to ASO.

    Input
    -----
    tensor: np.ndarray
        An array of integrals.
    index_pair: (int, int)
        The two indices to be transformed to the AO basis.

    Output
    ------
    np.ndarray
    """

    # Move our two electron indices to the end, for the kron call.
    tensor = np.moveaxis(tensor, index_pair, (-2, -1))
    # Now, convert. All alpha indices precede all beta indices...
    # ...but the spatial ordering is preserved.
    tensor = np.kron(np.eye(2), tensor)
    # Reverse the previous axis transform.
    tensor = np.moveaxis(tensor, (-2, -1), index_pair)
    return tensor


