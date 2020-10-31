import numpy as np
from opt_einsum import contract

def einsum(*args, **kwargs):
    """ Wrapper for einsum. This way, I can change the default arguments easily."""
    return contract(*args, **kwargs)

def contra_transform(tensor, matrix, exclude=set()):
    """
    Perform a change of basis. Spatial vs spin orbitals are NOT changed by this function.

    Input
    -----
    tensor: np.ndarray
        Tensor to transform. All axes had better have the same length.
    matrix: np.ndarray
        Change of basis matrix in active convention. Had better be 2-D. Need not be square.
    exclude: set of int
        Axes to exclude from the transformation.

    Output
    ------
    np.ndarray
    """
    for axis in range(tensor.ndim):
        if axis not in exclude:
            tensor = one_index_transform(tensor, axis, matrix)
    return tensor


def one_index_transform(tensor, axis, matrix):
    """
    Change the basis for a single component in the tensor.
    Can be used to restrict to a subspace of that vector space.

    Input
    -----
    tensor: np.ndarray
        Tensor to transform. All axes had better have the same length.
    axis: int
        Index of the axis to transform.
    matrix: np.ndarray
        Change of basis matrix in active convention. Had better be 2-D. Need not be square.

    Output
    ------
    np.ndarray
    """
    to_move = np.tensordot(tensor, matrix, (axis, 0))
    return np.moveaxis(to_move, -1, axis)

def broadcaster(multiplicity, first_tensor, second_tensor):
    """ Convenience function for full_broadcaster.
    Call full_broadcaster with multiplicity copies of first_tensor, then multiplicity copies of second_tensor

    Input
    ----
    multiplicity: int
        The number of times to use first_tensor and second_tensor
    first_tensor: np.ndarray
    second_tensor: np.ndarray
        1D tensors.

    Output
    ------
    np.ndarray
    """
    axis_tuple = [first_tensor] * multiplicity + [second_tensor] * multiplicity
    return full_broadcaster(axis_tuple)

def full_broadcaster(axes):
    """ Create a tensor where element ijkl is elt. i of axis 1 + elt. j of axis 2, etc.

    Input
    -----
    axes: iterable of 1D np.ndarray

    Output
    ------
    np.ndarray"""
    tensor = np.zeros(tuple(len(axis) for axis in axes))
    for i, axis in enumerate(axes):
        dim_tuple = tuple((-1 if i == j else 1) for j in range(len(axes)))
        tensor += axis.reshape(dim_tuple) # Reshape axis for easy broadcasting into final target
    return tensor

def transform_all_indices(tensor, matrix_tuple):
    """ Perform a separate change of basis on each index of the tensor, using the matrices in matrix_tuple
    Input
    -----
    tensor: np.ndarray
        Tensor to transform. n axes.
    matrix_tuple: iterable of np.ndarray
        Contains n 2D arrays.

    Output
    -----
    np.ndarray
    """
    assert len(matrix_tuple) == len(tensor.shape)
    for i, matrix in enumerate(matrix_tuple):
        tensor = one_index_transform(tensor, i, matrix)
    return tensor

def compute_basis_transformation(from_basis, target_basis):
    """
    Given two "matrices" of basis vectors for a space, construct the matrix U where Umn is the coefficient
    of basis vector m (of from_basis") in basis vector n (of "target_basis").
    target_basis = from_basis @ U; each basis vector is a column, and each row vector is some other basis

    Input
    -----
    from_basis: np.ndarray
    to_basis: np.ndarray

    Output
    ------
    np.ndarray
    """
    return np.linalg.inv(from_basis) @ target_basis


def read_tensor(name, num_indices, nina, nvir, failfast = False):
    """ Attempt to read a tensor from disk. Default to just zeros.

    Input
    -----
    name: str
        The name of the tensor to load.
    num_indices: int
    nina: int
    nvir: int
        Parameters used to define the target shape of the tensor.
    failfast: bool
        Fail if the tensor can't be read?

    Output
    ------
    np.ndarray
    """
    target_shape = (nina,) * num_indices + (nvir,) * num_indices
    try:
        tensor = np.load(name + ".npy")
        if tensor.shape == target_shape: return tensor
    except IOError:
        pass
    if failfast:
        raise AssertionError
    return np.zeros(target_shape)
