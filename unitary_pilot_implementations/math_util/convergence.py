from functools import partial
from itertools import starmap
import numpy as np
from copy import deepcopy

class DirectSumDiis():
    """
    A class to perform DIIS extrapolation on a direct sum of vectors.
    Needed to couple orbital and amplitude amplitudes.
    """

    def __init__(self, min_vec, max_vec):
        self.min = min_vec # Minimum vectors at once. Constant.
        self.max = max_vec # Maximum vectors at once. Constant.
        self.residuals = []
        self.trials = []
        self.skipped0 = False

    def diis(self, r, t):
        """
        Input
        -----
        r, t: np.ndarray or tuple of np.ndarray
            The current residuals and amplitudes

        Output
        ------
        np.ndarray or tuple of np.ndarray
            The new amplitudes.
        """
        if not self.skipped0:
            self.skipped0 = True
            return deepcopy(t)
        self.residuals.append(deepcopy(r))
        self.trials.append(deepcopy(t))
        # Enforce the maximum number of residuals and trial vectors with first in, first out.
        if len(self.residuals) > self.max:
            self.residuals.pop(0)
            self.trials.pop(0)
        # Perform DIIS if we have at least the minimum number of extrapolation points.
        if len(self.residuals) >= self.min:
            B_dim = 1 + len(self.residuals)
            B = np.empty((B_dim, B_dim))
            B[-1, :] = B[:, -1] = -1
            B[-1, -1] = 0
            for i, ri in enumerate(self.residuals):
                for j, rj in enumerate(self.residuals):
                    if i > j: continue
                    B[i, j] = B[j, i] = direct_sum_dot(ri, rj)
            # Normalize the matrix for numerical stability.
            B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()
            rhs = np.zeros((B_dim))
            rhs[-1] = -1
            coeffs = np.linalg.solve(B, rhs)[:-1]
            trials_to_new = partial(np.tensordot, b=coeffs, axes=(0,0))
            if isinstance(self.trials[0], np.ndarray):
                return trials_to_new(self.trials)
            else:
                return list(map(trials_to_new, zip(*self.trials)))
        else:
            return t


def direct_sum_dot(r1, r2):
    """
    Input
    -----
    r1, r2: np.ndarray or tuple of np.ndarray

    Output
    ------
    float
        The sum of the dot products for each vector in the direct sum.
    """
    if not isinstance(r1, np.ndarray):
        return sum(starmap(np.vdot, zip(r1, r2)))
    else:
        return np.vdot(r1, r2)
