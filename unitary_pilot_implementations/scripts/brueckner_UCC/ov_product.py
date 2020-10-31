import numpy as np
from .determinant import test_bit

def pq_action(ci_vec, det_list, alpha_ann, alpha_cre, ab_idx_map):
    """ Compute the action of a^p a_q on the CI vector.

    Input
    -----
    ci_vec: np.ndarray
    det_list: list of Determinants
    alpha_ann: int
    alpha_cre: int
        Indices of the relevant orbitals.
    ab_idx_map:
        Map from an alpha string, beta string tuple to its index in the det list.

    Output
    ------
    np.ndarray
        The new CI vector.
    """
    new_ci_vec = np.zeros(ci_vec.shape)
    if alpha_ann == alpha_cre:
        for i, (coeff, det) in enumerate(zip(ci_vec, det_list)):
            if test_bit(det.alpha, alpha_ann):
               new_ci_vec[i] = coeff
    else:
        for coeff, det in zip(ci_vec, det_list):
            alpha = det.alpha
            if test_bit(det.alpha, alpha_ann) and not test_bit(det.alpha, alpha_cre):
                # Create the string for the new determinant.
                # Elsewhere, I need to create a map from alpha, beta to i
                new_alpha = (alpha | (1 << alpha_cre)) ^ (1 << alpha_ann)
                phase = det.phase_alpha_beta(new_alpha, det.beta, twobody=False)
                new_ci_vec[ab_idx_map[(new_alpha, det.beta)]] += coeff * phase
                pass
    return new_ci_vec
