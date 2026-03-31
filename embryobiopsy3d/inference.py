"""
This file calculates the likelihood of a tree composition given the phylogeny
and the meiotic/mitotic error rates.
"""

import numpy as np


def _leaf_probability(obs, false_negative=0.0, false_positive=0.0):
    """Return [P(obs|state=0), P(obs|state=1)] for a leaf observation."""
    assert false_negative >= 0 and false_negative <= 1
    assert false_positive >= 0 and false_positive <= 1

    # P(obs | true=0), P(obs | true=1) ] for a leaf observation obs∈{0,1}.
    if obs not in (0, 1):
        raise ValueError("Leaf obs must be either euploid (0) or aneuploid (1)")
    if obs == 1:
        return np.array([false_positive, 1.0 - false_negative], dtype=float)
    else:
        return np.array([1.0 - false_positive, false_negative], dtype=float)


# if p is mitotic error rate
# let 0 be euploid, 1 be aneuploid
# The probability of observed:

#   - P(c = 0 | parent = 0) = 1-p
#   - P(c = 1 | parent = 0) = p
#   - P(c = 0 | parent = 1) = 0
#   - P(c = 1 | parent = 1) = 1


def _rate_for_generation(p, p_by_div, generation):
    """Boundaries and null handling for returning mitotic error rate for a given generation."""
    if p_by_div is None:
        return p
    if generation < 0 or generation >= len(p_by_div):
        raise ValueError(
            f"p_by_div length {len(p_by_div)} is incompatible with generation {generation}"
        )
    return p_by_div[generation]


# keeping false negative and false positive probabilities?
def likelihood(
    root,
    observed_leaves,
    p=None,
    p_meio=0.0,
    false_negative=0.0,
    false_positive=0.0,
    p_by_div=None,
):
    """
    Compute the tree likelihood given mitotic error rate(s) and meiotic rate p_meio.

    observed_leaves: dict mapping Cell -> {0,1} (euploid/aneuploid)
    p: scalar mitotic error rate applied to all divisions (if p_by_div is None)
    p_by_div: optional iterable of per-division mitotic rates indexed by parent generation
    """
    # parameter checking
    assert root
    if p is None and p_by_div is None:
        raise ValueError("Provide either scalar p or array p_by_div.")
    if p is not None:
        assert p >= 0 and p <= 1
    if p_by_div is not None:
        p_by_div = np.asarray(p_by_div, dtype=float)
        if p_by_div.ndim != 1:
            raise ValueError("p_by_div must be a 1-D sequence of rates.")
        if np.any((p_by_div < 0) | (p_by_div > 1)):
            raise ValueError("p_by_div entries must be in [0, 1].")
    assert 0 <= p_meio <= 1
    assert 0 <= false_negative <= 1
    assert 0 <= false_positive <= 1

    # transition matrix
    def _check_cell_likelihood(cell):

        # leaf node
        if not cell.children:
            if cell in observed_leaves:
                return _leaf_probability(
                    observed_leaves[cell],
                    false_negative=false_negative,
                    false_positive=false_positive,
                )
            else:  # not in observed leaf. Likelihood as 1?
                return np.array([1.0, 1.0], dtype=float)

        # Internal node (consider both children together)

        # L(current state) = P(all subtrees ∣ current cell's aneuploidy state)
        # = Sum (P (each child's state ∣ parent's state) * (P(children's subtrees ∣ child's states)))

        # If parent is euploid (0):
        # - with prob (1-p), both children stay euploid (0,0);
        # - with prob p, both children become aneuploid (1,1).
        # - cannot have mixed children states like (1,0) (For now, consider other cases later)
        #
        # If parent is aneuploid (1):
        # - both children are aneuploid with probability 1
        # - currently consider irreversible aneuploidy in this model
        #

        if len(cell.children) != 2:
            raise ValueError(
                "likelihood() currently assumes a binary division tree (each internal node has exactly 2 children)."
            )

        c0 = np.asarray(_check_cell_likelihood(cell.children[0]), dtype=float)
        c1 = np.asarray(_check_cell_likelihood(cell.children[1]), dtype=float)

        # pick rate for this division based on parent generation
        p_here = _rate_for_generation(p, p_by_div, cell.generation)

        # Likelihood given parent state = 0
        L0 = (1.0 - p_here) * (c0[0] * c1[0]) + p_here * (c0[1] * c1[1])

        # Likelihood given parent state = 1 (forced aneuploid children)
        L1 = c0[1] * c1[1]

        return np.array([L0, L1], dtype=float)

    root_likelihood = _check_cell_likelihood(root)
    first_division = np.array([1 - p_meio, p_meio])
    return first_division @ root_likelihood


def likelihood_from_embryo(
    embryo,
    observed_leaves,
    p=None,
    p_meio=0.0,
    false_negative=0.0,
    false_positive=0.0,
    p_by_div=None,
):
    """
    Wrapper to compute likelihood using an Embryo container.

    observed_leaves: dict mapping Cell -> {0,1} (euploid/aneuploid)
    """
    return likelihood(
        embryo.root,
        observed_leaves,
        p=p,
        p_meio=p_meio,
        false_negative=false_negative,
        false_positive=false_positive,
        p_by_div=p_by_div,
    )
