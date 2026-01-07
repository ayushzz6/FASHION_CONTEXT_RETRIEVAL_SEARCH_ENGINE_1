"""Phrase-to-patch assignment scoring."""

from __future__ import annotations

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover
    linear_sum_assignment = None


def assignment_score(phrase_embeds: np.ndarray, patch_embeds: np.ndarray, method: str = "greedy") -> float:
    """Compute compositional score by assigning each phrase to a distinct patch."""
    if phrase_embeds.size == 0 or patch_embeds.size == 0:
        return 0.0
    if phrase_embeds.shape[1] != patch_embeds.shape[1]:
        d = min(phrase_embeds.shape[1], patch_embeds.shape[1])
        phrase_embeds = phrase_embeds[:, :d]
        patch_embeds = patch_embeds[:, :d]
    sim_matrix = phrase_embeds @ patch_embeds.T
    if method == "hungarian" and linear_sum_assignment is not None:
        row_ind, col_ind = linear_sum_assignment(-sim_matrix)
        return float(sim_matrix[row_ind, col_ind].sum() / len(row_ind))
    return float(_greedy_score(sim_matrix))


def _greedy_score(sim_matrix: np.ndarray) -> float:
    sim = sim_matrix.copy()
    phrases, patches = sim.shape
    used_cols = set()
    total = 0.0
    matched = 0
    for i in range(phrases):
        best_col = None
        best_val = -1e9
        for j in range(patches):
            if j in used_cols:
                continue
            if sim[i, j] > best_val:
                best_val = sim[i, j]
                best_col = j
        if best_col is not None:
            used_cols.add(best_col)
            total += best_val
            matched += 1
    if matched == 0:
        return 0.0
    return total / matched
