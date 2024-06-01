import torch

from utils.distmat import compute_distmat


def init_feedback_indices(q, g, device=None):
    return torch.zeros((q, g), dtype=torch.bool, device=device)


def init_feedback_indices_qg(q, g, positive=False, device=None):
    indices = torch.zeros(q, q + g, dtype=torch.bool, device=device)
    if positive:
        indices[torch.arange(q), torch.arange(q)] = True
    return indices


def greedy_feedback(distmat, q_pids, g_pids, positive_indices, negative_indices, inplace=True):
    """
    Update positive_indices, negative_indices with one round of feedback. Provide feedback for top-ranked gallery.
    Note that distmat is corrupted if inplace=True.

    :param distmat: q x g Tensor (adjusted query to gallery)
    :param q_pids: q
    :param g_pids: g
    :param positive_indices: q x g
    :param negative_indices: q x g
    :return:
    (positive_indices, negative_indices, matches)
    """
    q, g = tuple(distmat.shape)

    if not inplace:
        distmat = distmat.clone().detach()
        positive_indices = positive_indices.copy()
        negative_indices = negative_indices.copy()

    distmat[positive_indices] = float("inf")
    distmat[negative_indices] = float("inf")

    indices = distmat.argmin(dim=1)
    pmap = g_pids[indices] == q_pids
    positive_q = torch.arange(0, q)[pmap]
    negative_q = torch.arange(0, q)[pmap == False]
    positive_g = indices[pmap]
    negative_g = indices[pmap == False]

    existing = positive_indices[positive_q, positive_g]
    assert (not existing.any())
    positive_indices[positive_q, positive_g] = True
    existing = negative_indices[negative_q, negative_g]
    assert (not existing.any())
    negative_indices[negative_q, negative_g] = True

    return positive_indices, negative_indices, pmap


def naive_round(qf, gf, q_pids, g_pids, positive_indices=None, negative_indices=None,
                inplace=True, previous_distmat=None, device=None):
    """
    qf: q x m
    gf: g x m
    q_pids: q
    g_pids: g
    positive_indices: q x g
    negative_indices: q x g
    previous_distmat: adjusted distmat (== compute_distmat(qf, gf) only at init)
    """
    q, g = qf.shape[0], gf.shape[0]
    assert (qf.shape[1] == gf.shape[1])

    if positive_indices is None: positive_indices = init_feedback_indices(q, g, device=device)
    if negative_indices is None: negative_indices = init_feedback_indices(q, g, device=device)

    if previous_distmat is None:
        distmat = compute_distmat(qf, gf)
    else:
        distmat = previous_distmat

    res = greedy_feedback(distmat, q_pids, g_pids, positive_indices, negative_indices, inplace=inplace)
    positive_indices, negative_indices, matches = res

    distmat = compute_distmat(qf, gf)
    distmat[positive_indices] = 0
    distmat[negative_indices] = float("inf")

    return distmat, positive_indices, negative_indices, matches
