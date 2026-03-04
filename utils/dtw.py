"""
DTW (Dynamic Time Warping) implementation.
Author: Brian Iwana
"""

import numpy as np
import math

RETURN_VALUE = 0
RETURN_PATH = 1
RETURN_ALL = -1


def _traceback(DTW, slope_constraint):
    i, j = np.array(DTW.shape) - 1
    p, q = [i - 1], [j - 1]

    if slope_constraint == "asymmetric":
        while i > 1:
            tb = np.argmin((DTW[i - 1, j], DTW[i - 1, j - 1], DTW[i - 1, j - 2]))
            if tb == 0:
                i = i - 1
            elif tb == 1:
                i = i - 1
                j = j - 1
            elif tb == 2:
                i = i - 1
                j = j - 2
            p.insert(0, i - 1)
            q.insert(0, j - 1)
    elif slope_constraint == "symmetric":
        while i > 1 or j > 1:
            tb = np.argmin((DTW[i - 1, j - 1], DTW[i - 1, j], DTW[i, j - 1]))
            if tb == 0:
                i = i - 1
                j = j - 1
            elif tb == 1:
                i = i - 1
            elif tb == 2:
                j = j - 1
            p.insert(0, i - 1)
            q.insert(0, j - 1)
    else:
        raise ValueError("Unknown slope constraint: %s" % slope_constraint)

    return (np.array(p), np.array(q))


def dtw(prototype, sample, return_flag=RETURN_VALUE, slope_constraint="asymmetric", window=None):
    """Computes the DTW of two sequences.

    Args:
        prototype: np array [0..b]
        sample: np array [0..t]
        return_flag: RETURN_VALUE, RETURN_PATH, or RETURN_ALL
        slope_constraint: "asymmetric" or "symmetric"
        window: Sakoe-Chiba band width (None = no constraint)
    """
    p = prototype.shape[0]
    assert p != 0, "Prototype empty!"
    s = sample.shape[0]
    assert s != 0, "Sample empty!"

    if window is None:
        window = s

    cost = np.full((p, s), np.inf)
    for i in range(p):
        start = max(0, i - window)
        end = min(s, i + window) + 1
        cost[i, start:end] = np.linalg.norm(sample[start:end] - prototype[i], axis=1)

    DTW = _cummulative_matrix(cost, slope_constraint, window)

    if return_flag == RETURN_ALL:
        return DTW[-1, -1], cost, DTW[1:, 1:], _traceback(DTW, slope_constraint)
    elif return_flag == RETURN_PATH:
        return _traceback(DTW, slope_constraint)
    else:
        return DTW[-1, -1]


def _cummulative_matrix(cost, slope_constraint, window):
    p = cost.shape[0]
    s = cost.shape[1]

    DTW = np.full((p + 1, s + 1), np.inf)
    DTW[0, 0] = 0.0

    if slope_constraint == "asymmetric":
        for i in range(1, p + 1):
            if i <= window + 1:
                DTW[i, 1] = cost[i - 1, 0] + min(DTW[i - 1, 0], DTW[i - 1, 1])
            for j in range(max(2, i - window), min(s, i + window) + 1):
                DTW[i, j] = cost[i - 1, j - 1] + min(DTW[i - 1, j - 2], DTW[i - 1, j - 1], DTW[i - 1, j])
    elif slope_constraint == "symmetric":
        for i in range(1, p + 1):
            for j in range(max(1, i - window), min(s, i + window) + 1):
                DTW[i, j] = cost[i - 1, j - 1] + min(DTW[i - 1, j - 1], DTW[i, j - 1], DTW[i - 1, j])
    else:
        raise ValueError("Unknown slope constraint: %s" % slope_constraint)

    return DTW


def shape_dtw(prototype, sample, return_flag=RETURN_VALUE, slope_constraint="asymmetric", window=None, descr_ratio=0.05):
    """Computes the shapeDTW of two sequences.

    Args:
        prototype: np array [0..b]
        sample: np array [0..t]
        return_flag: RETURN_VALUE, RETURN_PATH, or RETURN_ALL
        slope_constraint: "asymmetric" or "symmetric"
        window: Sakoe-Chiba band width (None = no constraint)
        descr_ratio: ratio for shape descriptor length
    """
    p = prototype.shape[0]
    assert p != 0, "Prototype empty!"
    s = sample.shape[0]
    assert s != 0, "Sample empty!"

    if window is None:
        window = s

    p_feature_len = np.clip(np.round(p * descr_ratio), 5, 100).astype(int)
    s_feature_len = np.clip(np.round(s * descr_ratio), 5, 100).astype(int)

    p_pad_front = (np.ceil(p_feature_len / 2.)).astype(int)
    p_pad_back = (np.floor(p_feature_len / 2.)).astype(int)
    s_pad_front = (np.ceil(s_feature_len / 2.)).astype(int)
    s_pad_back = (np.floor(s_feature_len / 2.)).astype(int)

    prototype_pad = np.pad(prototype, ((p_pad_front, p_pad_back), (0, 0)), mode="edge")
    sample_pad = np.pad(sample, ((s_pad_front, s_pad_back), (0, 0)), mode="edge")

    cost = np.full((p, s), np.inf)
    for i in range(p):
        for j in range(max(0, i - window), min(s, i + window)):
            cost[i, j] = np.linalg.norm(sample_pad[j:j + s_feature_len] - prototype_pad[i:i + p_feature_len])

    DTW = _cummulative_matrix(cost, slope_constraint=slope_constraint, window=window)

    if return_flag == RETURN_ALL:
        return DTW[-1, -1], cost, DTW[1:, 1:], _traceback(DTW, slope_constraint)
    elif return_flag == RETURN_PATH:
        return _traceback(DTW, slope_constraint)
    else:
        return DTW[-1, -1]


def accelerated_dtw(x, y, dist, warp=1):
    """Compute DTW distance with a specified distance function.

    Args:
        x: input array 1
        y: input array 2
        dist: distance function (e.g., lambda x, y: np.abs(x - y))
        warp: warp step size
    Returns:
        (minimum distance, cost matrix, accumulated cost matrix, wrap path)
    """
    assert len(x)
    assert len(y)
    r, c = len(x), len(y)
    D0 = np.zeros((r + 1, c + 1))
    D0[0, 1:] = np.inf
    D0[1:, 0] = np.inf
    D1 = D0[1:, 1:]
    for i in range(r):
        for j in range(c):
            D1[i, j] = dist(x[i], y[j])
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                min_list += [D0[min(i + k, r), j], D0[i, min(j + k, c)]]
            D1[i, j] += min(min_list)
    return D1[-1, -1], C, D1, _traceback_accelerated(D0)


def _traceback_accelerated(D):
    i, j = np.array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = np.argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return np.array(p), np.array(q)


def draw_graph1d(cost, DTW, path, prototype, sample):
    """Visualize 1D DTW alignment."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    p_steps = np.arange(prototype.shape[0])
    s_steps = np.arange(sample.shape[0])

    plt.subplot(2, 3, 1)
    plt.imshow(cost.T, cmap=plt.cm.gray, interpolation='none', origin='lower')
    plt.plot(path[0], path[1], 'y')
    plt.xlim((-0.5, cost.shape[0] - 0.5))
    plt.ylim((-0.5, cost.shape[0] - 0.5))

    plt.subplot(2, 3, 2)
    plt.imshow(DTW.T, cmap=plt.cm.gray, interpolation='none', origin='lower')
    plt.plot(path[0] + 1, path[1] + 1, 'y')
    plt.xlim((-0.5, DTW.shape[0] - 0.5))
    plt.ylim((-0.5, DTW.shape[0] - 0.5))

    plt.subplot(2, 3, 4)
    plt.plot(p_steps, prototype[:, 0], 'b-o')

    plt.subplot(2, 3, 5)
    for i in range(0, path[0].shape[0]):
        plt.plot([path[0][i], path[1][i]], [prototype[path[0][i], 0], sample[path[1][i], 0]], 'y-')
    plt.plot(p_steps, sample[:, 0], 'g-o')
    plt.plot(s_steps, prototype[:, 0], 'b-o')

    plt.subplot(2, 3, 6)
    plt.plot(s_steps, sample[:, 0], 'g-o')

    plt.tight_layout()
    plt.show()
