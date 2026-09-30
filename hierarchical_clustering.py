# hierarchical_clustering.py
import copy
import numpy as np
from collections.abc import Iterable


# ================== NumPy 2.x compatibility patch ==================
# Some legacy code may refer to np.NINF (removed in NumPy 2.0)
if not hasattr(np, "NINF"):
    np.NINF = -np.inf  # keep legacy reference working


def flatten(items):
    """Yield items from any nested iterable."""
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x


def calculating_adjacency(clients_idxs, U):
    """
    Build client-to-client "distance" matrix using principal angles between subspaces.
    Returned matrix is in degrees, smaller => more similar.

    Notes:
    - Robust to NaN/Inf by mapping them to a large distance.
    - Robust to empty U_i by mapping to a large distance.
    """
    nclients = len(clients_idxs)
    sim_mat = np.zeros((nclients, nclients), dtype=np.float64)

    BIG = 180.0  # max angle in degrees for arccos

    for idx1 in range(nclients):
        for idx2 in range(nclients):
            U1 = copy.deepcopy(U[clients_idxs[idx1]])
            U2 = copy.deepcopy(U[clients_idxs[idx2]])

            U1 = np.asarray(U1, dtype=np.float64)
            U2 = np.asarray(U2, dtype=np.float64)

            # empty guard
            if U1.size == 0 or U2.size == 0:
                sim_mat[idx1, idx2] = BIG
                continue

            # principal angle proxy: min arccos of pairwise dot products
            mul = U1.T @ U2
            mul = np.clip(mul, a_min=-1.0, a_max=1.0)

            ang = np.arccos(mul) * 180.0 / np.pi  # degrees
            finite = ang[np.isfinite(ang)]
            if finite.size == 0:
                sim_mat[idx1, idx2] = BIG
            else:
                sim_mat[idx1, idx2] = float(np.min(finite))

    # sanitize any residual NaN/Inf
    sim_mat = np.nan_to_num(sim_mat, nan=BIG, posinf=BIG, neginf=BIG)
    return sim_mat


def hierarchical_clustering(A, thresh=1.5, linkage="maximum"):
    """
    Hierarchical Clustering over an adjacency (distance) matrix A.

    This is an agglomerative procedure that repeatedly merges the two closest clusters
    (minimum inter-cluster distance), then updates distances using a linkage rule.

    Supported linkage values (compatible with your fkpl args):
      - 'single'   == 'minimum' : min distance (single linkage)
      - 'complete' == 'maximum' : max distance (complete linkage)
      - 'average'              : average distance
      - 'minimum'/'maximum'    : explicit aliases
      - 'ward'                 : NOT well-defined for a pure distance matrix here;
                                 we fallback to 'average' with a warning.

    Important fix:
      Your sweep passed linkage='complete', but the old implementation only had
      'maximum'/'minimum'/'average'. That caused Z to be undefined and raised:
      UnboundLocalError: cannot access local variable 'Z'...
    """
    A = np.array(A, dtype=np.float64, copy=True)

    # normalize + sanitize
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be a square matrix, got shape={A.shape}")

    n = A.shape[0]
    if n <= 1:
        return [list(range(n))]

    # make symmetric and finite
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)

    finite = A[np.isfinite(A)]
    if finite.size == 0:
        return [list(range(n))]
    max_f = float(finite.max())
    A = np.nan_to_num(A, nan=max_f * 1.01, posinf=max_f * 1.01, neginf=max_f * 1.01)

    # linkage mapping (key fix for 'complete' / 'single')
    lk = str(linkage).lower().strip()
    if lk in ("complete", "maximum", "max"):
        lk_mode = "maximum"
    elif lk in ("single", "minimum", "min"):
        lk_mode = "minimum"
    elif lk in ("average", "avg", "mean"):
        lk_mode = "average"
    elif lk in ("ward",):
        print("[WARN] linkage='ward' is not supported for this adjacency-matrix HC; fallback to 'average'.")
        lk_mode = "average"
    else:
        raise ValueError(
            f"Unknown linkage='{linkage}'. Supported: single/complete/average/ward "
            f"(ward->average fallback), or minimum/maximum."
        )

    label_assg = {i: i for i in range(A.shape[0])}

    step = 0
    while A.shape[0] > 1:
        # exclude diagonal from argmin by setting it to +inf
        np.fill_diagonal(A, -np.NINF)  # -(-inf)=+inf with np.NINF=-inf
        step += 1

        ind = np.unravel_index(np.argmin(A, axis=None), A.shape)
        best = A[ind[0], ind[1]]

        # stopping condition
        if not np.isfinite(best) or best > float(thresh):
            # print('Breaking HC')
            break

        # restore diagonal to 0 for linkage update
        np.fill_diagonal(A, 0.0)

        if lk_mode == "maximum":
            Z = np.maximum(A[:, ind[0]], A[:, ind[1]])
        elif lk_mode == "minimum":
            Z = np.minimum(A[:, ind[0]], A[:, ind[1]])
        else:  # 'average'
            Z = (A[:, ind[0]] + A[:, ind[1]]) / 2.0

        # merge cluster ind[1] into ind[0]
        A[:, ind[0]] = Z
        A[:, ind[1]] = Z
        A[ind[0], :] = Z
        A[ind[1], :] = Z

        # drop ind[1]
        A = np.delete(A, ind[1], axis=0)
        A = np.delete(A, ind[1], axis=1)

        # update label assignments
        if isinstance(label_assg[ind[0]], list):
            label_assg[ind[0]].append(label_assg[ind[1]])
        else:
            label_assg[ind[0]] = [label_assg[ind[0]], label_assg[ind[1]]]
        label_assg.pop(ind[1], None)

        # re-index keys after deletion
        temp = []
        for k, v in label_assg.items():
            if k > ind[1]:
                temp.append((k - 1, v))
            else:
                temp.append((k, v))
        label_assg = dict(temp)

    clusters = []
    for k in label_assg.keys():
        if isinstance(label_assg[k], list):
            clusters.append(list(flatten(label_assg[k])))
        else:
            clusters.append([label_assg[k]])

    return clusters


def error_gen(actual, rounded):
    divisor = np.sqrt(1.0 if actual < 1.0 else actual)
    return abs(rounded - actual) ** 2 / divisor


def round_to(percents, budget=100):
    if not np.isclose(sum(percents), budget):
        raise ValueError("percents must sum to budget exactly.")
    n = len(percents)
    rounded = [int(x) for x in percents]
    up_count = budget - sum(rounded)
    errors = [
        (error_gen(percents[i], rounded[i] + 1) - error_gen(percents[i], rounded[i]), i)
        for i in range(n)
    ]
    rank = sorted(errors)
    for i in range(up_count):
        rounded[rank[i][1]] += 1
    return rounded
