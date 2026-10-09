"""Statistics shared by verifier selection and by the evaluation protocol.

``cp_lower`` is the exact Clopper-Pearson lower confidence bound behind the conservative ("hard") update rule
(paper App. B.3, Eq. empirical-admission) and the held-out diagnostics recorded at every checkpoint.
``sign_test_p``, ``block_bootstrap_ci`` and ``holm`` implement the preference-score evaluation (App. "Evaluation -
Preference score": exact two-sided sign test, contiguous-block bootstrap interval, Holm correction over the
family of arm-vs-baseline comparisons).
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Sequence, Tuple


def cp_lower(k: int, n: int, alpha: float) -> float:
    """Clopper-Pearson lower bound on a binomial proportion (k successes of n) at level ``alpha``.

    Found by bisection on the upper tail probability (80 iterations); k == 0 gives 0.0.
    """
    if k == 0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        tail = sum(math.comb(n, i) * mid ** i * (1 - mid) ** (n - i) for i in range(k, n + 1))
        if tail > alpha:
            hi = mid
        else:
            lo = mid
    return lo


def sign_test_p(w: int, l_: int) -> float:
    """Exact two-sided sign test p-value for ``w`` wins against ``l_`` losses (ties excluded)."""
    n = w + l_
    if n == 0:
        return 1.0
    k = min(w, l_)
    tail = sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def block_bootstrap_ci(seq: Sequence[float], block: int = 64, reps: int = 50000,
                       seed: int = 42) -> Tuple[float, List[float]]:
    """Mean of ``seq`` with a 95% contiguous-block bootstrap interval.

    The sequence is cut into consecutive blocks of ``block`` items (the last one may be shorter); each
    replicate resamples as many blocks as there are, with replacement, and takes the mean of their items.
    Returns ``(round(mean, 4), [round(lo, 4), round(hi, 4)])``.
    """
    seq = list(seq)
    if not seq:
        raise ValueError("block_bootstrap_ci: empty sequence")
    rng = random.Random(seed)
    obs = sum(seq) / len(seq)
    blocks = [seq[j:j + block] for j in range(0, len(seq), block)]
    sums = [sum(b) for b in blocks]
    sizes = [len(b) for b in blocks]
    nb = len(blocks)
    means = []
    for _ in range(reps):
        tot, cnt = 0.0, 0
        for _ in range(nb):
            j = rng.randrange(nb)
            tot += sums[j]
            cnt += sizes[j]
        means.append(tot / cnt)
    means.sort()
    return round(obs, 4), [round(means[int(.025 * reps)], 4), round(means[int(.975 * reps) - 1], 4)]


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    """Holm step-down adjusted p-values for a family of tests ({name: p} -> {name: adjusted p})."""
    order = sorted(pvals.items(), key=lambda x: x[1])
    m = len(order)
    out: Dict[str, float] = {}
    running = 0.0
    for rank, (name, p) in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p))
        out[name] = running
    return {name: out[name] for name in pvals}
