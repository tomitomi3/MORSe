"""
Pareto front extraction utilities
パレートフロント抽出ユーティリティ
"""

from typing import Tuple

import numpy as np


def _dominates_normalized(
    normalized_a: np.ndarray,
    normalized_b: np.ndarray,
    eps: float,
) -> bool:
    """
    Return whether one normalized objective vector dominates another.
    正規化済み目的関数ベクトルが別のベクトルを支配するかどうかを返す。
    """
    return bool(
        np.all(normalized_a <= normalized_b + eps)
        and np.any(normalized_a < normalized_b - eps)
    )


def extract_pareto_front(objectives: np.ndarray, directions: list, eps: float = 1.2e-7) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract and return Pareto objective values with their original indices.
    パレート目的関数値と元のインデックスを抽出して返す。
    """
    n = len(objectives)
    if n == 0:
        return objectives, np.array([], dtype=int)

    # 全目的を最小化へ揃えるため、目的関数ごとの符号を作成する。
    sign_values = []
    for direction in directions:
        sign_values.append(1 if direction == "minimize" else -1)
    sign = np.array(sign_values, dtype=float)
    normalized = objectives * sign
    n_objectives = normalized.shape[1]

    # パレートフロントを求める
    # 高速化のため O(n log n) で求める方法を適用
    # https://en.wikipedia.org/wiki/Maxima_of_a_point_set
    # KUNG, Hsiang-Tsung; LUCCIO, Fabrizio; PREPARATA, Franco P. On finding the maxima of a set of vectors. Journal of the ACM (JACM), 1975, 22.4: 469-476.
    if n_objectives == 2:
        # x -> y の辞書順ソートで同一xを先に最良yから走査し、同一xの劣解残留を防ぐ。
        order = np.lexsort((normalized[:, 1], normalized[:, 0]))
        is_pareto = np.zeros(n, dtype=bool)
        best_y = float("inf")
        best_x_at_best_y = float("inf")

        for idx in order:
            x_val = float(normalized[idx, 0])
            y_val = float(normalized[idx, 1])

            if y_val < best_y - eps:
                is_pareto[idx] = True
                best_y = y_val
                best_x_at_best_y = x_val
                continue

            if abs(y_val - best_y) <= eps and abs(x_val - best_x_at_best_y) <= eps:
                # 同一点（許容誤差内）は非支配として残す
                is_pareto[idx] = True

        indices = np.where(is_pareto)[0]
        return objectives[is_pareto], indices

    # 3目的以上は既存の汎用判定（全最小化へ正規化済み）を維持
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            if _dominates_normalized(normalized[j], normalized[i], eps):
                is_pareto[i] = False
                break

    indices = np.where(is_pareto)[0]
    return objectives[is_pareto], indices
