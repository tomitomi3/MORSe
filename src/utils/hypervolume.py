"""
Calculate Hypervolume
ハイパーボリューム計算
"""

from typing import List, Optional

import numpy as np


def compute_hypervolume(
    front: np.ndarray,
    reference_point: np.ndarray,
    directions: Optional[List[str]] = None,
) -> float:
    """
    Compute and return the two-objective hypervolume.
    2目的のハイパーボリュームを計算して返す。
    """

    # 現状 2目的のみとする
    num_objectives = front.shape[1]
    if num_objectives != 2:
        return 0.0

    if directions is None:
        directions = []
        for _ in range(num_objectives):
            directions.append("minimize")

    # 目的空間 最小化として扱う
    signs = []
    for direction in directions:
        if direction == "minimize":
            signs.append(1.0)
        else:
            signs.append(-1.0)

    signs_array = np.array(signs, dtype=float)
    front_min = np.asarray(front, dtype=float) * signs_array
    reference_min = np.asarray(reference_point, dtype=float) * signs_array

    # 基準点に支配される領域に入っていない点を除外
    ref_f1 = reference_min[0]
    ref_f2 = reference_min[1]
    valid_mask = (front_min[:, 0] < ref_f1) & (front_min[:, 1] < ref_f2)
    valid_front = front_min[valid_mask]
    if len(valid_front) == 0:
        return 0.0

    # f1昇順に並べると、非劣解はf2が単調減少する点
    order = np.argsort(valid_front[:, 0])
    sorted_front = valid_front[order]

    non_dominated_points = []
    best_f2 = float("inf")
    for point in sorted_front:
        f2_value = point[1]
        if f2_value < best_f2:
            non_dominated_points.append(point)
            best_f2 = f2_value

    # 各非劣解から次のf1までを1列として面積を積み上げる。
    non_dominated_front = np.array(non_dominated_points, dtype=float)
    hv = 0.0
    num_points = len(non_dominated_front)
    for index in range(num_points):
        current_f1 = non_dominated_front[index, 0]
        current_f2 = non_dominated_front[index, 1]

        if index < num_points - 1:
            next_f1 = non_dominated_front[index + 1, 0]
            width = next_f1 - current_f1
        else:
            width = ref_f1 - current_f1

        height = ref_f2 - current_f2
        hv += width * height

    return hv


def compute_auto_reference_point(
    objectives: np.ndarray,
    directions: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Compute and return an automatic reference point from objective values.
    目的関数値から基準点を自動計算して返す。
    """

    # 目的数を取得し、方向が指定されていない場合は全て最小化
    num_objectives = objectives.shape[1]
    if directions is None:
        directions = []
        for _ in range(num_objectives):
            directions.append("minimize")

    # 基準点計算もHV本体と同じ向きで最小化空間へそろえる。
    signs = []
    for direction in directions:
        if direction == "minimize":
            signs.append(1.0)
        else:
            signs.append(-1.0)

    signs_array = np.array(signs, dtype=float)
    minimization_objectives = np.asarray(objectives, dtype=float) * signs_array

    # ちょっと値を足して基準点とする
    nadir_point = np.max(minimization_objectives, axis=0)
    margin = np.maximum(np.abs(nadir_point) * 0.1, 0.01)
    reference_min = nadir_point + margin

    return reference_min * signs_array
