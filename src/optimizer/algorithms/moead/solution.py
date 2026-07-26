"""
MOEA/D solution data structure
MOEA/D解のデータ構造
"""

from typing import Any, Dict

import numpy as np


class Solution:
    """
    Store parameters and objective values for one multi-objective optimization solution.
    1つの多目的最適化解のパラメータと目的関数値を保持する。
    """

    def __init__(self, num_objectives: int):
        """
        Initialize empty parameters and zero-valued objectives.
        空のパラメータとゼロ初期化した目的関数値を設定する。
        """
        # パラメータ
        # 例: {"spectral_radius": 0.9, "reservoir_size": 50}
        self.params: Dict[str, Any] = {}

        # 目的毎の評価値 ゼロ初期化
        self.obj_values = np.zeros(num_objectives)

    def __repr__(self) -> str:
        """
        Return a debug representation of objective values.
        目的関数値のデバッグ表現を返す。
        """
        return f"{self.obj_values})"
