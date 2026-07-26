"""
MOEA/D subproblem data structure
MOEA/Dサブ問題のデータ構造
"""

from typing import List, Optional

import numpy as np

from src.optimizer.algorithms.moead.solution import Solution


class Subproblem:
    """
    Store one MOEA/D solution, weight vector, and neighborhood.
    MOEA/Dの1解、重みベクトル、近傍を保持する。
    """

    def __init__(self):
        """
        Initialize an empty MOEA/D subproblem.
        空のMOEA/Dサブ問題を初期化する。
        """
        # 1サブ問題に対して1個体を割り当て
        self.solution: Optional[Solution] = None
        
        # 重みベクトル 0～1
        self.weight: Optional[np.ndarray] = None
        
        # 隣接サブ問題インデックス
        self.neighbors: List[int] = []

    def __repr__(self) -> str:
        """
        Return a debug representation of subproblem state.
        サブ問題状態のデバッグ表現を返す。
        """        
        ret_str = f"{self.solution}\n{self.weight}\n{self.neighbors}"
        return ret_str
