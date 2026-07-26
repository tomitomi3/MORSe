"""
Abstract class for multi-objective optimization adopting the Ask-Tell interface
多目的最適化の抽象クラス Ask-Tellインターフェースを採用
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np

from src.optimizer.core.variable_config import VariableConfig


class BaseOptimizer(ABC):
    """
    Define the abstract Ask-Tell interface for multi-objective optimization algorithms.
    多目的最適化アルゴリズムの抽象Ask-Tellインターフェースを定義する。
    """

    @abstractmethod
    def setup(self, var_config: VariableConfig, **kwargs) -> None:
        """
        Initialize multi-objective optimization state from variables and parameters.
        変数定義とパラメータから多目的最適化状態を初期化する。
        """
        pass

    @abstractmethod
    def ask(self) -> List[Dict[str, Any]]:
        """
        Generate and return candidate parameter dictionaries.
        候補パラメータ辞書を生成して返す。
        """
        pass

    @abstractmethod
    def tell(self, values: List[np.ndarray]) -> None:
        """
        Accept evaluated objective values and update multi-objective optimization state.
        評価済み目的関数値を受け取り、多目的最適化状態を更新する。
        """
        pass

    @abstractmethod
    def get_result(self) -> Any:
        """
        Return the current multi-objective optimization result solutions.
        現在の多目的最適化結果となる解を返す。
        """
        pass

