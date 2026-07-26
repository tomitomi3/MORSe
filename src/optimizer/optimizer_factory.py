"""
Multi-objective optimization factory
多目的最適化ファクトリ
"""

from src.optimizer.algorithms.moead.moead import MOEAD
from src.optimizer.core.base_optimizer import BaseOptimizer
from src.utils.common_messages import unsupported_value_message


def create_optimizer(name: str) -> BaseOptimizer:
    """
    Create and return the requested BaseOptimizer implementation.
    指定されたBaseOptimizer実装を生成して返す。
    """
    
    # 多目的最適化アルゴリズムのインスタンスを生成する
    if name == "MOEAD":
        # MOEA/D方式の多目的最適化アルゴリズムを生成する
        return MOEAD()
    
    # MOEA/D以外のアルゴリズムは未実装 例外スロー
    raise ValueError(
        unsupported_value_message(
            "optimizer",
            name,
            ["MOEAD"],
        )
    )
