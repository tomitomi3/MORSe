"""
Abstract base class for multi-objective optimization tasks
多目的最適化タスクの抽象基底クラス
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from src.tasks.core.data_type_network_metrics import NetworkMetrics


class BaseTask(ABC):
    """
    Define the abstract interface for multi-objective optimization tasks.
    多目的最適化タスクの抽象インターフェースを定義する。
    """

    # 継承先では目的関数、目的名、最適化方向、決定変数の各抽象メソッドを実装する。

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize a task with its configuration.
        設定を受け取ってタスクを初期化する。
        """
        self.config = config
        self.name = self.__class__.__name__

    @abstractmethod
    def objective_function(
        self,
        params: Dict[str, Any]
    ) -> Tuple[Tuple[float, ...], Optional[NetworkMetrics]]:
        """
        Return the required objective tuple and optional NetworkMetrics for reservoir analysis.
        目的関数値タプル（必須）と、リザバーのネットワーク分析に使用する任意のNetworkMetricsを返す。
        """
        pass

    @abstractmethod
    def get_objective_names(self) -> Tuple[str, ...]:
        """
        Return framework axis labels as a tuple of objective names.
        フレームワークの軸名として使用する目的関数名を常にタプル形式で返す。
        """
        pass

    @abstractmethod
    def get_directions(self) -> Tuple:
        """
        Return a tuple of Direction.MINIMIZE or Direction.MAXIMIZE values.
        各目的関数の最適化方向をDirection.MINIMIZEまたはDirection.MAXIMIZEを並べたタプル形式で返す。
        """
        pass

    @abstractmethod
    def get_variables(self) -> Any:
        """
        Return the decision variable configuration.
        決定変数の設定を返す。
        """
        # VariableConfigオブジェクトをビルダーメソッドで定義して返す。
        pass
