"""
Mixed-variable ZDT3 benchmark task
混合変数 ZDT3 ベンチマークタスク
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.optimizer.core.variable_config import VariableConfig
from src.tasks.core.base_task import BaseTask
from src.tasks.core.data_type_network_metrics import NetworkMetrics
from src.tasks.core.direction import Direction


class ZDT3MvTask(BaseTask):
    """
    Evaluate the mixed-variable ZDT3 benchmark.
    混合変数ZDT3ベンチマークを評価する。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Override BaseTask.__init__ and retain mixed-variable settings.
        BaseTask.__init__をオーバーライドして混合変数設定を保持する。
        """
        super().__init__(config)
        self.n_x = config.get('n_x', 10)
        self.n_z = config.get('n_z', 5)
        self.n_c = config.get('n_c', 3)
        self.K_z = config.get('K_z', 5)
        self.K_c = config.get('K_c', 4)
        self.name = "ZDT3_MV"

    def _evaluate(self, X: np.ndarray, Z: np.ndarray, C: np.ndarray) -> np.ndarray:
        """
        Calculate both ZDT3 objective values.
        ZDT3の2つの目的関数値を計算する。
        """
        X = np.asarray(X, dtype=float)
        Z = np.asarray(Z, dtype=float)
        C = np.asarray(C, dtype=float)

        x0_norm = X[0]
        z0_norm = (
            Z[0] / (self.K_z - 1)
            if self.K_z > 1
            else Z[0]
        )
        c0_norm = 1 - C[0, 0]
        f1 = (x0_norm + z0_norm + c0_norm) / 3.0

        term_x = float(np.mean(X[1:])) if self.n_x > 1 else 0.0
        if self.n_z > 1 and self.K_z > 1:
            term_z = float(np.mean(Z[1:] / (self.K_z - 1)))
        elif self.n_z > 1:
            term_z = float(np.mean(Z[1:]))
        else:
            term_z = 0.0
        term_c = float(np.mean(1 - C[1:, 0])) if self.n_c > 1 else 0.0
        mean_all = float(np.mean(np.array([term_x, term_z, term_c])))
        g = 1 + 9 * mean_all

        ratio = f1 / g
        f2 = g * (
            1
            - np.sqrt(ratio)
            - ratio * np.sin(10.0 * np.pi * f1)
        )
        return np.array([f1, f2])

    def objective_function(
        self,
        params: Dict[str, Any]
    ) -> Tuple[Tuple[float, ...], Optional[NetworkMetrics]]:
        """
        Implement BaseTask.objective_function and return the objective tuple with None.
        BaseTask.objective_functionを実装し、目的関数値タプルとNoneを返す。
        """
        # パラメータ辞書から、目的関数の評価に使う各変数のベクトルを作成する。
        x_values = [params[f'x{i}'] for i in range(self.n_x)]
        X = np.array(x_values)

        z_values = [params[f'z{i}'] for i in range(self.n_z)]
        Z = np.array(z_values)

        C = np.zeros((self.n_c, self.K_c))
        for i in range(self.n_c):
            cat_value = params[f'c{i}']
            if isinstance(cat_value, (list, np.ndarray)):
                C[i] = cat_value
            else:
                C[i, int(cat_value)] = 1

        result = self._evaluate(X, Z, C)
        f1, f2 = float(result[0]), float(result[1])
        return ((f1, f2), None)

    def get_objective_names(self) -> Tuple[str, ...]:
        """
        Implement BaseTask.get_objective_names and return axis labels as a tuple.
        BaseTask.get_objective_namesを実装し、軸名をタプルで返す。
        """
        return ("f1", "f2")

    def get_directions(self) -> Tuple:
        """
        Implement BaseTask.get_directions and return two Direction.MINIMIZE values as a tuple.
        BaseTask.get_directionsを実装し、2つのDirection.MINIMIZEをタプルで返す。
        """
        return (Direction.MINIMIZE, Direction.MINIMIZE)

    def get_variables(self) -> VariableConfig:
        """
        Implement BaseTask.get_variables and return mixed variables as VariableConfig.
        BaseTask.get_variablesを実装し、混合変数定義をVariableConfigで返す。
        """
        config = VariableConfig()

        # 連続変数
        for i in range(self.n_x):
            config.add_continuous(f'x{i}', 0.0, 1.0)

        # 整数変数
        for i in range(self.n_z):
            config.add_integer(f'z{i}', 0, self.K_z - 1)
        category_names = []

        # カテゴリ変数
        for j in range(self.K_c):
            category_names.append(str(j))
        for i in range(self.n_c):
            config.add_categorical(f'c{i}', category_names)
        return config

    @staticmethod
    def get_true_pareto_front(n_points: int = 500) -> np.ndarray:
        """
        Return the discontinuous analytical ZDT3 Pareto front.
        不連続な解析的ZDT3パレートフロントを返す。
        """
        regions = (
            (0.0, 0.0830),
            (0.1822, 0.2577),
            (0.4093, 0.4538),
            (0.6183, 0.6525),
            (0.8233, 0.8518),
        )
        points_per_region = n_points // len(regions)
        pareto_points: List[List[float]] = []

        for lower, upper in regions:
            f1_region = np.linspace(lower, upper, points_per_region)
            f2_region = (
                1.0
                - np.sqrt(f1_region)
                - f1_region * np.sin(10.0 * np.pi * f1_region)
            )
            for i in range(len(f1_region)):
                pareto_points.append([float(f1_region[i]), float(f2_region[i])])

        return np.asarray(pareto_points)
