"""
ReservoirPy NARMA10 prediction task with selectable activation functions
ReservoirPy による活性化関数選択付き NARMA10 予測タスク
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
from reservoirpy.datasets import narma
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.observables import nrmse

from src.optimizer.core.variable_config import VariableConfig
from src.tasks.core.base_task import BaseTask
from src.tasks.core.data_type_network_metrics import NetworkMetrics
from src.tasks.core.direction import Direction
from src.tasks.core.network_metrics import (
    compute_average_shortest_path_length,
    extract_adjacency_matrix,
)
from src.utils.common_messages import unsupported_value_message

ACTIVATIONS = [
    'tanh',
    'tanh_input_bias',
    'tanh_shift_1',
    'sigmoid',
]
NRMSE_PENALTY = 1e12


def stable_sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Apply a numerically stable sigmoid transformation.
    数値的に安定したシグモイド変換を適用する。
    """
    x_array = np.asarray(x, dtype=np.float64)
    result = np.empty_like(x_array)
    positive_mask = x_array >= 0.0

    result[positive_mask] = 1.0 / (1.0 + np.exp(-x_array[positive_mask]))
    negative_exp = np.exp(x_array[~positive_mask])
    result[~positive_mask] = negative_exp / (1.0 + negative_exp)
    return result


class Narma10PredictionActivationTask(BaseTask):
    """
    Evaluate NARMA10 prediction while selecting a reservoir activation.
    リザバーの活性化関数を選択しながらNARMA10予測を評価する。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Override BaseTask.__init__ and retain NARMA10 activation settings.
        BaseTask.__init__をオーバーライドしてNARMA10活性化設定を保持する。
        """
        super().__init__(config)
        self.seed = int(config.get('seed', 1234))
        self.n_eval_seeds = int(config.get('n_eval_seeds', 5))
        self.train_length = int(config.get('train_length', 900))
        self.test_length = int(config.get('test_length', 100))
        self.reservoir_size_lower = int(config.get('reservoir_size_lower', 20))
        self.reservoir_size_upper = int(config.get('reservoir_size_upper', 100))
        self.spectral_radius_lower = float(config.get('spectral_radius_lower', 0.2))
        self.spectral_radius_upper = float(config.get('spectral_radius_upper', 0.999))
        self.rc_connectivity_lower = float(config.get('rc_connectivity_lower', 0.05))
        self.rc_connectivity_upper = float(config.get('rc_connectivity_upper', 0.6))
        self.input_scaling_lower = float(config.get('input_scaling_lower', 0.01))
        self.input_scaling_upper = float(config.get('input_scaling_upper', 1.0))
        self.leaking_rate_lower = float(config.get('leaking_rate_lower', 0.1))
        self.leaking_rate_upper = float(config.get('leaking_rate_upper', 1.0))
        self.ridge_log10_lower = float(config.get('ridge_log10_lower', -4.0))
        self.ridge_log10_upper = float(config.get('ridge_log10_upper', 1.0))

        # リザバー状態の安定化とtrainからtestへの過渡応答除去に使う
        self.train_warmup = 100
        self.test_washout = 30

    def _resolve_activation(
        self,
        activation_index: int
    ) -> Tuple[str, Union[str, Callable[[np.ndarray], np.ndarray]], float, bool]:
        """
        Resolve an activation category to ReservoirPy arguments.
        活性化カテゴリをReservoirPyの引数へ変換する。
        """
        if activation_index < 0 or activation_index >= len(ACTIVATIONS):
            raise ValueError(
                unsupported_value_message(
                    "activation index",
                    activation_index,
                )
            )

        activation_name = ACTIVATIONS[activation_index]
        if activation_name == 'tanh':
            return activation_name, 'tanh', 0.0, False
        if activation_name == 'tanh_input_bias':
            return activation_name, 'tanh', 0.0, True
        if activation_name == 'tanh_shift_1':
            return activation_name, 'tanh', 1.0, False
        if activation_name == 'sigmoid':
            return activation_name, stable_sigmoid, 0.0, False

        raise ValueError(
            unsupported_value_message(
                "activation index",
                activation_index,
            )
        )

    def objective_function(
        self,
        params: Dict[str, Any]
    ) -> Tuple[Tuple[float, ...], Optional[NetworkMetrics]]:
        """
        Implement BaseTask.objective_function and return an objective tuple with NetworkMetrics.
        BaseTask.objective_functionを実装し、目的関数値タプルとNetworkMetricsを返す。
        """
        reservoir_size = int(params['reservoir_size'])
        spectral_radius = float(params['spectral_radius'])
        rc_connectivity = float(params['rc_connectivity'])
        input_scaling = float(params['input_scaling'])
        leaking_rate = float(params['leaking_rate'])
        ridge = 10.0 ** float(params['ridge_log10'])
        activation_index = int(params['activation'])
        _, activation, reservoir_bias, use_input_bias = self._resolve_activation(
            activation_index
        )

        if self.test_length <= self.test_washout:
            raise ValueError(
                'test_length must be greater than test_washout(30) '
                'to keep NRMSE window non-empty.'
            )

        nrmse_values: List[float] = []
        clustering_values_first = 0.0
        adjacency_matrix_first = None
        n_timesteps = self.train_length + self.test_length

        for i in range(self.n_eval_seeds):
            data_seed = self.seed + 10 * i
            reservoir_seed = self.seed + 1000 * i

            x_raw, y_raw = narma(
                n_timesteps=n_timesteps,
                order=10,
                a1=0.3,
                a2=0.05,
                b=1.5,
                c=0.1,
                seed=data_seed,
            )
            x_all = np.asarray(x_raw[-n_timesteps:], dtype=np.float64).reshape(-1, 1)
            y_all = np.asarray(y_raw, dtype=np.float64).reshape(-1, 1)
            if use_input_bias:
                constant_column = np.ones((x_all.shape[0], 1), dtype=np.float64)
                x_all = np.hstack((x_all, constant_column))

            x_train = x_all[: self.train_length]
            y_train = y_all[: self.train_length]
            x_test = x_all[self.train_length :]
            y_test = y_all[self.train_length :]
            if len(x_test) == 0 or len(y_test) == 0:
                raise ValueError('test split is empty.')

            reservoir = Reservoir(
                units=reservoir_size,
                sr=spectral_radius,
                lr=leaking_rate,
                input_scaling=input_scaling,
                rc_connectivity=rc_connectivity,
                activation=activation,
                bias=reservoir_bias,
                seed=reservoir_seed,
            )
            reservoir.initialize(x_train)

            if i == 0:
                adjacency_matrix_first = extract_adjacency_matrix(reservoir.W)
                topology_graph = nx.from_numpy_array(
                    np.asarray(adjacency_matrix_first, dtype=int),
                    create_using=nx.DiGraph,
                )
                clustering_values_first = float(nx.average_clustering(topology_graph))

            model = reservoir >> Ridge(ridge=ridge, fit_bias=True)
            model.fit(x_train, y_train, warmup=self.train_warmup)

            y_pred = np.asarray(model.run(x_test), dtype=np.float64).reshape(-1, 1)
            if not np.isfinite(y_pred).all():
                nrmse_values.append(NRMSE_PENALTY)
                continue

            nrmse_value = float(
                nrmse(
                    y_test[self.test_washout :],
                    y_pred[self.test_washout :],
                    norm='minmax',
                )
            )
            if not np.isfinite(nrmse_value):
                nrmse_value = NRMSE_PENALTY
            nrmse_values.append(nrmse_value)

        size_objective = float(reservoir_size)
        nrmse_mean = float(np.mean(nrmse_values))

        if adjacency_matrix_first is None:
            raise ValueError('adjacency_matrix must be captured at the first evaluation seed')
        average_shortest_path_length = compute_average_shortest_path_length(
            adjacency_matrix_first
        )
        if average_shortest_path_length is None:
            raise ValueError('average_shortest_path_length could not be computed')

        # Objective SpaceではX軸をreservoir_size、Y軸をnrmse_meanとする
        objective_values = (size_objective, nrmse_mean)
        metadata: NetworkMetrics = {
            'clustering': clustering_values_first,
            'adjacency_matrix': adjacency_matrix_first,
            'average_shortest_path_length': average_shortest_path_length,
        }
        return (objective_values, metadata)

    def get_objective_names(self) -> Tuple[str, ...]:
        """
        Implement BaseTask.get_objective_names and return NARMA10 axis labels as a tuple.
        BaseTask.get_objective_namesを実装し、NARMA10の軸名をタプルで返す。
        """
        return ('Reservoir Node', f'nrmse_mean(n={self.n_eval_seeds})')

    def get_directions(self) -> Tuple:
        """
        Implement BaseTask.get_directions and return two Direction.MINIMIZE values as a tuple.
        BaseTask.get_directionsを実装し、2つのDirection.MINIMIZEをタプルで返す。
        """
        return (Direction.MINIMIZE, Direction.MINIMIZE)

    def get_variables(self) -> VariableConfig:
        """
        Implement BaseTask.get_variables and return NARMA10 variables as VariableConfig.
        BaseTask.get_variablesを実装し、NARMA10の変数定義をVariableConfigで返す。
        """
        config = VariableConfig()

        # 整数変数
        config.add_integer(
            'reservoir_size',
            self.reservoir_size_lower,
            self.reservoir_size_upper,
        )

        # 連続変数
        config.add_continuous(
            'spectral_radius',
            self.spectral_radius_lower,
            self.spectral_radius_upper,
        )
        config.add_continuous(
            'rc_connectivity',
            self.rc_connectivity_lower,
            self.rc_connectivity_upper,
        )
        config.add_continuous(
            'input_scaling',
            self.input_scaling_lower,
            self.input_scaling_upper,
        )
        config.add_continuous(
            'leaking_rate',
            self.leaking_rate_lower,
            self.leaking_rate_upper,
        )
        config.add_continuous(
            'ridge_log10',
            self.ridge_log10_lower,
            self.ridge_log10_upper,
        )

        # カテゴリ変数
        config.add_categorical('activation', ACTIVATIONS)
        return config
