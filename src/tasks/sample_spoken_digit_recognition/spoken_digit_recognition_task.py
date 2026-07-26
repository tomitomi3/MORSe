"""
ReservoirPy spoken digit classification task
ReservoirPy による発話数字分類タスク
"""

import glob
import os
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np
from reservoirpy.nodes import Reservoir, Ridge
from scipy.io import loadmat
from sklearn.metrics import accuracy_score

from src.optimizer.core.variable_config import VariableConfig
from src.tasks.core.base_task import BaseTask
from src.tasks.core.data_type_network_metrics import NetworkMetrics
from src.tasks.core.direction import Direction
from src.tasks.core.network_metrics import (
    compute_average_shortest_path_length,
    extract_adjacency_matrix,
)
from src.utils.common_messages import (
    files_not_found_message,
    folder_not_found_message,
)

N_LABEL = 10
TRAIN_UTTERANCES = (1, 2, 3, 4, 5)
DATA_DIR_RELATIVE = Path('datasets') / 'Lyon_decimation_128'


class SpokenDigitRecognitionTask(BaseTask):
    """
    Evaluate spoken digit recognition with a ReservoirPy model.
    ReservoirPyモデルによる発話数字認識を評価する。
    """

    _DATA_CACHE: ClassVar[
        Dict[Tuple[str, Tuple[int, ...]], Tuple[np.ndarray, ...]]
    ] = {}

    def __init__(self, config: Dict[str, Any]):
        """
        Override BaseTask.__init__ and cache the speech dataset.
        BaseTask.__init__をオーバーライドして音声データセットをキャッシュする。
        """
        super().__init__(config)
        self.seed = int(config.get('seed', 1234))
        self.n_eval_seeds = int(config.get('n_eval_seeds', 3))
        self.data_dir = Path(__file__).resolve().parent / DATA_DIR_RELATIVE
        cache_key = (
            str(self.data_dir.resolve()),
            tuple(int(utterance) for utterance in TRAIN_UTTERANCES),
        )
        if cache_key not in self._DATA_CACHE:
            self._DATA_CACHE[cache_key] = self._read_speech_data(
                self.data_dir,
                TRAIN_UTTERANCES,
            )

        cached_data = self._DATA_CACHE[cache_key]
        self._train_input = cached_data[0]
        self._train_output = cached_data[1]
        self._test_input = cached_data[4]
        self._test_length = cached_data[6]
        self._test_label = cached_data[7]

        self.name = "SpokenDigitRecognition"

    def _read_speech_data(
        self,
        dir_name: Path,
        utterance_train_list: Iterable[int]
    ) -> Tuple[np.ndarray, ...]:
        """
        Load cochleagrams and split them into training and test data.
        音声信号のコクリアグラムを読み込み、学習データとテストデータへ分割する。
        """
        data_dir = Path(dir_name)
        if not data_dir.exists():
            raise FileNotFoundError(folder_not_found_message(data_dir))

        # .matファイルのみを読み込み対象とする
        data_files = sorted(glob.glob(os.path.join(str(data_dir), '*.mat')))
        if len(data_files) == 0:
            raise FileNotFoundError(files_not_found_message(".mat", data_dir))

        train_input_parts: List[np.ndarray] = []
        train_output_parts: List[np.ndarray] = []
        train_length: List[int] = []
        train_label: List[int] = []
        test_input_parts: List[np.ndarray] = []
        test_output_parts: List[np.ndarray] = []
        test_length: List[int] = []
        test_label: List[int] = []
        train_utterance_set = set(int(value) for value in utterance_train_list)

        for mat_file_path in data_files:
            file_name_with_ext = os.path.basename(mat_file_path)
            file_name, _ = os.path.splitext(file_name_with_ext)

            _speaker_part, utterance_part, digit_part = file_name.split('_')
            utterance = int(utterance_part[1:])
            digit = int(digit_part[1:])

            data = loadmat(mat_file_path)
            spec = np.asarray(data['spec'], dtype=np.float32)

            input_sequence = spec.T
            output_sequence = -np.ones((spec.shape[1], N_LABEL), dtype=np.float32)
            output_sequence[:, digit] = 1.0

            if utterance in train_utterance_set:
                train_input_parts.append(input_sequence)
                train_output_parts.append(output_sequence)
                train_length.append(spec.shape[1])
                train_label.append(digit)
            else:
                test_input_parts.append(input_sequence)
                test_output_parts.append(output_sequence)
                test_length.append(spec.shape[1])
                test_label.append(digit)

        train_input = np.vstack(train_input_parts)
        train_output = np.vstack(train_output_parts)
        test_input = np.vstack(test_input_parts)
        test_output = np.vstack(test_output_parts)

        return (
            train_input,
            train_output,
            np.asarray(train_length, dtype=int),
            np.asarray(train_label, dtype=int),
            test_input,
            test_output,
            np.asarray(test_length, dtype=int),
            np.asarray(test_label, dtype=int),
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
        leaking_rate = float(params['leaking_rate'])
        rc_connectivity = float(params['rc_connectivity'])
        input_connectivity = float(params['input_connectivity'])
        input_scaling = 10.0 ** float(params['input_scaling_log10'])
        ridge = 10.0 ** float(params['ridge_log10'])

        accuracy_values: List[float] = []
        clustering_first = 0.0
        adjacency_matrix_first = None

        for i in range(self.n_eval_seeds):
            reservoir_seed = self.seed + i

            reservoir = Reservoir(
                units=reservoir_size,
                sr=spectral_radius,
                lr=leaking_rate,
                input_scaling=input_scaling,
                rc_connectivity=rc_connectivity,
                input_connectivity=input_connectivity,
                activation='tanh',
                seed=reservoir_seed,
            )
            reservoir.initialize(self._train_input)

            # ReservoirPyの再帰重みWと入力重みWinを密行列に変換
            for attribute_name in ('W', 'Win'):
                matrix = getattr(reservoir, attribute_name, None)
                if hasattr(matrix, 'toarray'):
                    setattr(reservoir, attribute_name, matrix.toarray())

            if i == 0:
                adjacency_matrix_first = extract_adjacency_matrix(reservoir.W)
                topology_graph = nx.from_numpy_array(
                    np.asarray(adjacency_matrix_first, dtype=int),
                    create_using=nx.DiGraph,
                )
                clustering_first = float(nx.average_clustering(topology_graph))

            model = reservoir >> Ridge(ridge=ridge, fit_bias=False)
            model.fit(self._train_input, self._train_output, warmup=0)

            test_scores = np.asarray(model.run(self._test_input), dtype=np.float32)
            pred_test: List[int] = []
            start = 0
            for length in self._test_length:
                end = start + int(length)
                segment = test_scores[start:end, :]
                max_index = np.argmax(segment, axis=1)
                histogram = np.bincount(max_index, minlength=N_LABEL)
                pred_test.append(int(np.argmax(histogram)))
                start = end
            pred_test_array = np.asarray(pred_test, dtype=int)

            accuracy_values.append(
                float(accuracy_score(self._test_label, pred_test_array))
            )

        # 第1目的はノード数を使い、接続数近似へ切り替える場合は次の式を使う
        size_objective = reservoir_size
        # 接続数近似を使う場合の式: float((reservoir_size ** 2) * rc_connectivity)
        accuracy_mean = float(np.mean(accuracy_values))

        if adjacency_matrix_first is None:
            raise ValueError('adjacency_matrix must be captured at the first evaluation seed')
        average_shortest_path_length = compute_average_shortest_path_length(
            adjacency_matrix_first
        )
        if average_shortest_path_length is None:
            raise ValueError('average_shortest_path_length could not be computed')

        objective_values = (size_objective, accuracy_mean)
        metadata: NetworkMetrics = {
            'clustering': clustering_first,
            'adjacency_matrix': adjacency_matrix_first,
            'average_shortest_path_length': average_shortest_path_length,
        }
        return (objective_values, metadata)

    def get_objective_names(self) -> Tuple[str, ...]:
        """
        Implement BaseTask.get_objective_names and return spoken digit axis labels as a tuple.
        BaseTask.get_objective_namesを実装し、発話数字認識の軸名をタプルで返す。
        """
        return ('Reservoir Node', f'accuracy_mean(n={self.n_eval_seeds})')

    def get_directions(self) -> Tuple:
        """
        Implement BaseTask.get_directions and return Direction.MINIMIZE and Direction.MAXIMIZE.
        BaseTask.get_directionsを実装し、Direction.MINIMIZEとDirection.MAXIMIZEをタプルで返す。
        """
        return (Direction.MINIMIZE, Direction.MAXIMIZE)

    def get_variables(self) -> VariableConfig:
        """
        Implement BaseTask.get_variables and return spoken digit variables as VariableConfig.
        BaseTask.get_variablesを実装し、発話数字認識の変数定義をVariableConfigで返す。
        """
        config = VariableConfig()

        # 整数変数
        config.add_integer('reservoir_size', 20, 100)

        # 連続変数
        config.add_continuous('spectral_radius', 0.3, 1.2)
        config.add_continuous('leaking_rate', 0.01, 1.0)
        config.add_continuous('rc_connectivity', 0.01, 0.5)
        config.add_continuous('input_connectivity', 0.01, 1.0)
        config.add_continuous('input_scaling_log10', 1.0, 3.3)
        config.add_continuous('ridge_log10', -5.0, 1.0)
        return config
