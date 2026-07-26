"""
Execution of multi-objective optimization tasks
多目的最適化タスクの実行
"""

import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from src.optimizer import create_optimizer
from src.tasks.core.base_task import BaseTask
from src.tasks.core.data_type_network_metrics import NetworkMetrics
from src.tasks.core.direction import Direction
from src.utils.common_messages import unsupported_value_message


class TaskRunner:
    """
    Execute a BaseTask through the configured multi-objective optimization algorithm.
    設定された多目的最適化アルゴリズムでBaseTaskを実行する。
    """

    def __init__(
        self,
        task: BaseTask,
        n_generations: int = 50,
        seed: int = 123456,
        optimizer_params: Optional[Dict[str, Any]] = None,
        on_generation_end: Optional[callable] = None,
    ):
        """
        Initialize task execution, normalization, and callback state.
        タスク実行、正規化、コールバックの状態を初期化する。
        """
        self.task = task
        self.n_generations = n_generations
        self.seed = seed
        self.optimizer_params = optimizer_params or {}
        self.optimizer_instance = None
        self.trials: List[Dict[str, Any]] = []
        self.on_generation_end = on_generation_end
        self._sign_multipliers: Optional[np.ndarray] = None
        self._solution_metadata: Dict[tuple, NetworkMetrics] = {}  # パラメータキー -> ネットワーク指標
        self._requested_normalization_mode = str(
            self.optimizer_params.get("normalization_mode", "static_first_generation")
        ).lower()
        supported_modes = {"none", "basic", "sdn", "static_first_generation"}
        if self._requested_normalization_mode not in supported_modes:
            raise ValueError(
                unsupported_value_message(
                    "normalization_mode",
                    self._requested_normalization_mode,
                    sorted(supported_modes),
                )
            )
        effective_default_mode = (
            "none" if self._requested_normalization_mode == "static_first_generation"
            else self._requested_normalization_mode
        )
        # static_first_generation は TaskRunner 側のみで処理し、optimizer へは none を渡す。
        self._mode_for_optimizer = effective_default_mode
        self._normalization_eps = float(self.optimizer_params.get("normalization_eps", 1.0))
        self._f1_static_upper_coeff = float(
            self.optimizer_params.get("f1_static_normalization_upper_coeff", 1.0)
        )
        self._f1_static_lower_coeff = float(
            self.optimizer_params.get("f1_static_normalization_lower_coeff", 1.0)
        )
        self._f2_static_upper_coeff = float(
            self.optimizer_params.get("f2_static_normalization_upper_coeff", 1.0)
        )
        self._f2_static_lower_coeff = float(
            self.optimizer_params.get("f2_static_normalization_lower_coeff", 1.0)
        )
        self._use_static_first_generation = (
            self._requested_normalization_mode == "static_first_generation"
        )
        if self._use_static_first_generation:
            self._validate_static_normalization_coeff(
                self._f1_static_upper_coeff,
                "f1_static_normalization_upper_coeff",
            )
            self._validate_static_normalization_coeff(
                self._f1_static_lower_coeff,
                "f1_static_normalization_lower_coeff",
            )
            self._validate_static_normalization_coeff(
                self._f2_static_upper_coeff,
                "f2_static_normalization_upper_coeff",
            )
            self._validate_static_normalization_coeff(
                self._f2_static_lower_coeff,
                "f2_static_normalization_lower_coeff",
            )
        self._static_ranges_initialized = False
        self._static_lower_bounds: Optional[np.ndarray] = None
        self._static_denominators: Optional[np.ndarray] = None

    def _init_directions(self, directions: Tuple) -> None:
        """
        Initialize objective directions and minimization sign multipliers.
        目的関数の方向と最小化用の符号係数を初期化する。
        """
        multipliers = []
        for d in directions:
            if d == Direction.MAXIMIZE or d == "maximize":
                multipliers.append(-1.0)
            else:
                multipliers.append(1.0)
        self._sign_multipliers = np.array(multipliers)

    def _params_to_key(self, params: Dict[str, Any]) -> tuple:
        """
        Convert parameters into a hashable key for retrieving each solution's NetworkMetrics.
        各解のNetworkMetricsを検索するため、パラメータをハッシュ可能なキーへ変換して返す。
        """
        return tuple(sorted((k, float(v) if isinstance(v, (int, float)) else v)
                           for k, v in params.items()))

    @staticmethod
    def _validate_objective_values(raw_values: Any, num_objectives: int) -> Tuple[float, ...]:
        """
        Validate objective count and numeric values, then return them as a float tuple.
        目的関数の数と数値を検証し、floatへ変換したタプルを返す。
        """
        if not isinstance(raw_values, (tuple, list)):
            raise ValueError("objective_function must return objective values as tuple/list")

        values: List[float] = []
        for value in raw_values:
            try:
                value_float = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"objective value must be numeric, got: {value!r}") from exc
            values.append(value_float)

        if len(values) != num_objectives:
            raise ValueError(
                "objective value count mismatch: "
                f"expected {num_objectives}, got {len(values)}"
            )
        return tuple(values)

    @staticmethod
    def _validate_network_metrics(metadata: Any) -> Optional[NetworkMetrics]:
        """
        Validate NetworkMetrics keys and values, then return converted metrics or None.
        NetworkMetricsのキーと値を検証し、型を揃えた指標またはNoneを返す。
        """
        if metadata is None:
            return None
        if not isinstance(metadata, dict):
            raise ValueError("network metrics must be dict or None")

        required_keys = {
            "clustering",
            "average_shortest_path_length",
            "adjacency_matrix",
        }
        actual_keys = set(metadata.keys())
        missing = required_keys - actual_keys
        extra = actual_keys - required_keys
        if missing or extra:
            message_parts = []
            if missing:
                message_parts.append(f"missing keys: {sorted(missing)}")
            if extra:
                message_parts.append(f"unsupported keys: {sorted(extra)}")
            raise ValueError(f"invalid network metrics keys ({', '.join(message_parts)})")

        try:
            clustering = float(metadata["clustering"])
            average_shortest_path_length = float(metadata["average_shortest_path_length"])
        except (TypeError, ValueError) as exc:
            raise ValueError("network metrics values must be numeric") from exc

        if not np.isfinite(clustering):
            raise ValueError("network metric 'clustering' must be finite")
        if not np.isfinite(average_shortest_path_length):
            raise ValueError("network metric 'average_shortest_path_length' must be finite")

        adjacency_matrix = metadata["adjacency_matrix"]
        if not isinstance(adjacency_matrix, (list, tuple)):
            raise ValueError("network metric 'adjacency_matrix' must be a 2D integer square matrix")
        matrix_size = len(adjacency_matrix)
        if matrix_size == 0:
            raise ValueError("network metric 'adjacency_matrix' must not be empty")

        normalized_matrix: List[List[int]] = []
        for row in adjacency_matrix:
            if not isinstance(row, (list, tuple)) or len(row) != matrix_size:
                raise ValueError("network metric 'adjacency_matrix' must be square")
            normalized_row: List[int] = []
            for value in row:
                if not isinstance(value, (int, np.integer)):
                    raise ValueError(
                        "network metric 'adjacency_matrix' must contain integer values"
                    )
                normalized_row.append(int(value))
            normalized_matrix.append(normalized_row)

        validated_metrics: NetworkMetrics = {
            "clustering": clustering,
            "average_shortest_path_length": average_shortest_path_length,
            "adjacency_matrix": normalized_matrix,
        }
        return validated_metrics

    @staticmethod
    def _validate_static_normalization_coeff(coeff: float, name: str) -> None:
        """
        Validate a static-first-generation axis expansion coefficient.
        static_first_generationの軸範囲拡張係数を検証する。
        """
        if not np.isfinite(coeff):
            raise ValueError(f"{name} must be finite for static_first_generation")
        if coeff < 1.0:
            raise ValueError(
                f"{name} must be greater than or equal to 1.0 for static_first_generation"
            )

    @staticmethod
    def _expand_static_lower_bound(lb: float, lower_coeff: float) -> float:
        """
        Expand and return a lower bound only in the downward direction.
        下限を下方向だけに拡張して返す。
        """
        if lb > 0.0:
            return lb / lower_coeff
        if lb < 0.0:
            return lb * lower_coeff
        return 0.0

    @staticmethod
    def _expand_static_upper_bound(ub: float, upper_coeff: float) -> float:
        """
        Expand and return an upper bound only in the upward direction.
        上限を上方向だけに拡張して返す。
        """
        if ub > 0.0:
            return ub * upper_coeff
        if ub < 0.0:
            return ub / upper_coeff
        return 0.0

    @staticmethod
    def _print_static_normalization_ranges(
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        expanded_lower_bounds: np.ndarray,
        expanded_upper_bounds: np.ndarray,
    ) -> None:
        """
        Print original and expanded static normalization ranges.
        元の固定正規化範囲と拡張後の範囲を表示する。
        """
        # 正規化範囲 gen0で生成した個体を基準値
        for idx in range(len(lower_bounds)):
            print(
                f"f{idx + 1} normalization range(gen 0): "
                f"[{float(lower_bounds[idx]):.6g}, {float(upper_bounds[idx]):.6g}] -> "
                f"[{float(expanded_lower_bounds[idx]):.6g}, "
                f"{float(expanded_upper_bounds[idx]):.6g}](adjusted)"
            )

    def _initialize_static_ranges(
        self,
        external_values: List[Tuple[float, ...]],
        num_objectives: int,
    ) -> None:
        """
        Initialize static normalization ranges from generation zero values.
        第0世代の評価値から固定正規化範囲を初期化する。
        """
        if not external_values:
            raise ValueError("static_first_generation requires non-empty generation 0 objective values")

        matrix = np.asarray(external_values, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != num_objectives:
            raise ValueError(
                "static_first_generation objective value shape mismatch: "
                f"expected second dim {num_objectives}, got {matrix.shape}"
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError("static_first_generation objective values must be finite in generation 0")
        if self._normalization_eps <= 0.0:
            raise ValueError("normalization_eps must be greater than 0 for static_first_generation")

        lower_bounds = np.min(matrix, axis=0)
        upper_bounds = np.max(matrix, axis=0)
        upper_coeffs = np.ones(num_objectives, dtype=float)
        lower_coeffs = np.ones(num_objectives, dtype=float)
        if num_objectives > 0:
            upper_coeffs[0] = self._f1_static_upper_coeff
            lower_coeffs[0] = self._f1_static_lower_coeff
        if num_objectives > 1:
            upper_coeffs[1] = self._f2_static_upper_coeff
            lower_coeffs[1] = self._f2_static_lower_coeff

        expanded_lower_bounds = lower_bounds.copy()
        expanded_upper_bounds = upper_bounds.copy()
        for idx in range(num_objectives):
            expanded_lower_bounds[idx] = self._expand_static_lower_bound(
                lb=float(lower_bounds[idx]),
                lower_coeff=float(lower_coeffs[idx]),
            )
            expanded_upper_bounds[idx] = self._expand_static_upper_bound(
                ub=float(upper_bounds[idx]),
                upper_coeff=float(upper_coeffs[idx]),
            )

        denominators = np.maximum(
            expanded_upper_bounds - expanded_lower_bounds,
            self._normalization_eps,
        )
        self._print_static_normalization_ranges(
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            expanded_lower_bounds=expanded_lower_bounds,
            expanded_upper_bounds=expanded_upper_bounds,
        )
        raw_ranges = upper_bounds - lower_bounds
        eps_dominated_objectives = np.where(raw_ranges < self._normalization_eps)[0]
        if eps_dominated_objectives.size > 0:
            details = ", ".join(
                (
                    f"f{idx + 1}(raw_range={float(raw_ranges[idx]):.6g}, "
                    f"eps={self._normalization_eps:.6g}, "
                    f"denominator={float(denominators[idx]):.6g})"
                )
                for idx in eps_dominated_objectives
            )
            warnings.warn(
                "static_first_generation normalization range is epsilon-dominated: "
                f"{details}. Consider reducing normalization_eps for better axis balance.",
                RuntimeWarning,
                stacklevel=2,
            )

        self._static_lower_bounds = expanded_lower_bounds
        self._static_denominators = denominators
        self._static_ranges_initialized = True

    def _apply_static_normalization(self, values: Tuple[float, ...]) -> Tuple[float, ...]:
        """
        Apply static-range normalization and return an objective tuple.
        固定範囲正規化を適用して目的関数値タプルを返す。
        """
        if not self._use_static_first_generation:
            return values
        if not self._static_ranges_initialized:
            raise ValueError("static_first_generation ranges are not initialized")

        value_array = np.asarray(values, dtype=float)
        if self._static_lower_bounds is None or self._static_denominators is None:
            raise ValueError("static_first_generation range buffers are missing")

        normalized = (value_array - self._static_lower_bounds) / self._static_denominators
        return tuple(float(v) for v in normalized)

    def _apply_static_denormalization(self, values: np.ndarray) -> np.ndarray:
        """
        Apply static-range denormalization and return original-scale values.
        固定範囲の逆正規化を適用して元尺度の値を返す。
        """
        if not self._use_static_first_generation:
            return values
        if not self._static_ranges_initialized:
            raise ValueError("static_first_generation ranges are not initialized")
        if self._static_lower_bounds is None or self._static_denominators is None:
            raise ValueError("static_first_generation range buffers are missing")

        value_array = np.asarray(values, dtype=float)
        return value_array * self._static_denominators + self._static_lower_bounds

    def _to_internal(self, values: Tuple[float, ...]) -> Tuple[float, ...]:
        """
        Apply configured normalization and convert task objectives to minimization.
        タスクが返した目的関数値へ設定された正規化を適用し、全目的を最小化方向へ変換して返す。
        """
        normalized_values = self._apply_static_normalization(values)

        # 方向変換
        if self._sign_multipliers is None:
            return normalized_values
        return tuple(v * m for v, m in zip(normalized_values, self._sign_multipliers))

    def _to_external(self, values: np.ndarray) -> np.ndarray:
        """
        Restore optimizer objectives to task-defined directions and scale.
        多目的最適化アルゴリズムの目的関数値をタスク定義の方向と尺度へ戻して返す。
        """
        if self._sign_multipliers is None:
            restored = np.asarray(values, dtype=float)
        else:
            restored = np.asarray(values, dtype=float) * self._sign_multipliers
        return self._apply_static_denormalization(restored)

    def optimize(self) -> Dict[str, Any]:
        """
        Execute multi-objective optimization and return trials, Pareto results, and metadata.
        多目的最適化を実行し、試行、パレート結果、メタデータを含む辞書を返す。
        """
        start_time = time.time()

        num_objectives = len(self.task.get_objective_names())
        max_generations = self.optimizer_params.get("max_generations", self.n_generations)
        population_size = self.optimizer_params.get("population_size", 100)
        crossover_type = self.optimizer_params.get("crossover_type", "SBX")
        scalar_function_type = self.optimizer_params.get("scalar_function_type", "TCH")
        neighborhood_size = self.optimizer_params.get("neighborhood_size", 20)
        delta = self.optimizer_params.get("delta", 0.9)
        nr = self.optimizer_params.get("nr", 1)
        eta_c = self.optimizer_params.get("eta_c", 20)
        eta_m = self.optimizer_params.get("eta_m", 20)
        use_polynomial_mutation = self.optimizer_params.get("use_polynomial_mutation", True)
        use_stochastic_rounding = self.optimizer_params.get("use_stochastic_rounding", True)
        gen_num_child = self.optimizer_params.get("gen_num_child", 1)
        is_save_external_population = self.optimizer_params.get("is_save_external_population", False)
        normalization_mode = self._mode_for_optimizer
        sdn_final_generation = self.optimizer_params.get("sdn_final_generation")
        normalization_eps = self.optimizer_params.get("normalization_eps", 1.0)
        boundary_repair_mode = self.optimizer_params.get("boundary_repair_mode", "clip")
        boundary_repair_reflect_ratio = self.optimizer_params.get(
            "boundary_repair_reflect_ratio", 0.3
        )
        pbi_theta = self.optimizer_params.get("pbi_theta", 5.0)
        undx_alpha = self.optimizer_params.get("undx_alpha", 0.5)
        undx_beta = self.optimizer_params.get("undx_beta", 0.35)
        undx_m_num_parents = self.optimizer_params.get("undx_m_num_parents", 2)
        # None の場合は MOEAD.setup() 側で自動計算
        undx_m_sigma_xi = self.optimizer_params.get("undx_m_sigma_xi")
        undx_m_sigma_eta = self.optimizer_params.get("undx_m_sigma_eta")
        spx_epsilon = self.optimizer_params.get("spx_epsilon")
        tblx_alpha = self.optimizer_params.get("tblx_alpha", 0.5)

        if self._use_static_first_generation and self._normalization_eps <= 0.0:
            raise ValueError("normalization_eps must be greater than 0 for static_first_generation")

        var_config = self.task.get_variables()

        # タスクから最適化方向を取得
        self._init_directions(self.task.get_directions())

        self.optimizer_instance = create_optimizer("MOEAD")
        self.optimizer_instance.setup(
            var_config=var_config,
            num_objectives=num_objectives,
            population_size=population_size,
            scalar_function_type=scalar_function_type,
            neighborhood_size=neighborhood_size,
            nr=nr,
            pbi_theta=pbi_theta,
            delta=delta,
            is_save_external_population=is_save_external_population,
            num_children_per_crossover=gen_num_child,
            use_polynomial_mutation=use_polynomial_mutation,
            use_stochastic_rounding=use_stochastic_rounding,
            eta_c=eta_c,
            eta_m=eta_m,
            crossover_type=crossover_type,
            undx_alpha=undx_alpha,
            undx_beta=undx_beta,
            undx_m_num_parents=undx_m_num_parents,
            undx_m_sigma_xi=undx_m_sigma_xi,
            undx_m_sigma_eta=undx_m_sigma_eta,
            spx_epsilon=spx_epsilon,
            tblx_alpha=tblx_alpha,
            normalization_mode=normalization_mode,
            sdn_final_generation=sdn_final_generation,
            normalization_eps=normalization_eps,
            boundary_repair_mode=boundary_repair_mode,
            boundary_repair_reflect_ratio=boundary_repair_reflect_ratio,
            seed=self.seed,
            verbose=False,
        )

        self.trials = []
        self._solution_metadata = {}  # 各世代でリセット
        self._static_ranges_initialized = False
        self._static_lower_bounds = None
        self._static_denominators = None

        def _run_generation(generation: int) -> None:
            """
            Evaluate one generation and feed its values to the multi-objective optimization algorithm.
            1世代を評価し、その値を多目的最適化アルゴリズムへ渡す。
            """
            solutions = self.optimizer_instance.ask()
            # objective_function は (Tuple[float, ...], Optional[NetworkMetrics]) を返す
            # 各候補パラメータをタスクの目的関数で評価する。
            results = []
            for params in solutions:
                results.append(self.task.objective_function(params))

            external_values: List[Tuple[float, ...]] = []
            for params, result in zip(solutions, results):
                if not isinstance(result, tuple) or len(result) != 2:
                    raise ValueError(
                        "objective_function must return exactly two elements: "
                        "(objective_values, network_metrics_or_none)"
                    )

                values = self._validate_objective_values(result[0], num_objectives)
                metadata = self._validate_network_metrics(result[1])

                external_values.append(values)
                if metadata is not None:
                    params_key = self._params_to_key(params)
                    self._solution_metadata[params_key] = metadata

            if self._use_static_first_generation and not self._static_ranges_initialized:
                self._initialize_static_ranges(external_values, num_objectives)

            # タスクが返した目的関数値を、設定に応じて正規化して全目的を最小化方向へ変換する。
            internal_values = []
            for values in external_values:
                internal_values.append(self._to_internal(values))
            self.optimizer_instance.tell(internal_values)

            # 試行記録にはタスクが返した目的関数値を保存する。
            for params, values in zip(solutions, external_values):
                self.trials.append({"params": params, "values": values})

            # コールバック呼び出し（tell後の状態を渡す）
            if self.on_generation_end:
                result_population = self.optimizer_instance.get_result()
                # 各解の決定変数と、タスク定義の方向・尺度へ戻した目的関数値を収集する。
                population_solutions = []
                population_objectives = []
                for individual in result_population:
                    population_solutions.append(individual.params)
                    population_objectives.append(
                        tuple(self._to_external(individual.obj_values))
                    )

                # 各解のメタデータ（clustering等）を取得
                population_metadata: List[Optional[NetworkMetrics]] = []
                for params in population_solutions:
                    params_key = self._params_to_key(params)
                    metadata = self._solution_metadata.get(params_key)
                    population_metadata.append(metadata)

                callback_metadata = (
                    population_metadata
                    if any(meta is not None for meta in population_metadata)
                    else None
                )

                self.on_generation_end(
                    generation=generation,
                    solutions=population_solutions,
                    objective_values=population_objectives,
                    pareto_size=len(result_population),
                    metadata=callback_metadata
                )

        first_progress_generation = 0
        initial_progress = 0
        if self._use_static_first_generation and max_generations > 0:
            _run_generation(0)
            first_progress_generation = 1
            initial_progress = 1

        if max_generations > 0:
            with tqdm(
                total=max_generations,
                initial=initial_progress,
                desc="Generation",
                dynamic_ncols=True,
                file=sys.stdout,
            ) as progress:
                for generation in range(first_progress_generation, max_generations):
                    _run_generation(generation)
                    progress.update(1)

        optimization_time = time.time() - start_time
        result_population = self.optimizer_instance.get_result()
        # 多目的最適化アルゴリズムの値をタスク定義の方向と尺度へ戻す。
        pareto_values = []
        for individual in result_population:
            pareto_values.append(self._to_external(individual.obj_values))
        pareto_front = np.array(pareto_values)

        results = {
            "task": self.task,
            "study": None,
            "trials": self.trials,
            "pareto_front": pareto_front,
            "optimization_time": optimization_time,
            "n_trials": len(self.trials),
            "optimizer_result": result_population,
        }

        print("=== Optimization Results ===")
        print(f"Optimization time: {optimization_time:.2f} seconds")
        return results

    def get_pareto_solutions(self) -> List[Dict[str, Any]]:
        """
        Return Pareto solutions with objectives restored to task-defined directions and scale.
        目的関数値をタスク定義の方向と尺度へ戻し、パレート解の辞書として返す。
        """
        pareto_solutions: List[Dict[str, Any]] = []
        if self.optimizer_instance is None:
            return pareto_solutions

        result = self.optimizer_instance.get_result()
        for ind in result:
            # 多目的最適化アルゴリズムの値をタスク定義の方向と尺度へ戻す。
            external_obj = self._to_external(ind.obj_values)
            solution = {
                "params": ind.params,
                "objectives": external_obj,
                "trial": ind,
            }
            pareto_solutions.append(solution)

        return pareto_solutions

