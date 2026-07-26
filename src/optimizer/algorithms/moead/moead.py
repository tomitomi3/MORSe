"""
MOEA/D Multiobjective Evolutionary Algorithm Based on Decomposition algorithm
多目的進化アルゴリズム
"""

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from src.optimizer.algorithms.moead.crossover import CrossoverOperator
from src.optimizer.algorithms.moead.repair import RepairOperator
from src.optimizer.algorithms.moead.solution import Solution
from src.optimizer.algorithms.moead.subproblem import Subproblem
from src.optimizer.algorithms.moead.utils import EPS, random_sample
from src.optimizer.core.base_optimizer import BaseOptimizer
from src.optimizer.core.variable_config import VariableConfig
from src.utils.common_messages import unsupported_value_message


class MOEAD(BaseOptimizer):
    """
    MOEA/D algorithm supporting the Ask-Tell interface
    MOEA/Dアルゴリズム Ask-Tellインターフェース対応ver
    """

    def __init__(self):
        """
        Initialize empty MOEA/D parameters, populations, and Ask-Tell state.
        空のMOEA/Dパラメータ、集団、Ask-Tell状態を初期化する。
        """
        # Reference:
        # ZHANG, Qingfu; LI, Hui. MOEA/D: A multiobjective evolutionary algorithm based on decomposition. IEEE Transactions on evolutionary computation, 2007, 11.6: 712-731.

        # 基本パラメータ
        self.var_config: Optional[VariableConfig] = None
        self.num_objectives: int = 0
        self.scalar_function_type: str = ""
        self.nr: int = 0
        self.pbi_theta: float = 0.0
        self.eta_c: int = 0
        self.eta_m: int = 0
        self.neighborhood_size: int = 0
        self.is_save_external_population: bool = True
        self.rng: np.random.Generator = None  # type: ignore
        self.eps: float = EPS
        self.normalization_mode: str = "none"
        self.sdn_final_generation: Optional[int] = None
        # 論文では ε=1 も比較されるため、既定値は 1.0 を採用する（問題依存）。
        self.normalization_eps: float = 1.0
        self.boundary_repair_mode: str = "clip"
        self.boundary_repair_reflect_ratio: float = 0.2

        # 交叉パラメータ
        self.crossover_type: str = "SBX"
        self.undx_alpha: float = 0.5
        self.undx_beta: float = 0.35
        self.undx_m_num_parents: int = 2
        self.undx_m_sigma_xi: float = 0.0
        self.undx_m_sigma_eta: float = 0.0
        self.spx_epsilon: float = 0.0
        self.tblx_alpha: float = 0.5
        self.use_polynomial_mutation: bool = True
        self.use_stochastic_rounding: bool = True
        self.delta: float = 0.9

        # 集団管理
        self.num_children_per_crossover: int = 1
        self.ideal_point_global: np.ndarray = None  # type: ignore
        self.indiv_point: List[Solution] = []
        self.population: List[Subproblem] = []
        self.population_size: int = 0
        self.external_population: List[Solution] = []
        self.nadir_point_current: np.ndarray = None  # type: ignore

        # Ask-and-Tell用の状態管理
        self._is_setup_called: bool = False
        self._is_first_cycle: bool = True
        self._pending_solutions: List[Dict[str, Any]] = []
        self._pending_metadata: List[dict] = []
        self._has_pending: bool = False
        self._generation: int = 0
        self.verbose: bool = False

        # オペレータ (setup() で初期化)
        self._crossover: Optional[CrossoverOperator] = None
        self._repair: Optional[RepairOperator] = None

    @staticmethod
    def population_size_to_subdivisions(population_size: int, num_objectives: int) -> int:
        """
        Calculate and return weight subdivisions for a population size.
        集団サイズから重み分割数を計算して返す。
        """
        
        ## MOEA/D 2 or 3目的としておく
        if num_objectives == 2:
            return population_size - 1
        elif num_objectives == 3:
            s = 1
            while (s + 1) * (s + 2) // 2 < population_size:
                s += 1
            if (s + 1) * (s + 2) // 2 > population_size:
                s -= 1
            return s
        else:
            raise ValueError(
                f"num_objectives must be 2 or 3, but got {num_objectives}"
            )

    @staticmethod
    def generate_weight_vectors(subdivisions: int, num_objectives: int) -> List[np.ndarray]:
        """
        Generate and return weight vectors for scalarizing each subproblem.
        各サブ問題のスカラー化に使用する重みベクトルを生成して返す。
        """
        weights = []
        if num_objectives == 2:
            for i in range(subdivisions + 1):
                weight = np.array([i / subdivisions, (subdivisions - i) / subdivisions])
                weights.append(weight)
        elif num_objectives == 3:
            for i in range(subdivisions + 1):
                for j in range(subdivisions + 1):
                    if i + j <= subdivisions:
                        k = subdivisions - i - j
                        weight = np.array([i / subdivisions, j / subdivisions, k / subdivisions])
                        weights.append(weight)
        return weights

    def update_ideal_points(self, solution: Solution):
        """
        Update the global ideal point from a candidate solution.
        候補解から大域的理想点を更新する。
        """
        for i in range(self.num_objectives):
            if solution.obj_values[i] < self.ideal_point_global[i]:
                self.ideal_point_global[i] = solution.obj_values[i]
                self.indiv_point[i].params = solution.params.copy()
                self.indiv_point[i].obj_values = solution.obj_values.copy()

    def _validate_normalization_settings(self) -> None:
        """
        Validate the configured objective normalization settings.
        設定された目的関数正規化条件を検証する。
        """
        if self.normalization_mode not in {"none", "basic", "sdn"}:
            raise ValueError(
                f"normalization_mode must be one of ['none', 'basic', 'sdn'], got '{self.normalization_mode}'"
            )

        if self.normalization_mode == "sdn":
            if self.sdn_final_generation is None:
                raise ValueError("sdn_final_generation is required when normalization_mode='sdn'")
            if self.sdn_final_generation <= 1:
                raise ValueError("sdn_final_generation must be greater than 1")

    def _extract_nondominated_objectives(self) -> np.ndarray:
        """
        Extract and return nondominated objectives from the current population.
        現在の集団から非支配目的関数値を抽出して返す。
        """
        if self.population_size == 0:
            return np.empty((0, self.num_objectives))

        # 現在の各解について、非支配判定に使う目的関数値をまとめる。
        objective_rows = []
        for subproblem in self.population:
            objective_rows.append(subproblem.solution.obj_values)
        obj_matrix = np.array(objective_rows, dtype=float)
        nd_mask = np.ones(self.population_size, dtype=bool)

        for i in range(self.population_size):
            if not nd_mask[i]:
                continue
            for j in range(self.population_size):
                if i == j:
                    continue
                if self._dominates(obj_matrix[j], obj_matrix[i], self.eps):
                    nd_mask[i] = False
                    break

        return obj_matrix[nd_mask]

    def _compute_sdn_alpha(self) -> float:
        """
        Compute and return the generation-dependent SDN alpha value.
        世代依存のSDN alpha値を計算して返す。
        """
        if self.sdn_final_generation is None or self.sdn_final_generation <= 1:
            raise ValueError("sdn_final_generation must be greater than 1 for SDN mode")

        # MOEA/Dの多目的最適化における動的正規化式
        # 8もパラメータ
        t = self._generation + 1
        T = self.sdn_final_generation
        ratio = (t - 1.0) / float(T - 1)
        return float(1.0 / (1.0 + np.exp(-8.0 * (ratio - 0.5))))

    def _update_normalization_state(self) -> None:
        """
        Update the current nadir estimate used for normalization.
        正規化に使用する現在の最悪点推定値を更新する。
        """
        if self.normalization_mode == "none":
            return

        nd_objectives = self._extract_nondominated_objectives()
        if nd_objectives.shape[0] == 0:
            self.nadir_point_current = self.ideal_point_global.copy()
        else:
            self.nadir_point_current = np.max(nd_objectives, axis=0)

    def _get_scalarization_diff(self, obj_values: np.ndarray) -> np.ndarray:
        """
        Return ideal-point differences scaled by the configured normalization.
        目的関数値と理想点の差を求め、設定された正規化を適用して返す。
        """
        obj_array = np.asarray(obj_values, dtype=float)
        diff = obj_array - self.ideal_point_global

        if self.normalization_mode == "none":
            return diff

        if self.nadir_point_current is None:
            return diff

        ranges = self.nadir_point_current - self.ideal_point_global

        if self.normalization_mode == "basic":
            denominator = np.maximum(ranges + self.normalization_eps, self.normalization_eps)
            return diff / denominator

        alpha = self._compute_sdn_alpha()

        l_denominator = (1.0 - alpha) * (ranges - 1.0) + 1.0
        safe_l_denominator = np.where(
            np.abs(l_denominator) < self.normalization_eps,
            self.normalization_eps,
            l_denominator
        )
        L = ranges / safe_l_denominator
        denominator = np.maximum(L + self.normalization_eps, self.normalization_eps)
        return diff / denominator

    def scalar_func(self, obj_values: np.ndarray, weight: np.ndarray) -> float:
        """
        Evaluate and return the configured scalarization value.
        設定されたスカラー化関数を評価して値を返す。
        """
        diff = self._get_scalarization_diff(obj_values)

        if self.scalar_function_type == "TCH":
            diff = np.abs(diff)
            w = weight.copy()
            for i in range(len(w)):
                if np.isclose(w[i], 0.0):
                    w[i] = 1e-12
            return float(np.max(diff * w))
        elif self.scalar_function_type == "PBI":
            norm_weight = weight / np.linalg.norm(weight)
            d1 = abs(np.dot(diff, norm_weight))
            d2_vec = diff - d1 * norm_weight
            d2 = np.linalg.norm(d2_vec)
            return d1 + self.pbi_theta * d2
        else:
            raise ValueError(
                unsupported_value_message(
                    "scalar_function_type",
                    self.scalar_function_type,
                    ["TCH", "PBI"],
                )
            )

    @staticmethod
    def _dominates(obj_a: np.ndarray, obj_b: np.ndarray, eps: float = EPS) -> bool:
        """
        Return whether one objective vector dominates another.
        一方の目的関数ベクトルが他方を支配するかどうかを返す。
        """
        return bool(np.all(obj_a <= obj_b + eps) and np.any(obj_a < obj_b - eps))

    def update_external_population(self, solution: Solution):
        """
        Update the nondominated solution archive retained across generations.
        候補解を使い、世代をまたいで保持する非支配解集合を更新する。
        """
        if not self.is_save_external_population:
            return

        candidate_objs = solution.obj_values

        for existing in self.external_population:
            if self._dominates(existing.obj_values, candidate_objs, self.eps):
                return

        filtered = []
        for existing in self.external_population:
            if not self._dominates(candidate_objs, existing.obj_values, self.eps):
                filtered.append(existing)

        clone = Solution(self.num_objectives)
        clone.params = solution.params.copy()
        clone.obj_values = solution.obj_values.copy()
        filtered.append(clone)
        self.external_population = filtered

    def repair_solution(self, solution: Solution):
        """
        Repair a solution in place to satisfy variable bounds.
        変数境界を満たすように解をその場で修復する。
        """
        if not self._is_setup_called:
            # setup() 未呼び出し時はオンデマンドで生成（テスト互換）
            RepairOperator(
                rng=self.rng,
                var_config=self.var_config,
                boundary_repair_mode=self.boundary_repair_mode,
                boundary_repair_reflect_ratio=self.boundary_repair_reflect_ratio,
                use_stochastic_rounding=self.use_stochastic_rounding,
            ).repair_solution(solution)
            return
        self._repair.repair_solution(solution)

    def update_neighbors(self, child: Solution, subproblem_id: int):
        """
        Update neighboring subproblem solutions with an offspring.
        子の解を使用して近傍サブ問題の解を更新する。
        """
        replaced_count = 0
        for neighbor_index in self.population[subproblem_id].neighbors:
            neighbor = self.population[neighbor_index]
            current_scalar = self.scalar_func(neighbor.solution.obj_values, neighbor.weight)
            child_scalar = self.scalar_func(child.obj_values, neighbor.weight)

            if child_scalar < current_scalar:
                neighbor.solution.params = child.params.copy()
                neighbor.solution.obj_values = child.obj_values.copy()
                replaced_count += 1

                if replaced_count >= self.nr:
                    break

    def setup(self,
              var_config: VariableConfig,
              num_objectives: int = 2,
              population_size: Optional[int] = None,
              subdivisions: int = 99,
              scalar_function_type: str = "TCH",
              seed: Optional[int] = None,
              nr: int = 1,
              pbi_theta: float = 5.0,
              is_save_external_population: bool = False,
              eta_c: int = 20,
              eta_m: int = 20,
              neighborhood_size: int = 20,
              crossover_type: str = "SBX",
              undx_alpha: float = 0.5,
              undx_beta: float = 0.35,
              undx_m_num_parents: int = 2,
              undx_m_sigma_xi: Optional[float] = None,
              undx_m_sigma_eta: Optional[float] = None,
              spx_epsilon: Optional[float] = None,
              tblx_alpha: float = 0.5,
              delta: float = 0.9,
              num_children_per_crossover: int = 1,
              use_polynomial_mutation: bool = True,
              use_stochastic_rounding: bool = True,
              normalization_mode: str = "none",
              sdn_final_generation: Optional[int] = None,
              normalization_eps: float = 1.0,
              boundary_repair_mode: str = "clip",
              boundary_repair_reflect_ratio: float = 0.3,
              verbose: bool = False,
              **kwargs) -> None:
        """
        Implement BaseOptimizer.setup and initialize MOEA/D from parameters.
        BaseOptimizer.setupを実装し、パラメータからMOEA/Dを初期化する。
        """
        self.var_config = var_config
        self.num_objectives = num_objectives
        if population_size is not None:
            subdivisions = self.population_size_to_subdivisions(population_size, num_objectives)

        self.scalar_function_type = scalar_function_type
        self.nr = nr
        self.pbi_theta = pbi_theta
        self.eta_c = eta_c
        self.eta_m = eta_m
        self.neighborhood_size = neighborhood_size
        self.is_save_external_population = is_save_external_population

        # rngの初期化
        self.rng = np.random.default_rng(seed)

        self.crossover_type = crossover_type
        self.undx_alpha = undx_alpha
        self.undx_beta = undx_beta
        self.undx_m_num_parents = undx_m_num_parents
        self.tblx_alpha = tblx_alpha
        self.delta = delta
        self.num_children_per_crossover = num_children_per_crossover
        self.use_polynomial_mutation = use_polynomial_mutation
        self.use_stochastic_rounding = bool(use_stochastic_rounding)
        self.normalization_mode = normalization_mode.lower()
        self.sdn_final_generation = sdn_final_generation
        self.normalization_eps = normalization_eps
        self.boundary_repair_mode = str(boundary_repair_mode).lower()
        self.boundary_repair_reflect_ratio = float(boundary_repair_reflect_ratio)
        self.verbose = verbose
        self._validate_normalization_settings()

        n_numeric = len(var_config.get_continuous_names()) + len(var_config.get_integer_names())

        m = undx_m_num_parents
        if undx_m_sigma_xi is None:
            self.undx_m_sigma_xi = 1.0 / np.sqrt(m) if m > 0 else 1.0
        else:
            self.undx_m_sigma_xi = undx_m_sigma_xi

        if undx_m_sigma_eta is None:
            n_minus_m = max(1, n_numeric - m)
            self.undx_m_sigma_eta = 0.35 / np.sqrt(n_minus_m)
        else:
            self.undx_m_sigma_eta = undx_m_sigma_eta

        if spx_epsilon is None:
            self.spx_epsilon = np.sqrt(n_numeric + 2)
        else:
            self.spx_epsilon = spx_epsilon

        if self.verbose:
            print(f"MOEA/D setup: {self.num_objectives} objectives")
            print(f"Variables: {self.var_config.get_num_variables()} "
                  f"(continuous: {len(self.var_config.get_continuous_names())}, "
                  f"integer: {len(self.var_config.get_integer_names())}, "
                  f"categorical: {len(self.var_config.get_categorical_names())})")

        self.ideal_point_global = np.full(self.num_objectives, float('inf'))
        self.indiv_point = []
        for _ in range(self.num_objectives):
            ind = Solution(self.num_objectives)
            self.indiv_point.append(ind)

        self.external_population = []
        self.nadir_point_current = np.full(self.num_objectives, float('inf'))

        weights = self.generate_weight_vectors(subdivisions, self.num_objectives)
        self.population_size = len(weights)

        self.population = []
        for weight in weights:
            subproblem = Subproblem()
            subproblem.weight = weight
            subproblem.solution = Solution(self.num_objectives)
            self.population.append(subproblem)

        if self.neighborhood_size < 2:
            self.neighborhood_size = 2
        if self.population_size < 2:
            raise ValueError(
                "Population size must be at least 2. Check subdivisions"
            )
        self.neighborhood_size = min(self.neighborhood_size, self.population_size)

        if self.verbose:
            print(f"Population size: {self.population_size}")

        for i in range(self.population_size):
            distances = []
            for j in range(self.population_size):
                dist = np.linalg.norm(self.population[i].weight - self.population[j].weight)
                distances.append((dist, j))
            distances.sort()
            # 距離が近い順に近傍サブ問題のインデックスを抽出する。
            neighbors_list = []
            for _, idx in distances[:self.neighborhood_size]:
                neighbors_list.append(idx)
            self.population[i].neighbors = neighbors_list

        for subproblem in self.population:
            subproblem.solution.params = self.var_config.generate_random_params(self.rng)

        # 交叉オペレータの生成
        self._crossover = CrossoverOperator(
            rng=self.rng,
            var_config=self.var_config,
            num_objectives=self.num_objectives,
            eta_c=self.eta_c,
            eta_m=self.eta_m,
            undx_alpha=self.undx_alpha,
            undx_beta=self.undx_beta,
            undx_m_sigma_xi=self.undx_m_sigma_xi,
            undx_m_sigma_eta=self.undx_m_sigma_eta,
            spx_epsilon=self.spx_epsilon,
            tblx_alpha=self.tblx_alpha,
            num_children_per_crossover=self.num_children_per_crossover,
            use_polynomial_mutation=self.use_polynomial_mutation,
        )

        self._repair = RepairOperator(
            rng=self.rng,
            var_config=self.var_config,
            boundary_repair_mode=self.boundary_repair_mode,
            boundary_repair_reflect_ratio=self.boundary_repair_reflect_ratio,
            use_stochastic_rounding=self.use_stochastic_rounding,
        )

        self._is_setup_called = True
        self._is_first_cycle = True
        self._has_pending = False
        self._generation = 0

        if self.verbose:
            print("Setup is complete. Call ask() to create solutions.")

    def ask(self, objective_func: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """
        Implement BaseOptimizer.ask and return candidate parameter dictionaries.
        BaseOptimizer.askを実装し、候補パラメータ辞書を返す。
        """
        if not self._is_setup_called:
            raise RuntimeError("Optimizer not initialized. Call setup() first.")

        if self._has_pending:
            raise RuntimeError("Call tell() before ask()")

        if self._is_first_cycle:
            self._pending_solutions = []
            self._pending_metadata = []

            for i, subproblem in enumerate(self.population):
                self._pending_solutions.append(subproblem.solution.params.copy())
                self._pending_metadata.append({
                    'subproblem_id': i,
                    'is_initial': True
                })

            self._has_pending = True
            return self._pending_solutions

        self._pending_solutions = []
        self._pending_metadata = []
        rng_index = self.rng.permutation(self.population_size)

        for i in rng_index:
            neighbors = self.population[i].neighbors

            if self.rng.random() < self.delta:
                mating_pool = neighbors
            else:
                mating_pool = list(range(self.population_size))

            mutation_rate = 1.0 / self.var_config.get_num_variables()

            if self.crossover_type == "UNDX":
                num_parents = 3
                if len(mating_pool) >= num_parents:
                    idxs = random_sample(self.rng, mating_pool, num_parents)
                else:
                    idxs = self.rng.choice(self.population_size, size=num_parents, replace=False)
                parent1 = self.population[idxs[0]].solution
                parent2 = self.population[idxs[1]].solution
                parent3 = self.population[idxs[2]].solution
                children = self._crossover.crossover_undx(parent1, parent2, parent3, mutation_rate)

            elif self.crossover_type == "UNDX-m":
                num_parents = self.undx_m_num_parents + 2
                if len(mating_pool) >= num_parents:
                    idxs = random_sample(self.rng, mating_pool, num_parents)
                else:
                    idxs = self.rng.choice(self.population_size, size=num_parents, replace=False)
                # 選択したインデックスからUNDX-mの親解を収集する。
                parents = []
                for idx in idxs:
                    parents.append(self.population[idx].solution)
                children = self._crossover.crossover_undx_m(parents, mutation_rate)

            elif self.crossover_type == "SPX":
                n_numeric = len(self.var_config.get_continuous_names()) + len(self.var_config.get_integer_names())
                num_parents = n_numeric + 1
                if len(mating_pool) >= num_parents:
                    idxs = random_sample(self.rng, mating_pool, num_parents)
                else:
                    idxs = self.rng.choice(self.population_size, size=num_parents, replace=False)
                # 選択したインデックスからSPXの親解を収集する。
                parents = []
                for idx in idxs:
                    parents.append(self.population[idx].solution)
                children = self._crossover.crossover_spx(parents, mutation_rate)

            elif self.crossover_type == "TBLX":
                children = []
                for _ in range(self.num_children_per_crossover):
                    children.append(self._crossover.crossover_tblx(
                        i, mutation_rate, self.population, self.population_size
                    ))

            elif self.crossover_type == "SBX":
                num_parents = 2
                if len(mating_pool) >= num_parents:
                    idxs = random_sample(self.rng, mating_pool, num_parents)
                else:
                    idxs = self.rng.choice(self.population_size, size=num_parents, replace=False)
                parent1 = self.population[idxs[0]].solution
                parent2 = self.population[idxs[1]].solution
                children = []
                for _ in range(self.num_children_per_crossover):
                    children.append(self._crossover.crossover_sbx(parent1, parent2, mutation_rate))

            else:
                raise ValueError(
                    unsupported_value_message(
                        "crossover type",
                        self.crossover_type,
                    )
                )

            selected_children = []
            if objective_func is not None and len(children) > 1:
                best_child = None
                best_scalar = float('inf')
                weight = self.population[i].weight

                for child in children:
                    self.repair_solution(child)
                    obj_val = objective_func(child.params)
                    scalar = self.scalar_func(obj_val, weight)
                    if scalar < best_scalar:
                        best_scalar = scalar
                        best_child = child

                if best_child is not None:
                    selected_children.append(best_child)
            else:
                for child in children:
                    self.repair_solution(child)
                    selected_children.append(child)

            for child in selected_children:
                self._pending_solutions.append(child.params.copy())
                self._pending_metadata.append({
                    'subproblem_id': i,
                    'child': child,
                    'is_initial': False
                })

        self._has_pending = True
        return self._pending_solutions

    def tell(self, objective_values: List[np.ndarray]) -> None:
        """
        Implement BaseOptimizer.tell and update state from evaluated objectives.
        BaseOptimizer.tellを実装し、評価済み目的関数値から状態を更新する。
        """
        if not self._has_pending:
            raise RuntimeError("Call ask() before tell()")

        if len(objective_values) != len(self._pending_solutions):
            raise ValueError(
                f"Objective value count does not match: expected {len(self._pending_solutions)}, "
                f"got {len(objective_values)}"
            )

        if self._is_first_cycle:
            for idx, metadata in enumerate(self._pending_metadata):
                i = metadata['subproblem_id']
                obj_val = np.array(objective_values[idx])
                self.population[i].solution.obj_values = obj_val
                self.update_ideal_points(self.population[i].solution)
                self.update_external_population(self.population[i].solution)

            self._is_first_cycle = False
            self._has_pending = False
            self._update_normalization_state()
            self._generation += 1
            return

        for idx, metadata in enumerate(self._pending_metadata):
            i = metadata['subproblem_id']
            child = metadata['child']
            obj_val = np.array(objective_values[idx])
            child.obj_values = obj_val
            self.update_ideal_points(child)
            self.update_neighbors(child, i)
            self.update_external_population(child)

        self._has_pending = False
        self._update_normalization_state()
        self._generation += 1

    def get_result(self) -> List[Solution]:
        """
        Implement BaseOptimizer.get_result and return cloned result solutions.
        BaseOptimizer.get_resultを実装し、複製した結果解を返す。
        """
        solutions = []
        if self.external_population:
            # 世代をまたいで保持した非支配解集合がある場合は、その解を返す。
            solutions = self.external_population
        else:
            # 非支配解集合を保存しない場合は、最終世代を近似パレートフロントとして返す。
            for sub in self.population:
                solutions.append(sub.solution)

        result = []
        for solution in solutions:
            clone = Solution(self.num_objectives)
            clone.params = solution.params.copy()
            clone.obj_values = solution.obj_values.copy()
            result.append(clone)

        return result
