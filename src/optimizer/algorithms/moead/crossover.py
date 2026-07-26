"""
Crossover operators used by MOEA/D subproblems
MOEA/Dのサブ問題で用いる交叉オペレータ
"""

from typing import List, Tuple

import numpy as np

from src.optimizer.algorithms.moead.solution import Solution
from src.optimizer.algorithms.moead.subproblem import Subproblem
from src.optimizer.algorithms.moead.utils import EPS, random_select
from src.optimizer.core.variable_config import VariableConfig


class CrossoverOperator:
    """
    Generate offspring with supported mixed-variable crossover methods.
    対応する混合変数交叉方式で子個体を生成する。
    """

    def __init__(
        self,
        rng: np.random.Generator,
        var_config: VariableConfig,
        num_objectives: int,
        eta_c: int,
        eta_m: int,
        undx_alpha: float,
        undx_beta: float,
        undx_m_sigma_xi: float,
        undx_m_sigma_eta: float,
        spx_epsilon: float,
        tblx_alpha: float,
        num_children_per_crossover: int,
        use_polynomial_mutation: bool,
    ):
        """
        Initialize crossover behavior from variables and algorithm parameters.
        変数定義とアルゴリズムパラメータから交叉処理を初期化する。
        """
        self.rng = rng
        self.var_config = var_config
        self.num_objectives = num_objectives
        self.eta_c = eta_c
        self.eta_m = eta_m
        self.undx_alpha = undx_alpha
        self.undx_beta = undx_beta
        self.undx_m_sigma_xi = undx_m_sigma_xi
        self.undx_m_sigma_eta = undx_m_sigma_eta
        self.spx_epsilon = spx_epsilon
        self.tblx_alpha = tblx_alpha
        self.num_children_per_crossover = num_children_per_crossover
        self.use_polynomial_mutation = use_polynomial_mutation

    def crossover_continuous(self, parent1_val: float, parent2_val: float,
                            lower: float, upper: float) -> Tuple[float, float]:
        """
        Apply continuous SBX and return two offspring values.
        連続変数へSBXを適用し、2つの子の値を返す。
        """
        if self.rng.random() <= 0.5:
            if abs(parent1_val - parent2_val) > EPS:
                if parent1_val < parent2_val:
                    y1, y2 = parent1_val, parent2_val
                else:
                    y1, y2 = parent2_val, parent1_val

                rand = self.rng.random()

                beta = 1.0 + (2.0 * (y1 - lower) / (y2 - y1))
                alpha = 2.0 - beta ** (-(self.eta_c + 1.0))
                if rand <= (1.0 / alpha):
                    betaq = (rand * alpha) ** (1.0 / (self.eta_c + 1.0))
                else:
                    betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (self.eta_c + 1.0))
                c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))

                beta = 1.0 + (2.0 * (upper - y2) / (y2 - y1))
                alpha = 2.0 - beta ** (-(self.eta_c + 1.0))
                if rand <= (1.0 / alpha):
                    betaq = (rand * alpha) ** (1.0 / (self.eta_c + 1.0))
                else:
                    betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (self.eta_c + 1.0))
                c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))

                if self.rng.random() <= 0.5:
                    return c2, c1
                else:
                    return c1, c2
            else:
                return parent1_val, parent2_val
        else:
            return parent1_val, parent2_val

    def mutate_continuous(self, value: float, lower: float, upper: float,
                         mutation_rate: float) -> float:
        """
        Apply polynomial mutation and return one continuous value.
        多項式突然変異を適用して連続値を1つ返す。
        """
        if self.rng.random() <= mutation_rate:
            y = np.clip(value, lower, upper)
            delta1 = (y - lower) / (upper - lower)
            delta2 = (upper - y) / (upper - lower)
            rnd = self.rng.random()
            mut_pow = 1.0 / (self.eta_m + 1.0)

            if rnd <= 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (xy ** (self.eta_m + 1.0))
                deltaq = (max(0.0, val) ** mut_pow) - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (xy ** (self.eta_m + 1.0))
                deltaq = 1.0 - (max(0.0, val) ** mut_pow)

            y = y + deltaq * (upper - lower)
            return y
        else:
            return value

    def crossover_undx_m(self, parents: List[Solution], mutation_rate: float) -> List[Solution]:
        """
        Apply UNDX-m and return generated offspring solutions.
        UNDX-mを適用して生成した子の解を返す。
        """
        children = []
        continuous_names = self.var_config.get_continuous_names()
        integer_names = self.var_config.get_integer_names()
        numeric_names = continuous_names + integer_names
        n = len(numeric_names)
        num_parents = len(parents)
        m = num_parents - 2

        for _ in range(self.num_children_per_crossover):
            child = Solution(self.num_objectives)

            if n > 0 and num_parents >= 3:
                # 交叉元として選択された各解の決定変数をUNDX-m計算用のベクトルにまとめる。
                parent_vector_rows = []
                for parent in parents:
                    parent_vector = []
                    for name in numeric_names:
                        parent_vector.append(float(parent.params[name]))
                    parent_vector_rows.append(parent_vector)
                parent_vectors = np.array(parent_vector_rows)
                p = np.mean(parent_vectors[:m+1], axis=0)
                d_vectors = parent_vectors[:m+1] - p
                d_m_plus_2 = parent_vectors[m+1] - p
                d_m_vectors = d_vectors[:m]

                if m > 0 and m <= n:
                    Q_m, R = np.linalg.qr(d_m_vectors.T)
                elif m == 0:
                    Q_m = np.zeros((n, 0))
                else:
                    if d_m_vectors.T.shape[1] > 0:
                        Q_temp, R = np.linalg.qr(d_m_vectors.T)
                        Q_m = Q_temp[:, :n]
                    else:
                        Q_m = np.zeros((n, 0))

                if Q_m.shape[1] > 0:
                    proj = Q_m @ (Q_m.T @ d_m_plus_2)
                else:
                    proj = np.zeros(n)

                perp_component = d_m_plus_2 - proj
                D = np.linalg.norm(perp_component)

                if m < n:
                    if Q_m.shape[1] > 0:
                        U, S, Vt = np.linalg.svd(Q_m, full_matrices=True)
                        e_perp = U[:, m:]
                    else:
                        e_perp = np.eye(n)
                else:
                    e_perp = np.zeros((n, 0))

                W = self.rng.normal(0.0, self.undx_m_sigma_xi, size=m) if m > 0 else np.array([])
                n_minus_m = max(0, n - m)
                V = self.rng.normal(0.0, self.undx_m_sigma_eta, size=n_minus_m) if n_minus_m > 0 else np.array([])

                child_vec = p.copy()
                if m > 0:
                    for k in range(m):
                        child_vec += W[k] * d_m_vectors[k]
                if n_minus_m > 0 and D > 1e-10:
                    for k in range(n_minus_m):
                        if k < e_perp.shape[1]:
                            child_vec += V[k] * D * e_perp[:, k]

                for i, name in enumerate(numeric_names):
                    child.params[name] = child_vec[i]

                if self.use_polynomial_mutation:
                    for name in numeric_names:
                        var_def = self.var_config.variables[name]
                        lower, upper = var_def['bounds']
                        value = child.params[name]
                        child.params[name] = self.mutate_continuous(value, float(lower), float(upper), mutation_rate)

            categorical_names = self.var_config.get_categorical_names()
            for var_name in categorical_names:
                parent = random_select(self.rng, parents)
                child.params[var_name] = parent.params[var_name]
                var_def = self.var_config.variables[var_name]
                if self.rng.random() < mutation_rate:
                    n_cat = var_def['n_categories']
                    child.params[var_name] = self.rng.integers(0, n_cat)

            children.append(child)
        return children

    def crossover_undx(self, parent1: Solution, parent2: Solution,
                      parent3: Solution, mutation_rate: float) -> List[Solution]:
        """
        Apply UNDX and return generated offspring solutions.
        UNDXを適用して生成した子の解を返す。
        """
        children = []
        continuous_names = self.var_config.get_continuous_names()
        integer_names = self.var_config.get_integer_names()
        numeric_names = continuous_names + integer_names

        if len(numeric_names) > 0:
            n_numeric = len(numeric_names)
            # 交叉元として選択された3つの解の決定変数をUNDX計算用のベクトルにまとめる。
            p1_values = []
            for name in numeric_names:
                p1_values.append(float(parent1.params[name]))
            p1_vec = np.array(p1_values)

            p2_values = []
            for name in numeric_names:
                p2_values.append(float(parent2.params[name]))
            p2_vec = np.array(p2_values)

            p3_values = []
            for name in numeric_names:
                p3_values.append(float(parent3.params[name]))
            p3_vec = np.array(p3_values)
            g = (p1_vec + p2_vec + p3_vec) / 3.0
            d = p2_vec - p1_vec
            d_norm = np.linalg.norm(d)

            for _ in range(self.num_children_per_crossover):
                child = Solution(self.num_objectives)

                if d_norm < 1e-10:
                    child_vec = p1_vec.copy()
                else:
                    e_d = d / d_norm
                    diff_p3 = p3_vec - p1_vec
                    proj_length = np.dot(diff_p3, e_d)
                    proj_vec = proj_length * e_d
                    perp_vec = diff_p3 - proj_vec
                    D = np.linalg.norm(perp_vec)
                    child_vec = g.copy()
                    r_alpha = self.rng.normal(0.0, self.undx_alpha)
                    child_vec += r_alpha * d

                    if n_numeric > 1 and D > 1e-10:
                        for _ in range(n_numeric - 1):
                            r_beta = self.rng.normal(0.0, self.undx_beta * D)
                            random_vec = self.rng.standard_normal(n_numeric)
                            random_vec = random_vec - np.dot(random_vec, e_d) * e_d
                            random_vec_norm = np.linalg.norm(random_vec)
                            if random_vec_norm > 1e-10:
                                e_perp = random_vec / random_vec_norm
                                child_vec += r_beta * e_perp

                for i, name in enumerate(numeric_names):
                    child.params[name] = child_vec[i]

                if self.use_polynomial_mutation:
                    for name in numeric_names:
                        var_def = self.var_config.variables[name]
                        lower, upper = var_def['bounds']
                        value = child.params[name]
                        child.params[name] = self.mutate_continuous(value, float(lower), float(upper), mutation_rate)

                for var_name in self.var_config.get_categorical_names():
                    choice = self.rng.integers(0, 3)
                    if choice == 0:
                        child.params[var_name] = parent1.params[var_name]
                    elif choice == 1:
                        child.params[var_name] = parent2.params[var_name]
                    else:
                        child.params[var_name] = parent3.params[var_name]
                    if self.rng.random() < mutation_rate:
                        n_cat = self.var_config.variables[var_name]['n_categories']
                        child.params[var_name] = self.rng.integers(0, n_cat)

                children.append(child)
        else:
            for _ in range(self.num_children_per_crossover):
                child = Solution(self.num_objectives)
                for var_name in self.var_config.get_categorical_names():
                    choice = self.rng.integers(0, 3)
                    if choice == 0:
                        child.params[var_name] = parent1.params[var_name]
                    elif choice == 1:
                        child.params[var_name] = parent2.params[var_name]
                    else:
                        child.params[var_name] = parent3.params[var_name]
                    if self.rng.random() < mutation_rate:
                        n_cat = self.var_config.variables[var_name]['n_categories']
                        child.params[var_name] = self.rng.integers(0, n_cat)
                children.append(child)

        return children

    def crossover_spx(self, parents: List[Solution], mutation_rate: float) -> List[Solution]:
        """
        Apply SPX and return generated offspring solutions.
        SPXを適用して生成した子の解を返す。
        """
        children = []
        n = len(self.var_config.get_continuous_names()) + len(self.var_config.get_integer_names())

        if n + 1 != len(parents):
            raise ValueError(
                f"SPX needs {n + 1} parents, but got {len(parents)}"
            )

        for _ in range(self.num_children_per_crossover):
            child = Solution(self.num_objectives)
            numeric_names = self.var_config.get_continuous_names() + self.var_config.get_integer_names()

            if n > 0:
                # 交叉元として選択された各解の決定変数をSPX計算用のベクトルにまとめる。
                parent_vector_rows = []
                for parent in parents:
                    parent_vector = []
                    for name in numeric_names:
                        parent_vector.append(float(parent.params[name]))
                    parent_vector_rows.append(parent_vector)
                parent_vectors = np.array(parent_vector_rows)
                G = np.mean(parent_vectors, axis=0)
                x_bar = G + self.spx_epsilon * (parent_vectors - G)

                C = np.zeros(n)
                for k in range(1, n + 1):
                    u = self.rng.random()
                    r_k_minus_1 = u ** (1.0 / k)
                    C = r_k_minus_1 * (x_bar[k-1] - x_bar[k] + C)

                child_vec = x_bar[n] + C

                for i, name in enumerate(numeric_names):
                    child.params[name] = child_vec[i]

                if self.use_polynomial_mutation:
                    for name in numeric_names:
                        var_def = self.var_config.variables[name]
                        lower, upper = var_def['bounds']
                        value = child.params[name]
                        child.params[name] = self.mutate_continuous(value, float(lower), float(upper), mutation_rate)

            for var_name in self.var_config.get_categorical_names():
                parent = random_select(self.rng, parents)
                child.params[var_name] = parent.params[var_name]
                var_def = self.var_config.variables[var_name]
                if self.rng.random() < mutation_rate:
                    n_cat = var_def['n_categories']
                    child.params[var_name] = self.rng.integers(0, n_cat)

            children.append(child)
        return children

    def crossover_blx_alpha(self, y1: float, y2: float) -> float:
        """
        Apply BLX-alpha and return one offspring value.
        BLX-alphaを適用して子の値を1つ返す。
        """
        if y1 > y2:
            y1, y2 = y2, y1
        interval_width = y2 - y1
        lower = y1 - self.tblx_alpha * interval_width
        upper = y2 + self.tblx_alpha * interval_width
        return self.rng.uniform(lower, upper)

    def crossover_tblx(self, subproblem_id: int, mutation_rate: float,
                       population: List[Subproblem], population_size: int) -> Solution:
        """
        Apply TBLX-alpha and return one offspring solution.
        TBLX-alphaを適用して子の解を1つ返す。
        """
        child = Solution(self.num_objectives)
        numeric_names = (self.var_config.get_continuous_names() +
                        self.var_config.get_integer_names())

        if len(numeric_names) > 0:
            n_numeric = len(numeric_names)
            # 探索中の各解について、TBLXで平均と共分散を求める決定変数ベクトルを作成する。
            population_vector_rows = []
            for subproblem in population:
                population_vector = []
                for name in numeric_names:
                    population_vector.append(float(subproblem.solution.params[name]))
                population_vector_rows.append(population_vector)
            X = np.array(population_vector_rows)
            mu = np.mean(X, axis=0)
            Sigma = np.cov(X, rowvar=False)

            if n_numeric == 1:
                Sigma = np.array([[Sigma + 1e-8]])
            else:
                Sigma = Sigma + np.eye(n_numeric) * 1e-8

            try:
                L = np.linalg.cholesky(Sigma)
            except np.linalg.LinAlgError:
                Sigma = Sigma + np.eye(n_numeric) * 1e-6
                L = np.linalg.cholesky(Sigma)

            L_inv = np.linalg.inv(L)
            # 対象サブ問題に割り当てられた解の決定変数をTBLX計算用のベクトルにまとめる。
            x_i_values = []
            for name in numeric_names:
                x_i_values.append(float(population[subproblem_id].solution.params[name]))
            x_i = np.array(x_i_values)
            y_i = L_inv @ (x_i - mu)

            # 対象サブ問題以外から交叉相手を選択する。
            other_indices = []
            for idx in range(population_size):
                if idx != subproblem_id:
                    other_indices.append(idx)
            random_idx = random_select(self.rng, other_indices)
            # 交叉相手として選択した解の決定変数をTBLX計算用のベクトルにまとめる。
            x_r_values = []
            for name in numeric_names:
                x_r_values.append(float(population[random_idx].solution.params[name]))
            x_r = np.array(x_r_values)
            y_r = L_inv @ (x_r - mu)

            # 対象解と交叉相手の変換後ベクトルへ、成分ごとにBLX-alpha交叉を適用する。
            y_prime_values = []
            for j in range(n_numeric):
                y_prime_values.append(self.crossover_blx_alpha(y_i[j], y_r[j]))
            y_prime = np.array(y_prime_values)
            x_prime = L @ y_prime + mu

            for j, name in enumerate(numeric_names):
                child.params[name] = x_prime[j]

            if self.use_polynomial_mutation:
                for name in numeric_names:
                    var_def = self.var_config.variables[name]
                    lower, upper = var_def['bounds']
                    value = child.params[name]
                    child.params[name] = self.mutate_continuous(value, float(lower), float(upper), mutation_rate)

        categorical_names = self.var_config.get_categorical_names()
        if len(categorical_names) > 0:
            # 対象サブ問題以外からカテゴリ変数の交叉相手を選択する。
            other_indices = []
            for idx in range(population_size):
                if idx != subproblem_id:
                    other_indices.append(idx)
            random_idx = random_select(self.rng, other_indices)

            for var_name in categorical_names:
                if self.rng.random() < 0.5:
                    child.params[var_name] = population[subproblem_id].solution.params[var_name]
                else:
                    child.params[var_name] = population[random_idx].solution.params[var_name]
                var_def = self.var_config.variables[var_name]
                if self.rng.random() < mutation_rate:
                    n_cat = var_def['n_categories']
                    child.params[var_name] = self.rng.integers(0, n_cat)

        return child

    def crossover_sbx(self, parent1: Solution, parent2: Solution,
                      mutation_rate: float) -> Solution:
        """
        Apply SBX with optional mutation and return one offspring solution.
        任意の突然変異を伴うSBXを適用して子の解を1つ返す。
        """
        child = Solution(self.num_objectives)

        for var_name in self.var_config.variable_names:
            var_def = self.var_config.variables[var_name]

            if var_def['type'] == 'continuous':
                lower, upper = var_def['bounds']
                p1_val = parent1.params[var_name]
                p2_val = parent2.params[var_name]
                c1_val, _ = self.crossover_continuous(p1_val, p2_val, lower, upper)
                if self.use_polynomial_mutation:
                    c1_val = self.mutate_continuous(c1_val, lower, upper, mutation_rate)
                child.params[var_name] = c1_val

            elif var_def['type'] == 'integer':
                lower, upper = var_def['bounds']
                p1_val = float(parent1.params[var_name])
                p2_val = float(parent2.params[var_name])
                c1_val, _ = self.crossover_continuous(p1_val, p2_val, float(lower), float(upper))
                if self.use_polynomial_mutation:
                    c1_val = self.mutate_continuous(c1_val, float(lower), float(upper), mutation_rate)
                child.params[var_name] = c1_val

            elif var_def['type'] == 'categorical':
                if self.rng.random() < 0.5:
                    child.params[var_name] = parent1.params[var_name]
                else:
                    child.params[var_name] = parent2.params[var_name]
                if self.rng.random() < mutation_rate:
                    n_cat = var_def['n_categories']
                    child.params[var_name] = self.rng.integers(0, n_cat)

        return child
