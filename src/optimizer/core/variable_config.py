"""
Mixed-variable configuration used by multi-objective optimization algorithms
多目的最適化アルゴリズムで使う混合変数設定
"""

from typing import Any, Dict, List

import numpy as np


class VariableConfig:
    """
    Define continuous, integer, and categorical variables for multi-objective optimization.
    多目的最適化で用いる連続、整数、カテゴリ変数を定義する。
    """

    def __init__(self):
        """
        Initialize an empty ordered variable configuration.
        空の順序付き変数設定を初期化する。
        """
        self.variables: Dict[str, Dict[str, Any]] = {}
        self.variable_names: List[str] = []

    def add_continuous(self, name: str, lower: float, upper: float):
        """
        Add a continuous variable and return this configuration.
        連続変数を追加してこの設定自身を返す。
        """
        self.variables[name] = {
            'type': 'continuous',
            'bounds': [lower, upper]
        }
        self.variable_names.append(name)
        return self

    def add_integer(self, name: str, lower: int, upper: int):
        """
        Add an integer variable and return this configuration.
        整数変数を追加してこの設定自身を返す。
        """
        self.variables[name] = {
            'type': 'integer',
            'bounds': [lower, upper]
        }
        self.variable_names.append(name)
        return self

    def add_categorical(self, name: str, category_names: List[str]):
        """
        Add a categorical variable and return this configuration.
        カテゴリ変数を追加してこの設定自身を返す。
        """
        # カテゴリ変数は整数インデックス（0 〜 n_categories-1）で表現
        # 例: add_categorical('activation', ['tanh', 'relu']) → {0: 'tanh', 1: 'relu'}
        n_categories = len(category_names)
        self.variables[name] = {
            'type': 'categorical',
            'n_categories': n_categories,
            'category_names': list(category_names),
        }
        self.variable_names.append(name)
        return self

    def generate_random_params(self, rng: np.random.Generator = None) -> Dict[str, Any]:
        """
        Generate and return one random parameter dictionary from all variables.
        全変数からランダムなパラメータ辞書を1件生成して返す。
        """
        # 初期値を生成する        
        if rng is None:
            rng = np.random.default_rng()

        params = {}
        for name in self.variable_names:
            var_def = self.variables[name]

            if var_def['type'] == 'continuous':
                # 連続変数: [lower, upper] の一様分布
                lower, upper = var_def['bounds']
                params[name] = rng.uniform(lower, upper)

            elif var_def['type'] == 'integer':
                # 整数変数: [lower, upper] の整数一様分布（upper含む）
                lower, upper = var_def['bounds']
                params[name] = rng.integers(lower, upper + 1)

            elif var_def['type'] == 'categorical':
                # カテゴリ変数: [0, n_categories) から等確率で選択
                n_cat = var_def['n_categories']
                params[name] = rng.integers(0, n_cat)

        return params

    def get_num_variables(self) -> int:
        """
        Return the total number of configured variables.
        設定済み変数の総数を返す。
        """
        return len(self.variable_names)

    def get_continuous_names(self) -> List[str]:
        """
        Return continuous variable names in configuration order.
        連続変数名を設定順のリストで返す。
        """
        result = []
        for name in self.variable_names:
            if self.variables[name]['type'] == 'continuous':
                result.append(name)
        return result

    def get_integer_names(self) -> List[str]:
        """
        Return integer variable names in configuration order.
        整数変数名を設定順のリストで返す。
        """
        result = []
        for name in self.variable_names:
            if self.variables[name]['type'] == 'integer':
                result.append(name)
        return result

    def get_categorical_names(self) -> List[str]:
        """
        Return categorical variable names in configuration order.
        カテゴリ変数名を設定順のリストで返す。
        """
        result = []
        for name in self.variable_names:
            if self.variables[name]['type'] == 'categorical':
                result.append(name)
        return result

    def __repr__(self) -> str:
        """
        Return a debug representation grouped by variable type.
        変数型別にまとめたデバッグ表現を返す。
        """
        num_continuous = len(self.get_continuous_names())
        name_continuous = self.get_continuous_names()
        num_integer = len(self.get_integer_names())
        name_integer = self.get_integer_names()
        num_categorical = len(self.get_categorical_names())
        name_categorical = self.get_categorical_names()

        ret_str = (
            f"{num_continuous}, {name_continuous}\n"
            f"{num_integer}, {name_integer}\n"
            f"{num_categorical}, {name_categorical}"
        )
        return ret_str
