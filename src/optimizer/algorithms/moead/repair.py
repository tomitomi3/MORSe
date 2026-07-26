"""
MOEA/D repair operations
MOEA/D 修復処理
"""

import numpy as np

from src.optimizer.algorithms.moead.solution import Solution
from src.optimizer.core.variable_config import VariableConfig


class RepairOperator:
    """
    Repair continuous, integer, and categorical solution variables.
    解の連続、整数、カテゴリ変数を境界内へ修復する。
    """

    def __init__(
        self,
        rng: np.random.Generator,
        var_config: VariableConfig,
        boundary_repair_mode: str,
        boundary_repair_reflect_ratio: float,
        use_stochastic_rounding: bool,
    ):
        """
        Initialize repair behavior from variables and boundary settings.
        変数定義と境界設定から修復処理を初期化する。
        """
        self.rng = rng
        self.var_config = var_config
        self.boundary_repair_mode = boundary_repair_mode
        self.boundary_repair_reflect_ratio = boundary_repair_reflect_ratio
        self.use_stochastic_rounding = use_stochastic_rounding

    @staticmethod
    def _reflect_value(
        value: float,
        lower: float,
        upper: float,
        reflect_ratio: float = 1.0,
    ) -> float:
        """
        Reflect and return a numeric value within its bounds.
        数値を境界内へ反射して返す。
        """
        if lower > upper:
            raise ValueError(f"invalid bounds: lower={lower}, upper={upper}")
        if np.isclose(lower, upper):
            return float(lower)

        value_float = float(value)
        if value_float < lower:
            reflected = float(2.0 * lower - value_float)
        elif value_float > upper:
            reflected = float(2.0 * upper - value_float)
        else:
            reflected = value_float

        anchor = float(np.clip(value_float, lower, upper))
        candidate = float(anchor + reflect_ratio * (reflected - anchor))
        return float(np.clip(candidate, lower, upper))

    def repair_solution(self, solution: Solution):
        """
        Repair a solution in place to satisfy all variable bounds.
        全変数の境界を満たすように解をその場で修復する。
        """
        if self.var_config is None:
            raise RuntimeError("var_config is not initialized")

        for var_name in self.var_config.variable_names:
            var_def = self.var_config.variables[var_name]

            # solution.params 連続・整数は浮動小数点
            if var_def['type'] == 'continuous':
                # 連続変数を修復する
                lower, upper = var_def['bounds']
                raw = float(solution.params[var_name])

                # 選択した境界方式で値を修復する
                temp_value = 0.0
                if self.boundary_repair_mode == 'clip':
                    temp_value = raw
                elif self.boundary_repair_mode == 'reflect':
                    temp_value = RepairOperator._reflect_value(
                        value=raw,
                        lower=float(lower),
                        upper=float(upper),
                        reflect_ratio=self.boundary_repair_reflect_ratio,
                    )

                # clipで範囲内保証
                solution.params[var_name] = float(np.clip(temp_value, lower, upper))

            elif var_def['type'] == 'integer':
                # 整数変数を修復する
                lower, upper = var_def['bounds']
                raw = float(solution.params[var_name])

                # 確率的丸め 小数部分を切り上げ確率とする
                if self.use_stochastic_rounding:
                    floor_val = int(np.floor(raw))
                    frac = raw - floor_val  # fracは0以上1未満とする
                    if self.rng.random() < frac:
                        raw = float(floor_val + 1)
                    else:
                        raw = float(floor_val)

                # 選択した境界方式で値を修復する
                temp_value = 0.0
                if self.boundary_repair_mode == 'clip':
                    temp_value = raw
                elif self.boundary_repair_mode == 'reflect':
                    temp_value = RepairOperator._reflect_value(
                        value=float(raw),
                        lower=float(lower),
                        upper=float(upper),
                        reflect_ratio=self.boundary_repair_reflect_ratio,
                    )

                # clipで範囲内保証
                solution.params[var_name] = int(np.clip(temp_value, int(lower), int(upper)))

            elif var_def['type'] == 'categorical':
                # カテゴリ変数を修復する
                # clipで範囲内保証 0 から n_categories-1 の範囲に丸める
                n_cat = var_def['n_categories']
                solution.params[var_name] = int(np.clip(np.round(solution.params[var_name]), 0, n_cat - 1))
