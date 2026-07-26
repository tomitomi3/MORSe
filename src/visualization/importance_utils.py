"""
Variable importance display alignment utilities
変数重要度表示の整列ユーティリティ
"""

from typing import Any, Dict, List, Tuple


def _to_float(value: Any, default: float = 0.0) -> float:
    """
    Convert a value and return a finite float or the default.
    値を変換し、有限なfloatまたは既定値を返す。
    """
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default
    if converted != converted:  # NaNの場合
        return default
    return converted


def align_importance_series(
    param_names: List[str],
    obj_data: Dict[str, Any],
) -> Tuple[List[str], List[float], List[float]]:
    """
    Align importance data and return display names, means, and deviations.
    重要度データを整列し、表示名、平均、標準偏差のタプルを返す。
    """
    # param_namesはCondition.json、obj_dataはVariableImportance.jsonの対象目的関数データとする。
    if not isinstance(obj_data, dict):
        return [], [], []

    raw_means = obj_data.get("mean", [])
    raw_stds = obj_data.get("std", [])
    if not isinstance(raw_means, list):
        raw_means = []
    if not isinstance(raw_stds, list):
        raw_stds = []

    # 基準がある場合: parameter_names に合わせて名前ベースで再整列
    if param_names:
        n = len(param_names)
        aligned_means = [0.0] * n
        aligned_stds = [0.0] * n

        variables = obj_data.get("variables", [])
        if isinstance(variables, list) and variables:
            mean_map: Dict[str, float] = {}
            std_map: Dict[str, float] = {}
            for idx, raw_name in enumerate(variables):
                name = str(raw_name)
                mean_map[name] = _to_float(raw_means[idx] if idx < len(raw_means) else 0.0)
                std_map[name] = max(0.0, _to_float(raw_stds[idx] if idx < len(raw_stds) else 0.0))

            for idx, name in enumerate(param_names):
                aligned_means[idx] = mean_map.get(name, 0.0)
                aligned_stds[idx] = std_map.get(name, 0.0)
            return list(param_names), aligned_means, aligned_stds

        # variables が無い場合は従来互換で位置対応
        for idx in range(min(n, len(raw_means))):
            aligned_means[idx] = _to_float(raw_means[idx])
            aligned_stds[idx] = max(0.0, _to_float(raw_stds[idx] if idx < len(raw_stds) else 0.0))
        return list(param_names), aligned_means, aligned_stds

    # parameter_names が無い場合は variables/配列長をそのまま使用
    variables = obj_data.get("variables", [])
    # VariableImportance.jsonの変数名を表示用文字列へ揃える。
    if isinstance(variables, list) and variables:
        display_names = [str(variable) for variable in variables]
    else:
        display_names = [f"var_{index}" for index in range(len(raw_means))]

    n = len(display_names)
    # 変数ごとの平均と標準偏差をfloatへ変換し、不足値を0で補う。
    means = []
    stds = []
    for index in range(n):
        mean = raw_means[index] if index < len(raw_means) else 0.0
        std = raw_stds[index] if index < len(raw_stds) else 0.0
        means.append(_to_float(mean))
        stds.append(max(0.0, _to_float(std)))
    return display_names, means, stds
