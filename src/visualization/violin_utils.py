"""
Violin plot data utilities
バイオリンプロットデータユーティリティ
"""

from typing import Any, Dict, List, Optional

import numpy as np
from scipy.stats import gaussian_kde


def normalize_violin_values(
    values: List[Any],
    variable_info: Optional[Dict[str, Any]] = None,
) -> List[float]:
    """
    Normalize and return selected parameter values for violin display.
    選択したパラメータ値をバイオリン表示用に正規化して返す。
    """
    try:
        values_array = np.array(values, dtype=float)
    except (TypeError, ValueError):
        return []

    values_array = values_array[np.isfinite(values_array)]
    if len(values_array) == 0:
        return []

    info = variable_info if isinstance(variable_info, dict) else {}
    if info.get("type") == "categorical":
        try:
            n_categories = int(info.get("n_categories", 0))
        except (TypeError, ValueError):
            n_categories = 0
        if n_categories > 1:
            rounded_values = np.rint(values_array)
            clipped_values = np.clip(rounded_values, 0, n_categories - 1)
            return (clipped_values / float(n_categories - 1)).tolist()
        return [0.5] * len(values_array)

    v_min = values_array.min()
    v_max = values_array.max()
    if v_max - v_min > 1e-10:
        return ((values_array - v_min) / (v_max - v_min)).tolist()
    return [0.5] * len(values_array)


def create_violin_data(
    data_values: List[Any],
    center_x: float,
    max_width: float = 0.35,
    n_points: int = 100,
) -> Dict[str, List[float]]:
    """
    Create and return violin polygon data for one parameter.
    1パラメータのバイオリン形状データを生成して返す。
    """
    if len(data_values) < 2:
        return {"x1": [], "x2": [], "y": []}

    try:
        data_array = np.array(data_values, dtype=float)
    except (TypeError, ValueError):
        return {"x1": [], "x2": [], "y": []}

    data_array = data_array[np.isfinite(data_array)]
    if len(data_array) < 2:
        return {"x1": [], "x2": [], "y": []}

    if np.std(data_array) < 1e-10:
        y_center = float(np.mean(data_array))
        half_height = 0.02
        half_width = max_width * 0.10
        return {
            "y": [y_center - half_height, y_center + half_height],
            "x1": [center_x - half_width, center_x - half_width],
            "x2": [center_x + half_width, center_x + half_width],
        }

    # KDEの帯域幅はSciPyのScottルールで決まり、選択データに応じて変化する。
    # 表示グリッド数と最大幅は、n_pointsとmax_widthで固定する。
    try:
        kde = gaussian_kde(data_array, bw_method="scott")
    except np.linalg.LinAlgError:
        return {"x1": [], "x2": [], "y": []}

    v_min = np.min(data_array)
    v_max = np.max(data_array)
    margin = (v_max - v_min) * 0.1 if v_max > v_min else 0.1
    y_range = np.linspace(v_min - margin, v_max + margin, n_points)
    density = kde(y_range)
    density_max = density.max()
    if density_max > 0:
        density_norm = density / density_max * max_width
    else:
        density_norm = density
    return {
        "y": y_range.tolist(),
        "x1": (center_x - density_norm).tolist(),
        "x2": (center_x + density_norm).tolist(),
    }
