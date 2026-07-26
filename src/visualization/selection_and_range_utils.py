"""
Visualization selection and axis-range utilities
可視化の選択と軸範囲ユーティリティ
"""

import math
from typing import Any, List, Optional, Sequence, Tuple


def sanitize_fixed_range(
    min_text: str,
    max_text: str,
    *,
    use_log: bool,
    epsilon: float,
    fallback: Tuple[float, float],
) -> Tuple[float, float]:
    """
    Return axis bounds from axis-range text input values.
    軸範囲文字列の入力値から上下限を返す。
    """
    # 1e-2の対応、無効値はfallback、対数軸の非正値はepsilonを使用する。
    # 上下限が等しい場合は微小区間へ拡張する。
    try:
        start = float(str(min_text).strip())
        end = float(str(max_text).strip())
    except (TypeError, ValueError):
        start, end = fallback

    if use_log:
        if start <= 0:
            start = epsilon
        if end <= 0:
            end = epsilon

    if start > end:
        start, end = end, start

    if start == end:
        if use_log:
            end = max(start + epsilon, start * (1.0 + 1e-3))
        else:
            end = start + 1.0

    if use_log:
        start = max(start, epsilon)
        end = max(end, start * (1.0 + 1e-6))

    return float(start), float(end)


def resolve_text_input_value(committed_value: str, editing_value: Optional[str]) -> str:
    """
    Return a nonempty editing value or the committed TextInput value.
    空でない編集中の値、または確定済みTextInput値を返す。
    """
    if isinstance(editing_value, str) and editing_value.strip() != "":
        return editing_value
    return str(committed_value)


def set_text_input_value_and_editing(text_input: Any, value: str) -> None:
    """
    Synchronize committed and editing values on a TextInput.
    TextInputの確定値と編集中の値を同期する。
    """
    normalized = str(value)
    text_input.value = normalized
    current_editing = getattr(text_input, "value_input", None)
    if isinstance(current_editing, str):
        text_input.value_input = normalized


def round_up_to_power_of_ten(value: float, minimum: float = 1.0) -> float:
    """
    Round up and return a positive value as a power of ten.
    下限を考慮して正の値を10の累乗へ切り上げて返す。
    """
    safe_minimum = max(float(minimum), 1e-12)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = safe_minimum

    target = max(numeric, safe_minimum)
    exponent = math.ceil(math.log10(target))
    return float(10.0 ** exponent)


def compute_data_tolerance_from_pixels(
    x_range_start: float,
    x_range_end: float,
    y_range_start: float,
    y_range_end: float,
    plot_width: int,
    plot_height: int,
    radius_px: float,
) -> Tuple[float, float]:
    """
    Convert a pixel radius and return per-axis data tolerances.
    ピクセル半径を変換し、各軸のデータ空間許容誤差を返す。
    """
    try:
        x_start = float(x_range_start)
        x_end = float(x_range_end)
        y_start = float(y_range_start)
        y_end = float(y_range_end)
    except (TypeError, ValueError):
        return 0.0, 0.0

    width = max(int(plot_width), 1)
    height = max(int(plot_height), 1)
    radius = max(float(radius_px), 0.0)

    x_span = abs(x_end - x_start)
    y_span = abs(y_end - y_start)
    x_tol = (x_span / float(width)) * radius
    y_tol = (y_span / float(height)) * radius
    return float(x_tol), float(y_tol)


def collect_overlapped_indices_by_tolerance(
    anchor_idx: int,
    f1_values: Sequence[float],
    f2_values: Sequence[float],
    x_tolerance: float,
    y_tolerance: float,
) -> List[int]:
    """
    Collect and return indices within tolerance of an anchor point.
    アンカー点の許容誤差内にあるインデックスを収集して返す。
    """
    if anchor_idx < 0 or anchor_idx >= len(f1_values) or anchor_idx >= len(f2_values):
        return []
    try:
        anchor_f1 = float(f1_values[anchor_idx])
        anchor_f2 = float(f2_values[anchor_idx])
    except (TypeError, ValueError):
        return []

    x_tol = max(float(x_tolerance), 0.0)
    y_tol = max(float(y_tolerance), 0.0)
    overlapped: List[int] = []
    upper = min(len(f1_values), len(f2_values))
    for idx in range(upper):
        try:
            value_f1 = float(f1_values[idx])
            value_f2 = float(f2_values[idx])
        except (TypeError, ValueError):
            continue
        if abs(value_f1 - anchor_f1) <= x_tol and abs(value_f2 - anchor_f2) <= y_tol:
            overlapped.append(idx)
    return overlapped


def order_indices_by_original_index(
    indices: Sequence[int],
    original_indices: Sequence[int],
) -> List[int]:
    """
    Sort and return indices by original index then source index.
    元インデックス、次にソースインデックスの順で並べて返す。
    """
    unique_indices = sorted({int(idx) for idx in indices if int(idx) >= 0})

    def _key(idx: int) -> Tuple[int, int]:
        """
        Return an original-index and source-index sorting key.
        元インデックスとソースインデックスのソートキーを返す。
        """
        if 0 <= idx < len(original_indices):
            try:
                return int(original_indices[idx]), idx
            except (TypeError, ValueError):
                pass
        return idx, idx

    return sorted(unique_indices, key=_key)


def resolve_tap_cycle_selection(
    ordered_group: Sequence[int],
    clicked_idx: int,
    previous_group: Optional[Tuple[int, ...]],
    previous_position: int,
) -> Tuple[List[int], Optional[Tuple[int, ...]], int, bool]:
    """
    Resolve and return the next tap-cycle selection state.
    タップによる循環選択を解決し、選択結果と次の状態を返す。
    """
    group = tuple(int(idx) for idx in ordered_group)
    if not group:
        return [], None, -1, False
    if len(group) == 1:
        return [group[0]], None, -1, False

    if previous_group != group:
        try:
            position = group.index(int(clicked_idx))
        except ValueError:
            position = 0
        return [group[position]], group, int(position), False

    next_position = (int(previous_position) + 1) % len(group)
    return [group[next_position]], group, int(next_position), True
