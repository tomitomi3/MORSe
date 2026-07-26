"""
Network metric visualization utilities
ネットワーク指標可視化ユーティリティ
"""

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def collect_adjacency_matrices(raw_matrices: Sequence[Any]) -> List[np.ndarray]:
    """
    Collect and return valid square adjacency matrices.
    有効な隣接行列を収集して返す。
    """
    matrices: List[np.ndarray] = []
    for raw in raw_matrices:
        if not isinstance(raw, (list, tuple)):
            continue
        size = len(raw)
        if size == 0:
            continue

        rows: List[List[float]] = []
        valid = True
        for row in raw:
            if not isinstance(row, (list, tuple)) or len(row) != size:
                valid = False
                break
            converted_row: List[float] = []
            for value in row:
                try:
                    converted_row.append(float(value))
                except (TypeError, ValueError):
                    valid = False
                    break
            if not valid:
                break
            rows.append(converted_row)

        if not valid:
            continue
        matrices.append(np.asarray(rows, dtype=float))
    return matrices


def compute_degree_histogram_stats(
    matrices: Sequence[np.ndarray],
) -> Tuple[List[int], List[float], List[float]]:
    """
    Return degrees with mean and deviation histograms across matrices.
    複数行列の次数とヒストグラムの平均、標準偏差を返す。
    """
    histograms: List[np.ndarray] = []
    max_len = 0

    for matrix in matrices:
        if matrix.size == 0:
            continue
        degrees = matrix.sum(axis=1).astype(int)
        if len(degrees) == 0:
            continue
        histogram = np.bincount(degrees)
        histograms.append(histogram.astype(float))
        max_len = max(max_len, len(histogram))

    if not histograms or max_len == 0:
        return [], [], []

    aligned: List[np.ndarray] = []
    for histogram in histograms:
        if len(histogram) < max_len:
            padded = np.pad(histogram, (0, max_len - len(histogram)), mode="constant")
            aligned.append(padded)
        else:
            aligned.append(histogram)

    histogram_array = np.vstack(aligned)
    degrees = list(range(max_len))
    mean_values = histogram_array.mean(axis=0).tolist()
    std_values = histogram_array.std(axis=0).tolist()
    return degrees, mean_values, std_values


def _build_degree_bar_edges(centers: Sequence[float]) -> Tuple[List[float], List[float]]:
    """
    Build and return left and right degree-bar boundaries.
    次数ヒストグラムの棒の左右境界を構築して返す。
    """
    center_count = len(centers)
    if center_count == 0:
        return [], []

    if center_count == 1:
        center = float(centers[0])
        return [max(0.0, center - 0.5)], [center + 0.5]

    # 棒の中心位置をfloatへ変換して境界計算に使う。
    center_values = [float(center) for center in centers]
    safe_centers = np.asarray(center_values, dtype=float)
    boundaries = (safe_centers[:-1] + safe_centers[1:]) / 2.0
    left_edges: List[float] = []
    right_edges: List[float] = []
    for idx, center in enumerate(safe_centers):
        if idx == 0:
            left = max(0.0, float(center - (boundaries[0] - center)))
        else:
            left = float(boundaries[idx - 1])
        if idx == len(safe_centers) - 1:
            right = float(center + (center - boundaries[-1]))
        else:
            right = float(boundaries[idx])
        left_edges.append(left)
        right_edges.append(right)
    return left_edges, right_edges


def build_degree_histogram_bar_data(
    degrees: Sequence[float],
    mean_counts: Sequence[float],
    use_log: bool,
    epsilon: float,
) -> Dict[str, Any]:
    """
    Build Bokeh bar-chart coordinates for a linear or logarithmic degree histogram.
    線形または対数の次数ヒストグラムをBokehの棒グラフへ渡す座標データを返す。
    """
    raw_degrees: List[float] = []
    raw_counts: List[float] = []
    pair_count = min(len(degrees), len(mean_counts))
    for idx in range(pair_count):
        raw_degrees.append(float(degrees[idx]))
        raw_counts.append(float(mean_counts[idx]))

    raw_left, raw_right = _build_degree_bar_edges(raw_degrees)
    display_degrees: List[float] = []
    display_counts: List[float] = []
    display_left: List[float] = []
    display_right: List[float] = []
    display_bottom: List[float] = []
    zero_degree_count = 0.0

    for idx, degree in enumerate(raw_degrees):
        count = raw_counts[idx]
        if degree == 0.0 and count > 0.0:
            zero_degree_count += count
        if use_log and (degree < 1.0 or count <= 0.0):
            continue

        display_degrees.append(degree)
        display_counts.append(count)
        display_left.append(raw_left[idx])
        display_right.append(raw_right[idx])
        if use_log:
            display_bottom.append(float(epsilon))
        else:
            display_bottom.append(0.0)

    return {
        "degree": display_degrees,
        "left": display_left,
        "right": display_right,
        "bottom": display_bottom,
        "top": display_counts,
        "zero_degree_count": zero_degree_count,
    }


def build_adjacency_image_data(matrix: np.ndarray) -> Dict[str, List[Any]]:
    """
    Build and return Bokeh image-glyph data for an adjacency matrix.
    隣接行列のBokeh image glyph用データを構築して返す。
    """
    # 隣接行列表示高速化のために、画像にする
    # Bokeh image glyphの仕様はhttps://docs.bokeh.org/en/2.4.0/docs/reference/models/glyphs/image.htmlを参照する
    size = matrix.shape[0]
    if size == 0:
        return {"image": [], "x": [], "y": [], "dw": [], "dh": []}

    # image glyph は下端基準描画のため、上下反転
    flipped = np.flipud(matrix.astype(float, copy=False))
    return {
        "image": [flipped],
        "x": [-0.5],
        "y": [-0.5],
        "dw": [float(size)],
        "dh": [float(size)],
    }
