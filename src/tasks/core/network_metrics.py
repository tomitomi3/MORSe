"""
Network metric calculation utilities for tasks
タスク側のネットワーク指標計算ユーティリティ
"""

from typing import List, Optional

import networkx as nx
import numpy as np


def extract_adjacency_matrix(weight_matrix) -> List[List[int]]:
    """
    Extract a binary adjacency matrix from ReservoirPy recurrent weights.
    ReservoirPyの再帰重みから0/1隣接行列を抽出する。
    """
    # 結合密度に応じて返されるscipy疎行列とnumpy密行列の両方に対応する。
    # 位相から自己ループを除外するため、対角要素を0にする。
    dense = weight_matrix.toarray() if hasattr(weight_matrix, "toarray") else weight_matrix
    matrix_array = np.asarray(dense, dtype=np.float64)
    topology = (np.abs(matrix_array) > 0.0).astype(int)
    np.fill_diagonal(topology, 0)
    return topology.tolist()


def compute_average_shortest_path_length(adjacency_matrix) -> Optional[float]:
    """
    Calculate average shortest path length from a binary adjacency matrix.
    0/1隣接行列から平均最短経路長を計算する。
    """
    # 可視化側の従来実装と互換性のある計算規則を使用する。
    # 無向グラフとして扱い、非連結の場合は最大連結成分を使用する。
    # 選択したグラフのノード数が1以下の場合は0.0を返す。
    if not isinstance(adjacency_matrix, (list, tuple)):
        return None
    size = len(adjacency_matrix)
    if size == 0:
        return None

    rows = []
    for row in adjacency_matrix:
        if not isinstance(row, (list, tuple)) or len(row) != size:
            return None
        converted_row = []
        for value in row:
            converted_row.append(1.0 if float(value) > 0.0 else 0.0)
        rows.append(converted_row)

    matrix = np.asarray(rows, dtype=float)
    np.fill_diagonal(matrix, 0.0)
    graph = nx.from_numpy_array(matrix.astype(int))

    if graph.number_of_nodes() <= 1:
        return 0.0

    if nx.is_connected(graph):
        subgraph = graph
    else:
        largest_component = max(nx.connected_components(graph), key=len)
        subgraph = graph.subgraph(largest_component).copy()

    if subgraph.number_of_nodes() <= 1:
        return 0.0

    return float(nx.average_shortest_path_length(subgraph))
