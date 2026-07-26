"""
Type definition for network metrics returned by tasks
タスクが返すネットワーク指標の型定義
"""

from typing import List, TypedDict

class NetworkMetrics(TypedDict):
    """
    Define the dictionary type for task network metrics.
    タスクが返すネットワーク指標の辞書型を定義する。
    """

    # 平均クラスタリング係数、平均最短経路長、隣接行列を辞書形式で保持する。
    clustering: float
    average_shortest_path_length: float
    adjacency_matrix: List[List[int]]
