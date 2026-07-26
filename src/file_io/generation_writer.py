"""
Multi-objective optimization generation result writer
多目的最適化の世代結果ライター
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.file_io.atomic_io import write_json_atomic
from src.tasks.core.data_type_network_metrics import NetworkMetrics


class GenerationWriter:
    """
    Write each generation of multi-objective optimization to GenerationXXXXX.json.
    多目的最適化の各世代をGenerationXXXXX.jsonへ書き込む。
    """

    def __init__(self, result_dir: Path):
        """
        Initialize the writer for a result directory.
        結果ディレクトリを受け取ってライターを初期化する。
        """
        # result_dirは世代ファイルの書き込み先ディレクトリとする。
        self.result_dir = Path(result_dir)

    @staticmethod
    def _normalize_adjacency_matrix(matrix: Any) -> Optional[List[List[int]]]:
        """
        Normalize a square adjacency matrix and return integer rows or None.
        正方隣接行列を正規化し、整数行列またはNoneを返す。
        """
        if not isinstance(matrix, (list, tuple)):
            return None
        size = len(matrix)
        if size == 0:
            return None
        normalized: List[List[int]] = []
        for row in matrix:
            if not isinstance(row, (list, tuple)) or len(row) != size:
                return None
            normalized_row: List[int] = []
            for value in row:
                if not isinstance(value, (int, bool)):
                    return None
                normalized_row.append(int(value))
            normalized.append(normalized_row)
        return normalized

    def write(
        self,
        generation: int,
        solutions: List[Dict[str, Any]],
        objective_values: List[tuple],
        param_names: List[str],
        metadata: Optional[List[Optional[NetworkMetrics]]] = None
    ) -> None:
        """
        Write one generation with ordered parameters and optional NetworkMetrics.
        順序付きパラメータと任意のNetworkMetricsを含む1世代を書き込む。
        """
        # generationは0始まり、solutionsとobjective_valuesは各解の辞書と目的関数値タプルのリストとする。
        # param_namesはCondition.jsonのparameter_namesと同じ順序とし、metadataは各解のNetworkMetricsまたはNoneとする。
        # パラメータ不足によるKeyErrorと書き込み時のOSErrorは呼び出し元へ伝播する。
        # 各解の目的関数値をJSON保存用のfloatリストへ変換する。
        objectives = []
        for values in objective_values:
            objective_row = []
            for value in values:
                objective_row.append(float(value))
            objectives.append(objective_row)

        # 各解の決定変数をCondition.jsonのパラメータ順でfloatリストへ変換する。
        params = []
        for solution in solutions:
            param_row = []
            for name in param_names:
                param_row.append(float(solution[name]))
            params.append(param_row)

        payload = {
            "generation": generation,
            "objectives": objectives,
            "params": params,
        }

        # メタデータからネットワーク指標を抽出して追加
        if metadata:
            clustering = []
            adjacency_matrices = []
            average_shortest_path_length = []
            for meta in metadata:
                clustering_val = meta.get('clustering') if meta else None
                if clustering_val is not None:
                    try:
                        clustering_val = float(clustering_val)
                    except (TypeError, ValueError):
                        clustering_val = None
                clustering.append(clustering_val)
                adjacency_matrix = meta.get('adjacency_matrix') if meta else None
                adjacency_matrices.append(self._normalize_adjacency_matrix(adjacency_matrix))
                path_val = meta.get('average_shortest_path_length') if meta else None
                if path_val is not None:
                    try:
                        path_val = float(path_val)
                    except (TypeError, ValueError):
                        path_val = None
                average_shortest_path_length.append(path_val)
            # 全てNoneでなければ追加
            if any(c is not None for c in clustering):
                payload["clustering"] = clustering
            if any(m is not None for m in adjacency_matrices):
                payload["adjacency_matrices"] = adjacency_matrices
            if any(v is not None for v in average_shortest_path_length):
                payload["average_shortest_path_length"] = average_shortest_path_length

        filepath = self.result_dir / f"Generation{generation:05d}.json"
        write_json_atomic(payload, filepath)

