"""
Multi-objective optimization generation completion callback
多目的最適化の世代終了コールバック
"""

from typing import Any, Dict, List, Optional

from src.file_io.generation_writer import GenerationWriter
from src.tasks.core.data_type_network_metrics import NetworkMetrics

# memo callbackの処理検討

class GenerationCallback:
    """
    Write each completed generation to GenerationXXXXX.json.
    完了した各世代をGenerationXXXXX.jsonへ書き込む。
    """

    def __init__(self, generation_writer: GenerationWriter, param_names: List[str]):
        """
        Initialize the callback with its writer and ordered parameter names.
        Writerと順序付きパラメータ名を受け取ってコールバックを初期化する。
        """
        self.generation_writer = generation_writer
        self.param_names = param_names

    def __call__(
        self,
        generation: int,
        solutions: List[Dict[str, Any]],
        objective_values: List[tuple],
        pareto_size: int,
        metadata: Optional[List[Optional[NetworkMetrics]]] = None,
    ):
        """
        Write one generation through the configured GenerationWriter.
        設定済みのGenerationWriterを介して1世代を書き込む。
        """
        self.generation_writer.write(
            generation=generation,
            solutions=solutions,
            objective_values=objective_values,
            param_names=self.param_names,
            metadata=metadata,
        )

