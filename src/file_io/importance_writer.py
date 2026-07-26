"""
Variable importance result writer
変数重要度結果ライター
"""

from pathlib import Path
from typing import Any, Dict

from src.file_io.atomic_io import write_json_atomic


class ImportanceWriter:
    """
    Write variable importance results to VariableImportance.json.
    変数重要度結果をVariableImportance.jsonへ書き込む。
    """

    def __init__(self, result_dir: Path):
        """
        Initialize the writer for a result directory.
        結果ディレクトリを受け取ってライターを初期化する。
        """
        # result_dirはVariableImportance.jsonの書き込み先ディレクトリとする。
        self.result_dir = Path(result_dir)
        self.importance_path = self.result_dir / "VariableImportance.json"

    def write(self, importance_data: Dict[str, Any]) -> None:
        """
        Write variable importance data to VariableImportance.json.
        変数重要度データをVariableImportance.jsonへ書き込む。
        """
        # importance_dataはVariableImportanceCalculator.calculateの戻り値とし、I/O失敗は呼び出し元へ伝播する。
        write_json_atomic(importance_data, self.importance_path)

