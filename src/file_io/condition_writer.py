"""
Multi-objective optimization condition writer for Condition.json
Condition.jsonへの多目的最適化条件書き込み
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from src.file_io.atomic_io import write_json_atomic
from src.version import __version__


class ConditionWriter:
    """
    Write multi-objective optimization conditions to Condition.json.
    多目的最適化条件をCondition.jsonへ書き込む。
    """

    def __init__(self, result_dir: Path):
        """
        Initialize the writer for a result directory.
        結果ディレクトリを受け取ってライターを初期化する。
        """
        # result_dirはCondition.jsonの書き込み先ディレクトリとする。
        self.result_dir = Path(result_dir)
        self.condition_path = self.result_dir / "Condition.json"

    def write(self, task, optimizer_config: Dict[str, Any], ui_title: str = "") -> None:
        """
        Write Condition.json from a task and multi-objective optimization configuration.
        タスクと多目的最適化設定からCondition.jsonを書き込む。
        """
        # taskは必要なBaseTaskメソッドを実装し、optimizer_configは多目的最適化条件、ui_titleは可視化UI名を保持する。
        # 必須メソッドがなければAttributeError、書き込みに失敗すればOSErrorを呼び出し元へ伝播する。
        # タスクからメタデータを取得
        var_config = task.get_variables()
        objective_names = task.get_objective_names()
        objective_directions = task.get_directions()

        # VariableConfig からパラメータ情報を構築
        parameter_info = {}
        for name in var_config.variable_names:
            var_def = var_config.variables[name]
            info = {"type": var_def["type"]}
            if var_def["type"] in ("continuous", "integer"):
                info["bounds"] = var_def["bounds"]
            elif var_def["type"] == "categorical":
                info["n_categories"] = var_def["n_categories"]
                if "category_names" in var_def:
                    info["category_names"] = var_def["category_names"]
            parameter_info[name] = info

        # Condition.jsonへ書き込む辞書を構築する。
        payload = {
            "task_name": task.__class__.__name__,
            "task_instance_name": getattr(task, 'name', 'default'),
            "ui_title": ui_title,
            "app_version": __version__,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "parameter_names": list(var_config.variable_names),
            "parameter_info": parameter_info,
            "objective_names": list(objective_names),
            "objective_directions": list(objective_directions),
            "optimizer_config": optimizer_config,
        }

        # 書き込み途中のJSONを読み込ませないため、一時ファイル経由で置換する。
        write_json_atomic(payload, self.condition_path)

    def update_elapsed_time(self, elapsed_seconds: float) -> None:
        """
        Append multi-objective optimization elapsed time to Condition.json.
        多目的最適化の経過時間をCondition.jsonへ追記する。
        """
        # elapsed_secondsは多目的最適化の合計秒数とし、ファイル不在やI/O失敗は呼び出し元へ伝播する。
        # 既存の Condition.json を読み込む
        with open(self.condition_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)

        # 経過時間情報を追加
        payload['ended_at'] = datetime.now().isoformat(timespec="seconds")
        payload['elapsed_seconds'] = round(elapsed_seconds, 2)

        # 書き込み途中のJSONを読み込ませないため、一時ファイル経由で置換する。
        write_json_atomic(payload, self.condition_path)


