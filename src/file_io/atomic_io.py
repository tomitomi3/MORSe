"""
JSON writing utility using temporary-file replacement
一時ファイルを置換して保存するJSON書き込みユーティリティ
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict


def write_json_atomic(payload: Dict[str, Any], filepath: Path) -> None:
    """
    Write a dictionary as JSON to a temporary file and replace the specified path.
    辞書を一時ファイルへJSONとして書き込み、指定パスを置換する。
    """
    # 書き込み途中のJSONを読み込ませないため、一時ファイルへ保存してから指定パスを置換する。
    # payloadはJSON化する辞書、filepathは書き込み先パスとする。
    # PermissionErrorは最大5回まで再試行し、解消しない場合とその他のOSErrorは呼び出し元へ伝播する。
    filepath.parent.mkdir(parents=True, exist_ok=True)
    temp_path = filepath.with_suffix(filepath.suffix + ".tmp")
    max_retries = 5

    for attempt in range(max_retries):
        try:
            with temp_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())  # ディスクへの書き込みを強制

            # 一時ファイルを書き込み先へ置換する。
            os.replace(temp_path, filepath)
            return

        except PermissionError:
            if attempt >= max_retries - 1:
                raise
            # PermissionErrorの再試行ごとに待機時間を0.05秒ずつ延ばす。
            time.sleep(0.05 * (attempt + 1))
