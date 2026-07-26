"""
JSON reader for multi-objective optimization results
多目的最適化結果のJSON読み込み
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 高速なorjsonを使用
import orjson


class JSONReader:
    """
    Read Condition.json with caching and generation JSON with retries.
    Condition.jsonはキャッシュし、世代JSONは再試行付きで読み込む。
    """

    def __init__(self, result_dir: Path):
        """
        Initialize the reader for a result directory.
        結果ディレクトリを受け取ってリーダーを初期化する。
        """
        # result_dirはCondition.jsonと世代ファイルを含むディレクトリとする。
        self.result_dir = Path(result_dir)
        self.condition_path = result_dir / "Condition.json"
        self._condition_mtime: Optional[float] = None
        self._condition: Dict = self._default_condition()
        # 世代番号とパスだけを保持し、世代JSONの内容はキャッシュしない。
        self._generation_files: Dict[int, Path] = {}

    def _default_condition(self) -> Dict:
        """
        Return the default condition dictionary.
        既定のcondition辞書を返す。
        """
        return {
            "task_name": "Optimizer",
            "started_at": "",
            "parameter_names": [],
            "objective_names": ["f1", "f2"],
            "optimizer_config": {},
        }

    def read_condition(self, force: bool = False) -> Tuple[bool, Dict]:
        """
        Read Condition.json and return its changed flag with cached data.
        Condition.jsonを読み込み、更新有無とキャッシュ済みデータのタプルを返す。
        """
        # forceがTrueならファイル変更の有無にかかわらず再読み込みする。
        # 戻り値のchangedは再読み込み時にTrue、condition_dictは現在のConditionデータとする。
        if not self.condition_path.exists():
            return False, self._condition

        try:
            mtime = self.condition_path.stat().st_mtime
        except OSError:
            return False, self._condition

        if not force and self._condition_mtime == mtime:
            return False, self._condition

        condition = self._read_json_with_retry(self.condition_path)
        if condition:
            self._condition = condition
            self._condition_mtime = mtime
            return True, self._condition

        return False, self._condition

    def scan_generation_files(self) -> Tuple[bool, List[int]]:
        """
        Scan generation files and return a changed flag with sorted indices.
        世代ファイルを走査し、更新有無とソート済み世代インデックスのタプルを返す。
        """
        # changedは世代ファイル集合の変更時にTrue、available_generationsは利用可能な世代番号とする。
        if not self.result_dir.exists():
            return False, []

        generation_files: Dict[int, Path] = {}
        max_gen = self._condition.get("optimizer_config", {}).get("max_generations", 0) - 1

        for path in self.result_dir.glob("Generation*.json"):
            gen = self._parse_generation_index(path.stem)
            if gen is not None and (max_gen < 0 or gen <= max_gen):
                generation_files[gen] = path

        available = sorted(generation_files.keys())
        changed = generation_files.keys() != self._generation_files.keys()
        self._generation_files = generation_files

        return changed, available

    def read_generation(self, generation: int) -> Optional[Dict]:
        """
        Read a generation and return its data dictionary or None.
        指定世代を読み込み、データ辞書またはNoneを返す。
        """
        # generationは0始まりの世代インデックスとする。
        path = self._generation_files.get(generation)
        if not path:
            return None
        return self._read_json_with_retry(path)

    def _parse_generation_index(self, stem: str) -> Optional[int]:
        """
        Parse a filename stem and return its generation index or None.
        ファイル名のstemを解析し、世代インデックスまたはNoneを返す。
        """
        # Generation00000 とする。
        prefix = "Generation"
        if not stem.startswith(prefix):
            return None
        suffix = stem[len(prefix):]
        return int(suffix) if suffix.isdigit() else None

    def _read_json_with_retry(self, path: Path, max_retries: int = 3) -> Optional[Dict]:
        """
        Read JSON quickly with orjson and return a dictionary or None after permission retries.
        orjsonでJSONを高速に読み、権限エラー時は再試行して辞書またはNoneを返す。
        """
        # pathは読み込み先、max_retriesは最大試行回数とする。
        for attempt in range(max_retries):
            try:
                return orjson.loads(path.read_bytes())
            except PermissionError:
                if attempt >= max_retries - 1:
                    return None
                time.sleep(1.0)
            except (OSError, orjson.JSONDecodeError):
                return None
        return None
