"""
Bokeh server for multi-objective optimization result visualization
多目的最適化結果可視化用Bokehサーバー
"""

import html
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import orjson
from bokeh.events import DocumentReady, Tap as TapEvent
from bokeh.layouts import column, row
from bokeh.models import (
    BasicTicker,
    BoxSelectTool,
    Button,
    BasicTickFormatter,
    ColorBar,
    ColumnDataSource,
    CustomJS,
    DataRange1d,
    Div,
    HoverTool,
    InlineStyleSheet,
    LinearScale,
    LinearColorMapper,
    LogScale,
    LogTicker,
    PrintfTickFormatter,
    RadioButtonGroup,
    Range1d,
    ScrollBox,
    TabPanel,
    TapTool,
    Tabs,
    TextAreaInput,
    TextInput,
)
from bokeh.palettes import RdBu11
from bokeh.plotting import curdoc, figure

from src.optimizer.utils.pareto_utils import extract_pareto_front
from src.utils.hypervolume import compute_auto_reference_point, compute_hypervolume
from src.utils.common_messages import operation_failed_message, required_value_message
from src.version import __version__
from src.visualization.importance_utils import align_importance_series
from src.visualization.network_metrics_utils import (
    build_adjacency_image_data,
    build_degree_histogram_bar_data,
    collect_adjacency_matrices,
    compute_degree_histogram_stats,
)
from src.visualization.selection_and_range_utils import (
    collect_overlapped_indices_by_tolerance,
    compute_data_tolerance_from_pixels,
    order_indices_by_original_index,
    resolve_tap_cycle_selection,
    resolve_text_input_value,
    set_text_input_value_and_editing,
    round_up_to_power_of_ten,
    sanitize_fixed_range,
)
from src.visualization.violin_utils import create_violin_data, normalize_violin_values

def _generate_step_line_data(
    pareto_points: np.ndarray,
    directions: List[str]
) -> Tuple[List[float], List[float]]:
    """
    Convert Pareto points and return step-line coordinate lists.
    パレート点を変換し、階段状ラインの座標リストを返す。
    """
    # pareto_pointsは2目的の点、directionsはminimizeまたはmaximizeを並べた列とする。
    if len(pareto_points) < 2:
        if len(pareto_points) == 1:
            return [pareto_points[0, 0]], [pareto_points[0, 1]]
        return [], []

    x_dir = directions[0] if len(directions) > 0 else "minimize"

    # X軸でソート（minimizeなら昇順、maximizeなら降順）
    if x_dir == "minimize":
        sorted_idx = np.argsort(pareto_points[:, 0])
    else:
        sorted_idx = np.argsort(-pareto_points[:, 0])

    sorted_points = pareto_points[sorted_idx]

    # 階段状データを生成
    x_coords = []
    y_coords = []

    # 支配領域（worse side）の境界を示すため、常に水平→垂直（post）の階段にする

    for i in range(len(sorted_points)):
        x, y = sorted_points[i]
        x_coords.append(x)
        y_coords.append(y)

        if i < len(sorted_points) - 1:
            next_x, next_y = sorted_points[i + 1]
            # post: 水平移動→垂直移動
            x_coords.append(next_x)
            y_coords.append(y)

    return x_coords, y_coords
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_AUTO_UPDATE_INTERVAL_MS = 2000

# UI サイズ定数（ここで各要素のサイズを調整）
SECTION_SPACING = 20        # セクション間のスペース
SAVE_PANEL_WIDTH = 200      # 保存操作パネル幅

LEFT_PADDING = 24
TOP_PADDING = 16

SCATTER_WIDTH = 650
SCATTER_HEIGHT = 550
HYPERVOLUME_WIDTH = 600
HYPERVOLUME_HEIGHT = 180
VIOLIN_HEIGHT = 240
IMPORTANCE_WIDTH = 360
IMPORTANCE_HEIGHT = 280
CORRELATION_WIDTH = 400
CORRELATION_HEIGHT = 360
VIOLIN_WIDTH = 700
DEGREE_HISTOGRAM_WIDTH = 600
DEGREE_HISTOGRAM_HEIGHT = 320
ADJACENCY_HEATMAP_SIZE = 320
NETWORK_GRID_CELL_MIN_WIDTH = 320
NETWORK_TEXT_CELL_HEIGHT = 30
NETWORK_NOTE_CELL_HEIGHT = 22
SOLUTION_TEXT_WIDTH = 500
SOLUTION_TEXT_HEIGHT = 340
TITLE_BAR_HEIGHT = 56
CONTENT_TOP_OFFSET = TOP_PADDING + TITLE_BAR_HEIGHT + SECTION_SPACING
LEFT_PANEL_WIDTH = SCATTER_WIDTH
LEFT_PANEL_CONTENT_MIN_WIDTH = 750
RIGHT_PANEL_MIN_WIDTH = 560
AXIS_RANGE_INPUT_WIDTH = 120
PARETO_TAP_OVERLAP_RADIUS_PX = 10.0
NETWORK_METRICS_REPRESENTATIVE_NOTE = (
    "A plotted solution may average multiple task evaluations; "
    "Network Metrics show one evaluated reservoir."
)


def _create_section_header(title: str) -> Div:
    """
    Create and return a styled section-header Div.
    スタイル付きセクションヘッダーDivを生成して返す。
    """
    style = (
        "font-weight: bold; font-size: 16px; color: #333; "
        "padding: 10px 0 3px 0; border-bottom: 3px solid #1E2973; "
        "margin-bottom: 10px;"
    )
    return Div(text=f"<div style='{style}'>{title}</div>")


class VisualizerServer:
    """
    Serve interactive views of saved multi-objective optimization results.
    保存済み多目的最適化結果の対話的な表示を提供する。
    """

    def __init__(self, result_dir: Optional[str] = None):
        """
        Initialize visualization state for a result directory.
        結果ディレクトリを受け取って可視化状態を初期化する。
        """
        if isinstance(result_dir, str) and result_dir.strip():
            selected_result_dir = result_dir.strip()
        elif len(sys.argv) > 1 and str(sys.argv[1]).strip():
            selected_result_dir = str(sys.argv[1]).strip()
        else:
            raise ValueError(required_value_message("result_dir"))

        self._ui_title = self._load_setting_ui_title()
        self.result_dir = self._resolve_result_dir(selected_result_dir)
        self.condition_path = self.result_dir / "Condition.json"
        self.current_display_generation: Optional[int] = None
        self.auto_update = True
        self.auto_update_interval_ms = DEFAULT_AUTO_UPDATE_INTERVAL_MS

        self.condition = self._default_condition()
        self._param_names: List[str] = []
        self._generation_files: Dict[int, Path] = {}
        self._available_generations: List[int] = []
        self._latest_generation = -1
        self._condition_mtime: Optional[float] = None
        self._display_payload: Optional[Dict] = None
        self._error_message: Optional[str] = None
        self._halt_updates = False
        self._hypervolume_ref_point: Tuple[float, float] = (1.1, 1.1)
        self._hypervolume_ref_point_auto: bool = False  # 自動計算済みフラグ
        self._hypervolume_cache: Dict[int, float] = {}
        self._hypervolume_cache_ref: Optional[Tuple[float, float]] = None
        self._objective_names: List[str] = ["f1", "f2"]  # 既定値はCondition.jsonの読み込み時に更新する
        self._objective_directions: List[str] = ["minimize", "minimize"]  # 既定の最適化方向
        self._param_info: Dict = {}  # 変数型情報（parameter_info from Condition.json）
        self._display_clustering: List[Any] = []
        self._display_path_length: List[Any] = []
        self._display_adjacency: List[Any] = []

        # 変数重要度データ
        self._variable_importance: Optional[Dict] = None
        self._importance_mtime: Optional[float] = None

        self._load_condition(force=True)
        self._scan_generation_files()
        self._load_variable_importance()

    def _default_condition(self) -> Dict:
        """
        Build and return the default condition dictionary.
        既定の条件辞書を構築して返す。
        """
        return {
            "task_name": "Optimizer",
            "task_instance_name": "default",
            "ui_title": self._ui_title,
            "started_at": "",
            "parameter_names": [],
            "objective_names": ["f1", "f2"],
            "optimizer_config": {},
        }

    def _empty_objective_source_data(self) -> Dict[str, List[Any]]:
        """
        Return an empty lightweight Objective Space data source.
        空の軽量Objective Spaceデータソースを返す。
        """
        return {
            "f1": [],
            "f2": [],
            "params": [],
            "is_pareto": [],
            "original_index": [],
            "color": [],
            "line_color": [],
            "line_width": [],
        }

    def _build_objective_source_data(
        self,
        x_vals: List[float],
        y_vals: List[float],
        params: List[Any],
        pareto_set: set,
    ) -> Dict[str, List[Any]]:
        """
        Build and return lightweight Objective Space source data.
        軽量Objective Spaceデータソースを構築して返す。
        """
        all_f1: List[float] = []
        all_f2: List[float] = []
        all_params: List[Any] = []
        all_is_pareto: List[bool] = []
        all_original_index: List[int] = []
        all_color: List[str] = []
        all_line_color: List[str] = []
        all_line_width: List[float] = []

        for idx in range(len(x_vals)):
            all_f1.append(x_vals[idx])
            all_f2.append(y_vals[idx])
            all_params.append(params[idx] if idx < len(params) else [])
            all_original_index.append(idx)
            is_pareto = idx in pareto_set
            all_is_pareto.append(is_pareto)
            all_color.append("red" if is_pareto else "lightblue")
            all_line_color.append("black" if is_pareto else "lightblue")
            all_line_width.append(1.5 if is_pareto else 0)

        return {
            "f1": all_f1,
            "f2": all_f2,
            "params": all_params,
            "is_pareto": all_is_pareto,
            "original_index": all_original_index,
            "color": all_color,
            "line_color": all_line_color,
            "line_width": all_line_width,
        }

    def _clear_display_network_metrics(self) -> None:
        """
        Clear network metrics associated with the displayed generation.
        表示世代に対応するネットワーク指標を消去する。
        """
        self._display_clustering = []
        self._display_path_length = []
        self._display_adjacency = []

    def _set_display_network_metrics_from_payload(self, payload: Optional[Dict[str, Any]]) -> None:
        """
        Set displayed network metrics from a generation payload.
        世代payloadから表示対象のネットワーク指標を設定する。
        """
        if not isinstance(payload, dict):
            self._clear_display_network_metrics()
            return

        clustering = payload.get("clustering", [])
        path_length = payload.get("average_shortest_path_length", [])
        adjacency = payload.get("adjacency_matrices", [])

        self._display_clustering = list(clustering) if isinstance(clustering, list) else []
        self._display_path_length = list(path_length) if isinstance(path_length, list) else []
        self._display_adjacency = list(adjacency) if isinstance(adjacency, list) else []

    def _get_network_metrics_by_original_index(
        self,
        original_index: int,
    ) -> Tuple[Optional[float], Optional[float], Any]:
        """
        Return network metrics for an original population index.
        元の集団インデックスに対応するネットワーク指標を返す。
        """
        def _to_optional_float(values: List[Any], idx: int) -> Optional[float]:
            """
            Return one sequence value as float or None.
            列の1要素をfloatまたはNoneとして返す。
            """
            if idx < 0 or idx >= len(values):
                return None
            value = values[idx]
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        clustering_value = _to_optional_float(self._display_clustering, original_index)
        path_value = _to_optional_float(self._display_path_length, original_index)
        adjacency_value: Any = None
        if 0 <= original_index < len(self._display_adjacency):
            adjacency_value = self._display_adjacency[original_index]
        return clustering_value, path_value, adjacency_value

    def _get_network_metrics_by_selected_index(
        self,
        selected_index: int,
        original_indices: List[Any],
    ) -> Tuple[Optional[float], Optional[float], Any]:
        """
        Return network metrics for a selected source index.
        選択したソースインデックスに対応するネットワーク指標を返す。
        """
        if selected_index < 0 or selected_index >= len(original_indices):
            return None, None, None
        try:
            original_index = int(original_indices[selected_index])
        except (TypeError, ValueError):
            return None, None, None
        return self._get_network_metrics_by_original_index(original_index)

    def _load_setting_ui_title(self) -> str:
        """
        Load and return the visualization title from configs/setting.json.
        configs/setting.jsonから可視化タイトルを読み込んで返す。
        """
        setting_path = ROOT_DIR / "configs" / "setting.json"
        if setting_path.exists():
            try:
                setting = orjson.loads(setting_path.read_bytes())
                title = str(setting.get("ui_title", "")).strip()
                if title:
                    return title
            except (OSError, orjson.JSONDecodeError, TypeError, ValueError, AttributeError):
                pass
        return "Optimizer"

    def _resolve_result_dir(self, result_dir: str) -> Path:
        """
        Resolve and return an absolute result directory path.
        結果ディレクトリを解決して絶対パスで返す。
        """
        path = Path(result_dir)
        if not path.is_absolute():
            path = ROOT_DIR / path
        return path

    def _set_error(self, message: str) -> None:
        """
        Record the first fatal visualization error and halt updates.
        最初の致命的な可視化エラーを記録して更新を停止する。
        """
        if self._error_message:
            return
        self._error_message = message
        self._halt_updates = True
        self.auto_update = False

    def _read_json_with_retry(self, path: Path) -> Optional[Dict]:
        """
        Read JSON with lock retries and return a dictionary or None.
        ロック時に再試行してJSONを読み込み、辞書またはNoneを返す。
        """
        for attempt in range(3):
            try:
                return orjson.loads(path.read_bytes())
            except PermissionError:
                if attempt >= 2:
                    message = operation_failed_message("File read", path.name)
                    self._set_error(message)
                    return None
                time.sleep(1.0)
            except (OSError, orjson.JSONDecodeError):
                return None
        return None

    def _load_condition(self, force: bool = False) -> bool:
        """
        Load Condition.json and return whether state was updated.
        Condition.jsonを読み込み、状態を更新したかどうかを返す。
        """
        if self._halt_updates:
            return False
        if not self.condition_path.exists():
            return False
        try:
            mtime = self.condition_path.stat().st_mtime
        except OSError:
            return False
        if not force and self._condition_mtime == mtime:
            return False

        condition = self._read_json_with_retry(self.condition_path)
        if not isinstance(condition, dict):
            return False

        self.condition = condition
        self._param_names = condition.get("parameter_names", [])
        self._param_info = condition.get("parameter_info", {})
        self._objective_names = condition.get("objective_names", ["f1", "f2"])
        self._objective_directions = condition.get("objective_directions", ["minimize", "minimize"])
        loaded_ui_title = str(condition.get("ui_title", "")).strip()
        if loaded_ui_title:
            self._ui_title = loaded_ui_title
        self._condition_mtime = mtime
        return True

    def _format_param_value(self, name: str, value) -> str:
        """
        Format and return a parameter value according to its variable type.
        変数型に応じてパラメータ値を整形して返す。
        """
        info = self._param_info.get(name)
        if info is None:
            try:
                return f"{float(value):.6f}"
            except (TypeError, ValueError):
                return str(value)

        var_type = info.get("type")
        if var_type == "integer":
            try:
                return str(int(round(float(value))))
            except (TypeError, ValueError):
                return str(value)
        elif var_type == "categorical":
            try:
                idx = int(round(float(value)))
            except (TypeError, ValueError):
                return str(value)
            category_labels = self._get_category_labels(info)
            if category_labels and 0 <= idx < len(category_labels):
                return f"{category_labels[idx]} ({idx})"
            return str(idx)
        else:
            # continuous またはその他
            try:
                return f"{float(value):.6f}"
            except (TypeError, ValueError):
                return str(value)

    def _get_category_labels(self, info: Dict) -> List[str]:
        """
        Return display labels for a categorical variable.
        カテゴリ変数の表示ラベルを返す。
        """
        if not isinstance(info, dict):
            return []

        category_names = info.get("category_names")
        if isinstance(category_names, list) and category_names:
            return [str(name) for name in category_names]

        n_categories = info.get("n_categories")
        try:
            n_cat = int(n_categories)
        except (TypeError, ValueError):
            return []

        if n_cat <= 0:
            return []
        return [f"category_{i}" for i in range(n_cat)]

    def _convert_param_for_export(self, name: str, value):
        """
        Convert and return a parameter value for export.
        パラメータ値をダウンロード用に変換して返す。
        """
        info = self._param_info.get(name)
        if info is None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return value

        var_type = info.get("type")
        if var_type == "integer":
            try:
                return int(round(float(value)))
            except (TypeError, ValueError):
                return value
        elif var_type == "categorical":
            try:
                idx = int(round(float(value)))
            except (TypeError, ValueError):
                return value
            category_labels = self._get_category_labels(info)
            if category_labels and 0 <= idx < len(category_labels):
                return category_labels[idx]
            return idx
        else:
            # 連続変数の場合
            try:
                return float(value)
            except (TypeError, ValueError):
                return value

    def _compute_auto_reference_point(self) -> bool:
        """
        Compute an automatic reference point and return whether it succeeded.
        自動基準点を計算し、成功したかどうかを返す。
        """
        # Generation00000.jsonの目的関数値から最悪値を求め、10%の余白を追加する。
        if self._hypervolume_ref_point_auto:
            return True  # 既に計算済み

        gen0_path = self._generation_files.get(0)
        if gen0_path is None:
            return False

        payload = self._read_json_with_retry(gen0_path)
        if not isinstance(payload, dict):
            return False

        pairs = self._get_objective_pairs(payload)
        if not pairs:
            return False

        objectives = np.array(pairs)
        ref_point = compute_auto_reference_point(
            objectives, self._objective_directions[:2])

        self._hypervolume_ref_point = (float(ref_point[0]), float(ref_point[1]))
        self._hypervolume_ref_point_auto = True
        return True

    def _get_max_generations(self) -> int:
        """
        Return the configured nonnegative generation count.
        設定された非負の世代数を返す。
        """
        value = self.condition.get("optimizer_config", {}).get("max_generations", 0)
        try:
            max_generations = int(value)
        except (TypeError, ValueError):
            max_generations = 0
        return max_generations if max_generations > 0 else 0

    def _get_max_generation_index(self) -> Optional[int]:
        """
        Return the configured final generation index or None.
        設定上の最終世代インデックスまたはNoneを返す。
        """
        max_generations = self._get_max_generations()
        if max_generations <= 0:
            return None
        return max_generations - 1

    def _format_generation_label(self, generation_index: int) -> str:
        """
        Format and return a one-based generation label.
        1始まりの世代ラベルへ整形して返す。
        """
        if generation_index < 0:
            return "-"
        return str(generation_index + 1)

    def _get_title_text(self) -> str:
        """
        Build and return the application title text.
        アプリケーションのタイトル文字列を構築して返す。
        """
        task_name = self.condition.get("task_name") or "N/A"
        return f"{self._ui_title} - {task_name}"

    def _parse_generation_index(self, stem: str) -> Optional[int]:
        """
        Parse a generation filename stem and return its index or None.
        世代ファイル名のstemを解析し、インデックスまたはNoneを返す。
        """
        prefix = "Generation"
        if not stem.startswith(prefix):
            return None
        suffix = stem[len(prefix):]
        if not suffix.isdigit():
            return None
        return int(suffix)

    def _transform_objectives_for_display(self, objectives):
        """
        Extract objectives for display.

        値はそのまま表示（TaskRunnerが既に変換済み）。
        """
        if not objectives or len(objectives) == 0:
            return [], []

        x_vals = []
        y_vals = []
        for objective_row in objectives:
            x_val = objective_row[0]
            y_val = objective_row[1] if len(objective_row) > 1 else 0
            x_vals.append(x_val)
            y_vals.append(y_val)

        return x_vals, y_vals

    def _scan_generation_files(self) -> bool:
        """
        Scan generation files and return whether their set changed.
        世代ファイルを走査し、ファイル集合が変化したかどうかを返す。
        """
        if self._halt_updates:
            return False
        if not self.result_dir.exists():
            self._generation_files = {}
            self._available_generations = []
            self._latest_generation = -1
            return False
        generation_files: Dict[int, Path] = {}
        max_generation_index = self._get_max_generation_index()
        for path in self.result_dir.glob("Generation*.json"):
            gen = self._parse_generation_index(path.stem)
            if gen is not None and (max_generation_index is None or gen <= max_generation_index):
                generation_files[gen] = path
        available = sorted(generation_files.keys())
        latest = available[-1] if available else -1
        changed = generation_files.keys() != self._generation_files.keys()
        self._generation_files = generation_files
        self._available_generations = available
        self._latest_generation = latest

        # Generation00000.json が見つかったら参照点を自動計算
        if 0 in generation_files and not self._hypervolume_ref_point_auto:
            self._compute_auto_reference_point()

        return changed

    def _load_variable_importance(self, force: bool = False) -> bool:
        """
        Load VariableImportance.json and return whether data was updated.
        VariableImportance.jsonを読み込み、データを更新したかどうかを返す。
        """
        if self._halt_updates:
            return False

        importance_path = self.result_dir / "VariableImportance.json"
        if not importance_path.exists():
            return False

        try:
            mtime = importance_path.stat().st_mtime
        except OSError:
            return False

        if not force and self._importance_mtime == mtime:
            return False

        data = self._read_json_with_retry(importance_path)
        if data:
            self._variable_importance = data
            self._importance_mtime = mtime
            return True

        return False

    def _get_display_generation(self) -> int:
        """
        Return the explicit display generation or the latest generation.
        明示指定した表示世代、または最新世代を返す。
        """
        if self.current_display_generation is None:
            return self._latest_generation
        return self.current_display_generation

    def _get_previous_generation(self, steps: int = 1) -> Optional[int]:
        """
        Return an available generation before the current position or None.
        現在位置より前の利用可能な世代、またはNoneを返す。
        """
        if not self._available_generations:
            return None
        normalized_steps = max(1, int(steps))
        cursor = (
            self._available_generations[-1]
            if self.current_display_generation is None
            else self.current_display_generation
        )
        target: Optional[int] = None
        for _ in range(normalized_steps):
            found: Optional[int] = None
            for gen in reversed(self._available_generations):
                if gen < cursor:
                    found = gen
                    break
            if found is None:
                break
            target = found
            cursor = found
        return target

    def _get_next_generation(self, steps: int = 1) -> Optional[int]:
        """
        Return an available generation after the current position or None.
        現在位置より後の利用可能な世代、またはNoneを返す。
        """
        if not self._available_generations:
            return None
        if self.current_display_generation is None:
            return None
        normalized_steps = max(1, int(steps))
        cursor = self.current_display_generation
        target: Optional[int] = None
        for _ in range(normalized_steps):
            found: Optional[int] = None
            for gen in self._available_generations:
                if gen > cursor:
                    found = gen
                    break
            if found is None:
                break
            target = found
            cursor = found
        return target

    def _load_generation_payload(self, generation_index: int) -> Optional[Dict]:
        """
        Load and return a generation payload or None.
        指定世代のpayloadを読み込み、辞書またはNoneを返す。
        """
        if generation_index < 0:
            return None
        path = self._generation_files.get(generation_index)
        if path is None:
            return None
        return self._read_json_with_retry(path)

    def _get_objective_pairs(self, payload: Dict) -> List[Tuple[float, float]]:
        """
        Extract and return valid two-objective pairs from a payload.
        payloadから有効な2目的の値ペアを抽出して返す。
        """
        objectives = payload.get("objectives", [])
        pairs: List[Tuple[float, float]] = []
        if not isinstance(objectives, list):
            return pairs
        for objective_row in objectives:
            if not isinstance(objective_row, list) or len(objective_row) < 2:
                continue
            try:
                f1 = float(objective_row[0])
                f2 = float(objective_row[1])
            except (TypeError, ValueError):
                continue
            pairs.append((f1, f2))
        return pairs

    def _compute_hypervolume(self, payload: Dict, ref_point: Tuple[float, float]) -> float:
        """
        Compute and return hypervolume for one generation payload.
        1世代のpayloadからハイパーボリュームを計算して返す。
        """
        pairs = self._get_objective_pairs(payload)
        if not pairs:
            return 0.0

        front = np.array(pairs)
        ref = np.array(ref_point)
        return compute_hypervolume(front, ref, self._objective_directions[:2])

    def _update_hypervolume_source(self, source: ColumnDataSource, force: bool = False) -> None:
        """
        Update a Bokeh source with hypervolume history.
        Bokehデータソースをハイパーボリューム履歴で更新する。
        """
        if self._halt_updates:
            return
        if force or self._hypervolume_cache_ref != self._hypervolume_ref_point:
            self._hypervolume_cache = {}
            self._hypervolume_cache_ref = self._hypervolume_ref_point
        gens: List[int] = []
        values: List[float] = []
        for gen in self._available_generations:
            hv = self._hypervolume_cache.get(gen)
            if hv is None:
                payload = self._load_generation_payload(gen)
                if not isinstance(payload, dict):
                    continue
                hv = self._compute_hypervolume(payload, self._hypervolume_ref_point)
                self._hypervolume_cache[gen] = hv
            gens.append(gen + 1)
            values.append(hv)
        source.data = {"gen": gens, "hv": values}

    def _create_title_html(self) -> str:
        """
        Build and return escaped application title HTML.
        エスケープ済みのアプリケーションタイトルHTMLを構築して返す。
        """
        task_name = self.condition.get("task_name") or "N/A"
        escaped_title = html.escape(self._ui_title)
        escaped_task_name = html.escape(str(task_name))
        return (
            "<div style='font-size:16px; line-height:1.2; display:flex; "
            "align-items:center; gap:20px; color:#ffffff;'>"
            "<span style='display:inline-flex; align-items:baseline; gap:6px;'>"
            "<span style='font-size:21px; font-weight:600; letter-spacing:1px;'>"
            f"{escaped_title}</span>"
            f"<span style='font-size:16px;'>v{__version__}</span>"
            "</span>"
            f"<span>Task: {escaped_task_name}</span>"
            "</div>"
        )

    def _create_generation_html(self) -> str:
        """
        Build HTML showing the displayed and maximum generation numbers.
        表示中の世代番号と最大世代数を示すHTMLを構築して返す。
        """
        max_generations = self._get_max_generations()
        display_generation = self._get_display_generation()
        display_generation_label = self._format_generation_label(display_generation)
        max_generation_label = str(max_generations) if max_generations > 0 else "-"
        return (
            "<div style='font-size:14px; line-height:1.6;'>"
            f"<span style='display:inline-block; width:80px; color:#555;'>Generation</span>"
            f"<span>{display_generation_label} / {max_generation_label}</span>"
            "</div>"
        )

    def create_app(self):
        """
        Build and return the interactive Bokeh application layout.
        対話的なBokehアプリケーションレイアウトを構築して返す。
        """
        # 統合データソース（パレートフロント解と非パレートフロント解を統合）
        source = ColumnDataSource(data=self._empty_objective_source_data())
        hypervolume_source = ColumnDataSource(data={
            "gen": [],
            "hv": [],
        })

        title_div = Div(text=self._create_title_html())
        generation_div = Div(text=self._create_generation_html(), width=160)

        # 目的関数名へ最適化方向を加えた軸ラベルを構築する
        obj_names = self._objective_names
        obj_dirs = self._objective_directions

        def make_label(idx, default):
            """
            Build and return an objective-axis label with its direction.
            最適化方向を含む目的関数の軸ラベルを構築して返す。
            """
            name = obj_names[idx] if idx < len(obj_names) else default
            if idx < len(obj_dirs):
                return f"{name} ({obj_dirs[idx]})"
            return name

        x_label = make_label(0, "f1")
        y_label = make_label(1, "f2")

        pareto_front_fig = figure(
            title=None,
            x_axis_label=x_label,
            y_axis_label=y_label,
            width=SCATTER_WIDTH,
            height=SCATTER_HEIGHT,
            tools="pan,reset,save,tap,box_select,wheel_zoom,box_zoom",
            active_drag="box_select",  # デフォルトで選択モード
            active_scroll="wheel_zoom",
        )
        pareto_front_fig.toolbar.logo = None
        # グリッド線のスタイル設定
        pareto_front_fig.xgrid.grid_line_color = "#e0e0e0"
        pareto_front_fig.ygrid.grid_line_color = "#e0e0e0"
        pareto_front_fig.xgrid.grid_line_width = 1
        pareto_front_fig.ygrid.grid_line_width = 1
        # 統合された単一データソースで描画（パレート/非パレートは色で区別）
        scatter_renderer = pareto_front_fig.scatter(
            "f1",
            "f2",
            source=source,
            marker="circle",
            size=8,
            alpha=0.7,
            color="color",
            line_color="line_color",
            line_width="line_width",
            selection_color="orange",
            nonselection_alpha=0.3,
        )

        # パレートフロント用階段状 line データソース
        pareto_step_source = ColumnDataSource(data={'x': [], 'y': []})
        # 階段状 line を描画（パレートフロントを階段状に結ぶ）
        pareto_front_fig.line(
            'x',
            'y',
            source=pareto_step_source,
            line_color="red",
            line_width=1.5,
            alpha=0.8,
        )

        hover = HoverTool(
            tooltips=[(x_label, "@f1{0.0000}"), (y_label, "@f2{0.0000}")],
            renderers=[scatter_renderer],
        )
        pareto_front_fig.add_tools(hover)

        tap_tool = pareto_front_fig.select_one(TapTool)
        if tap_tool is not None:
            tap_tool.renderers = [scatter_renderer]

        box_select_tool = pareto_front_fig.select_one(BoxSelectTool)
        if box_select_tool is not None:
            box_select_tool.renderers = [scatter_renderer]

        # バイオリンプロット用Figure（Min-Max正規化表示）
        violin_fig = figure(
            x_axis_label="Parameter",
            y_axis_label="Normalized Value (0-1)",
            width=VIOLIN_WIDTH,
            height=VIOLIN_HEIGHT,
            tools="save",
            toolbar_location="above",
            x_range=Range1d(start=-0.5, end=max(len(self._param_names) - 0.5, 0.5)),
            y_range=Range1d(start=-0.05, end=1.05),
            sizing_mode="fixed",
        )
        violin_fig.toolbar.logo = None
        # グリッド線のスタイル設定
        violin_fig.xgrid.grid_line_color = "#e0e0e0"
        violin_fig.ygrid.grid_line_color = "#e0e0e0"

        # パラメータ名に対応した色リスト
        violin_colors = [
            "#FF69B4", "#87CEEB", "#98FB98", "#DDA0DD", "#F0E68C",
            "#FFA07A", "#20B2AA", "#778899", "#B0C4DE", "#FFDAB9",
        ]

        # 各パラメータ用のColumnDataSource
        violin_sources = {}
        for i, name in enumerate(self._param_names):
            vsource = ColumnDataSource(data={'x1': [], 'x2': [], 'y': []})
            violin_sources[name] = vsource
            color = violin_colors[i % len(violin_colors)]
            violin_fig.harea(
                x1='x1', x2='x2', y='y',
                source=vsource,
                fill_alpha=0.6,
                fill_color=color,
            )

        # X軸のカテゴリラベル設定
        if self._param_names:
            violin_fig.xaxis.ticker = list(range(len(self._param_names)))
            violin_fig.xaxis.major_label_overrides = {
                i: name[:8] for i, name in enumerate(self._param_names)
            }
            violin_fig.xaxis.major_label_orientation = 0.8

        # 変数重要度グラフ（目的関数ごと）
        importance_sources: Dict[str, ColumnDataSource] = {}
        importance_figs: Dict[str, figure] = {}

        # 表示するキーのリスト（目的関数名）
        importance_keys = list(self._objective_names)

        for obj_name in importance_keys:
            src = ColumnDataSource(data={
                'y': [],           # 変数位置
                'right': [],       # 重要度平均
                'left': [],        # エラーバー左端（mean - std）
                'right_err': [],   # エラーバー右端（mean + std）
                'color': [],       # バーの色
            })
            importance_sources[obj_name] = src

            display_title = f"Importance: {obj_name}"
            initial_n_vars = len(self._param_names) if self._param_names else 1
            fig = figure(
                title=display_title,
                x_axis_label="Importance",
                y_axis_label="",
                width=IMPORTANCE_WIDTH,
                height=IMPORTANCE_HEIGHT,
                tools="save",
                toolbar_location="above",
                x_range=Range1d(start=0, end=1),
                y_range=Range1d(start=initial_n_vars - 0.5, end=-0.5),
            )
            fig.toolbar.logo = None
            # グリッド線のスタイル設定
            fig.xgrid.grid_line_color = "#e0e0e0"
            fig.ygrid.grid_line_color = None  # 横棒グラフなのでY軸グリッドは非表示
            fig.hbar(y='y', right='right', height=0.6, color='color',
                     source=src, alpha=0.8)
            # エラーバー（Segment）
            fig.segment(x0='left', y0='y', x1='right_err', y1='y',
                        source=src, line_color="black", line_width=2)

            if self._param_names:
                fig.yaxis.ticker = list(range(len(self._param_names)))
                fig.yaxis.major_label_overrides = {
                    i: name[:8] for i, name in enumerate(self._param_names)
                }

            importance_figs[obj_name] = fig

        def update_importance_plots():
            """
            Update variable-importance plots from loaded analysis data.
            読み込み済み分析データから変数重要度グラフを更新する。
            """
            if not self._variable_importance:
                for obj_name, src in importance_sources.items():
                    src.data = {'y': [], 'right': [], 'left': [],
                               'right_err': [], 'color': []}
                    importance_figs[obj_name].visible = False
                return

            importance_data = self._variable_importance.get('importance', {})

            for obj_name, src in importance_sources.items():
                obj_data = importance_data.get(obj_name)
                if not obj_data:
                    src.data = {'y': [], 'right': [], 'left': [],
                               'right_err': [], 'color': []}
                    # データがないグラフは非表示
                    importance_figs[obj_name].visible = False
                    continue

                display_names, means, stds = align_importance_series(self._param_names, obj_data)
                n_vars = len(display_names)
                if n_vars == 0:
                    src.data = {'y': [], 'right': [], 'left': [],
                               'right_err': [], 'color': []}
                    importance_figs[obj_name].visible = False
                    continue

                colors = [violin_colors[i % len(violin_colors)] for i in range(n_vars)]

                src.data = {
                    'y': list(range(n_vars)),
                    'right': means,
                    'left': [max(0, m - s) for m, s in zip(means, stds)],
                    'right_err': [min(1, m + s) for m, s in zip(means, stds)],
                    'color': colors,
                }
                fig = importance_figs[obj_name]
                fig.y_range = Range1d(start=n_vars - 0.5, end=-0.5)
                fig.yaxis.ticker = list(range(n_vars))
                fig.yaxis.major_label_overrides = {
                    i: name[:8] for i, name in enumerate(display_names)
                }
                # データがあるグラフは表示
                fig.visible = True

        # パラメータ相関行列ヒートマップ
        correlation_source = ColumnDataSource(data={
            'x': [],
            'y': [],
            'corr': [],
            'corr_text': [],
            'x_name': [],
            'y_name': [],
        })

        # 色マッパー（赤=+1, 白=0, 青=-1）
        corr_color_mapper = LinearColorMapper(
            palette=list(RdBu11),
            low=-1.0,
            high=1.0
        )

        n_params = len(self._param_names)

        correlation_fig = figure(
            title="",
            x_axis_label="",
            y_axis_label="",
            width=CORRELATION_WIDTH,
            height=CORRELATION_HEIGHT,
            tools="save,hover",
            x_range=Range1d(start=-0.5, end=max(n_params - 0.5, 0.5)),
            y_range=Range1d(start=-0.5, end=max(n_params - 0.5, 0.5)),
        )

        correlation_fig.toolbar.logo = None
        correlation_fig.xgrid.grid_line_color = None
        correlation_fig.ygrid.grid_line_color = None

        correlation_fig.rect(
            x='x',
            y='y',
            width=1,
            height=1,
            source=correlation_source,
            fill_color={'field': 'corr', 'transform': corr_color_mapper},
            line_color=None,
        )

        correlation_fig.text(
            x='x',
            y='y',
            text='corr_text',
            source=correlation_source,
            text_align='center',
            text_baseline='middle',
            text_font_size='9px',
            text_color='black',
        )

        if self._param_names:
            param_indices = list(range(n_params))
            correlation_fig.xaxis.ticker = param_indices
            correlation_fig.xaxis.major_label_overrides = {
                i: name[:6] for i, name in enumerate(self._param_names)
            }
            correlation_fig.xaxis.major_label_orientation = 0.8

            correlation_fig.yaxis.ticker = param_indices
            correlation_fig.yaxis.major_label_overrides = {
                i: name[:6] for i, name in enumerate(self._param_names)
            }

        color_bar = ColorBar(
            color_mapper=corr_color_mapper,
            ticker=BasicTicker(desired_num_ticks=5),
            label_standoff=8,
            width=12,
            location=(0, 0),
        )
        correlation_fig.add_layout(color_bar, 'right')

        hover = correlation_fig.select(dict(type=HoverTool))
        if hover:
            hover[0].tooltips = [
                ("Parameter 1", "@x_name"),
                ("Parameter 2", "@y_name"),
                ("Correlation", "@corr{0.000}"),
            ]

        def update_correlation_heatmap():
            """
            Update the parameter-correlation heatmap.
            パラメータ相関行列のヒートマップを更新する。
            """
            if not self._variable_importance:
                correlation_source.data = {
                    'x': [], 'y': [], 'corr': [], 'corr_text': [],
                    'x_name': [], 'y_name': [],
                }
                return

            corr_data = self._variable_importance.get('correlation', {})
            matrix = corr_data.get('matrix', [])
            variables = corr_data.get('variables', [])

            if not matrix or not variables:
                correlation_source.data = {
                    'x': [], 'y': [], 'corr': [], 'corr_text': [],
                    'x_name': [], 'y_name': [],
                }
                return

            n = len(variables)
            x_list, y_list, corr_list, corr_text_list = [], [], [], []
            x_name_list, y_name_list = [], []

            for i in range(n):
                for j in range(n):
                    x_list.append(j)
                    y_list.append(n - 1 - i)
                    corr_val = matrix[i][j]
                    corr_list.append(corr_val)
                    corr_text_list.append(f"{corr_val:.2f}")
                    x_name_list.append(variables[j])
                    y_name_list.append(variables[i])

            correlation_source.data = {
                'x': x_list,
                'y': y_list,
                'corr': corr_list,
                'corr_text': corr_text_list,
                'x_name': x_name_list,
                'y_name': y_name_list,
            }

            # 軸ラベルを更新
            if variables:
                param_indices = list(range(n))
                correlation_fig.x_range.start = -0.5
                correlation_fig.x_range.end = n - 0.5
                correlation_fig.y_range.start = -0.5
                correlation_fig.y_range.end = n - 0.5
                correlation_fig.xaxis.ticker = param_indices
                correlation_fig.xaxis.major_label_overrides = {
                    i: name[:6] for i, name in enumerate(variables)
                }
                correlation_fig.yaxis.ticker = param_indices
                correlation_fig.yaxis.major_label_overrides = {
                    i: variables[n - 1 - i][:6] for i in range(n)
                }

        clustering_row_display = Div(
            text=(
                "<div style='height: 100%; display: flex; align-items: center;'>"
                "Clustering coefficient : N/A"
                "</div>"
            ),
            height=NETWORK_TEXT_CELL_HEIGHT,
            sizing_mode="stretch_width",
        )
        path_row_display = Div(
            text=(
                "<div style='height: 100%; display: flex; align-items: center;'>"
                "Average shortest path length : N/A"
                "</div>"
            ),
            height=NETWORK_TEXT_CELL_HEIGHT,
            sizing_mode="stretch_width",
        )
        network_metrics_note_display = Div(
            text="",
            height=NETWORK_NOTE_CELL_HEIGHT,
            sizing_mode="stretch_width",
        )
        network_metrics_note_display.visible = False

        degree_hist_source = ColumnDataSource(
            data={
                "degree": [],
                "left": [],
                "right": [],
                "bottom": [],
                "top": [],
            }
        )
        degree_hist_fig = figure(
            title="Degree Distribution",
            x_axis_label="Degree",
            y_axis_label="Node Count",
            width=DEGREE_HISTOGRAM_WIDTH,
            height=DEGREE_HISTOGRAM_HEIGHT,
            tools="save",
            toolbar_location="above",
            y_range=Range1d(start=0.0, end=1.0),
            sizing_mode="fixed",
        )
        degree_hist_fig.toolbar.logo = None
        degree_hist_fig.xgrid.grid_line_color = "#e0e0e0"
        degree_hist_fig.ygrid.grid_line_color = "#e0e0e0"
        degree_hist_fig.xgrid.minor_grid_line_color = None
        degree_hist_fig.ygrid.minor_grid_line_color = None
        degree_hist_fig.quad(
            left="left",
            right="right",
            bottom="bottom",
            top="top",
            source=degree_hist_source,
            fill_alpha=0.7,
            fill_color="#4a90e2",
            line_color="#1f77b4",
        )
        degree_scale_mode = RadioButtonGroup(labels=["Linear", "Log"], active=0, width=220)
        degree_x_min = [0.0]
        degree_x_max = [1.0]
        degree_y_max = [1.0]
        degree_log_x_min = 0.5
        degree_log_y_min = 0.5
        degree_axis_margin_ratio = 1.15

        adjacency_source = ColumnDataSource(data={"image": [], "x": [], "y": [], "dw": [], "dh": []})
        adjacency_color_mapper = LinearColorMapper(
            palette=["#ffffff", "#000000"],
            low=0.0,
            high=1.0,
        )
        adjacency_heatmap_fig = figure(
            title="Adjacency Matrix",
            x_axis_label="",
            y_axis_label="",
            width=ADJACENCY_HEATMAP_SIZE,
            height=ADJACENCY_HEATMAP_SIZE,
            tools="save",
            x_range=Range1d(start=-0.5, end=0.5),
            y_range=Range1d(start=-0.5, end=0.5),
            match_aspect=True,
            aspect_scale=1.0,
        )
        adjacency_heatmap_fig.toolbar.logo = None
        adjacency_heatmap_fig.min_width = NETWORK_GRID_CELL_MIN_WIDTH
        adjacency_heatmap_fig.xgrid.grid_line_color = None
        adjacency_heatmap_fig.ygrid.grid_line_color = None
        adjacency_heatmap_fig.image(
            image="image",
            x="x",
            y="y",
            dw="dw",
            dh="dh",
            source=adjacency_source,
            color_mapper=adjacency_color_mapper,
        )

        solution_text = TextAreaInput(value="", height=SOLUTION_TEXT_HEIGHT, width=SOLUTION_TEXT_WIDTH)
        download_source = ColumnDataSource(data={"json": [""], "filename": ["SelectedSolution.json"]})
        save_button = Button(label="Save Selection", width=SAVE_PANEL_WIDTH, button_type="default")
        save_button.disabled = True

        def update_solution_text(indices: List[int]) -> None:
            """
            Update solution details for one or more selected indices.
            1件以上の選択インデックスに対応する解の詳細を更新する。
            """
            payload = self._display_payload if isinstance(self._display_payload, dict) else None
            if payload is None:
                solution_text.value = ""
                return
            gen_value = payload.get("generation", -1)
            try:
                gen_label = self._format_generation_label(int(gen_value))
            except (TypeError, ValueError):
                gen_label = "-"

            lines = [f"Generation: {gen_label}"]

            if not indices:
                lines.append("")
                lines.append("Select solutions from the Pareto front on the left.")
                solution_text.value = "\n".join(lines)
                return

            lines.append("")

            objectives = payload.get("objectives", [])
            params = payload.get("params", [])
            original_indices = source.data.get("original_index", [])

            for i, sel_idx in enumerate(indices):
                # 選択インデックスからoriginal_indexを取得
                if original_indices and sel_idx < len(original_indices):
                    orig_idx = original_indices[sel_idx]
                else:
                    orig_idx = sel_idx

                lines.append(f"--- Solution {i+1} (ID: {orig_idx}) ---")

                if orig_idx < len(objectives):
                    obj = objectives[orig_idx]
                    if isinstance(obj, list) and len(obj) >= 2:
                        try:
                            f1 = float(obj[0])
                            f2 = float(obj[1])
                            name1 = self._objective_names[0] if len(self._objective_names) > 0 else "f1"
                            name2 = self._objective_names[1] if len(self._objective_names) > 1 else "f2"
                            lines.append(f"  Objectives: {name1}={f1:.6f}, {name2}={f2:.6f}")
                        except (TypeError, ValueError):
                            lines.append(f"  Objectives: f1={obj[0]}, f2={obj[1]}")

                if orig_idx < len(params):
                    values = params[orig_idx]
                    if isinstance(values, list):
                        for name, value in zip(self._param_names, values):
                            lines.append(f"  {name}: {self._format_param_value(name, value)}")
                lines.append("")

            solution_text.value = "\n".join(lines)

        def _build_selected_parameters(idx: int) -> Optional[Dict]:
            """
            Build and return export parameters for a selected index or None.
            選択インデックスの出力用パラメータを構築し、辞書またはNoneを返す。
            """
            payload = self._display_payload if isinstance(self._display_payload, dict) else None
            if payload is None:
                return None
            params_list = payload.get("params", [])
            if not isinstance(params_list, list) or idx >= len(params_list):
                return None
            values = params_list[idx]
            if not isinstance(values, list):
                return None
            parameters = {}
            if self._param_names:
                for name, value in zip(self._param_names, values):
                    parameters[name] = self._convert_param_for_export(name, value)
            else:
                for value_index, value in enumerate(values):
                    try:
                        parameters[f"param_{value_index}"] = float(value)
                    except (TypeError, ValueError):
                        parameters[f"param_{value_index}"] = value
            return parameters

        def _update_download_data(indices: List[int]) -> None:
            """
            Prepare download data for one or more selected solutions.
            1件以上の選択解をダウンロード用データへ整形する。
            """
            if not indices:
                download_source.data = {"json": [""], "filename": ["SelectedSolution.json"]}
                save_button.disabled = True
                return

            # 世代番号を取得
            generation = -1
            if isinstance(self._display_payload, dict):
                gen_value = self._display_payload.get("generation", -1)
                try:
                    generation = int(gen_value)
                except (TypeError, ValueError):
                    generation = -1

            # original_indexを取得してマッピング
            original_indices = source.data.get("original_index", [])

            if len(indices) == 1:
                # 単一選択: 従来のフォーマット
                sel_idx = indices[0]
                orig_idx = original_indices[sel_idx] if sel_idx < len(original_indices) else sel_idx
                params = _build_selected_parameters(orig_idx)
                if params is None:
                    download_source.data = {"json": [""], "filename": ["SelectedSolution.json"]}
                    save_button.disabled = True
                    return
                payload = {"parameters": params}
                filename = f"SelectedSolution_G{generation:05d}_I{orig_idx:04d}.json"
            else:
                # 複数選択: 配列フォーマット
                solutions = []
                for sel_idx in indices:
                    orig_idx = original_indices[sel_idx] if sel_idx < len(original_indices) else sel_idx
                    params = _build_selected_parameters(orig_idx)
                    if params is not None:
                        solutions.append({
                            "index": orig_idx,
                            "parameters": params
                        })

                if not solutions:
                    download_source.data = {"json": [""], "filename": ["SelectedSolution.json"]}
                    save_button.disabled = True
                    return

                payload = {
                    "generation": generation,
                    "count": len(solutions),
                    "solutions": solutions
                }
                filename = f"SelectedSolutions_G{generation:05d}_N{len(solutions):04d}.json"

            download_source.data = {
                "json": [orjson.dumps(payload, option=orjson.OPT_INDENT_2).decode("utf-8")],
                "filename": [filename],
            }
            save_button.disabled = False

        def _build_conditions_text() -> str:
            """
            Build and return variable names and ranges for Search Conditions.
            Search Conditions表示用の変数名と範囲を構築して返す。
            """
            param_names = self.condition.get("parameter_names", [])
            param_info = self.condition.get("parameter_info", {})

            if not isinstance(param_names, list) or not param_names:
                return "No variable information available."
            if not isinstance(param_info, dict):
                param_info = {}

            lines: List[str] = []
            for name in param_names:
                info = param_info.get(name, {})
                if not isinstance(info, dict):
                    lines.append(f"{name}: N/A")
                    continue

                var_type = info.get("type")
                if var_type in ("continuous", "integer"):
                    bounds = info.get("bounds")
                    if isinstance(bounds, (list, tuple)) and len(bounds) >= 2:
                        lines.append(f"{name}: [{bounds[0]}, {bounds[1]}]")
                    else:
                        lines.append(f"{name}: N/A")
                elif var_type == "categorical":
                    n_categories = info.get("n_categories")
                    try:
                        n_cat = int(n_categories)
                    except (TypeError, ValueError):
                        n_cat = 0

                    if n_cat > 0:
                        line = f"{name}: [0, {n_cat - 1}]"
                    else:
                        line = f"{name}: N/A"

                    category_labels = self._get_category_labels(info)
                    if category_labels:
                        category_names_text = ", ".join(category_labels)
                        line += f" (categories: {category_names_text})"
                    lines.append(line)
                else:
                    lines.append(f"{name}: N/A")

            return "\n".join(lines)

        def update_violin_plot(selected_indices):
            """
            Update Min-Max-normalized violin plots for selected solutions.
            選択解の決定変数分布をMin-Max正規化したバイオリンプロットへ更新する。
            """
            if not selected_indices or len(selected_indices) < 2:
                for vsource in violin_sources.values():
                    vsource.data = {'x1': [], 'x2': [], 'y': []}
                return
            params_list = source.data.get('params', [])
            if not params_list:
                return

            for var_idx, var_name in enumerate(self._param_names):
                values = []
                for idx in selected_indices:
                    if idx < len(params_list) and var_idx < len(params_list[idx]):
                        try:
                            values.append(float(params_list[idx][var_idx]))
                        except (TypeError, ValueError):
                            pass
                if len(values) >= 2:
                    # KDE処理をBokehサーバー本体から切り出し、描画データ生成を単体テスト可能にする。
                    normalized_values = normalize_violin_values(
                        values,
                        self._param_info.get(var_name),
                    )
                    violin_data = create_violin_data(normalized_values, center_x=var_idx)
                    violin_sources[var_name].data = violin_data
                else:
                    violin_sources[var_name].data = {'x1': [], 'x2': [], 'y': []}

        def _apply_degree_axis_scale() -> None:
            """
            Apply the selected linear or logarithmic degree-axis scale.
            選択された線形または対数の次数軸尺度を適用する。
            """
            use_log = degree_scale_mode.active == 1

            if use_log:
                degree_hist_fig.x_scale = LogScale()
                degree_hist_fig.y_scale = LogScale()
                degree_hist_fig.xaxis.ticker = LogTicker(num_minor_ticks=8)
                degree_hist_fig.yaxis.ticker = LogTicker(num_minor_ticks=8)
                degree_hist_fig.xaxis.formatter = PrintfTickFormatter(format="%.0f")
                degree_hist_fig.yaxis.formatter = PrintfTickFormatter(format="%.0f")
                degree_hist_fig.xaxis.major_label_orientation = 0.8
                degree_hist_fig.xgrid.minor_grid_line_color = "#eeeeee"
                degree_hist_fig.ygrid.minor_grid_line_color = "#eeeeee"
                degree_hist_fig.xgrid.minor_grid_line_alpha = 0.55
                degree_hist_fig.ygrid.minor_grid_line_alpha = 0.55
                degree_hist_fig.xgrid.minor_grid_line_width = 0.5
                degree_hist_fig.ygrid.minor_grid_line_width = 0.5
                x_start = max(degree_log_x_min, degree_x_min[0])
                y_start = degree_log_y_min
            else:
                degree_hist_fig.x_scale = LinearScale()
                degree_hist_fig.y_scale = LinearScale()
                degree_hist_fig.xaxis.ticker = BasicTicker(desired_num_ticks=8)
                degree_hist_fig.yaxis.ticker = BasicTicker(desired_num_ticks=8)
                degree_hist_fig.xaxis.formatter = BasicTickFormatter()
                degree_hist_fig.yaxis.formatter = BasicTickFormatter()
                degree_hist_fig.xaxis.major_label_orientation = 0.0
                degree_hist_fig.xgrid.minor_grid_line_color = None
                degree_hist_fig.ygrid.minor_grid_line_color = None
                x_start = 0.0
                y_start = 0.0

            x_end = max(1.0, degree_x_max[0])
            y_end = max(1.0, degree_y_max[0])
            if use_log:
                x_end = round_up_to_power_of_ten(x_end, minimum=degree_log_x_min)
                y_end = round_up_to_power_of_ten(y_end, minimum=degree_log_y_min)
                x_end = max(x_end, x_start * (1.0 + 1e-6))
                y_end = max(y_end, y_start * (1.0 + 1e-6))

            degree_hist_fig.x_range = Range1d(start=x_start, end=x_end)
            degree_hist_fig.y_range = Range1d(start=y_start, end=y_end)

        def _set_network_stats_html(
            path_value: Optional[float],
            clustering_value: Optional[float],
            note: str = "",
        ) -> None:
            """
            Update network-statistics HTML with optional explanatory text.
            ネットワーク統計HTMLを任意の補足文とともに更新する。
            """
            path_text = "N/A"
            if path_value is not None:
                path_text = f"{path_value:.6f}"

            clustering_text = "N/A"
            if clustering_value is not None:
                clustering_text = f"{clustering_value:.6f}"

            clustering_row_display.text = (
                "<div style='height: 100%; display: flex; align-items: center;'>"
                f"Clustering coefficient : {clustering_text}"
                "</div>"
            )
            path_row_display.text = (
                "<div style='height: 100%; display: flex; align-items: center;'>"
                f"Average shortest path length : {path_text}"
                "</div>"
            )
            network_metrics_note_display.text = (
                "<div style='height: 100%; display: flex; align-items: center; "
                "color: #666;'>"
                f"{note if note else ''}"
                "</div>"
            )
            network_metrics_note_display.visible = bool(note)

        def _clear_adjacency_heatmap() -> None:
            """
            Clear adjacency heatmap data and reset its ranges.
            隣接行列ヒートマップのデータを消去して範囲を初期化する。
            """
            adjacency_source.data = {"image": [], "x": [], "y": [], "dw": [], "dh": []}
            adjacency_heatmap_fig.x_range = Range1d(start=-0.5, end=0.5)
            adjacency_heatmap_fig.y_range = Range1d(start=-0.5, end=0.5)

        def update_network_metrics(selected_indices: List[int]) -> None:
            """
            Update network-metric views for selected solutions.
            選択解に対応するネットワーク指標表示を更新する。
            """
            original_indices = source.data.get("original_index", [])

            def _clear_degree_histogram() -> None:
                """
                Clear degree-histogram data and reset its ranges.
                次数ヒストグラムのデータを消去して範囲を初期化する。
                """
                degree_hist_source.data = {
                    "degree": [],
                    "left": [],
                    "right": [],
                    "bottom": [],
                    "top": [],
                }

            if len(selected_indices) != 1:
                _clear_degree_histogram()
                degree_x_min[0] = 0.0
                degree_x_max[0] = 1.0
                degree_y_max[0] = 1.0
                _apply_degree_axis_scale()
                _clear_adjacency_heatmap()
                _set_network_stats_html(None, None)
                return

            selected_idx = selected_indices[0]
            notes: List[str] = []
            fallback_clustering, fallback_path, raw_adjacency = self._get_network_metrics_by_selected_index(
                selected_idx,
                original_indices,
            )

            if raw_adjacency is None:
                _clear_degree_histogram()
                degree_x_min[0] = 0.0
                degree_x_max[0] = 1.0
                degree_y_max[0] = 1.0
                _apply_degree_axis_scale()
                _clear_adjacency_heatmap()
                _set_network_stats_html(fallback_path, fallback_clustering, note="<br/>".join(notes))
                return

            adjacency_matrices = collect_adjacency_matrices([raw_adjacency])

            if adjacency_matrices:
                degrees, hist_means, _ = compute_degree_histogram_stats(adjacency_matrices)
                raw_degree_values = [float(d) for d in degrees]
                raw_mean_values = [float(v) for v in hist_means]
                use_log_degree_scale = degree_scale_mode.active == 1
                degree_bar_data = build_degree_histogram_bar_data(
                    degrees=raw_degree_values,
                    mean_counts=raw_mean_values,
                    use_log=use_log_degree_scale,
                    epsilon=degree_log_y_min,
                )
                if use_log_degree_scale and degree_bar_data["zero_degree_count"] > 0.0:
                    notes.append(
                        "Log view excludes degree=0 bins. Zero-degree nodes: "
                        f"{int(round(degree_bar_data['zero_degree_count']))}"
                    )

                degree_hist_source.data = {
                    "degree": degree_bar_data["degree"],
                    "left": degree_bar_data["left"],
                    "right": degree_bar_data["right"],
                    "bottom": degree_bar_data["bottom"],
                    "top": degree_bar_data["top"],
                }
                if degree_bar_data["left"]:
                    degree_x_min[0] = min(degree_bar_data["left"])
                else:
                    degree_x_min[0] = 1.0 if use_log_degree_scale else 0.0
                if degree_bar_data["right"]:
                    degree_x_max[0] = max(degree_bar_data["right"]) * degree_axis_margin_ratio
                else:
                    degree_x_max[0] = 1.0
                if degree_bar_data["top"]:
                    degree_y_max[0] = max(degree_bar_data["top"]) * degree_axis_margin_ratio
                else:
                    degree_y_max[0] = 1.0
                _apply_degree_axis_scale()

                _set_network_stats_html(
                    path_value=fallback_path,
                    clustering_value=fallback_clustering,
                    note="<br/>".join(notes),
                )
            else:
                _clear_degree_histogram()
                degree_x_min[0] = 0.0
                degree_x_max[0] = 1.0
                degree_y_max[0] = 1.0
                _apply_degree_axis_scale()
                _set_network_stats_html(
                    path_value=fallback_path,
                    clustering_value=fallback_clustering,
                    note="<br/>".join(notes),
                )
                _clear_adjacency_heatmap()
                return

            adjacency_matrix = adjacency_matrices[0]
            adjacency_source.data = build_adjacency_image_data(adjacency_matrix)
            size = adjacency_matrix.shape[0]
            adjacency_heatmap_fig.x_range = Range1d(start=-0.5, end=size - 0.5)
            adjacency_heatmap_fig.y_range = Range1d(start=-0.5, end=size - 0.5)

        tap_cycle_state = {
            "internal_update": False,
            "group": None,
            "position": -1,
        }

        def _reset_tap_cycle_state() -> None:
            """
            Reset the overlap group and position used to cycle through coincident points.
            重複点を順番に選択するためのグループと現在位置を初期化する。
            """
            tap_cycle_state["internal_update"] = False
            tap_cycle_state["group"] = None
            tap_cycle_state["position"] = -1

        def _get_axis_bounds(axis_range, values: List[float]) -> Tuple[float, float]:
            """
            Resolve and return finite bounds for an axis range.
            軸範囲の有限な上下限を解決して返す。
            """
            start = getattr(axis_range, "start", None)
            end = getattr(axis_range, "end", None)
            try:
                start_val = float(start)
                end_val = float(end)
                if not np.isfinite(start_val) or not np.isfinite(end_val):
                    raise ValueError
                if start_val == end_val:
                    end_val = start_val + 1.0
                if start_val > end_val:
                    start_val, end_val = end_val, start_val
                return start_val, end_val
            except (TypeError, ValueError):
                pass

            valid_values = []
            for value in values:
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(numeric):
                    valid_values.append(numeric)
            if not valid_values:
                return 0.0, 1.0
            low = min(valid_values)
            high = max(valid_values)
            if low == high:
                high = low + 1.0
            return float(low), float(high)

        def _collect_tap_overlap_group(anchor_idx: int) -> List[int]:
            """
            Collect and return points overlapping a tapped anchor.
            タップしたアンカーと重なる点を収集して返す。
            """
            f1_values = source.data.get("f1", [])
            f2_values = source.data.get("f2", [])
            original_indices = source.data.get("original_index", [])
            if anchor_idx < 0 or anchor_idx >= len(f1_values) or anchor_idx >= len(f2_values):
                return []

            x_start, x_end = _get_axis_bounds(pareto_front_fig.x_range, f1_values)
            y_start, y_end = _get_axis_bounds(pareto_front_fig.y_range, f2_values)
            x_tolerance, y_tolerance = compute_data_tolerance_from_pixels(
                x_start,
                x_end,
                y_start,
                y_end,
                int(SCATTER_WIDTH),
                int(SCATTER_HEIGHT),
                PARETO_TAP_OVERLAP_RADIUS_PX,
            )
            candidates = collect_overlapped_indices_by_tolerance(
                anchor_idx,
                f1_values,
                f2_values,
                x_tolerance,
                y_tolerance,
            )
            return order_indices_by_original_index(candidates, original_indices)

        def _find_anchor_index_from_tap(event: TapEvent) -> Optional[int]:
            """
            Find and return the nearest tapped source index or None.
            タップ位置に最も近いソースインデックス、またはNoneを返す。
            """
            f1_values = source.data.get("f1", [])
            f2_values = source.data.get("f2", [])
            if not f1_values or not f2_values:
                return None

            x_start, x_end = _get_axis_bounds(pareto_front_fig.x_range, f1_values)
            y_start, y_end = _get_axis_bounds(pareto_front_fig.y_range, f2_values)
            x_tolerance, y_tolerance = compute_data_tolerance_from_pixels(
                x_start,
                x_end,
                y_start,
                y_end,
                int(SCATTER_WIDTH),
                int(SCATTER_HEIGHT),
                PARETO_TAP_OVERLAP_RADIUS_PX,
            )
            try:
                click_x = float(event.x)
                click_y = float(event.y)
            except (TypeError, ValueError):
                click_x = None
                click_y = None

            if click_x is not None and click_y is not None:
                best_idx: Optional[int] = None
                best_dist = float("inf")
                upper = min(len(f1_values), len(f2_values))
                safe_x_tol = max(x_tolerance, 1e-12)
                safe_y_tol = max(y_tolerance, 1e-12)
                for idx in range(upper):
                    try:
                        x_val = float(f1_values[idx])
                        y_val = float(f2_values[idx])
                    except (TypeError, ValueError):
                        continue
                    dx = abs(x_val - click_x)
                    dy = abs(y_val - click_y)
                    if dx <= x_tolerance and dy <= y_tolerance:
                        # 軸スケール差を均すため許容差で正規化した距離を使う
                        dist = (dx / safe_x_tol) ** 2 + (dy / safe_y_tol) ** 2
                        if dist < best_dist:
                            best_dist = dist
                            best_idx = idx
                if best_idx is not None:
                    return int(best_idx)

            selected_indices = [int(idx) for idx in source.selected.indices]
            if selected_indices:
                return selected_indices[0]
            return None

        def _handle_pareto_tap(event: TapEvent) -> None:
            """
            Handle a Pareto-plot tap and update cycle selection.
            パレートプロットのタップを処理して循環選択を更新する。
            """
            anchor_idx = _find_anchor_index_from_tap(event)
            if anchor_idx is None:
                _reset_tap_cycle_state()
                return

            ordered_group = _collect_tap_overlap_group(anchor_idx)
            selected_indices, next_group, next_position, _ = resolve_tap_cycle_selection(
                ordered_group=ordered_group,
                clicked_idx=anchor_idx,
                previous_group=tap_cycle_state["group"],
                previous_position=int(tap_cycle_state["position"]),
            )

            if not selected_indices:
                _reset_tap_cycle_state()
                return

            tap_cycle_state["group"] = next_group
            tap_cycle_state["position"] = next_position

            next_idx = int(selected_indices[0])
            if source.selected.indices != [next_idx]:
                tap_cycle_state["internal_update"] = True
                source.selected.indices = [next_idx]
            if next_group is None:
                tap_cycle_state["group"] = None
                tap_cycle_state["position"] = -1

        def update_param_panel(attr, old, new):
            """
            Update parameter panels after a selection change.
            選択変更後にパラメータ表示領域を更新する。
            """
            selected_indices = [int(idx) for idx in new]

            if tap_cycle_state["internal_update"]:
                tap_cycle_state["internal_update"] = False
            else:
                if len(selected_indices) != 1:
                    _reset_tap_cycle_state()
                else:
                    current_group = tap_cycle_state["group"]
                    if current_group is not None and selected_indices[0] not in current_group:
                        _reset_tap_cycle_state()

            update_violin_plot(selected_indices)
            update_network_metrics(selected_indices)
            if not selected_indices:
                update_solution_text([])
                _update_download_data([])
                return
            update_solution_text(selected_indices)
            _update_download_data(selected_indices)

        save_button.js_on_click(CustomJS(args={"download": download_source}, code="""
            const data = download.data;
            const jsonText = data.json[0] || "";
            if (!jsonText) {
                return;
            }
            const filename = data.filename[0] || "SelectedSolution.json";
            const blob = new Blob([jsonText], {type: "application/json"});
            const url = URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.href = url;
            link.download = filename;
            link.click();
            URL.revokeObjectURL(url);
        """))

        source.selected.on_change("indices", update_param_panel)
        pareto_front_fig.on_event(TapEvent, _handle_pareto_tap)
        degree_scale_mode.on_change(
            "active",
            lambda attr, old, new: update_network_metrics(source.selected.indices),
        )

        first_button = Button(label="<<", width=40, button_type="default")
        prev_button = Button(label="<", width=40, button_type="default")
        next_button = Button(label=">", width=40, button_type="default")
        final_button = Button(label=">>", width=40, button_type="default")
        auto_button = Button(label="Stop Auto", width=100, button_type="default")
        refresh_button = Button(label="Refresh", width=80, button_type="default")

        hypervolume_fig = figure(
            x_axis_label="Generation",
            y_axis_label="HV",
            width=HYPERVOLUME_WIDTH,
            height=HYPERVOLUME_HEIGHT,
            tools="save",
            toolbar_location="above",
            y_range=DataRange1d(start=0),  # Y軸を0始まりに固定
        )
        hypervolume_fig.toolbar.logo = None
        # グリッド線のスタイル設定
        hypervolume_fig.xgrid.grid_line_color = "#e0e0e0"
        hypervolume_fig.ygrid.grid_line_color = "#e0e0e0"
        hypervolume_fig.line("gen", "hv", source=hypervolume_source, line_width=2)
        hypervolume_fig.scatter("gen", "hv", source=hypervolume_source, marker="circle", size=6)

        ref_x_input = TextInput(value=str(self._hypervolume_ref_point[0]), width=120)
        ref_y_input = TextInput(value=str(self._hypervolume_ref_point[1]), width=120)
        hv_apply_button = Button(label="Apply", width=80)

        def apply_hv_settings():
            """
            Apply hypervolume reference-point input and refresh history.
            ハイパーボリューム基準点の入力を適用して履歴を更新する。
            """
            try:
                ref_x = float(ref_x_input.value)
                ref_y = float(ref_y_input.value)
            except (TypeError, ValueError):
                return
            self._hypervolume_ref_point = (ref_x, ref_y)
            self._update_hypervolume_source(hypervolume_source, force=True)
            hv_y_range_manual[0] = False
            _sync_hv_y_range_inputs(hypervolume_source)

        hv_apply_button.on_click(apply_hv_settings)

        # --- Hypervolume Y Range 設定 ---
        hv_y_min_input = TextInput(value="0", width=120)
        hv_y_max_input = TextInput(value="1", width=120)
        hv_y_range_apply_button = Button(label="Apply", width=80, button_type="default")
        hv_y_range_manual = [False]  # Y Range Apply押下後にTrue

        def _sync_hv_y_range_inputs(hv_source: ColumnDataSource) -> None:
            """
            Synchronize HV Y-range inputs unless manually fixed.
            手動固定されていない場合にHVのY範囲入力を同期する。
            """
            if hv_y_range_manual[0]:
                return
            hv_values = hv_source.data.get("hv", [])
            if hv_values:
                max_hv = max(hv_values)
                hv_y_max_input.value = str(max_hv if max_hv > 0 else 1)
            else:
                hv_y_max_input.value = "1"
            hv_y_min_input.value = "0"

        def apply_hv_y_range():
            """
            Validate and apply the manually entered hypervolume Y range.
            手入力されたハイパーボリュームのY範囲を検証して適用する。
            """
            min_text = _get_pending_input_value(hv_y_min_input)
            max_text = _get_pending_input_value(hv_y_max_input)
            start, end = sanitize_fixed_range(
                min_text, max_text,
                use_log=False, epsilon=1e-12, fallback=(0.0, 1.0),
            )
            hv_y_min_input.value = str(start)
            hv_y_max_input.value = str(end)
            hypervolume_fig.y_range = Range1d(start=start, end=end)
            hv_y_range_manual[0] = True

        hv_y_range_apply_button.on_click(apply_hv_y_range)

        x_mode = RadioButtonGroup(labels=["Auto", "Manual"], active=0)
        x_min_input = TextInput(value="", width=AXIS_RANGE_INPUT_WIDTH)
        x_max_input = TextInput(value="", width=AXIS_RANGE_INPUT_WIDTH)
        y_mode = RadioButtonGroup(labels=["Auto", "Manual"], active=0)
        y_min_input = TextInput(value="", width=AXIS_RANGE_INPUT_WIDTH)
        y_max_input = TextInput(value="", width=AXIS_RANGE_INPUT_WIDTH)
        axis_range_apply_button = Button(label="Apply", width=80, button_type="default")

        last_valid_x_range = [0.0, 1.0]
        last_valid_y_range = [0.0, 1.0]

        def _format_axis_range_value(value: float) -> str:
            """
            Format and return an axis-range value for text input.
            軸範囲値をテキスト入力用に整形して返す。
            """
            return format(float(value), ".12g")

        def _capture_current_x_bounds() -> Tuple[float, float]:
            """
            Capture and return current Objective Space X bounds.
            現在のObjective SpaceのX軸範囲を取得して返す。
            """
            x_start, x_end = _get_axis_bounds(
                pareto_front_fig.x_range,
                source.data.get("f1", []),
            )
            last_valid_x_range[0] = x_start
            last_valid_x_range[1] = x_end
            return x_start, x_end

        def _capture_current_y_bounds() -> Tuple[float, float]:
            """
            Capture and return current Objective Space Y bounds.
            現在のObjective SpaceのY軸範囲を取得して返す。
            """
            y_start, y_end = _get_axis_bounds(
                pareto_front_fig.y_range,
                source.data.get("f2", []),
            )
            last_valid_y_range[0] = y_start
            last_valid_y_range[1] = y_end
            return y_start, y_end

        def _set_x_inputs(start: float, end: float) -> None:
            """
            Set Objective Space X-range input fields.
            Objective SpaceのX軸範囲入力欄を設定する。
            """
            set_text_input_value_and_editing(x_min_input, _format_axis_range_value(start))
            set_text_input_value_and_editing(x_max_input, _format_axis_range_value(end))

        def _set_y_inputs(start: float, end: float) -> None:
            """
            Set Objective Space Y-range input fields.
            Objective SpaceのY軸範囲入力欄を設定する。
            """
            set_text_input_value_and_editing(y_min_input, _format_axis_range_value(start))
            set_text_input_value_and_editing(y_max_input, _format_axis_range_value(end))

        def _get_pending_input_value(text_input: TextInput) -> str:
            """
            Return the pending or committed value of a TextInput.
            TextInputの編集中または確定済みの値を返す。
            """
            return resolve_text_input_value(
                committed_value=text_input.value,
                editing_value=getattr(text_input, "value_input", None),
            )

        def _apply_axis_ranges():
            """
            Apply automatic or fixed Objective Space axis ranges.
            Objective Spaceの自動または固定の軸範囲を適用する。
            """
            if x_mode.active == 0:
                pareto_front_fig.x_range = DataRange1d()
            else:
                start, end = sanitize_fixed_range(
                    _get_pending_input_value(x_min_input),
                    _get_pending_input_value(x_max_input),
                    use_log=False,
                    epsilon=1e-12,
                    fallback=(last_valid_x_range[0], last_valid_x_range[1]),
                )
                last_valid_x_range[0] = start
                last_valid_x_range[1] = end
                _set_x_inputs(start, end)
                pareto_front_fig.x_range = Range1d(start=start, end=end)
            if y_mode.active == 0:
                pareto_front_fig.y_range = DataRange1d()
            else:
                start, end = sanitize_fixed_range(
                    _get_pending_input_value(y_min_input),
                    _get_pending_input_value(y_max_input),
                    use_log=False,
                    epsilon=1e-12,
                    fallback=(last_valid_y_range[0], last_valid_y_range[1]),
                )
                last_valid_y_range[0] = start
                last_valid_y_range[1] = end
                _set_y_inputs(start, end)
                pareto_front_fig.y_range = Range1d(start=start, end=end)
            # 軸範囲の変更をRefresh操作なしでグラフへ反映する。
            _update_source()

        def _on_x_mode_change(attr, old, new):
            """
            Synchronize X-range inputs after an axis-mode change.
            X軸モード変更後に範囲入力を同期する。
            """
            if int(new) == 1:
                start, end = _capture_current_x_bounds()
                _set_x_inputs(start, end)
            else:
                set_text_input_value_and_editing(x_min_input, "")
                set_text_input_value_and_editing(x_max_input, "")

        def _on_y_mode_change(attr, old, new):
            """
            Synchronize Y-range inputs after an axis-mode change.
            Y軸モード変更後に範囲入力を同期する。
            """
            if int(new) == 1:
                start, end = _capture_current_y_bounds()
                _set_y_inputs(start, end)
            else:
                set_text_input_value_and_editing(y_min_input, "")
                set_text_input_value_and_editing(y_max_input, "")

        x_mode.on_change("active", _on_x_mode_change)
        y_mode.on_change("active", _on_y_mode_change)
        axis_range_apply_button.on_click(_apply_axis_ranges)

        def _update_source():
            """
            Load the display generation and update all dependent sources.
            表示世代を読み込み、依存する全データソースを更新する。
            """
            gen_index = self._get_display_generation()
            payload = self._load_generation_payload(gen_index)
            if not isinstance(payload, dict):
                payload = None
            self._display_payload = payload
            self._set_display_network_metrics_from_payload(payload)
            solution_text.value = ""
            _update_download_data([])
            _reset_tap_cycle_state()
            update_conditions_panel()

            empty_data = self._empty_objective_source_data()

            if not payload:
                source.data = empty_data
                source.selected.indices = []
                pareto_step_source.data = {'x': [], 'y': []}
                update_network_metrics([])
                title_div.text = self._create_title_html()
                generation_div.text = self._create_generation_html()
                curdoc().title = self._get_title_text()
                return

            objectives = payload.get("objectives", [])
            params = payload.get("params", [])

            if not objectives:
                source.data = empty_data
                source.selected.indices = []
                pareto_step_source.data = {'x': [], 'y': []}
            else:
                x_vals, y_vals = self._transform_objectives_for_display(objectives)

                # パレートフロントを計算
                pareto_set = set()
                pareto_indices = []
                if len(x_vals) > 0 and len(self._objective_directions) >= 2:
                    objectives_array = np.column_stack([x_vals, y_vals])
                    _, pareto_indices = extract_pareto_front(objectives_array, self._objective_directions)
                    pareto_set = set(pareto_indices)
                elif len(x_vals) > 0:
                    # 方向情報がない場合は全てパレートフロントとして扱う
                    pareto_indices = list(range(len(x_vals)))
                    pareto_set = set(pareto_indices)

                # パレートフロント階段状 line データを更新
                if len(pareto_indices) >= 2 and len(self._objective_directions) >= 2:
                    pareto_x = [x_vals[i] for i in pareto_indices]
                    pareto_y = [y_vals[i] for i in pareto_indices]
                    pareto_points = np.column_stack([pareto_x, pareto_y])
                    step_x, step_y = _generate_step_line_data(
                        pareto_points, self._objective_directions
                    )
                    pareto_step_source.data = {'x': step_x, 'y': step_y}
                else:
                    pareto_step_source.data = {'x': [], 'y': []}

                source.data = self._build_objective_source_data(
                    x_vals=x_vals,
                    y_vals=y_vals,
                    params=params,
                    pareto_set=pareto_set,
                )
                source.selected.indices = []
            update_network_metrics(source.selected.indices)

            title_div.text = self._create_title_html()
            generation_div.text = self._create_generation_html()
            curdoc().title = self._get_title_text()

        # 参照点UI更新用のフラグ（初期化時に既に計算済みなら同期済みとする）
        ref_point_ui_synced = [self._hypervolume_ref_point_auto]

        def periodic_update():
            """
            Poll result files and refresh the display when auto-update is active.
            自動更新中に結果ファイルを確認して表示を更新する。
            """
            if self._halt_updates:
                return
            if self.auto_update:
                condition_updated = self._load_condition()
                if self._halt_updates:
                    title_div.text = self._create_title_html()
                    generation_div.text = self._create_generation_html()
                    curdoc().title = self._get_title_text()
                    return
                files_changed = self._scan_generation_files()
                if self._halt_updates:
                    title_div.text = self._create_title_html()
                    generation_div.text = self._create_generation_html()
                    curdoc().title = self._get_title_text()
                    return
                if condition_updated:
                    update_conditions_panel()

                # 参照点が自動計算されたらUIに反映
                if self._hypervolume_ref_point_auto and not ref_point_ui_synced[0]:
                    ref_x_input.value = str(self._hypervolume_ref_point[0])
                    ref_y_input.value = str(self._hypervolume_ref_point[1])
                    ref_point_ui_synced[0] = True

                # 変数重要度の読み込みと更新
                if self._load_variable_importance():
                    update_importance_plots()
                    update_correlation_heatmap()

                if files_changed or condition_updated:
                    self._update_hypervolume_source(hypervolume_source)
                    _sync_hv_y_range_inputs(hypervolume_source)
                if self.current_display_generation is None:
                    if files_changed or condition_updated or self._display_payload is None:
                        _update_source()
                    else:
                        title_div.text = self._create_title_html()
                        generation_div.text = self._create_generation_html()
                        curdoc().title = self._get_title_text()
                else:
                    if self.current_display_generation not in self._generation_files:
                        if self._latest_generation >= 0:
                            self.current_display_generation = None
                            _update_source()
                        else:
                            title_div.text = self._create_title_html()
                            generation_div.text = self._create_generation_html()
                            curdoc().title = self._get_title_text()
                    elif condition_updated or files_changed:
                        title_div.text = self._create_title_html()
                        generation_div.text = self._create_generation_html()
                        curdoc().title = self._get_title_text()

        def refresh_callback():
            """
            Rescan result files and refresh the current display.
            結果ファイルを再走査して現在の表示を更新する。
            """
            if self._halt_updates:
                return
            self._load_condition(force=True)
            if self._halt_updates:
                title_div.text = self._create_title_html()
                generation_div.text = self._create_generation_html()
                curdoc().title = self._get_title_text()
                return
            update_conditions_panel()
            self._scan_generation_files()
            if self._halt_updates:
                title_div.text = self._create_title_html()
                generation_div.text = self._create_generation_html()
                curdoc().title = self._get_title_text()
                return
            self._load_variable_importance(force=True)
            update_importance_plots()
            update_correlation_heatmap()
            self._update_hypervolume_source(hypervolume_source, force=True)
            _sync_hv_y_range_inputs(hypervolume_source)
            _update_source()

        def toggle_auto():
            """
            Toggle automatic result-file polling.
            結果ファイルの自動確認を切り替える。
            """
            if self._halt_updates:
                return
            self.auto_update = not self.auto_update
            auto_button.label = "Start Auto" if not self.auto_update else "Stop Auto"

        def show_first():
            """
            Display the first available generation.
            最初の利用可能な世代を表示する。
            """
            if self._halt_updates:
                return
            if not self._available_generations:
                return
            self.current_display_generation = self._available_generations[0]
            _update_source()

        def show_final():
            """
            Display the latest available generation.
            最新の利用可能な世代を表示する。
            """
            if self._halt_updates:
                return
            max_generation = self._get_max_generation_index()
            if max_generation is None:
                max_generation = self._latest_generation
            if self._latest_generation >= 0 and max_generation > self._latest_generation:
                max_generation = self._latest_generation
            if max_generation < 0:
                return
            self.current_display_generation = max_generation
            _update_source()

        def move_prev(steps: int = 1):
            """
            Move backward by available generations and refresh the display.
            利用可能な世代を指定数だけ戻って表示を更新する。
            """
            if self._halt_updates:
                return
            target = self._get_previous_generation(steps=steps)
            if target is None or target == self.current_display_generation:
                return
            self.current_display_generation = target
            _update_source()

        def move_next(steps: int = 1):
            """
            Move forward by available generations and refresh the display.
            利用可能な世代を指定数だけ進んで表示を更新する。
            """
            if self._halt_updates:
                return
            target = self._get_next_generation(steps=steps)
            if target is None or target == self.current_display_generation:
                return
            self.current_display_generation = target
            _update_source()

        first_button.on_click(show_first)
        final_button.on_click(show_final)
        prev_button.on_click(move_prev)
        next_button.on_click(move_next)
        refresh_button.on_click(refresh_callback)
        auto_button.on_click(toggle_auto)

        # キーボードショートカット用の隠しウィジェット
        key_receiver = TextInput(value="", visible=False)

        def on_key_action(attr, old, new):
            """
            Handle a keyboard navigation action from the hidden input.
            非表示入力から受け取ったキーボード移動操作を処理する。
            """
            if not new:
                return
            if new.startswith("prev5_"):
                move_prev(steps=5)
            elif new.startswith("next5_"):
                move_next(steps=5)
            elif new.startswith("prev_"):
                move_prev()
            elif new.startswith("next_"):
                move_next()

        key_receiver.on_change("value", on_key_action)

        visualization_controls = row(
            first_button,
            prev_button,
            next_button,
            final_button,
            refresh_button,
            auto_button,
            key_receiver,
        )

        hv_ref_row = row(
            Div(text="Ref", width=50),
            Div(text="X", width=15),
            ref_x_input,
            Div(text="Y", width=15),
            ref_y_input,
            hv_apply_button,
        )
        hv_y_range_row = row(
            Div(text="Y Range", width=50),
            Div(text="min", width=25),
            hv_y_min_input,
            Div(text="max", width=25),
            hv_y_max_input,
            hv_y_range_apply_button,
        )
        hypervolume_settings = column(hv_ref_row, hv_y_range_row)

        # 3ブロックUI構成（タイトル + 左右レイアウト）
        header_section = column(title_div, sizing_mode="stretch_width")
        header_section.styles = {
            "position": "fixed",
            "top": "0",
            "left": "0",
            "right": "0",
            "z-index": "1000",
            "background-color": "rgba(30, 41, 115, 0.85)",
            "color": "#ffffff",
            "padding": f"8px {LEFT_PADDING}px",
            "border-bottom": "1px solid rgba(255, 255, 255, 0.2)",
            "box-shadow": "0 2px 6px rgba(0, 0, 0, 0.18)",
            "box-sizing": "border-box",
        }

        nav_controls = row(
            visualization_controls,
            generation_div,
            spacing=SECTION_SPACING,
            width=LEFT_PANEL_CONTENT_MIN_WIDTH,
        )
        nav_controls.styles = {
            "align-items": "center",
            "flex-wrap": "nowrap",
        }

        left_content = column(
            _create_section_header("Objective Space (Pareto Front)"),
            nav_controls,
            pareto_front_fig,
            width=LEFT_PANEL_CONTENT_MIN_WIDTH,
            sizing_mode="stretch_height",
        )
        left_panel = ScrollBox(
            child=left_content,
            vertical_scrollbar="auto",
            horizontal_scrollbar="auto",
            width=LEFT_PANEL_WIDTH,
            sizing_mode="stretch_height",
        )
        left_panel.styles = {
            "height": "100%",
            "min-height": "0",
            "flex-shrink": "0",
        }

        save_controls = column(save_button, spacing=8, width=SAVE_PANEL_WIDTH)

        selection_header = _create_section_header("Selected Parameters")
        conditions_header = _create_section_header("Search Conditions")
        conditions_text = Div(sizing_mode="stretch_width")

        def update_conditions_panel() -> None:
            """
            Update the Search Conditions panel from current condition data.
            現在の条件データからSearch Conditions領域を更新する。
            """
            formatted = _build_conditions_text()
            conditions_text.text = (
                "<pre style='margin: 0; font-family: Consolas, monospace; "
                "font-size: 13px; white-space: pre;'>"
                f"{html.escape(formatted)}"
                "</pre>"
            )

        selection_content = column(
            selection_header,
            solution_text,
            save_controls,
            spacing=SECTION_SPACING,
            sizing_mode="stretch_width",
        )
        selection_tab = TabPanel(
            title="Select Parameter",
            child=ScrollBox(
                child=selection_content,
                vertical_scrollbar="auto",
                horizontal_scrollbar="auto",
                sizing_mode="stretch_both",
            ),
        )

        importance_items = list(importance_figs.values())
        if importance_items:
            importance_rows = [
                row(*importance_items[i:i + 2], spacing=SECTION_SPACING)
                for i in range(0, len(importance_items), 2)
            ]
            importance_layout = column(*importance_rows, spacing=SECTION_SPACING)
        else:
            importance_layout = Div(text="No variable importance data available.")

        analysis_header = _create_section_header("Parameter Importance")
        violin_note = Div(
            text=(
                "<div style='font-size: 12px; color: #555; line-height: 1.5;'>"
                "* Scaled variables are shown in 0-1. Categoricals are encoded as 0 to n-1."
                "</div>"
            )
        )
        network_metrics_representative_note = Div(
            text=(
                "<div style='font-size: 12px; color: #555; line-height: 1.5;'>"
                f"{NETWORK_METRICS_REPRESENTATIVE_NOTE}"
                "</div>"
            ),
            sizing_mode="stretch_width",
        )

        network_metrics_table = column(
            column(
                clustering_row_display,
                min_width=NETWORK_GRID_CELL_MIN_WIDTH,
                sizing_mode="stretch_width",
            ),
            column(
                path_row_display,
                min_width=NETWORK_GRID_CELL_MIN_WIDTH,
                min_height=NETWORK_TEXT_CELL_HEIGHT,
                sizing_mode="stretch_width",
            ),
            column(
                row(Div(text="Degree Scale", width=100), degree_scale_mode),
                degree_hist_fig,
                network_metrics_note_display,
                spacing=4,
                sizing_mode="stretch_width",
            ),
            column(
                adjacency_heatmap_fig,
                spacing=4,
                sizing_mode="stretch_width",
            ),
            spacing=4,
            sizing_mode="stretch_width",
        )

        analysis_content = column(
            analysis_header,
            importance_layout,
            _create_section_header("Parameter Correlation"),
            correlation_fig,
            conditions_header,
            conditions_text,
            _create_section_header("Violin Plot"),
            violin_fig,
            violin_note,
            _create_section_header("Network Metrics"),
            network_metrics_representative_note,
            network_metrics_table,
            spacing=SECTION_SPACING,
            sizing_mode="stretch_width",
        )
        analysis_tab = TabPanel(
            title="Analysis",
            child=ScrollBox(
                child=analysis_content,
                vertical_scrollbar="auto",
                horizontal_scrollbar="auto",
                sizing_mode="stretch_both",
            ),
        )

        settings_header = _create_section_header("Axis Range")

        settings_content = column(
            settings_header,
            row(Div(text="X Range", width=60), x_mode, x_min_input, x_max_input),
            row(Div(text="Y Range", width=60), y_mode, y_min_input, y_max_input),
            row(Div(text="", width=60), axis_range_apply_button),
            spacing=SECTION_SPACING,
            sizing_mode="stretch_width",
        )
        settings_tab = TabPanel(
            title="Settings",
            child=ScrollBox(
                child=settings_content,
                vertical_scrollbar="auto",
                horizontal_scrollbar="auto",
                sizing_mode="stretch_both",
            ),
        )

        hypervolume_header = _create_section_header("Hypervolume")

        hypervolume_content = column(
            hypervolume_header,
            hypervolume_fig,
            hypervolume_settings,
            spacing=SECTION_SPACING,
            sizing_mode="stretch_width",
        )
        hypervolume_tab = TabPanel(
            title="Hypervolume",
            child=ScrollBox(
                child=hypervolume_content,
                vertical_scrollbar="auto",
                horizontal_scrollbar="auto",
                sizing_mode="stretch_both",
            ),
        )

        tabs = Tabs(
            tabs=[selection_tab, analysis_tab, hypervolume_tab, settings_tab],
            sizing_mode="stretch_both",
        )
        tabs.stylesheets = [
            InlineStyleSheet(css="""
                .bk-tab {
                    font-size: 18px;
                }
                .bk-tabs-header {
                    position: sticky;
                    top: 0;
                    z-index: 20;
                    background-color: #ffffff;
                }
                .bk-tab-panels {
                    min-height: 0;
                    overflow: auto;
                }
                .bk-tab-panel {
                    min-height: 0;
                }
            """)
        ]
        tabs.styles = {
            "height": "100%",
            "min-height": "0",
            "overflow": "hidden",
        }

        right_panel = column(
            tabs,
            spacing=SECTION_SPACING,
            sizing_mode="stretch_both",
        )
        right_panel.styles = {
            "height": "100%",
            "overflow": "hidden",
            "min-height": "0",
            "min-width": f"{RIGHT_PANEL_MIN_WIDTH}px",
            "flex-shrink": "0",
        }

        main_row = row(
            left_panel,
            right_panel,
            spacing=SECTION_SPACING,
            sizing_mode="stretch_width",
        )
        main_row.styles = {
            "height": f"calc(100vh - {CONTENT_TOP_OFFSET}px)",
            "overflow-x": "auto",
            "overflow-y": "hidden",
            "align-items": "stretch",
            "min-height": "0",
        }

        layout = column(
            header_section,
            main_row,
            spacing=SECTION_SPACING,
            sizing_mode="stretch_width",
        )
        layout.styles = {
            "padding-left": f"{LEFT_PADDING}px",
            "padding-top": f"{TOP_PADDING + TITLE_BAR_HEIGHT}px",
            "height": "100vh",
            "overflow": "hidden",
            "box-sizing": "border-box",
        }


        curdoc().add_periodic_callback(periodic_update, self.auto_update_interval_ms)
        self._update_hypervolume_source(hypervolume_source, force=True)
        _sync_hv_y_range_inputs(hypervolume_source)
        update_importance_plots()  # 変数重要度グラフを初期化
        update_correlation_heatmap()  # 相関行列ヒートマップを初期化
        update_conditions_panel()  # Select Parameter内のSearch Conditionsを初期化
        _apply_axis_ranges()
        _update_source()

        curdoc().js_on_event(DocumentReady, CustomJS(
            args=dict(key_recv=key_receiver),
            code="""
            document.addEventListener('keydown', function(e) {
                function getDeepActiveElement() {
                    let active = document.activeElement;
                    while (
                        active &&
                        active instanceof Element &&
                        active.shadowRoot &&
                        active.shadowRoot.activeElement
                    ) {
                        active = active.shadowRoot.activeElement;
                    }
                    return active;
                }

                function isEditableElement(el) {
                    if (!el || !(el instanceof Element)) return false;
                    const tag = (el.tagName || '').toLowerCase();
                    if (tag === 'input' || tag === 'textarea' || tag === 'select') return true;
                    const contentEditableAttr = (el.getAttribute('contenteditable') || '').toLowerCase();
                    if (el.isContentEditable && contentEditableAttr !== 'false') return true;
                    const role = (el.getAttribute('role') || '').toLowerCase();
                    if (role === 'textbox' || role === 'combobox') return true;
                    if (typeof el.closest === 'function') {
                        const editableParent = el.closest(
                            'input, textarea, select, [contenteditable]:not([contenteditable=\"false\"]), [role=\"textbox\"], [role=\"combobox\"]'
                        );
                        if (editableParent) return true;
                    }
                    return false;
                }

                function eventPathHasEditable(eventObj) {
                    if (!eventObj || typeof eventObj.composedPath !== 'function') return false;
                    const path = eventObj.composedPath();
                    for (const node of path) {
                        if (node instanceof Element && isEditableElement(node)) return true;
                    }
                    return false;
                }

                if (eventPathHasEditable(e) || isEditableElement(getDeepActiveElement())) return;
                if (e.isComposing || e.altKey || e.ctrlKey || e.metaKey) return;
                if (e.key === 'ArrowLeft') {
                    const prefix = e.shiftKey ? 'prev5_' : 'prev_';
                    key_recv.value = prefix + Date.now();
                } else if (e.key === 'ArrowRight') {
                    const prefix = e.shiftKey ? 'next5_' : 'next_';
                    key_recv.value = prefix + Date.now();
                }
            });
            """,
        ))

        curdoc().add_root(layout)
        curdoc().title = self._get_title_text()


visualizer = VisualizerServer()
visualizer.create_app()

