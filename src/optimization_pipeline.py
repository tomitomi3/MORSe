"""
Multi-objective optimization processing pipeline by MORSe
MORSeによる多目的最適化処理パイプライン
"""

import json
import shutil
from pathlib import Path

from src.file_io.condition_writer import ConditionWriter
from src.file_io.generation_writer import GenerationWriter
from src.generation_callback import GenerationCallback
from src.post_processor import PostProcessor
from src.task_runner import TaskRunner
from src.tasks import create_task
from src.utils.common_messages import (
    file_not_found_message,
    required_value_message,
    unsupported_value_message,
)
from src.utils.visualizer_launcher import (
    DEFAULT_PORT,
    release_bokeh_port,
    reserve_bokeh_port,
    start_bokeh_server,
    stop_bokeh_server,
    wait_for_bokeh_server,
)
from src.version import __version__


class OptimizationPipeline:
    """
    Orchestrate the complete multi-objective optimization workflow.
    多目的最適化ワークフロー全体を統括する。
    """

    def __init__(
        self,
        config: dict,
        result_dir: Path,
        setting_path: Path | None = None,
    ):
        """
        Initialize the pipeline with configuration and output paths.
        設定と出力パスを受け取ってパイプラインを初期化する。
        """
        self.config = config
        self.result_dir = Path(result_dir)
        self.setting_path = Path(setting_path) if setting_path is not None else None
        self.optimizer_params = self._build_optimizer_params()
        self.defaults = self._build_defaults(config)

    @classmethod
    def from_setting_file(cls, setting_path: Path):
        """
        Create and return a pipeline from a setting file.
        設定ファイルからパイプラインを生成して返す。
        """
        setting_path = Path(setting_path)
        if not setting_path.exists():
            raise FileNotFoundError(file_not_found_message(setting_path))

        with open(setting_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        result_dir = str(config.get("result_dir", "Result")).strip() or "Result"
        result_dir_path = Path(result_dir)
        result_dir_path.mkdir(parents=True, exist_ok=True)
        return cls(config=config, result_dir=result_dir_path, setting_path=setting_path)

    def run(self, visualization_port: int = DEFAULT_PORT) -> dict:
        """
        Execute the complete workflow and return its result dictionary.
        ワークフロー全体を実行して結果辞書を返す。
        """
        defaults = self.defaults

        port_reservation = None
        bokeh_process = None
        try:
            if defaults["realtime_visualize"]:
                port_reservation = reserve_bokeh_port(visualization_port)

            print("=== Starting Optimization ===")
            print(f"MORSe v{__version__}")

            # 実行に使用した設定を結果フォルダへ保存（上書き）
            self._copy_setting_file()

            # I/O Writer初期化
            condition_writer = ConditionWriter(self.result_dir)
            generation_writer = GenerationWriter(self.result_dir)

            # タスク作成
            task = self._create_task()

            # 多目的最適化設定
            optimizer_params = dict(self.optimizer_params)

            # Condition.json書き出し
            condition_writer.write(
                task,
                optimizer_params,
                ui_title=defaults["ui_title"],
            )

            # Bokehサーバー起動
            if defaults["realtime_visualize"]:
                release_bokeh_port(port_reservation)
                port_reservation = None
                bokeh_process = start_bokeh_server(
                    str(self.result_dir),
                    auto_open=True,
                    port=visualization_port,
                )

            # コールバック作成
            var_config = task.get_variables()
            param_names = list(var_config.variable_names)
            callback = GenerationCallback(generation_writer, param_names)

            # 多目的最適化実行
            runner = TaskRunner(
                task,
                n_generations=defaults["generations"],
                seed=defaults["seed"],
                optimizer_params=optimizer_params,
                on_generation_end=callback,
            )
            print("=== Starting Task Optimization ===")
            self._show_bokeh_status(bokeh_process)
            print(f"Starting optimization for {task.name}...")
            results = runner.optimize()

            # Condition.json更新
            condition_writer.update_elapsed_time(results["optimization_time"])

            # パレート解取得
            pareto_solutions = runner.get_pareto_solutions()
            results["pareto_solutions"] = pareto_solutions

            # 後処理
            post_processor = PostProcessor(self.result_dir, self.config)
            post_processor.run_all(results, task, defaults)

            # Bokehサーバー待機
            self._wait_bokeh(bokeh_process)
        except BaseException:
            release_bokeh_port(port_reservation)
            stop_bokeh_server(bokeh_process)
            raise

        return results

    def _build_defaults(self, config: dict) -> dict:
        """
        Build runtime settings for the task, seed, generations, and visualization.
        タスク、seed、世代数、可視化設定をまとめた実行設定を返す。
        """
        task_cfg = config.get("task", {})
        task_name = task_cfg.get("name")
        if not task_name:
            raise ValueError(required_value_message("task.name"))
        task_params = task_cfg.get("params", {})

        generations = self.optimizer_params["max_generations"]
        ui_title = str(config.get("ui_title", "")).strip()
        if not ui_title:
            ui_title = task_name

        return {
            "task": task_name,
            "task_params": task_params,
            "generations": generations,
            "seed": config.get("seed", 1234),
            "ui_title": ui_title,
            "save_plots": config.get("save_plots", True),
            "realtime_visualize": config.get("realtime_visualize", False),
        }

    def _create_task(self):
        """
        Dynamically create and return the configured task.
        設定されたタスクを動的に生成して返す。
        """
        defaults = self.defaults

        task_config = {
            "seed": defaults["seed"],
        }

        # タスク固有パラメータを追加（全タスク共通の汎用処理）
        task_config.update(defaults["task_params"])

        return create_task(defaults["task"], task_config)

    def _build_optimizer_params(self) -> dict:
        """
        Validate configuration and return multi-objective optimization parameters.
        設定を検証して多目的最適化パラメータ辞書を返す。
        """
        optimizer_cfg = self.config.get("optimizer", {})
        if not isinstance(optimizer_cfg, dict):
            raise ValueError("optimizer must be an object in config")

        algorithm = optimizer_cfg.get("algorithm")
        if not isinstance(algorithm, str) or not algorithm.strip():
            raise ValueError(required_value_message("optimizer.algorithm"))

        if algorithm != "MOEAD":
            raise ValueError(
                unsupported_value_message(
                    "optimizer algorithm",
                    algorithm,
                    ["MOEAD"],
                )
            )

        algorithm_cfg = optimizer_cfg.get("MOEAD")
        if not isinstance(algorithm_cfg, dict):
            raise ValueError(
                "optimizer.MOEAD is required when optimizer.algorithm='MOEAD'"
            )

        normalization_cfg = algorithm_cfg.get("normalization", {})
        if normalization_cfg is None:
            normalization_cfg = {}
        if not isinstance(normalization_cfg, dict):
            raise ValueError("optimizer.MOEAD.normalization must be an object")

        boundary_cfg = algorithm_cfg.get("boundary", {})
        if boundary_cfg is None:
            boundary_cfg = {}
        if not isinstance(boundary_cfg, dict):
            raise ValueError("optimizer.MOEAD.boundary must be an object")

        crossover_cfg = algorithm_cfg.get("crossover", {})
        if crossover_cfg is None:
            crossover_cfg = {}
        if not isinstance(crossover_cfg, dict):
            raise ValueError("optimizer.MOEAD.crossover must be an object")

        # 自動計算指定（None）を許容するパラメータ
        undx_m_sigma_xi_raw = crossover_cfg.get("undx_m_sigma_xi")
        undx_m_sigma_eta_raw = crossover_cfg.get("undx_m_sigma_eta")
        spx_epsilon_raw = crossover_cfg.get("spx_epsilon")

        max_generations = algorithm_cfg.get("generations", 100)
        normalization_mode = str(
            normalization_cfg.get("mode", "static_first_generation")
        ).lower()
        supported_modes = {"none", "basic", "sdn", "static_first_generation"}
        if normalization_mode not in supported_modes:
            raise ValueError(
                unsupported_value_message(
                    "normalization_mode",
                    normalization_mode,
                    sorted(supported_modes),
                )
            )
        sdn_final_generation = normalization_cfg.get("sdn_final_generation")
        if normalization_mode == "sdn" and sdn_final_generation is None:
            sdn_final_generation = max_generations

        return {
            "max_generations": max_generations,
            "population_size": algorithm_cfg.get("population_size", 100),
            "crossover_type": algorithm_cfg.get("crossover_type", "SBX"),
            "scalar_function_type": algorithm_cfg.get("scalar_function_type", "TCH"),
            "neighborhood_size": algorithm_cfg.get("neighborhood_size", 20),
            "delta": algorithm_cfg.get("delta", 0.9),
            "nr": algorithm_cfg.get("nr", 1),
            "eta_c": algorithm_cfg.get("eta_c", 20),
            "eta_m": algorithm_cfg.get("eta_m", 20),
            "use_polynomial_mutation": algorithm_cfg.get(
                "use_polynomial_mutation", True
            ),
            "use_stochastic_rounding": algorithm_cfg.get(
                "use_stochastic_rounding", True
            ),
            "gen_num_child": algorithm_cfg.get("gen_num_child", 1),
            "normalization_mode": normalization_mode,
            "sdn_final_generation": sdn_final_generation,
            "normalization_eps": normalization_cfg.get("eps", 1.0),
            "f1_static_normalization_upper_coeff": float(
                normalization_cfg.get("f1_static_upper_coeff", 1.0)
            ),
            "f1_static_normalization_lower_coeff": float(
                normalization_cfg.get("f1_static_lower_coeff", 1.0)
            ),
            "f2_static_normalization_upper_coeff": float(
                normalization_cfg.get("f2_static_upper_coeff", 1.0)
            ),
            "f2_static_normalization_lower_coeff": float(
                normalization_cfg.get("f2_static_lower_coeff", 1.0)
            ),
            "boundary_repair_mode": str(
                boundary_cfg.get("repair_mode", "clip")
            ).lower(),
            "boundary_repair_reflect_ratio": float(
                boundary_cfg.get("reflect_ratio", 0.3)
            ),
            "is_save_external_population": bool(
                algorithm_cfg.get("is_save_external_population", False)
            ),
            "pbi_theta": float(algorithm_cfg.get("pbi_theta", 5.0)),
            "undx_alpha": float(crossover_cfg.get("undx_alpha", 0.5)),
            "undx_beta": float(crossover_cfg.get("undx_beta", 0.35)),
            "undx_m_num_parents": int(crossover_cfg.get("undx_m_num_parents", 2)),
            "undx_m_sigma_xi": (
                None if undx_m_sigma_xi_raw is None else float(undx_m_sigma_xi_raw)
            ),
            "undx_m_sigma_eta": (
                None if undx_m_sigma_eta_raw is None else float(undx_m_sigma_eta_raw)
            ),
            "spx_epsilon": None if spx_epsilon_raw is None else float(spx_epsilon_raw),
            "tblx_alpha": float(crossover_cfg.get("tblx_alpha", 0.5)),
        }

    def _show_bokeh_status(self, process) -> None:
        """
        Show the confirmed Bokeh child process and URL before optimization.
        最適化前に起動確認済みBokeh子プロセスとURLを表示する。
        """
        if not process:
            return

        url = getattr(process, "morse_url", "http://localhost:5006/visualizer_server")
        print(f"Starting Bokeh server with process id: {process.pid}")
        print(url)

    def _wait_bokeh(self, process) -> None:
        """
        Wait for the Bokeh server process until completion or interruption.
        完了または割り込みまでBokehサーバープロセスを待機する。
        """
        if not process:
            return

        wait_for_bokeh_server(process)

    def _copy_setting_file(self) -> None:
        """
        Copy the source setting file into the result directory.
        元の設定ファイルを結果ディレクトリへコピーする。
        """
        if self.setting_path is None:
            return
        self.result_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.setting_path, self.result_dir / "setting.json")
