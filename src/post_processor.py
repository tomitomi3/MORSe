"""
Multi-objective optimization result post-processing
多目的最適化結果の後処理
"""

import json
import os
from datetime import datetime
from pathlib import Path

from src.analysis.variable_importance import VariableImportanceCalculator
from src.file_io.importance_writer import ImportanceWriter
from src.utils.plotter import ParetoPlotter


class PostProcessor:
    """
    Orchestrate analysis and output after multi-objective optimization.
    多目的最適化後の分析と出力を統括する。
    """

    def __init__(self, result_dir: Path, config: dict):
        """
        Initialize post-processing with the result directory and configuration.
        結果ディレクトリと設定を受け取って後処理を初期化する。
        """
        self.result_dir = Path(result_dir)
        self.config = config

    def run_all(self, results: dict, task, defaults: dict) -> None:
        """
        Run all configured post-processing steps.
        設定された全ての後処理を実行する。
        """
        self.run_variable_importance()
        self.generate_plots(results, task, defaults)
        self.save_text_results(results, defaults)

    def run_variable_importance(self) -> None:
        """
        Calculate and save variable importance when enabled.
        有効な場合に変数重要度を計算して保存する。
        """
        va_config = self.config.get('variable_analysis', {})
        if not va_config.get('enabled', True):
            return

        print("Calculating variable importance...")
        calculator = VariableImportanceCalculator(va_config)
        params_array, objectives_array = calculator.load_generation_data(self.result_dir)

        if len(params_array) == 0:
            return

        condition_data = self._load_condition()
        param_names = condition_data.get('parameter_names', [])
        obj_names = condition_data.get('objective_names', [])
        obj_directions = condition_data.get('objective_directions', [])

        importance_result = calculator.calculate(
            params_array, objectives_array, param_names, obj_names, obj_directions
        )

        importance_writer = ImportanceWriter(self.result_dir)
        importance_writer.write(importance_result)

    def generate_plots(self, results: dict, task, defaults: dict) -> None:
        """
        Generate and save configured static plots.
        設定された静的プロットを生成して保存する。
        """
        if not defaults['save_plots']:
            return

        plotter = ParetoPlotter()

        condition_data = self._load_condition()
        objective_directions = condition_data.get('objective_directions', [])

        adapted_results = {
            'trials': results['trials'],
            'pareto_front': results['pareto_front'],
            'optimization_time': results['optimization_time'],
            'objective_names': task.get_objective_names(),
            'objective_directions': objective_directions,
            'pareto_solutions': results.get('pareto_solutions', []),
        }

        result_dir_str = str(self.result_dir)
        os.makedirs(result_dir_str, exist_ok=True)

        save_path_pareto = (
            os.path.join(result_dir_str, f"{defaults['task']}_pareto_front.png")
            if defaults['save_plots'] else None
        )
        plotter.plot_pareto_front(adapted_results, save_path=save_path_pareto)

    def save_text_results(self, results: dict, defaults: dict) -> None:
        """
        Save multi-objective optimization summaries and solutions as text files.
        多目的最適化の概要と解をテキストファイルへ保存する。
        """
        if not defaults['save_plots']:
            return

        task = results['task']
        # 見出し用の表示名は単語区切りへ変換し、ファイル名にはタスクフォルダ名をそのまま使う。
        task_display_name = defaults['task'].replace('_', ' ').title()
        task_file_name = defaults['task']
        result_dir_str = str(self.result_dir)

        os.makedirs(result_dir_str, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        summary_filename = os.path.join(result_dir_str, f"{task_file_name}_results_{timestamp}.txt")
        with open(summary_filename, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write(f"{task_display_name} Multi-Objective Optimization Results\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("Configuration:\n")
            f.write(f"  Task: {task.name}\n")
            f.write(f"  Generations: {defaults['generations']}\n")
            f.write(f"  Random seed: {defaults['seed']}\n\n")

            f.write("Optimization Results:\n")
            f.write(f"  Total evaluations: {len(results['trials'])}\n")
            f.write(f"  Pareto solutions: {len(results['pareto_front'])}\n")
            f.write(f"  Optimization time: {results['optimization_time']:.2f} seconds\n\n")

            if len(results['pareto_front']) > 0:
                f.write("Pareto Front Range:\n")
                pf = results['pareto_front']
                obj_names = task.get_objective_names()

                for i, obj_name in enumerate(obj_names):
                    f.write(f"  {obj_name}: [{pf[:, i].min():.4f}, {pf[:, i].max():.4f}]\n")
                f.write("\n")

        pareto_solutions = results.get('pareto_solutions', [])
        solutions_filename = os.path.join(result_dir_str, f"{task_file_name}_solutions_{timestamp}.txt")
        if pareto_solutions:
            with open(solutions_filename, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write(f"DETAILED {task_display_name.upper()} SOLUTIONS\n")
                f.write("=" * 60 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                for i, solution in enumerate(pareto_solutions):
                    f.write(f"Solution {i + 1}:\n")
                    f.write(f"  Params: {solution['params']}\n")
                    f.write(f"  Objective Values: {solution['objectives']}\n")
                    f.write("-" * 40 + "\n")

    def _load_condition(self) -> dict:
        """
        Load and return Condition.json or an empty dictionary.
        Condition.jsonを読み込み、存在しなければ空の辞書を返す。
        """
        path = self.result_dir / "Condition.json"
        if not path.exists():
            return {}
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

