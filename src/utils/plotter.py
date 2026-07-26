"""
Static multi-objective optimization image saving utilities
多目的最適化結果の静的画像保存ユーティリティ
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.optimizer.utils.pareto_utils import extract_pareto_front

# matplotlibで散布図表示


def _sort_pareto_for_step_plot(
    pareto_points: np.ndarray,
    directions: List[str]
) -> Tuple[np.ndarray, str]:
    """
    Sort Pareto points and return them with a step-plot style.
    パレート点をソートし、階段状プロット形式とのタプルで返す。
    """
    # pareto_pointsは2目的の点、directionsはminimizeまたはmaximizeを並べた列とする。
    if len(pareto_points) < 2:
        return pareto_points, "steps-post"

    x_dir = directions[0] if len(directions) > 0 else "minimize"

    # X軸でソート（minimizeなら昇順、maximizeなら降順）
    if x_dir == "minimize":
        sorted_idx = np.argsort(pareto_points[:, 0])
    else:
        sorted_idx = np.argsort(-pareto_points[:, 0])

    sorted_points = pareto_points[sorted_idx]

    # 支配領域（worse side）の境界を示すため、常に水平→垂直（post）の階段にする
    step_style = "steps-post"

    return sorted_points, step_style

class ParetoPlotter:
    """
    Plot multi-objective optimization Pareto fronts and summary metrics.
    多目的最適化のパレートフロントと概要指標を描画する。
    """

    def __init__(self):
        """
        Initialize default plotting settings.
        既定のプロット設定を初期化する。
        """
        plt.rcParams['font.size'] = 12

    def plot_pareto_front(self, results: Dict, save_path: Optional[str] = None):
        """
        Plot a Pareto front and optionally save the figure.
        パレートフロントを描画し、必要に応じて図を保存する。
        """
        trials = results['trials']
        pareto_front = results['pareto_front']
        objective_names = results.get('objective_names', ['f1', 'f2'])
        objective_directions = results.get('objective_directions', [])

        if not trials:
            print("No trials to plot")
            return

        # Optuna trials（.values属性あり）と辞書形式の多目的最適化結果の両方に対応
        values_list = [t['values'] if isinstance(t, dict) else t.values for t in trials]
        values_list = [np.atleast_1d(v) for v in values_list]

        try:
            values = np.vstack(values_list)
        except ValueError as e:
            print(f"Warning: vstack failed ({e}), trying np.array")
            values = np.array(values_list)

        if values.ndim == 1:
            values = values.reshape(-1, 1)

        if values.shape[1] < 2:
            print("Error: values array needs at least 2 columns for pareto front plot")
            return

        # 3つのサブプロットを作成: 全世代、最終世代、パレートフロント
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # 目的関数名へ最適化方向を加えた軸ラベルを構築する
        def make_label(idx, default):
            """
            Build and return an objective-axis label with its direction.
            最適化方向を含む目的関数の軸ラベルを構築して返す。
            """
            name = objective_names[idx] if idx < len(objective_names) else default
            if idx < len(objective_directions):
                direction = objective_directions[idx]
                return f"{name} ({direction})"
            return name

        x_label = make_label(0, "Objective 1")
        y_label = make_label(1, "Objective 2")

        # 左プロット: 全世代
        ax1.scatter(values[:, 0], values[:, 1],
                    color="lightblue", alpha=0.6, s=50, label="All solutions")

        if len(pareto_front) > 0:
            if pareto_front.ndim == 1:
                pareto_front = pareto_front.reshape(1, -1)
            ax1.scatter(pareto_front[:, 0], pareto_front[:, 1],
                        color="red", edgecolor="black", s=50,
                        label=f"Final generation ({len(pareto_front)})")

        ax1.set_xlabel(x_label)
        ax1.set_ylabel(y_label)
        ax1.set_title("All Generations")
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # 中央プロット: 最終世代
        if len(pareto_front) > 0:
            ax2.scatter(pareto_front[:, 0], pareto_front[:, 1],
                        color="red", edgecolor="black", s=50)

        ax2.set_xlabel(x_label)
        ax2.set_ylabel(y_label)
        ax2.set_title("Final Generation")
        ax2.grid(True, alpha=0.3)

        # 右プロット: 真のパレートフロント
        if len(pareto_front) > 0 and len(objective_directions) >= 2:
            true_pareto, pareto_indices = extract_pareto_front(pareto_front, objective_directions)
            ax3.scatter(pareto_front[:, 0], pareto_front[:, 1],
                        color="lightgray", alpha=0.6, s=30,
                        label=f"Final generation ({len(pareto_front)})")
            ax3.scatter(true_pareto[:, 0], true_pareto[:, 1],
                        color="red", edgecolor="black", s=50,
                        label=f"Pareto front ({len(true_pareto)})")
            # パレートフロントを階段状の線で結ぶ
            if len(true_pareto) >= 2:
                sorted_pareto, step_style = _sort_pareto_for_step_plot(
                    true_pareto, objective_directions
                )
                ax3.plot(sorted_pareto[:, 0], sorted_pareto[:, 1],
                         color="red", linewidth=1.5, drawstyle=step_style,
                         alpha=0.8)
            ax3.legend(loc='upper right')
        elif len(pareto_front) > 0:
            ax3.scatter(pareto_front[:, 0], pareto_front[:, 1],
                        color="red", edgecolor="black", s=50,
                        label=f"Final generation ({len(pareto_front)})")
            ax3.legend(loc='upper right')

        ax3.set_xlabel(x_label)
        ax3.set_ylabel(y_label)
        ax3.set_title("Pareto Front")
        ax3.grid(True, alpha=0.3)

        # 軸範囲を統一
        all_x = np.concatenate([values[:, 0], pareto_front[:, 0]]) if len(pareto_front) > 0 else values[:, 0]
        all_y = np.concatenate([values[:, 1], pareto_front[:, 1]]) if len(pareto_front) > 0 else values[:, 1]
        x_min, x_max = all_x.min(), all_x.max()
        y_min, y_max = all_y.min(), all_y.max()
        x_margin = (x_max - x_min) * 0.05
        y_margin = (y_max - y_min) * 0.05
        ax1.set_xlim(x_min - x_margin, x_max + x_margin)
        ax1.set_ylim(y_min - y_margin, y_max + y_margin)
        ax2.set_xlim(x_min - x_margin, x_max + x_margin)
        ax2.set_ylim(y_min - y_margin, y_max + y_margin)
        ax3.set_xlim(x_min - x_margin, x_max + x_margin)
        ax3.set_ylim(y_min - y_margin, y_max + y_margin)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.close()

