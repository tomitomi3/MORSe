"""
PED-ANOVA variable importance using KDE and Pearson divergence
KDEとピアソン発散を使用するPED-ANOVA変数重要度
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.stats import gaussian_kde

from src.utils.common_messages import operation_failed_message

logger = logging.getLogger(__name__)

# 参考文献:
# WATANABE, Shuhei; BANSAL, Archit; HUTTER, Frank. PED-ANOVA: Efficiently quantifying hyperparameter importance in arbitrary subspaces. arXiv preprint arXiv:2304.10255, 2023.
# https://github.com/nabenabe0928/ped-anova

def calculate_ped_anova(
    X: np.ndarray,
    objectives: np.ndarray,
    param_names: List[str],
    objective_names: List[str],
    objective_directions: List[str],
    config: Dict[str, Any],
    common_config: Dict[str, Any],
) -> Dict[str, Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
    """
    Calculate PED-ANOVA and return importance tuples keyed by objective name.
    PED-ANOVAを計算し、目的関数名ごとの重要度タプルを辞書で返す。
    """
    # Xの各列はparam_namesの決定変数、objectivesの各列は目的関数名とDirectionに対応する。
    # configはグリッド点数と上位解比率を保持し、common_configは共通APIとの互換性のため受け取る。
    # Directionに従って選んだ上位解と全解のKDEからピアソン発散重要度を算出する。
    # 戻り値は目的関数名ごとの重要度、標準偏差、解件数メタデータの辞書とする。
    n_samples, n_params = X.shape

    n_grid_points = config.get("n_grid_points", 100)
    top_ratio = config.get("top_ratio", 0.1)

    if n_samples < 2:
        logger.warning("PED-ANOVA needs at least two samples (n=%d)", n_samples)
        return _empty_results(objective_names, n_params, n_samples, top_ratio)

    results = {}

    for obj_idx, obj_name in enumerate(objective_names):
        y = objectives[:, obj_idx]
        direction = objective_directions[obj_idx]

        if direction == "minimize":
            sorted_indices = np.argsort(y)  # 小さい順
        else:  # 最大化の場合
            sorted_indices = np.argsort(-y)  # 大きい順

        n_top = max(1, int(n_samples * top_ratio))
        top_indices = sorted_indices[:n_top]

        X_top = X[top_indices]
        X_all = X

        # 各変数について重要度を計算
        importance = np.zeros(n_params)
        for d in range(n_params):
            importance[d] = _compute_variable_importance(
                X_top[:, d],
                X_all[:, d],
                n_grid_points,
            )

        importance = _normalize_importance(importance)

        # 上位解と全解の件数および上位比率を記録する。
        metadata = {
            "n_top_solutions": n_top,
            "n_total_solutions": n_samples,
            "top_ratio": top_ratio,
        }

        # 1回の発散計算結果を使用するため、標準偏差は0とする。
        std_importance = np.zeros(n_params)

        results[obj_name] = (importance, std_importance, metadata)

    return results


def _empty_results(
    objective_names: List[str],
    n_params: int,
    n_samples: int,
    top_ratio: float,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
    """
    Build and return empty importance results for every objective.
    各目的関数の空の重要度結果を構築して返す。
    """
    results = {}
    for obj_name in objective_names:
        results[obj_name] = (
            np.zeros(n_params),
            np.zeros(n_params),
            {
                "n_top_solutions": 0,
                "n_total_solutions": n_samples,
                "top_ratio": top_ratio,
                "warning": "Insufficient data",
            },
        )
    return results


def _compute_variable_importance(
    x_top: np.ndarray,
    x_all: np.ndarray,
    n_grid_points: int,
) -> float:
    """
    Calculate variable importance.
    変数重要度を計算する。
    """
    # x_topは上位解、x_allは全解の1次元値、n_grid_pointsは発散計算のグリッド点数とする。
    # 全解に変動がなく分布差を定義できない場合は重要度を0とする。
    if np.std(x_all) < 1e-10:
        return 0.0

    # 上位解だけが一点に集中する場合は未正規化重要度を1とする。
    if np.std(x_top) < 1e-10:
        return 1.0

    try:
        # カーネル密度推定
        # KDEの参考実装はhttps://github.com/nabenabe0928/local-anovaを参照する
        bandwidth = "scott"
        kde_top = gaussian_kde(x_top, bw_method=bandwidth)
        kde_all = gaussian_kde(x_all, bw_method=bandwidth)

        x_min = min(x_all.min(), x_top.min())
        x_max = max(x_all.max(), x_top.max())
        margin = (x_max - x_min) * 0.1
        grid = np.linspace(x_min - margin, x_max + margin, n_grid_points)

        pdf_top = kde_top(grid)
        pdf_all = kde_all(grid)

        # ピアソン発散計算: Σ (P_top - P_all)² / P_all
        # 数値安定性のため、pdf_all が小さい場合は除外
        eps = 1e-10
        mask = pdf_all > eps
        if mask.sum() < 2:
            return 0.0

        divergence = np.sum(
            ((pdf_top[mask] / pdf_all[mask] - 1) ** 2) * pdf_all[mask]
        )

        return float(divergence)

    except (ValueError, np.linalg.LinAlgError) as exc:
        logger.warning(operation_failed_message("KDE calculation", exc))
        return 0.0


def _normalize_importance(importance: np.ndarray) -> np.ndarray:
    """
    Normalize importance values to a unit sum and return them.
    重要度を合計1へ正規化して返す。
    """
    total = importance.sum()
    if total > 0:
        return importance / total
    return importance
