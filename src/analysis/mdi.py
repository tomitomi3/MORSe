"""
MDI variable importance using RandomForest feature importances
RandomForestのfeature_importances_を使用するMDI変数重要度
"""

from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def calculate_mdi(
    X: np.ndarray,
    y: np.ndarray,
    param_names: List[str],
    config: Dict[str, Any],
    common_config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Calculate MDI importance and return mean, standard deviation, and metadata.
    MDI重要度を計算し、平均、標準偏差、メタデータのタプルを返す。
    """
    # Xの各列はparam_namesの決定変数に対応し、yは1つの目的関数値とする。
    # configは決定木条件、common_configは検証比率、反復回数、乱数seedを保持する。
    # 各反復でRandomForestのfeature_importances_を算出し、重要度の合計を1にする。
    # 戻り値は重要度の平均、標準偏差、評価スコアのメタデータとする。
    n_samples = len(X)
    n_params = len(param_names)

    if n_samples < 2:
        return (
            np.zeros(n_params),
            np.zeros(n_params),
            {"baseline_score": 0.0, "scores_per_iteration": []},
        )

    n_estimators = config.get("n_estimators", 100)
    max_depth = config.get("max_depth", None)
    min_samples_leaf = config.get("min_samples_leaf", 5)
    test_size = common_config.get("test_size", 0.2)
    n_iterations = common_config.get("n_iterations", 10)
    random_state = common_config.get("random_state", 123456)

    iteration_importances = []
    iteration_scores = []

    for i in range(n_iterations):
        iter_seed = random_state + i

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=iter_seed,
        )

        # RandomForest から重要度を得る。
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=iter_seed,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)

        importance = rf.feature_importances_
        importance = _normalize_importance(importance)
        iteration_importances.append(importance)

        # 性能を見るため検証データのR^2
        score = rf.score(X_test, y_test)
        iteration_scores.append(score)

    all_importances = np.array(iteration_importances)
    mean_importance = all_importances.mean(axis=0)
    std_importance = all_importances.std(axis=0)

    # 反復平均後の合計が1になるよう再正規化する。
    mean_importance = _normalize_importance(mean_importance)

    # 反復ごとの評価値をfloatへ変換し、平均評価値とともに記録する。
    metadata = {
        "baseline_score": float(np.mean(iteration_scores)),
        "scores_per_iteration": [float(score) for score in iteration_scores],
    }

    return mean_importance, std_importance, metadata


def _normalize_importance(importance: np.ndarray) -> np.ndarray:
    """
    Normalize importance values to a unit sum and return them.
    重要度を合計1へ正規化して返す。
    """
    total = importance.sum()
    if total > 0:
        return importance / total
    return importance
