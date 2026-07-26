"""
Variable importance analysis facade
変数重要度分析ファサード
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.utils import shuffle

from src.analysis.mdi import calculate_mdi
from src.analysis.pfi import calculate_pfi
from src.analysis.ped_anova import calculate_ped_anova
from src.utils.common_messages import required_value_message, unsupported_value_message

SUPPORTED_ALGORITHMS = ["MDI", "PFI", "PED_ANOVA"]

# 後方互換性のためのエイリアス
ALGORITHM_ALIASES = {
    "RandomForest": "MDI",
    "PermutationImportance": "PFI",
}


class VariableImportanceCalculator:
    """
    Calculate decision-variable importance with various algorithms.
    各種アルゴリズムで決定変数の重要度を計算する。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the calculator with variable-analysis configuration.
        変数分析設定を受け取って計算器を初期化する。
        """
        # configはMDI、PFI、PED_ANOVAのいずれか、乱数seed、各アルゴリズム固有設定を保持する。
        # アルゴリズム名の解決（エイリアス対応）
        raw_algorithm = config.get("algorithm", "MDI")
        self.algorithm = ALGORITHM_ALIASES.get(raw_algorithm, raw_algorithm)

        if self.algorithm not in SUPPORTED_ALGORITHMS:
            raise ValueError(
                unsupported_value_message(
                    "algorithm",
                    self.algorithm,
                    SUPPORTED_ALGORITHMS,
                )
            )

        self.random_state = config.get("random_state", 123456)

        # アルゴリズム固有の設定を先に取得
        self.algo_config = config.get(self.algorithm, {})

        # max_samples はアルゴリズム固有設定から取得（フォールバックでトップレベルも確認）
        self.max_samples = self.algo_config.get(
            "max_samples", config.get("max_samples", 5000)
        )

        # 共通設定（MDI/PFI用）
        if self.algorithm in ["MDI", "PFI"]:
            self.common_config = {
                "test_size": self.algo_config.get(
                    "test_size", config.get("test_size", 0.2)
                ),
                "n_iterations": self.algo_config.get(
                    "n_iterations", config.get("n_iterations", 10)
                ),
                "random_state": self.random_state,
            }
        else:
            # PED_ANOVA の場合は minimal
            self.common_config = {
                "random_state": self.random_state,
            }

        # サンプル数追跡用
        self._n_samples_total = 0
        self._n_samples_used = 0

    def calculate(
        self,
        params: np.ndarray,
        objectives: np.ndarray,
        param_names: List[str],
        objective_names: List[str],
        objective_directions: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Calculate and return importance, correlation, and configuration results.
        変数重要度、相関、設定情報を含む結果辞書を計算して返す。
        """
        # paramsとobjectivesはサンプル行を共有し、各名称は対応する列順とする。
        # objective_directionsはminimizeまたはmaximizeの列順で、PED_ANOVAでは必須とする。
        # importanceは目的関数ごとに変数名、平均、標準偏差、メタデータを保持する。
        n_samples = len(params)
        if n_samples < 2:
            return self._empty_result()

        # サンプル数を記録
        self._n_samples_total = n_samples

        # サンプリング（max_samples を超える場合、初回に1回だけ）
        params_sampled, objectives_sampled = self._sample_data(params, objectives)

        # 使用サンプル数を記録
        self._n_samples_used = len(params_sampled)

        result = {
            "importance": {},
            "correlation": {},
            "config": self._build_config_info(),
        }

        # PED_ANOVA は目的関数ごとに重要度を計算
        if self.algorithm == "PED_ANOVA":
            if objective_directions is None:
                raise ValueError(required_value_message("objective_directions"))
            ped_results = calculate_ped_anova(
                params_sampled,
                objectives_sampled,
                param_names,
                objective_names,
                objective_directions,
                self.algo_config,
                self.common_config,
            )
            # 目的ごとに結果を格納
            for obj_name, (mean_imp, std_imp, metadata) in ped_results.items():
                result["importance"][obj_name] = {
                    "variables": param_names,
                    "mean": mean_imp.tolist(),
                    "std": std_imp.tolist(),
                    "metadata": metadata,
                }
        else:
            # MDI/PFI: 各目的関数に対して重要度を計算
            for obj_idx, obj_name in enumerate(objective_names):
                y = objectives_sampled[:, obj_idx]

                if self.algorithm == "MDI":
                    mean_imp, std_imp, metadata = calculate_mdi(
                        params_sampled, y, param_names,
                        self.algo_config, self.common_config
                    )
                elif self.algorithm == "PFI":
                    mean_imp, std_imp, metadata = calculate_pfi(
                        params_sampled, y, param_names,
                        self.algo_config, self.common_config
                    )

                result["importance"][obj_name] = {
                    "variables": param_names,
                    "mean": mean_imp.tolist(),
                    "std": std_imp.tolist(),
                    "metadata": metadata,
                }

        # 相関行列を計算
        result["correlation"] = self._calculate_correlation(params_sampled, param_names)

        return result

    def _sample_data(
        self, params: np.ndarray, objectives: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample once and return aligned parameter and objective arrays.
        1回だけサンプリングし、対応を維持したパラメータ配列と目的関数値配列を返す。
        """
        n_samples = len(params)
        if n_samples <= self.max_samples:
            # シャッフルのみ
            return shuffle(params, objectives, random_state=self.random_state)

        # ランダムサンプリング
        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(n_samples, size=self.max_samples, replace=False)
        return params[indices], objectives[indices]

    def _build_config_info(self) -> Dict[str, Any]:
        """
        Build and return metadata for the active analysis configuration.
        使用中の分析設定メタデータを構築して返す。
        """
        return {
            "algorithm": self.algorithm,
            "n_samples_used": self._n_samples_used,
            "n_samples_total": self._n_samples_total,
            "max_samples": self.max_samples,
            **self.common_config,
            **self.algo_config,
        }

    def _calculate_correlation(
        self,
        params: np.ndarray,
        param_names: List[str],
    ) -> Dict[str, Any]:
        """
        Calculate and return the parameter correlation matrix with names.
        パラメータ間の相関行列と名前を計算して返す。
        """
        n_samples, n_params = params.shape

        if n_params < 2 or n_samples < 2:
            return {
                "matrix": [],
                "variables": param_names,
            }

        # ピアソン相関係数を計算
        with np.errstate(invalid="ignore"):
            corr_matrix = np.corrcoef(params.T)

        # NaN を 0 に置換
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        return {
            "matrix": corr_matrix.tolist(),
            "variables": param_names,
        }

    def _empty_result(self) -> Dict[str, Any]:
        """
        Return an empty analysis result for insufficient data.
        データ不足時の空の分析結果を返す。
        """
        return {
            "importance": {},
            "correlation": {"matrix": [], "variables": []},
            "config": {"algorithm": self.algorithm},
        }

    def load_generation_data(
        self, result_dir: Path
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load configured generations and return parameter and objective arrays.
        設定対象の全世代を読み込み、パラメータ配列と目的関数値配列を返す。
        """
        # result_dirはCondition.jsonと世代ファイルを含むResultディレクトリとする。
        # Condition.jsonのmax_generationsを参照し、対象世代のみを読み込む。
        all_params = []
        all_objectives = []

        # Condition.json から世代数を取得
        condition_path = result_dir / "Condition.json"
        with open(condition_path, "r", encoding="utf-8") as f:
            condition = json.load(f)
        max_generations = condition.get("optimizer_config", {}).get("max_generations", 0)

        # 世代番号を指定して読み込み
        for g in range(max_generations):
            gen_path = result_dir / f"Generation{g:05d}.json"
            with open(gen_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            all_params.extend(data.get("params", []))
            all_objectives.extend(data.get("objectives", []))

        if not all_params:
            return np.array([]), np.array([])

        return np.array(all_params), np.array(all_objectives)

