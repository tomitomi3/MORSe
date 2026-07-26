"""
Utilities for random selection
ランダム選択を行うユーティリティ
"""

from typing import List, TypeVar

import numpy as np


EPS = 1.2e-7    # 数値計算の微小値 共有利用

T = TypeVar('T')


def random_select(rng: np.random.Generator, items: List[T]) -> T:
    """
    Select and return one random item from a list.
    リストから1要素をランダムに選択して返す。
    """
    #  0 - len(items)の範囲でインデックスを1つ選ぶ
    idx = rng.integers(0, len(items))
    return items[idx]


def random_sample(rng: np.random.Generator, items: List[T], size: int) -> List[T]:
    """
    Select and return random items without replacement.
    複数要素を重複なしでランダムに選択して返す。
    """
    # 指定個数分のインデックスを選ぶ（重複無し）
    indices = rng.choice(len(items), size=size, replace=False)
    
    # リストで返す
    result = []
    for idx in indices:
        result.append(items[idx])
    return result
