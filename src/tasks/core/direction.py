"""
Enumeration of objective optimization directions
目的関数の最適化方向を表す列挙型
"""

from enum import Enum


class Direction(str, Enum):
    """
    Represent objective directions as string enumeration values.
    目的関数の最適化方向を文字列列挙値として表す。
    """

    # strを継承するため、JSONでは"minimize"または"maximize"として出力される。
    # Direction.MINIMIZE.valueと文字列との比較にはどちらも"minimize"を使用する。
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"
