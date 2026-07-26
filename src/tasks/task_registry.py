"""
Dynamic task discovery class
タスクの動的発見クラス
"""
import importlib
from pathlib import Path
from typing import Dict, Type, Optional

from src.tasks.core.base_task import BaseTask
from src.utils.common_messages import unsupported_value_message


# タスク発見から除外するフォルダ
EXCLUDE_FOLDERS = {'core', '__pycache__'}


def discover_tasks() -> Dict[str, Type[BaseTask]]:
    """
    Discover BaseTask subclasses and return a mapping from task folder names to classes.
    BaseTaskサブクラスを動的に検出し、タスクフォルダ名からクラスへのマッピングを返す。
    """
    # 各タスクフォルダにはBaseTaskサブクラスを公開する__init__.pyが必要となる
    tasks: Dict[str, Type[BaseTask]] = {}
    tasks_dir = Path(__file__).parent

    for folder in tasks_dir.iterdir():
        if not folder.is_dir():
            continue
        if folder.name.startswith('_'):
            continue
        if folder.name in EXCLUDE_FOLDERS:
            continue

        init_file = folder / '__init__.py'
        if not init_file.exists():
            continue

        module_name = f"src.tasks.{folder.name}"
        module = importlib.import_module(module_name)

        # タスクを検索（BaseTask を継承する最初のクラス）
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and
                issubclass(attr, BaseTask) and
                attr is not BaseTask):
                tasks[folder.name] = attr
                break

    return tasks


def get_task_class(task_name: str) -> Optional[Type[BaseTask]]:
    """
    Return the BaseTask subclass for a task folder name, or None when unavailable.
    タスクフォルダ名に対応するBaseTaskサブクラスを返し、見つからない場合はNoneを返す。
    """
    tasks = discover_tasks()
    return tasks.get(task_name)


def create_task(task_name: str, config: dict):
    """
    Create and return a task instance, raising ValueError for an unknown task name.
    タスク名に対応するインスタンスを生成して返し、不明なタスク名ではValueErrorを送出する。
    """
    task_class = get_task_class(task_name)

    if task_class is None:
        raise ValueError(unsupported_value_message("task", task_name))

    return task_class(config)

