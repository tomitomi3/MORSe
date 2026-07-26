"""
Export of the multi-objective optimization target task definitions
多目的最適化の対象タスク定義の公開
"""
from src.tasks.core.base_task import BaseTask
from src.tasks.task_registry import discover_tasks, get_task_class, create_task

__all__ = ['BaseTask', 'discover_tasks', 'get_task_class', 'create_task']

