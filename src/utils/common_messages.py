"""
Common runtime message builders
共通の実行時メッセージ生成関数
"""


def file_not_found_message(path: object) -> str:
    """
    Build a message for a missing file.
    存在しないファイルのメッセージを構築する。
    """
    return f"File not found: {path}"


def folder_not_found_message(path: object) -> str:
    """
    Build a message for a missing folder.
    存在しないフォルダのメッセージを構築する。
    """
    return f"Folder not found: {path}"


def files_not_found_message(file_type: str, folder: object) -> str:
    """
    Build a message when a folder has no files of a requested type.
    指定形式のファイルがフォルダにない場合のメッセージを構築する。
    """
    return f"No {file_type} files found in: {folder}"


def required_value_message(name: str) -> str:
    """
    Build a message for a required value.
    必須値がない場合のメッセージを構築する。
    """
    return f"{name} is required"


def unsupported_value_message(
    name: str,
    value: object,
    supported_values: object = None,
) -> str:
    """
    Build a message for an unsupported value.
    未対応値のメッセージを構築する。
    """
    message = f"Unsupported {name}: {value}"
    if supported_values is None:
        return message
    return f"{message}. Supported values: {supported_values}"


def operation_failed_message(operation: str, detail: object = None) -> str:
    """
    Build a message for a failed operation.
    処理失敗のメッセージを構築する。
    """
    message = f"{operation} failed"
    if detail is None:
        return message
    return f"{message}: {detail}"


def port_in_use_message(port: int) -> str:
    """
    Build a message for a TCP port that is already in use.
    使用中のTCPポートに対するメッセージを構築する。
    """
    return (
        f"Port {port} is already in use. "
        "Use the --port option to specify a different port."
    )
