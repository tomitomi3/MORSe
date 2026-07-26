"""
Bokeh visualization server launcher
Bokeh可視化サーバー起動ユーティリティ
"""

import queue
import socket
import subprocess
import sys
import threading
from typing import TextIO

from tornado.netutil import bind_sockets

from src.utils.common_messages import (
    operation_failed_message,
    port_in_use_message,
)

DEFAULT_PORT = 5006
BOKEH_STARTUP_TIMEOUT_SECONDS = 10.0
BOKEH_READY_LOG = "Starting Bokeh server with process id:"
BOKEH_SUPPRESSED_LOG_MESSAGES = (
    "Starting Bokeh server version",
    "User authentication hooks NOT provided",
    "Bokeh app running at:",
    BOKEH_READY_LOG,
    "WebSocket connection opened",
    "WebSocket connection closed",
    "ServerConnection created",
)


def _can_bind_bokeh_port(port: int) -> bool:
    """
    Return whether Tornado can bind the sockets used by Bokeh.
    Bokehが使用するソケットをTornadoで確保できるか返す。
    """
    try:
        bound_sockets = bind_sockets(port, address=None)
    except OSError:
        return False

    for bound_socket in bound_sockets:
        bound_socket.close()
    return True


def reserve_bokeh_port(port: int) -> list[socket.socket]:
    """
    Reserve and return the sockets Bokeh uses for the specified port.
    指定ポートでBokehが使用するソケットを確保して返す。
    """
    try:
        return bind_sockets(port, address=None)
    except OSError as exc:
        raise RuntimeError(port_in_use_message(port)) from exc


def release_bokeh_port(
    reserved_sockets: list[socket.socket] | None,
) -> None:
    """
    Release sockets held as a Bokeh port reservation.
    Bokehポートの一時確保に使用したソケットを解放する。
    """
    if reserved_sockets is None:
        return

    for reserved_socket in reserved_sockets:
        reserved_socket.close()


def ensure_port_available(port: int) -> None:
    """
    Raise an error when the specified local TCP port is unavailable.
    指定したローカルTCPポートが使用できない場合にエラーを送出する。
    """
    reserved_sockets = reserve_bokeh_port(port)
    release_bokeh_port(reserved_sockets)


def _is_suppressed_bokeh_log(line: str) -> bool:
    """
    Return whether a routine Bokeh lifecycle log should stay hidden.
    Bokehの定型ライフサイクルログを端末へ表示しない場合に真を返す。
    """
    for message in BOKEH_SUPPRESSED_LOG_MESSAGES:
        if message in line:
            return True
    return False


def _forward_bokeh_logs(
    stderr: TextIO,
    startup_result: queue.Queue[bool],
) -> None:
    """
    Forward Bokeh logs and report whether the child reached its ready state.
    Bokehログを転送し、子プロセスが起動完了したか通知する。
    """
    startup_reported = False
    for line in stderr:
        if not startup_reported and BOKEH_READY_LOG in line:
            startup_reported = True
            startup_result.put(True)
        if _is_suppressed_bokeh_log(line):
            continue
        sys.stderr.write(line)
        sys.stderr.flush()

    if not startup_reported:
        startup_result.put(False)


def _wait_for_bokeh_startup(
    process: subprocess.Popen,
    port: int,
) -> None:
    """
    Wait until this Bokeh child reports success or exits during startup.
    このBokeh子プロセスが起動成功を通知するか起動中に終了するまで待つ。
    """
    startup_result: queue.Queue[bool] = queue.Queue()
    stderr_thread = threading.Thread(
        target=_forward_bokeh_logs,
        args=(process.stderr, startup_result),
        daemon=True,
        name=f"bokeh-stderr-{process.pid}",
    )
    process.morse_stderr_thread = stderr_thread
    stderr_thread.start()

    try:
        startup_succeeded = startup_result.get(timeout=BOKEH_STARTUP_TIMEOUT_SECONDS)
    except queue.Empty as exc:
        stop_bokeh_server(process)
        detail = f"timed out on port {port}"
        raise RuntimeError(
            operation_failed_message("Bokeh server startup", detail)
        ) from exc

    if startup_succeeded:
        return

    return_code = process.wait()
    stderr_thread.join()
    if not _can_bind_bokeh_port(port):
        raise RuntimeError(port_in_use_message(port))

    detail = f"process exited with code {return_code}"
    raise RuntimeError(operation_failed_message("Bokeh server startup", detail))


def start_bokeh_server(
    result_dir: str,
    auto_open: bool = False,
    port: int = DEFAULT_PORT,
) -> subprocess.Popen:
    """
    Start a Bokeh server on the specified port and return its process handle.
    指定ポートでBokehサーバーを起動し、プロセスハンドルを返す。
    """
    # result_dirは結果パス、auto_openはブラウザ自動起動、portはBokehの待受ポートとする。
    ensure_port_available(port)

    # 実行形式はpython -m bokeh serve src/visualization/visualizer_server.pyとする
    cmd = [
        sys.executable,
        "-m",
        "bokeh",
        "serve",
        "src/visualization/visualizer_server.py",
        "--port",
        str(port),
        "--log-level",
        "info",
    ]
    if auto_open:
        cmd.append("--show")
    cmd.extend(["--args", result_dir])

    # Bokehサーバーをサブプロセスとして起動
    process = subprocess.Popen(
        cmd,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        bufsize=1,
    )
    process.morse_port = port
    process.morse_url = f"http://localhost:{port}/visualizer_server"
    try:
        _wait_for_bokeh_startup(process, port)
    except BaseException:
        stop_bokeh_server(process)
        raise

    return process


def stop_bokeh_server(process: subprocess.Popen | None) -> None:
    """
    Terminate and reap a running Bokeh server process.
    実行中のBokehサーバープロセスを終了して回収する。
    """
    if process is None:
        return

    if process.poll() is None:
        process.terminate()
        # terminate後に終了を待ち、Bokeh子プロセスを確実に回収する
        process.wait()

    stderr_thread = getattr(process, "morse_stderr_thread", None)
    if stderr_thread is not None:
        stderr_thread.join()


def wait_for_bokeh_server(process: subprocess.Popen) -> None:
    """
    Wait for a Bokeh server and stop it when interrupted.
    Bokehサーバーを待機し、割り込み時に終了する。
    """
    try:
        process.wait()
    except KeyboardInterrupt:
        pass
    finally:
        stop_bokeh_server(process)
