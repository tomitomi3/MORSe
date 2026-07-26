"""
MORSe command-line entry point
MORSeのコマンドラインエントリーポイント
"""

import argparse
from pathlib import Path

from src.optimization_pipeline import OptimizationPipeline
from src.utils.common_messages import folder_not_found_message
from src.utils.visualizer_launcher import (
    DEFAULT_PORT,
    start_bokeh_server,
    wait_for_bokeh_server,
)


def _port_number(value: str) -> int:
    """
    Parse and return a valid TCP port number.
    有効なTCPポート番号を解析して返す。
    """
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc

    return port


def _build_argument_parser() -> argparse.ArgumentParser:
    """
    Build and return the MORSe command-line argument parser.
    MORSeのコマンドライン引数パーサーを構築して返す。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "result_dir",
        nargs="?",
        help="Result folder to visualize",
    )
    parser.add_argument(
        "--port",
        type=_port_number,
        default=DEFAULT_PORT,
        help=f"Bokeh server port (default: {DEFAULT_PORT})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    Run optimization or visualize saved results and return an exit code.
    最適化または保存結果の可視化を実行し、終了コードを返す。
    """
    args = _build_argument_parser().parse_args(argv)

    if args.result_dir is not None:
        result_dir = Path(args.result_dir).resolve()
        if not result_dir.is_dir():
            print(folder_not_found_message(result_dir))
            return 1

        try:
            process = start_bokeh_server(
                str(result_dir),
                auto_open=True,
                port=args.port,
            )
            wait_for_bokeh_server(process)
        except Exception as exc:
            print(str(exc))
            return 1
        return 0

    setting_path = Path("configs/setting.json")
    try:
        pipeline = OptimizationPipeline.from_setting_file(setting_path)
        pipeline.run(visualization_port=args.port)
    except KeyboardInterrupt:
        print("\nOptimization interrupted.")
        return 130
    except Exception as exc:
        print(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
