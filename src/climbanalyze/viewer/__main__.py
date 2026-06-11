from __future__ import annotations

import argparse

from .server import serve


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="climbanalyze.viewer",
        description="Serve the ClimbAnalyze viewer for a previous analysis output.",
    )
    parser.add_argument(
        "output_dir", nargs="?", default="outputs",
        metavar="OUTPUT_DIR",
        help="Directory containing analysis.json (default: outputs/).",
    )
    parser.add_argument(
        "--port", "-p", type=int, default=8742,
        help="Local port (default: 8742).",
    )
    parser.add_argument(
        "--no-browser", action="store_true",
        help="Do not open a browser tab automatically.",
    )
    args = parser.parse_args()
    serve(args.output_dir, port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
