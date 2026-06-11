from __future__ import annotations

import argparse
import json
import sys

from .config import AnalysisConfig
from .pipeline import run
from .viewer import serve as viewer_serve


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="climbanalyze",
        description="Analyze a climbing video and output pose + movement analysis.",
    )
    parser.add_argument("video", help="Path to the input video file.")
    parser.add_argument(
        "--config", "-c", metavar="CONFIG_JSON",
        help="Path to a JSON config file (overrides defaults).",
    )
    parser.add_argument(
        "--output-dir", "-o", default="outputs", metavar="DIR",
        help="Directory for analysis.json and annotated frames (default: outputs/).",
    )
    parser.add_argument(
        "--serve", action="store_true",
        help="After analysis, open the viewer in a browser (port 8742).",
    )
    parser.add_argument(
        "--port", type=int, default=8742,
        help="Viewer port when --serve is used (default: 8742).",
    )
    args = parser.parse_args()

    # Load config
    if args.config:
        try:
            config = AnalysisConfig.from_file(args.config)
        except json.JSONDecodeError as exc:
            print(f"Config JSON parse error: {exc}", file=sys.stderr)
            sys.exit(5)
        except (TypeError, ValueError) as exc:
            print(f"Config field error: {exc}", file=sys.stderr)
            sys.exit(5)
    else:
        config = AnalysisConfig()

    result = run(args.video, config, args.output_dir)

    status = result.get("status", "ok")
    issue_count = len(result.get("issues", []))
    annotated = len(result.get("artifacts", {}).get("annotatedFrames", []))

    print(f"Status : {status}")
    print(f"Issues : {issue_count}")
    print(f"Frames : {annotated} annotated")
    print(f"Output : {args.output_dir}/analysis.json")

    for w in result.get("warnings", []):
        print(f"Warning: {w}", file=sys.stderr)

    exit_code = 4 if status == "no_person_detected" else 0

    if args.serve and exit_code == 0:
        viewer_serve(args.output_dir, port=args.port)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
