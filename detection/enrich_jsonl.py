"""Enrich canonical FlightState JSONL with results from the default detector.

Run ``python -m detection.enrich_jsonl input.jsonl output.jsonl``. The input
must be chronological per aircraft. Each output record preserves the original
flight state and adds a JSON-ready ``detection`` object for the frontend.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TextIO

from detection.engine import build_default_engine


def enrich_jsonl(input_file: TextIO, output_file: TextIO, *, source: str = "<stdin>") -> int:
    """Write each input flight state with its detection result attached."""
    engine = build_default_engine()
    count = 0
    for line_number, line in enumerate(input_file, start=1):
        if not line.strip():
            continue
        try:
            state = json.loads(line)
            if not isinstance(state, Mapping):
                raise ValueError("flight state line must be a JSON object")
            detection = engine.update_mapping(state)
            output_file.write(json.dumps({**state, "detection": detection}, separators=(",", ":")) + "\n")
            count += 1
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError(f"{source}:{line_number}: {type(error).__name__}: {error}") from error
    return count


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="chronological FlightState JSONL input")
    parser.add_argument("output", type=Path, help="enriched JSONL output consumed by the frontend")
    args = parser.parse_args(argv)

    try:
        if args.input.resolve() == args.output.resolve():
            raise ValueError("input and output paths must be different")
        with args.input.open("r", encoding="utf-8") as input_file:
            with args.output.open("w", encoding="utf-8", newline="\n") as output_file:
                count = enrich_jsonl(input_file, output_file, source=str(args.input))
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    print(f"Enriched {count} flight state(s) -> {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
