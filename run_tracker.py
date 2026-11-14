"""Run target-aware tennis auto-reframing on one clip."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import load_config
from src.pipeline import TrackingPipeline
from src.schemas import Box


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", help="input video")
    parser.add_argument("dst", help="9:16 output MP4")
    parser.add_argument("--targets", default="targets.json")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--log", required=True, help="per-frame JSONL output")
    parser.add_argument("--debug", help="optional annotated source-resolution MP4")
    return parser.parse_args()


def main():
    args = parse_args()
    # Keep CLI help and non-model tooling usable before the heavyweight
    # TensorFlow dependencies are installed.
    from src.detection.movenet import MoveNetMultiPoseDetector

    config = load_config(args.config)
    targets = json.loads(Path(args.targets).read_text(encoding="utf-8"))
    clip_id = Path(args.src).stem
    if clip_id not in targets:
        raise SystemExit(f"no target for {clip_id!r} in {args.targets}; run make_targets.py first")
    raw = targets[clip_id]
    initial_box = Box(float(raw["x"]), float(raw["y"]), float(raw["w"]), float(raw["h"]))

    detector = MoveNetMultiPoseDetector(
        config.detector.model_url,
        config.detector.input_size,
        config.detector.confidence_threshold,
    )
    summary = TrackingPipeline(detector, config).run(
        args.src, args.dst, args.log, initial_box, args.debug
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
