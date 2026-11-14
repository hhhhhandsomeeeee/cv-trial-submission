"""Create sparse, independent target boxes for honest metric evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    parser.add_argument("--step", type=int, default=30)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    annotations, frame_index = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % args.step == 0:
            window = f"frame {frame_index}: box target; C/ESC skips invisible"
            x, y, w, h = cv2.selectROI(window, frame, showCrosshair=True)
            cv2.destroyAllWindows()
            annotations.append({
                "frame_index": frame_index,
                "visible": bool(w and h),
                "box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)} if w and h else None,
            })
        frame_index += 1
    cap.release()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"clip": Path(args.video).stem, "annotations": annotations}, indent=2) + "\n",
                      encoding="utf-8")
    print(f"wrote {len(annotations)} references to {output}")


if __name__ == "__main__":
    main()
