from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.metrics import crop_center_error, summarise, target_center_outside_crop
from src.schemas import Box, CropWindow


def load_log(path: str | Path) -> dict[int, dict]:
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    if text.startswith("["):
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return {int(row["frame_index"]): row for row in rows}


def row_crop(row: dict) -> CropWindow:
    raw = row.get("crop") or {
        "x": row["crop_x"],
        "y": row.get("crop_y", 0),
        "w": row["crop_w"],
        "h": row.get("crop_h", 0),
    }
    return CropWindow(int(raw["x"]), int(raw.get("y", 0)), int(raw["w"]), int(raw.get("h", 0)))


def evaluate_log(refs: dict, rows: dict[int, dict]) -> tuple[dict, list[dict]]:
    errors, outside, xs, widths, details = [], [], [], [], []
    for annotation in refs["annotations"]:
        if not annotation.get("visible") or annotation.get("box") is None:
            continue
        frame_index = int(annotation["frame_index"])
        row = rows.get(frame_index)
        if row is None:
            continue
        crop = row_crop(row)
        target = Box(**{key: float(value) for key, value in annotation["box"].items()})
        error = crop_center_error(crop, target)
        is_outside = target_center_outside_crop(crop, target)
        errors.append(error)
        outside.append(is_outside)
        xs.append(crop.x)
        widths.append(crop.w)
        details.append({
            "frame_index": frame_index,
            "error": round(error, 6),
            "target_center_outside": is_outside,
            "crop_x": crop.x,
            "crop_w": crop.w,
        })
    return summarise(errors, outside, xs, widths), details
