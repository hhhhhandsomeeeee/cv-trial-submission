"""Compare baseline and candidate crops against sparse independent references."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.report import evaluate_log, load_log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refs", required=True)
    parser.add_argument("--baseline-log", required=True)
    parser.add_argument("--candidate-log", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    refs = json.loads(Path(args.refs).read_text(encoding="utf-8"))
    baseline, baseline_details = evaluate_log(refs, load_log(args.baseline_log))
    candidate, candidate_details = evaluate_log(refs, load_log(args.candidate_log))
    report = {
        "clip": refs.get("clip"),
        "baseline": baseline,
        "candidate": candidate,
        "improvement": {
            "mean_center_error_reduction": round(
                baseline["mean_center_error"] - candidate["mean_center_error"], 6
            ),
            "relative_mean_reduction": round(
                (baseline["mean_center_error"] - candidate["mean_center_error"])
                / max(baseline["mean_center_error"], 1e-12), 6
            ),
        },
        "per_frame": {
            "baseline": baseline_details,
            "candidate": candidate_details,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("clip", "baseline", "candidate", "improvement")}, indent=2))


if __name__ == "__main__":
    main()
