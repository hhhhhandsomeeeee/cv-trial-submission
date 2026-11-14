# elevenset CV trial — target-aware tennis auto-reframe

**Track:** Build  
**Environment:** CPU-only, Windows, Python 3.11  
**CPU:** AMD Ryzen 9 6900HX with Radeon Graphics  
**Models:** MoveNet SinglePose Lightning (supplied baseline) and MoveNet
MultiPose Lightning (candidate), loaded from TensorFlow Hub

## 1. Executive summary

The task is to convert landscape tennis footage to a 9:16 vertical video while
staying on the player specified by a frame-0 box in `targets.json`. The supplied
baseline runs SinglePose independently on every frame and has no target
identity. My candidate pipeline uses MultiPose plus motion, appearance, IoU and
body-size association, explicit loss/reacquisition states, and a smoothed crop
planner.

I evaluated both methods against the same 21 independently hand-labelled frames
per clip. The result is mixed and is reported without hiding the failure:

- Indoor clip: candidate mean centre error improves by **5.9%**.
- `9055`: candidate mean centre error is **23.0% worse**, mainly because the
  small player is missed after full-frame downscaling and motion-only prediction
  drifts near the end.
- The target centre remains inside the crop on all 42 labelled frames for both
  methods.
- Sparse centre-error labels miss a real two-frame baseline crop jump at indoor
  frames 391–392. The evidence and a proposed complementary metric are included.

## 2. Trial requirements and where they are satisfied

| Trial requirement | Submission evidence | Status |
|---|---|---|
| Follow the correct player from `targets.json` | `targets.json`, `run_tracker.py`, `src/tracking/` | Implemented |
| Produce one 9:16 output per clip | `out/videos/9055_tracked.mp4`, `out/videos/indoor_tracked.mp4` | Implemented and validated |
| Handle misses/occlusion reasonably | `tracked/predicted/lost/reacquired` states in `src/tracking/tracker.py` | Implemented; small-target weakness documented |
| Report before vs after metrics | `out/metrics/*_comparison.json` and Section 7 | Complete on 21 independent frames per clip |
| Challenge the metric with real evidence | `METRIC_FAILURE_CASE.md` and `out/evidence/metric_failure_sparse_sampling.jpg` | Complete |
| Report CPU cost | Section 9 | Complete |
| One-command output reproduction | `run_all.ps1` | Complete |
| Code clarity and reproducibility | modular `src/`, config, JSONL logs, 7 tests | Complete |
| Do not submit private source clips | `clips/` contains only `.gitkeep`; ZIP audit found zero source MP4s | Complete |

## 3. Target definition and inputs

The tracker receives only a frame-0 target box, not future labels:

| Source filename | Target used in this submission |
|---|---|
| `9055.mp4` | Near-side outdoor player in a dark top and light shorts |
| `iBbVocyi7jM8QqB_VID_20260509144917376_rallies.mp4` | Near-side indoor player in dark clothing |

`targets.json` contains those two initial boxes. The private source videos are
intentionally excluded from this submission. To reproduce, place them in
`clips/` using the exact filenames above.

## 4. Setup and one-command reproduction

Python 3.10–3.12 is supported. On Windows:

```powershell
powershell -ExecutionPolicy Bypass -File setup_windows.ps1
powershell -ExecutionPolicy Bypass -File run_all.ps1
```

`setup_windows.ps1` creates `.venv`, installs the pinned CPU dependencies, and
verifies TensorFlow/TF-Hub imports. The first run needs internet once to download
MoveNet models. `run_all.ps1` processes both private clips and writes the final
videos and per-frame logs to `out/`.

A single clip can also be run directly:

```powershell
.\.venv\Scripts\python.exe run_tracker.py `
  clips\9055.mp4 `
  out\videos\9055_tracked.mp4 `
  --targets targets.json `
  --config configs\default.json `
  --log out\logs\9055_tracked.jsonl
```

The generated MP4 is 1080×1920, preserves source FPS and frame count, and does
not preserve source audio because OpenCV `VideoWriter` writes the video stream
only.

## 5. Method: before and after

### Before — supplied baseline

`baseline.py` uses MoveNet SinglePose on the full frame, averages confident
keypoint x-coordinates and centres a 9:16 crop there. It does not read
`targets.json`, does not associate identities across frames, and does not smooth
the crop. On detection failure it holds the previous centre.

### After — target-aware candidate

1. `MoveNetMultiPoseDetector` detects up to six people and maps letterboxed model
   coordinates back to source pixels.
2. `TargetTracker` initialises appearance and location from `targets.json`.
3. Each candidate receives a weighted association score:
   motion proximity 0.35, HSV appearance 0.35, IoU 0.20 and size consistency 0.10.
4. A motion gate rejects physically implausible jumps. Appearance is updated
   only on high-confidence matches to reduce template drift.
5. Missing detections transition through `predicted` and `lost`; a valid later
   match becomes `reacquired`.
6. `CropPlanner` builds a bounded 9:16 crop. `CropSmoother` applies EMA smoothing
   and a maximum per-frame movement.

Configuration is in `configs/default.json`; frame-by-frame decisions and match
components are saved as JSONL for diagnosis.

## 6. Output and evidence paths

| Path | What it contains | How to use it |
|---|---|---|
| `out/videos/9055_tracked.mp4` | Final 9:16 candidate output for outdoor clip | Primary deliverable; visually follows the near-side outdoor player |
| `out/videos/indoor_tracked.mp4` | Final 9:16 candidate output for indoor clip | Primary deliverable; follows the near-side dark-clothed player |
| `out/logs/9055_baseline.json` | Baseline crop x/width for every outdoor frame | “Before” positions used by evaluation |
| `out/logs/9055_tracked.jsonl` | Candidate crop, target state, confidence and detections per frame | “After” positions and failure diagnosis |
| `out/logs/indoor_baseline.json` | Baseline per-frame crop log for indoor clip | “Before” evaluation input |
| `out/logs/indoor_tracked.jsonl` | Candidate per-frame diagnostic log for indoor clip | “After” evaluation input |
| `refs/9055.refs.json` | 21 independent manual target boxes, frames 0–400 every 20 frames | Ground truth for outdoor comparison |
| `refs/indoor.refs.json` | 21 independent manual target boxes, frames 0–1000 every 50 frames | Ground truth for indoor comparison |
| `out/metrics/9055_comparison.json` | Aggregate and per-reference-frame before/after metrics | Exact outdoor numbers; worst candidate frame is 400 |
| `out/metrics/indoor_comparison.json` | Aggregate and per-reference-frame before/after metrics | Exact indoor numbers |
| `out/evidence/metric_failure_sparse_sampling.jpg` | Frames 390–394 showing the missed crop jump | Main metric-failure evidence |
| `out/evidence/9055_comparison.jpg` | Sampled baseline/candidate visual comparison | Quick visual review |
| `out/evidence/indoor_comparison.jpg` | Sampled baseline/candidate visual comparison | Quick visual review |
| `METRIC_FAILURE_CASE.md` | Detailed failure-pattern argument and alternative metric | Written evidence for the metric critique |

Baseline videos and large debug videos were not included because they are not
required to inspect the metrics; their crop logs and the final candidate videos
are included. Source clips are never included.

## 7. Independent evaluation

The manual boxes were drawn on original frames without using tracker boxes as
ground truth. This avoids circular evaluation. The supplied metric is:

```text
crop-centre error = abs(crop centre x - manual target centre x) / crop width
```

`0` is perfectly centred; `0.5` places the target centre on the crop edge;
greater than `0.5` places it outside. `evaluate.py` reports mean, P95, maximum
and target-centre-outside rate. Both methods use exactly the same references.

Reproduce the stored metrics:

```powershell
.\.venv\Scripts\python.exe evaluate.py `
  --refs refs\9055.refs.json `
  --baseline-log out\logs\9055_baseline.json `
  --candidate-log out\logs\9055_tracked.jsonl `
  --output out\metrics\9055_comparison.json

.\.venv\Scripts\python.exe evaluate.py `
  --refs refs\indoor.refs.json `
  --baseline-log out\logs\indoor_baseline.json `
  --candidate-log out\logs\indoor_tracked.jsonl `
  --output out\metrics\indoor_comparison.json
```

| Clip | Method | Mean | P95 | Max | Centre outside |
|---|---|---:|---:|---:|---:|
| `9055` | Baseline | 0.0225 | 0.0531 | 0.0617 | 0% |
| `9055` | Tracked | 0.0277 | 0.0716 | 0.2333 | 0% |
| Indoor rallies | Baseline | 0.0199 | 0.0321 | 0.0847 | 0% |
| Indoor rallies | Tracked | 0.0187 | 0.0567 | 0.0732 | 0% |

Interpretation:

- `9055`: mean error is 23.0% worse after tracking. The maximum error is caused
  by frame 400, where state is `lost`, detections are 0, confidence is 0 and
  error is 0.2333. The target remains inside the crop, but prediction has drifted.
- Indoor: mean error improves by 5.9% and maximum error improves, but P95 is
  worse. The methods are broadly comparable rather than a decisive candidate win.
- These clips do not strongly exercise persistent identity switching. The main
  candidate weakness observed is small-player detector recall, not re-ID.

## 8. Metric failure pattern

The indoor baseline looks excellent under the 21-frame evaluation: mean error
**0.0199**, P95 **0.0321**, sampled centre-outside rate **0%**. However, the full
crop log and real output show:

| Frame | Time | `crop_x` | Change |
|---:|---:|---:|---:|
| 390 | 13.01 s | 377 | — |
| 391 | 13.04 s | 805 | **+428 px** |
| 392 | 13.08 s | 807 | +2 px |
| 393 | 13.11 s | 384 | **−423 px** |
| 394 | 13.14 s | 385 | +1 px |

The crop visibly leaves the near-side player for two frames and jumps back. The
uniform manual labels nearest this event are frames 350 and 400, so crop-centre
error never evaluates the failure. This is a concrete disagreement between a
good summary score and viewer experience.

I would add a full-frame normalised jump metric:

```text
normalised_jump[i] = abs(crop_x[i] - crop_x[i-1]) / crop_width
```

P95/max jump and a threshold exceedance count require no new target labels and
negligible compute. Identity-switch count is stronger but requires denser human
identity labels. A practical labelling strategy is sparse uniform boxes plus
event-triggered review of frames with large crop jumps.

## 9. CPU cost

CPU-only wall-clock timing on **AMD Ryzen 9 6900HX with Radeon Graphics**:

| Clip | Duration | Baseline | Candidate | Candidate cost per video minute |
|---|---:|---:|---:|---:|
| `9055` | 13.74 s | 66.5 s | 149.3 s | 651.9 s/min |
| Indoor rallies | 33.82 s | 167.5 s | 392.0 s | 695.3 s/min |

First model download is excluded. Candidate timing includes optional debug-video
encoding, so it is conservative. Per-frame full-image MultiPose is the dominant
cost and is not production-ready.

## 10. Failures, decisions and next steps

Known limitations:

- On `9055`, MultiPose finds a person on 268/413 frames; tracker states are 233
  tracked, 125 predicted, 35 reacquired and 20 lost.
- Indoor detection is much stronger: person detected on 1009/1014 frames; 1005
  tracked, 5 predicted, 4 reacquired and 0 lost.
- HSV appearance is inexpensive but weak when people wear similar colours.
- Motion-only extrapolation can drift during prolonged detector misses.
- Centre error does not measure full-body visibility, identity correctness,
  vertical composition, latency or brief failures between labels.
- OpenCV output does not preserve audio.

With another day I would:

1. Run MultiPose every 3–5 frames and use optical flow/CSRT between detections.
2. Re-detect in an enlarged ROI around the predicted player for small targets.
3. Freeze rather than extrapolate after prolonged misses, then reacquire with
   stricter appearance and motion gating.
4. Add full-frame jump/acceleration, full-body coverage and event-triggered review.
5. Re-encode final output with H.264 and copy source audio when required.

## 11. Tests, privacy and licences

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Result: **7 passed**. Tests cover association gating, crop geometry/smoothing,
metric calculations and an end-to-end video/log smoke test.

Private assessment clips are excluded by `.gitignore` and from the submission
ZIP. `clips/` contains only `.gitkeep`. MoveNet is loaded from TensorFlow Hub;
TensorFlow and upstream MoveNet assets are Apache-2.0 licensed.
