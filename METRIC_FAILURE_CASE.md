# Metric failure case — sparse labels miss a transient identity/crop jump

## Failure pattern

Mean crop-centre error on sparse hand-labelled frames can miss short but
visually severe failures between labelled frames. A low mean therefore does
not guarantee a stable or identity-correct output throughout the clip.

## Evidence from the indoor clip

The baseline is scored on 21 independent reference frames, sampled every 50
frames. Its reported result looks strong:

- Mean crop-centre error: **0.019893**
- P95 crop-centre error: **0.032072**
- Target-centre-outside rate: **0%**

However, the full per-frame crop log shows this sequence:

| Frame | Time | `crop_x` | Change from previous frame |
|---:|---:|---:|---:|
| 390 | 13.01 s | 377 | — |
| 391 | 13.04 s | 805 | **+428 px** |
| 392 | 13.08 s | 807 | +2 px |
| 393 | 13.11 s | 384 | **−423 px** |
| 394 | 13.14 s | 385 | +1 px |

At frames 391–392 the crop visibly jumps away from the near-side target and
then jumps back. Neither failure frame was among the labelled frames (the
nearest labels are frames 350 and 400), so the mean error and outside rate do
not reflect it.

See `out/evidence/metric_failure_sparse_sampling.jpg`.

## Why the metric and viewing experience disagree

The crop-centre metric is only evaluated where an independent target box is
available. With sparse periodic labels, it measures spatial accuracy at those
sampled instants but not temporal stability or identity correctness between
them. A viewer still sees the two-frame camera/identity jump even though the
reported mean is low and the sampled outside rate is zero.

## What to measure alongside it

Add a full-frame crop-jump or crop-acceleration metric computed from the crop
log, for example:

```text
normalised_jump[i] = abs(crop_x[i] - crop_x[i-1]) / crop_width
```

Report at least the P95/max normalised jump and the number of frames exceeding
a viewer-calibrated threshold. Also report identity-switch count when denser
identity labels are available.

## Cost

- Crop-jump metric: no additional target labels and negligible computation;
  it uses the existing per-frame crop log.
- Identity-switch metric: requires denser human identity labels around hard
  events, so annotation cost is higher.
- A practical compromise is uniform sparse boxes plus event-triggered review
  of frames with unusually large crop jumps.
