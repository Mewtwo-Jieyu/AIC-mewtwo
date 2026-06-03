# Phase126: Drain Boundary Design

## Decision

| Item | Result |
|---|---|
| Scope | Define marker close point |
| Full capture | No-Go |
| Target patch dry-run | PASS |
| Benchmark metric use | No-Go |
| Default AIC | No-Go |
| PerfDatabase | No-Go |

## Close Point

| Step | Marker state |
|---|---|
| Before service readiness | Disabled |
| After service readiness | Enabled |
| During traffic generation | Enabled |
| After benchmark client returns | Still enabled |
| Drain wait | Enabled until stable row count |
| After drain stable | Disabled |
| Timeout | Fail, then cleanup |

The runner must not remove `ENABLE_FILE` immediately after the benchmark client returns. It must first wait for marker rows to stop increasing.

## Drain Contract

| Parameter | Default |
|---|---:|
| `DRAIN_POLL_SECONDS` | 5 |
| `DRAIN_STABLE_POLLS` | 3 |
| `DRAIN_TIMEOUT_SECONDS` | 900 |

Stable means marker row count does not increase for `DRAIN_STABLE_POLLS` consecutive polls.

## Drain Log

| File | Content |
|---|---|
| `drain_boundary.log` | `poll_index,marker_rows,stable_polls,close_reason` |

Allowed close reasons:

| close_reason | Meaning |
|---|---|
| `pending` | More polling needed |
| `stable` | Accepted marker close point |
| `timeout` | Failure |
| `count_decreased` | Failure |

## Failure Rules

| Condition | Action |
|---|---|
| Timeout before stable | Fail current run |
| Marker row count decreases | Fail current run |
| No marker rows after drain | Fail parser step |
| Cleanup fails | Do not accept artifact |

Timeout output must not be parsed as an accepted artifact.

## Evidence Boundary

| Boundary | Rule |
|---|---|
| Marker schema | Unchanged |
| `timing=false` | Kept only as parser gate |
| Marker time fields | Not added |
| Benchmark result | Traffic health only |
| Default AIC | No-Go |
| PerfDatabase | No-Go |
| Interpolation / extrapolation | No-Go |

## Next Gate

Full capture remains No-Go until a separate capture2 plan is accepted. Target `patch-dry-run` has passed for the drain-updated runner.
