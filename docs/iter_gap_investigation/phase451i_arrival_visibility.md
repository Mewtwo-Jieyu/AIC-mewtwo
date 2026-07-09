# Phase451-I arrival visibility

结论: EngineCore 侧判别为 `engine_burst`, 下一步路线 `debug_observation_required`。本报告只做判别,不改 runtime、PerfDB 或 validate gate。

- Default AIC remains `No-Go`.

## Summary

| section | metric | value | target | status | note |
|---|---|---:|---|---|---|
| i1_discriminator | metrics_sample_count | 751 |  |  |  |
| i1_discriminator | max_running_per_engine | 56 |  |  |  |
| i1_discriminator | max_waiting_per_engine_all_run | 67 |  |  |  |
| i1_discriminator | max_kv_usage | 1 |  |  |  |
| i1_discriminator | running_p50 | 47 |  |  |  |
| i1_discriminator | waiting_p50 | 15 |  |  |  |
| i1_discriminator | active_window_samples | 59 |  |  |  |
| i1_discriminator | active_start_s | 0 |  |  |  |
| i1_discriminator | active_window_max_waiting_per_engine | 67 | >=50 burst; <=5 drizzle |  |  |
| i1_discriminator | active_window_max_waiting_global | 122 | >=100 global burst |  |  |
| i1_discriminator | active_window_max_running_per_engine | 53 |  |  |  |
| i1_discriminator | engine_visibility | engine_burst | burst or drizzle | pass | EngineCore-visible waiting queue jumps to burst scale |
| i1_discriminator | next_route | debug_observation_required |  | open |  |
| i1_added_lines | added_request_line_count | 0 | >0 gives per-request engine-visible arrival | blocked | no Added request lines at INFO-level serve.log |
| i3_debug_run | debug_short_run_needed | True |  | open | metrics reject drizzle; need DEBUG/victim/waiting observation before any semantic fix |
| i2_boundary | engine_visible_replay | not_applied |  | blocked | drizzle route not supported by existing metrics |
| i4_accept | runtime_patch | not_applied |  | blocked | Phase451-I Step1 is report-only until route is proven |
| i4_accept | default_aic | No-Go |  | blocked |  |

## Boundary

- Waiting/running gauges are sampled at metrics poll cadence; they can prove burst-scale visibility, but not victim identity.
- INFO serve.log has no Added request lines in the current raw, so per-request EngineCore visible arrival is unavailable offline.
- If the route is `debug_observation_required`, the next valid step is a short DEBUG ramp run, not a scheduler fix.
