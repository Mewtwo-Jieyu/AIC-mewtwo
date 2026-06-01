# Phase122: Required Evidence Matrix

## Evidence Matrix

| Evidence | Current state | Missing piece | Required before default AIC |
|---|---|---|---|
| Config provenance | SHA256 `15c797eed1dd441e1d6d50fcf7584b2264d248d5ebc1144e1c95e6c4e885d853` | None for this exact config | Recheck on every new timing or integration run |
| Config load | Non-fallback load smoke passed | Default runtime lookup not wired | Prove default process reads the same config without package-dir mutation |
| Loaded weight | DeepseekV2MoE / SharedFusedMoE boundary passed | No default path ownership proof | Record owner and module path in runtime key |
| Token buckets | 128/248/512/1024 measured | Real scheduler MoE token bucket distribution | Capture or derive bucket set from same-source runtime rows |
| Shape coverage | Single hidden/intermediate/topk/topology | No adjacent topology/model coverage | Keep default limited to exact key, or collect every default target key |
| Input distribution | synthetic_random_hidden_states | Real loaded-model activation evidence | Collect real activation state-smoke or document why synthetic is equivalent |
| Repeatability | Diagnostic repeats exist for base shape | More repeatability across token buckets | Define acceptable coefficient of variation before default |
| Holdout | 512/1024 diagnostic holdouts exist | No end-to-end holdout after integration | Validate aggregate prediction error before enabling default |
| Query policy | Experimental exact-match query exists | Default query path may interpolate today | Default MoE WNA16 table must fail on missing token/key |
| Safety flags | diagnostic_only=true, valid_for_default=false, perf_database=false | No valid_for_default=true evidence | Only flip after all gates pass |

## Phase123 Evidence Plan Boundary

| Candidate evidence | Allowed in Phase123 | Not allowed in Phase123 |
|---|---|---|
| Real token bucket inventory | Yes | No latency modeling |
| Real activation state-smoke | Yes | No full default integration |
| Additional diagnostic MoE timing | Only if gate is written first | No broad sweep without exact key plan |
| End-to-end closure design | Yes | No PerfDatabase update |
| Default rollback plan | Yes | No run_static or IterationLatencyCalculator change |

## Review Decision

Current evidence is enough to keep the experimental exact-key table. It is not enough to mark the table valid_for_default.
