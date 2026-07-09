# Phase451K router harness

结论: 真实 `get_core_engine_for_request` 路由块本身仍是交替,不是成簇。63/1 只能在 stats task 反复用旧 coordinator snapshot 覆盖本地递增时复现。因此本轮不改 sim 路由;下一步应测 stats 覆盖与 add_request 的交错时序。

| Section | Metric | Value | Target | Status | Note |
|---|---|---:|---|---|---|
| k1_boundary | serve_api_server_count | data_parallel_size(2) | api_server_count=2 | pass | serve.log confirms internal DP default when present |
| k1_boundary | api_server_processes_seen | 2 | 2 | pass | two ApiServer processes share the internal DP path |
| k1_boundary | client_class_source | DPLBAsyncMPClient | core_client.py:107-129 | pass | internal DP without external LB selects DPLBAsyncMPClient |
| k1_boundary | client_config_source | client_count/client_index | utils.py:202-208 | pass | APIServerProcessManager passes per-process client_index |
| k2_harness | direct_import | blocked | import vllm.v1.engine.core_client | blocked | local import requires torch; harness uses source-faithful extracted route block |
| k2_harness | persistent_increment_routes | [32, 32] | alternating | alternating | final_counts=[[64, 1], [64, 0]] |
| k2_harness | two_api_persistent_routes | [64, 64] | alternating | alternating | two API clients with independent start indices still balance if increments persist |
| k2_harness | stats_overwrite_routes | [0, 64] | clustered | clustered | clump appears only when stats task overwrites local increments with an old snapshot between route calls |
| k2_observation | debug_first_asymmetric_counts | [[0,1],[63,1]] | [[0,1],[63,1]] | observed | Phase451-I real run |
| k2_verdict | route_loop_verdict | alternates | direct route loop behavior | pass | get_core_engine_for_request itself does not produce 63/1 |
| k2_verdict | clump_mechanism_candidate | stats_overwrite_interleaving | explain code+DEBUG contradiction | candidate | source line core_client.py:1271 assigns self.lb_engines=sliced_counts and can discard local increments |
| k5_decision | sim_route_alignment | not_applied | harness-proven actual behavior | blocked | stats-overwrite interleaving cadence is not yet measured; changing route policy would be a guessed model |
| k5_decision | cascade_gates | not_run |  | blocked | no runtime change was made, so H/450-A/fingerprint/preemption/--ab cascade is not meaningful |
| k5_decision | phase443b_452 | unchanged |  | blocked | 443-B and Phase452 decisions remain pending until stats-overwrite or upstream distribution is measured |

## Source boundary

- `serve.py:83-103`: internal DP defaults `api_server_count` to `data_parallel_size`.
- `core_client.py:107-129`: internal DP selects `DPLBAsyncMPClient`.
- `core_client.py:1337-1360`: route score is `waiting*4+running`, then local waiting is incremented by `client_count`.
- `core_client.py:1268-1275`: stats task assigns `self.lb_engines = sliced_counts`, which is the only source-local way to discard local increments.

## Decision

Do not change `DPAdmissionRouter` to no-increment clumping. That would contradict the route block. The next source-grounded probe is request-level instrumentation around `add_request_async`: log the route choice, pre/post `lb_engines`, and whether a stats update overwrote the list between adjacent adds.
