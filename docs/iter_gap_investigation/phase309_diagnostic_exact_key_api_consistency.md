# Phase309 Diagnostic Exact-Key API Consistency

This audit checks that the two diagnostic exact-key lookup APIs expose fixed evidence only.

| Item | Value |
|---|---|
| actual scheduled token family candidates | 4 / 4 |
| deeper trace topology family candidates | 2 / 2 |
| unknown key behavior | KeyError |
| Default AIC | No-Go |
| diagnostic_only | true |
| valid_for_default | false |
| perf_database | false |

Both APIs load only their committed diagnostic sources: Phase258 for actual scheduled tokens and Phase303 for deeper trace topology. Unknown topology, shape, or budget keys fail instead of interpolation or extrapolation.

Verdict: diagnostic exact-key lookup consistency is established for these fixed evidence rows only. This is not ready for Default AIC and does not write PerfDatabase data.
