# Phase397w required buckets

| Item | Result |
|---|---|
| Rows analyzed | 8 |
| Rows with missing buckets | 8 |
| bucket 128 required by current runtime | true |
| planned scenario without 0.19-real measurement | K2.5-tp4ep8dp2-32k3k-bt65536 |
| GPU collection | GPU collection is not started in this phase397w enumeration step |
| Default AIC | No-Go |

## Missing buckets

| Scenario | Boundary | Missing count | Missing buckets |
|---|---|---:|---|
| K2.5-tp4ep8dp2-8k2k | ep8_comm_dispatch_combine | 17 | 2/3/4/5/6/7/8/9/10/11/12/13/14/31/504/992/2000 |
| K2.5-tp4ep8dp2-8k2k | fusedmoe_runner_compute | 63 | 4/6/8/10/12/14/18/20/22/24/26/28/34/36/38/40/42/44/46/48/50/52/54/56/58/60/62/64/66/68/70/72/74/76/78/80/82/84/86/88/90/92/94/96/98/100/102/104/106/108/110/112/114/116/118/120/122/124/126/128/1008/1984/4000 |
| K2.5-tp4ep8dp2-32k3k | ep8_comm_dispatch_combine | 17 | 2/3/4/5/6/7/8/9/10/11/12/13/14/31/504/992/8000 |
| K2.5-tp4ep8dp2-32k3k | fusedmoe_runner_compute | 63 | 4/6/8/10/12/14/18/20/22/24/26/28/34/36/38/40/42/44/46/48/50/52/54/56/58/60/62/64/66/68/70/72/74/76/78/80/82/84/86/88/90/92/94/96/98/100/102/104/106/108/110/112/114/116/118/120/122/124/126/128/1008/1984/16000 |
| K2.5-tp4ep8dp2-8k2k-bt65536 | ep8_comm_dispatch_combine | 15 | 2/4/6/8/10/12/13/14/13368/14000/15273/16000/16014/16384/18000 |
| K2.5-tp4ep8dp2-8k2k-bt65536 | fusedmoe_runner_compute | 16 | 48/64/80/96/98/110/112/114/128/26736/28000/30546/32000/32028/32768/36000 |
| K2.5-tp4ep8dp2-32k3k-bt65536 | ep8_comm_dispatch_combine | 20 | 2/3/4/5/6/7/8/9/10/11/12/13/14/4346/8000/12328/16000/16015/16384/24000 |
| K2.5-tp4ep8dp2-32k3k-bt65536 | fusedmoe_runner_compute | 47 | 4/8/12/20/24/28/36/40/44/48/52/56/60/64/68/72/76/80/84/88/90/92/94/96/98/100/102/104/106/108/110/112/114/116/118/120/122/124/126/128/8692/16000/24656/32000/32030/32768/48000 |

## Verdict

The current tp4dp2ep8 simulator emits a broad bucket family, not only bucket 2000. A naive GPU bucket-widen pass would have to materialize many scheduler-shaped buckets, including bucket 128, before exact lookup can pass.

This is diagnostic evidence only. It does not write `vllm_module_perf.txt`, does not relax exact lookup, and does not enable Default AIC.
