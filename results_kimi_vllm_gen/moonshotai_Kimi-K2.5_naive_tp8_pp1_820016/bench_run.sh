#!/bin/bash
set -euo pipefail

concurrency_array=(1 2 8 16 32 64 121 128 134)

ARTIFACT_DIR="${BENCH_ARTIFACT_DIR:-/tmp/bench_artifacts}"

for concurrency in "${concurrency_array[@]}"; do
  echo "Run concurrency: $concurrency"
  aiperf profile \
    --artifact-dir "${ARTIFACT_DIR}/concurrency_${concurrency}" \
    -m  \
    --endpoint-type  \
    -u http://: \
    --tokenizer  \
    --isl  --isl-stddev  \
    --osl  --osl-stddev  \
    --extra-inputs ignore_eos:true \
    --extra-inputs "{\"nvext\":{\"ignore_eos\":true}}" \
    --concurrency ${concurrency} \
    --num-requests $(($concurrency*50)) \
    --warmup-request-count $(($concurrency*2)) \
    --random-seed 100 \
    --ui  \
    --streaming
done