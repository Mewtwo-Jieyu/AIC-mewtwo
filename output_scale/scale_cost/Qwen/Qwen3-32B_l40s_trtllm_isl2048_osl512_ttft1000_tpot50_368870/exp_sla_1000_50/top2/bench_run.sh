concurrency_array=(1 2 8 16 32 64 128 307 324 340)

for concurrency in "${concurrency_array[@]}"; do
  echo "Run concurrency: $concurrency"
  aiperf profile \
    -m Qwen/Qwen3-32B \
    --endpoint-type chat \
    -u http://0.0.0.0:8000 \
    --tokenizer Qwen/Qwen3-32B \
    --isl 2048 --isl-stddev 0 \
    --osl 512 --osl-stddev 0 \
    --extra-inputs ignore_eos:true \
    --extra-inputs "{\"nvext\":{\"ignore_eos\":true}}" \
    --concurrency ${concurrency} \
    --num-requests $(($concurrency*50)) \
    --warmup-request-count $(($concurrency*2)) \
    --random-seed 100 \
    --ui simple \
    --streaming
done