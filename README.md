
# Novita.AI Fork

This fork of SGLang implements comprehensive optimizations for GLM-MOE models, including shared expert fusion, QK norm fusion, asynchronous transfer, and suffix decoding. Reproduction scripts are provided below for interested users.



## General Optimization
### Client
```
python3 -m sglang.bench_serving \
    --backend sglang-oai-chat \
    --tokenizer /model \
    --dataset-name random \
    --random-input 4096\
    --random-output 1000\
    --num-prompts 1000 --port 8123 --request-rate 14
```


### OURS
``` bash
# PREFILL
SGLANG_ENABLE_SPEC_V2=1 \
SGLANG_DISAGGREGATION_QUEUE_SIZE=1 \
SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=1 \
MC_TE_METRIC=1 \
SGLANG_SET_CPU_AFFINITY=true \
python -m sglang.launch_server \
  --model /models/models/GLM-4.7-FP8/ \
  --trust-remote-code \
  --watchdog-timeout "1000000" \
  --mem-fraction-static 0.9 \
  --max-running-requests 128 \
  --disaggregation-mode prefill \
  --tp-size 8 \
  --kv-cache-dtype fp8_e4m3 \
  --host 0.0.0.0 \
  --chunked-prefill-size 16384 \
  --attention-backend fa3 \
  --enable-metrics \
  --disaggregation-ib-device mlx5_ib7s400p0,mlx5_ib7s400p1,mlx5_ib7s400p2,mlx5_ib7s400p3,mlx5_ib7s400p4,mlx5_ib7s400p5,mlx5_ib7s400p6,mlx5_ib7s400p7 \
  --page-size 128 \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --enable-flashinfer-allreduce-fusion \
 --enable-fused-qk-norm-rope \
 --disaggregation-async-transfer

#Decode
CUDA_SCALE_LAUNCH_QUEUES=2x \
SGLANG_ENABLE_SPEC_V2=1 \
SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION=512 \
SGLANG_SET_CPU_AFFINITY=true \
python -m sglang.launch_server \
  --model /models/models/GLM-4.7-FP8 \
  --trust-remote-code \
  --watchdog-timeout "1000000" \
  --mem-fraction-static 0.9 \
  --tp-size 8 \
  --kv-cache-dtype fp8_e4m3 \
  --disaggregation-mode decode \
  --prefill-round-robin-balance \
  --host 0.0.0.0 \
  --chunked-prefill-size 16384 \
  --attention-backend fa3 \
  --max-running-requests 256 \
  --enable-metrics \
  --page-size 128 \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --enable-flashinfer-allreduce-fusion \
  --disaggregation-ib-device mlx5_ib7s400p0,mlx5_ib7s400p1,mlx5_ib7s400p2,mlx5_ib7s400p3,mlx5_ib7s400p4,mlx5_ib7s400p5,mlx5_ib7s400p6,mlx5_ib7s400p7 \
   --enable-fused-qk-norm-rope
```

### Baseline
```
SGLANG_ENABLE_SPEC_V2=1 \
SGLANG_DISAGGREGATION_QUEUE_SIZE=1 \
SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=1 \
MC_TE_METRIC=1 \
SGLANG_SET_CPU_AFFINITY=true \
python -m sglang.launch_server \
  --model /models/models/GLM-4.7-FP8/ \
  --trust-remote-code \
  --watchdog-timeout "1000000" \
  --mem-fraction-static 0.9 \
  --max-running-requests 128 \
  --disaggregation-mode prefill \
  --tp-size 8 \
  --kv-cache-dtype fp8_e4m3 \
  --host 0.0.0.0 \
  --chunked-prefill-size 16384 \
  --attention-backend fa3 \
  --enable-metrics \
  --disaggregation-ib-device mlx5_ib7s400p0,mlx5_ib7s400p1,mlx5_ib7s400p2,mlx5_ib7s400p3,mlx5_ib7s400p4,mlx5_ib7s400p5,mlx5_ib7s400p6,mlx5_ib7s400p7 \
  --page-size 128 \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --enable-flashinfer-allreduce-fusion \
  --disable-shared-experts-fusion


CUDA_SCALE_LAUNCH_QUEUES=2x \
SGLANG_ENABLE_SPEC_V2=1 \
SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION=512 \
SGLANG_SET_CPU_AFFINITY=true \
python -m sglang.launch_server \
  --model /models/models/GLM-4.7-FP8 \
  --trust-remote-code \
  --watchdog-timeout "1000000" \
  --mem-fraction-static 0.9 \
  --tp-size 8 \
  --kv-cache-dtype fp8_e4m3 \
  --disaggregation-mode decode \
  --prefill-round-robin-balance \
  --host 0.0.0.0 \
  --chunked-prefill-size 16384 \
  --attention-backend fa3 \
  --max-running-requests 256 \
  --enable-metrics \
  --page-size 128 \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --enable-flashinfer-allreduce-fusion \
  --disaggregation-ib-device mlx5_ib7s400p0,mlx5_ib7s400p1,mlx5_ib7s400p2,mlx5_ib7s400p3,mlx5_ib7s400p4,mlx5_ib7s400p5,mlx5_ib7s400p6,mlx5_ib7s400p7 \
  --enable-fused-qk-norm-rope \
  --disable-shared-experts-fusion
```

## Agentic Scenarios

### Suffix Decoding
```
python3 -m sglang.launch_server \
    --host 0.0.0.0 \
    --port 18011 \
    --model /workspace/models/glm4.6-fp8/snapshots/c064d336a8d0b0f59071f77eafdcdfca40f4b54c  \
    --served-model-name glm46 \
    --tp 8 \
    --trust-remote-code \
    --watchdog-timeout "1000000" \
    --page-size "128" \
    --tool-call-parser glm \
    --reasoning-parser glm45 \
    --enable-metrics \
    --collect-tokens-histogram \
    --enable-request-time-stats-logging \
    --kv-cache-dtype fp8_e4m3 \
    --enable-cache-report \
    --mem-fraction-static "0.8" \
    --max-running-requests "128" \
    --context-length "202752" \
    --chunked-prefill-size "32768" \
    --attention-backend fa3 \
    --speculative-algorithm SUFFIX \
    --speculative-eagle-topk 1 \
    --speculative-suffix-cache-max-depth 64 \
    --speculative-suffix-max-spec-factor 1.0 \
    --speculative-suffix-max-spec-offset 0.0 \
    --speculative-suffix-min-token-prob 0.1 \
    --speculative-num-draft-tokens 8
```


### MTP Baseline
```
python3 -m sglang.launch_server \
    --host 0.0.0.0 \
    --port 18011 \
    --model /workspace/models/glm4.6-fp8/  \
    --served-model-name glm46 \
    --tp 8 \
    --trust-remote-code \
    --watchdog-timeout "1000000" \
    --page-size "128" \
    --tool-call-parser glm \
    --reasoning-parser glm45 \
    --enable-metrics \
    --collect-tokens-histogram \
    --enable-request-time-stats-logging \
    --kv-cache-dtype fp8_e4m3 \
    --enable-cache-report \
    --mem-fraction-static "0.8" \
    --max-running-requests "128" \
    --context-length "202752" \
    --chunked-prefill-size "32768" \
    --attention-backend fa3 \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 1

```


The dataset pertaining to agentic behavior has been open-sourced on Hugging Face: https://huggingface.co/datasets/novita/agentic_code_dataset_22. For usage details and documentation, please refer to the dataset repository.
