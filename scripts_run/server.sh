export NO_PROXY=127.0.0.1,localhost,::1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

export MASTER_PORT=29502
export CUDA_VISIBLE_DEVICES=0,1,2,3

swift rollout \
  --model /path/to/Qwen2.5-VL-7B-Instruct \
  --vllm_data_parallel_size 4 \

