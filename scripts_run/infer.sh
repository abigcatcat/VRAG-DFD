NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MAX_PIXELS=1003520 \
swift infer \
    --model "" \
    --infer_backend vllm \
    --val_dataset "" \
    --model_type internvl3 \
    --vllm_data_parallel_size 8 \
    --system 'scripts_run/prompts.txt' \
    --result_path "" \
    --max_new_tokens 1024


