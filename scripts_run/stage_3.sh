export CUDA_VISIBLE_DEVICES=4,5,6,7
export NPROC_PER_NODE=4
export MASTER_PORT=29506
export WANDB_MODE=offline
export no_proxy="localhost,127.0.0.1"

swift rlhf \
    --rlhf_type grpo \
    --model_type qwen2_5_vl \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs final_meta_reward multi_format \
    --reward_weights 1.0 0.25 \
    --model /path/to/model \
    --dataset ./VRAG_DFD/ms-swift/jsons/rag_grpo_data.jsonl \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000 \
    --train_type lora \
    --freeze_vit false \
    --freeze_aligner false \
    --lora_rank 128 \
    --lora_alpha 256 \
    --torch_dtype bfloat16 \
    --num_generations 4 \
    --temperature 1.0 \
    --deepspeed zero2 \
    --log_completions true \
    --learning_rate 1e-6 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --max_completion_length 1024 \
    --eval_strategy "no" \
    --save_steps 281 \
    --save_total_limit 10 \
    --logging_steps 10 \
    --system 'scripts_run/prompts.txt' \
    --output_dir output/stage_3_output \
    --dataloader_num_workers 8 \
    --num_iterations 1 \
    --attn_impl flash_attn \
    --report_to wandb \
    --logging_dir output/stage_3_log \
    --beta 0.001
