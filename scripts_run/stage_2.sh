#开不开aug注意下
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift sft \
    --model "stage1_model_path" \
    --model_type qwen2_5_vl \
    --template qwen2_5_vl \
    --train_type lora \
    --dataset '' \
    --torch_dtype bfloat16 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --learning_rate 3e-5 \
    --weight_decay 0.01 \
    --lora_rank 128 \
    --lora_alpha 256 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner false \
    --gradient_accumulation_steps 2 \
    --eval_strategy "no" \
    --save_steps 70 \
    --save_total_limit 5 \
    --logging_steps 10 \
    --output_dir output/stage_2_output \
    --logging_dir output/stage_2_log \
    --system 'scripts_run/prompts.txt' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --deepspeed zero2 \
