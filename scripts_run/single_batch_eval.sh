#!/bin/bash
set -e 


NUM_GPUS=8
PYTHON_SCRIPT="single_eval.py" 

MODEL_PATH=""

VLLM_BATCH_SIZE=512 

MAX_TOKENS=512
LOGPROBS=10
DTYPE="bfloat16"
LIMIT_ARG="" 

DATASET_PAIRS=(
    "input_path1.jsonl:output_path1.jsonl"
    "input_path2.jsonl:output_path2.jsonl"
    "input_path3.jsonl:output_path3.jsonl"
    )

echo "--- 自动化推理脚本启动 ---"
echo "--- 将处理 ${#DATASET_PAIRS[@]} 个数据集..."
echo "--- 使用模型: ${MODEL_PATH}"
echo "------------------------------------------------"


# --- 遍历任务队列并执行 ---
for pair in "${DATASET_PAIRS[@]}"; do
    # 解析输入和输出路径
    IFS=':' read -r INPUT_JSON FINAL_OUTPUT_JSONL <<< "$pair"
    
    echo ""
    echo "================================================"
    echo "▶️ [任务开始] 正在处理: ${INPUT_JSON}"
    echo "▶️ [任务开始] 输出目标: ${FINAL_OUTPUT_JSONL}"
    echo "================================================"

    PART_FILE_PREFIX="${FINAL_OUTPUT_JSONL}.part"

    #在后台启动 ${NUM_GPUS} 个独立的进程 ---
    echo "正在启动 ${NUM_GPUS} 个 vLLM 推理进程..."
    i=0
    while [ "$i" -lt "$NUM_GPUS" ]
    do
        # 为每个进程设置独立的环境变量
        export CUDA_VISIBLE_DEVICES=$i
        export RANK=$i  # [!!] 确保这里是 RANK
        export WORLD_SIZE=$NUM_GPUS
        
        # 每个进程写入自己的 .part 文件
        OUTPUT_PART_FILE="${PART_FILE_PREFIX}${i}"
        
        echo "  -> 启动 Rank ${i} (GPU ${i}), 输出到 ${OUTPUT_PART_FILE}"
        
        # 在后台运行 Python 脚本
        python ${PYTHON_SCRIPT} \
            --model_path ${MODEL_PATH} \
            --input_jsonl ${INPUT_JSON} \
            --output_jsonl ${OUTPUT_PART_FILE} \
            --vllm_batch_size ${VLLM_BATCH_SIZE} \
            --max_new_tokens ${MAX_TOKENS} \
            --logprobs_top_k ${LOGPROBS} \
            --dtype ${DTYPE} \
            ${LIMIT_ARG} &
        
        i=$((i + 1))
    done

    #等待所有后台进程完成
    echo "等待所有 ${NUM_GPUS} 个进程完成... (这可能需要很长时间)"
    wait
    echo "所有进程已完成。"

    # 验证所有 part 文件都存在 
    echo "正在验证所有 ${NUM_GPUS} 个分片文件是否都已创建..."
    ALL_PARTS_EXIST=true
    i=0
    while [ "$i" -lt "$NUM_GPUS" ]
    do
        PART_FILE_TO_CHECK="${PART_FILE_PREFIX}${i}"
        if [ ! -f "${PART_FILE_TO_CHECK}" ]; then
            echo " [!!! 严重错误 !!!] 文件 ${PART_FILE_TO_CHECK} 丢失!"
            echo " [!!! 严重错误 !!!] Rank ${i} 进程可能已崩溃 (OOM 或数据错误)。"
            echo " [!!! 严重错误 !!!] 将跳过此数据集的合并与排序。"
            ALL_PARTS_EXIST=false
            break 
        fi
        i=$((i + 1))
    done

    # 仅在所有分片都存在时，才执行合并、排序、清理
    if [ "$ALL_PARTS_EXIST" = true ]; then
        
        echo "✅ 验证通过。所有分片文件均存在。"

        # 合并所有分片文件 (使用循环保证顺序) 
        echo "正在按顺序合并所有 ${NUM_GPUS} 个分片文件..."
        rm -f ${FINAL_OUTPUT_JSONL} # 删除可能存在的旧文件
        i=0
        while [ "$i" -lt "$NUM_GPUS" ]
        do
            cat "${PART_FILE_PREFIX}${i}" >> ${FINAL_OUTPUT_JSONL}
            i=$((i + 1))
        done

        # 对合并后的文件进行排序 (确保 id 顺序正确)
        echo "正在对合并后的文件进行最终排序..."
        SORTED_OUTPUT_JSONL="${FINAL_OUTPUT_JSONL}.sorted"

        python -c "
import json
import sys
records = []
print('  -> 正在从 ${FINAL_OUTPUT_JSONL} 加载记录...')
with open('${FINAL_OUTPUT_JSONL}', 'r', encoding='utf-8') as f_in:
    for line_num, line in enumerate(f_in):
        try:
            rec = json.loads(line)
            rec['_sort_key'] = rec.get('id', f'line_{line_num}') 
            records.append(rec)
        except Exception as e:
            print(f'警告: 跳过无效的 JSON 行: {e}', file=sys.stderr)

print(f'  -> 加载了 {len(records)} 条记录。正在按 id 排序...')
records.sort(key=lambda x: str(x['_sort_key'])) 

print('  -> 排序完成。正在写入到 ${SORTED_OUTPUT_JSONL}...')
with open('${SORTED_OUTPUT_JSONL}', 'w', encoding='utf-8') as f_out:
    for rec in records:
        if '_sort_key' in rec: del rec['_sort_key']
        f_out.write(json.dumps(rec, ensure_ascii=False) + '\n')
"

        # 用排好序的文件覆盖原文件
        mv ${SORTED_OUTPUT_JSONL} ${FINAL_OUTPUT_JSONL}
        echo "排序完成。最终输出文件: ${FINAL_OUTPUT_JSONL}"

        # 清理分片文件
        echo "正在清理分片文件..."
        rm ${PART_FILE_PREFIX}*
        
        echo "✅ [任务完成] ${INPUT_JSON}"

    else
        echo "❌ [任务失败] 由于分片文件丢失, 未能完成 ${INPUT_JSON}"
    fi

done

echo ""
echo "================================================"
echo "🎉 --- 全部任务队列已处理完毕 --- 🎉"
echo "================================================"