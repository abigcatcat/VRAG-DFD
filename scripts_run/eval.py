#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
import logging
from typing import List, Dict, Any, Optional
from PIL import Image
import math
import faulthandler
import traceback

# enable faulthandler early to help diagnose native crashes
faulthandler.enable(all_threads=True)

import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info # (仍然需要)
from tqdm import tqdm


system_prompt = '''
You are a world-class Deepfake Detection expert.
Your task is to perform **"Cross-Validation"** based on "visual evidence" and a "retrieval report" to determine whether an image is `Real` or `Fake`.

You will receive two inputs:
1.  **Query Image**: This is the primary "physical evidence" you need to analyze.
2.  **RAG Context (Retrieval Report)**: This is secondary "reference information" provided by an auxiliary system, containing annotations and similarity scores for the 5 images most similar to the query image.
    * **Warning**: The RAG report may contain **noise** (incorrect retrieval results). You must not blindly trust it; you must use your visual analysis to **verify** it.

**Your Reasoning Steps:**
You must strictly follow these three steps to generate your "Chain-of-Thought":
1.  **[Preliminary Visual Analysis]**: Independently analyze the image to identify potential forgery artifacts or authentic features.
2.  **[RAG Reference Information Analysis]**: Objectively evaluate the RAG report, pointing out which evidence supports your visual judgment and which conflicts with it.
3.  **[Fusion, Reasoning, and Decision]** (Most Critical): **Cross-verify** the information from both sources.
    * If RAG provides high-scoring evidence that can be visually **confirmed**, please **adopt** it (even if it corrects your initial intuition).
    * If RAG provides evidence that **contradicts** visual facts, please identify it as "retrieval noise" and **reject** it.
    * Make a final judgment based on the verified evidence.

Please use the following format to output your analysis report:
<Preliminary Visual Analysis>
...
</Preliminary Visual Analysis>
<RAG Reference Information Analysis>
...
</RAG Reference Information Analysis>
<Fusion, Reasoning, and Decision>
...
</Fusion, Reasoning, and Decision>
<verdict>Real/Fake</verdict>
'''


# -------------------------
# Helper: find sublist index (反向搜索版本)
# -------------------------
def find_sublist_index_reverse(main_list: List[int], sub_list: List[int]) -> Optional[int]:
    """
    (v12 版) 
    从末尾反向搜索, 返回 *决策步骤* 的索引 (即 sub_list 之后一个元素的索引)。
    如果未找到或序列在末尾，返回 None。
    """
    n = len(main_list)
    m = len(sub_list)
    if m == 0 or m > n:
        return None

    # 从后往前找 (i 是序列 *开头* 的索引)
    for i in range(n - m, -1, -1): # 从 n-m 找到 0
        if main_list[i : i + m] == sub_list:
            # 找到了! 序列在 i+m-1 处结束 (e.g., "dict" at i+2)
            # 决策步骤是在 i + m
            decision_step_index = i + m
            
            # 检查是否越界 (如果序列是最后一个 token)
            if decision_step_index >= n:
                return None 
            
            return decision_step_index # 返回决策点的索引
            
    return None # 未找到


def parse_model_response(response_text: str) -> str:
    """(v12 版) 提取 <verdict> 标签内容"""
    verdict = "PARSE_ERROR"
    try:
        # re.IGNORECASE 忽略大小写 (比如 <Verdict>)
        # re.DOTALL 让 '.' 匹配换行符 (比如 <verdict>\nFake\n</verdict>)
        match = re.search(r'<verdict>(.*?)</verdict>', response_text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            result = match.group(1).strip().upper()
            if "FAKE" in result:
                verdict = "FAKE"
            elif "REAL" in result:
                verdict = "REAL"
            else:
                verdict = "PARSE_ERROR" 
    except Exception as e:
        print(f"Error parsing model response: {e}")
    return verdict


def safe_get_logprob(step_logprobs: Dict, token_id: int, tokenizer) -> float:
    """
    (v12 - 保持不变)
    Try multiple key types in step_logprobs and return logprob or -inf.
    """
    if not step_logprobs:
        return -math.inf

    def _get_float_from_value(value: Any) -> float:
        """Helper to extract logprob float from a value."""
        try:
            return float(value)
        except (TypeError, ValueError):
            try:
                if hasattr(value, "logprob"):
                    return float(value.logprob)
            except Exception:
                pass
            raise TypeError(f"Cannot convert value ({type(value)}) to float")

    try:
        if token_id in step_logprobs:
            return _get_float_from_value(step_logprobs[token_id])
    except Exception: pass

    try:
        s_token_id = str(token_id)
        if s_token_id in step_logprobs:
            return _get_float_from_value(step_logprobs[s_token_id])
    except Exception: pass

    try:
        tok_str = tokenizer.convert_ids_to_tokens([token_id])[0]
        if tok_str in step_logprobs:
            return _get_float_from_value(step_logprobs[tok_str])
    except Exception: pass
    
    return -math.inf

def safe_construct_prompt(processor, messages: List[Dict[str, Any]]) -> str:
    try:
        if hasattr(processor, "apply_chat_template"):
            try:
                return processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            except TypeError:
                return processor.apply_chat_template(messages)
        else:
            parts = []
            for msg in messages:
                if msg.get("role") == "system":
                    parts.append(f"System: {msg.get('content')}")
                elif msg.get("role") == "user":
                    parts.append("User:")
                    if isinstance(msg.get("content"), list):
                        for sub in msg.get("content"):
                            if sub.get("type") == "text":
                                parts.append(sub.get("text", ""))
                            elif sub.get("type") == "image":
                                parts.append(f"[Image: {sub.get('image')}]")
                    else:
                        parts.append(str(msg.get("content")))
            return "\n".join(parts)
    except Exception as e:
        print("Warning: constructing prompt via fallback due to:", e)
        return str(messages)

def calculate_fake_prob_vllm(
    token_ids: List[int], 
    logprobs_list: List[Dict[int, Any]], 
    tokenizer,
    prefix_ids: List[int],
    real_path_id: int,
    fake_path_id: int
) -> (float, str, bool):
    
    decision_step_index = find_sublist_index_reverse(token_ids, prefix_ids)
    
    if decision_step_index is None:
        return 0.5, "N/A", True 

    try:
        step_logprobs = logprobs_list[decision_step_index]
        if not step_logprobs:
            return 0.5, "N/A", True 
    except IndexError:
         return 0.5, "N/A", True 
            
    logprob_real_path = safe_get_logprob(step_logprobs, real_path_id, tokenizer)
    logprob_fake_path = safe_get_logprob(step_logprobs, fake_path_id, tokenizer)

    lp = torch.tensor([logprob_real_path, logprob_fake_path], dtype=torch.float64)
    probs = torch.softmax(lp, dim=0)
    score = float(probs[1].item()) # P(Fake)
    
    missing_tokens_flag = (logprob_real_path == -math.inf and logprob_fake_path == -math.inf)
    
    predicted_token_id = token_ids[decision_step_index]
    predicted_verdict = "N/A"
    if predicted_token_id == fake_path_id:
        predicted_verdict = "Fake"
    elif predicted_token_id == real_path_id:
        predicted_verdict = "Real"
    
    return score, predicted_verdict, missing_tokens_flag

def main():
    parser = argparse.ArgumentParser(description="vLLM Inference Script (JSONL a输入, v9 Logprob Logic)") # <--- 修改描述
    parser.add_argument("--input_jsonl", type=str, 
                        default="", 
                        help="输入的 .jsonl 文件路径")
    parser.add_argument("--output_jsonl", type=str, 
                        default="", 
                        help="*最终*输出的 .jsonl 文件路径")
    parser.add_argument("--model_path", type=str, 
                        default="")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max_new_tokens", type=int, default=1024) # (!! 修改默认值 !!)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--vllm_batch_size", type=int, default=16)
    parser.add_argument("--logprobs_top_k", type=int, default=10, help="在 logprobs 中请求多少个 top tokens (必须大于等于1)")
    parser.add_argument("--debug", action="store_true", help="打印更多调试信息")
    args = parser.parse_args()
    print(f"--- [DEBUG] ---")
    print(f"[DEBUG] 脚本接收到的 --model_path 参数是: {args.model_path}")
    print(f"--- [DEBUG] ---")

    try:
        local_rank = int(os.environ.get("WORKER_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
    except ValueError:
        local_rank = 0
        world_size = 1
    
    print(f"--- Starting Worker Rank {local_rank} of {world_size} ---")
    
    
    print(f"[Rank {local_rank}] Initializing vLLM with model: {args.model_path}")
    
    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        tensor_parallel_size=1, 
        dtype=args.dtype,
        max_loras=5,
        max_lora_rank=128,
        limit_mm_per_prompt={"image": 6}
    )

    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=True)
    tokenizer = processor.tokenizer
    
    try:
        VERDICT_PREFIX_IDS = tokenizer.convert_tokens_to_ids(['<', 'ver', 'dict'])
        REAL_PATH_ID = tokenizer.convert_tokens_to_ids('>')
        FAKE_PATH_ID = tokenizer.convert_tokens_to_ids('>F')

        if any(t == tokenizer.unk_token_id for t in VERDICT_PREFIX_IDS + [REAL_PATH_ID, FAKE_PATH_ID]):
            print(f"[Rank {local_rank} WARN] 关键 token (>, >F, <, ver, dict) 中有 UNK token!")
            
    except Exception as e:
        print(f"[Rank {local_rank} FATAL] 无法转换关键 token (>, >F, <, ver, dict). Error: {e}")
        raise e
    
    if args.debug:
        print(f"[Rank {local_rank} DEBUG] VERDICT_PREFIX_IDS={VERDICT_PREFIX_IDS} REAL_PATH_ID={REAL_PATH_ID} FAKE_PATH_ID={FAKE_PATH_ID}")

    print(f"[Rank {local_rank}] Loading input JSONL: {args.input_jsonl}")
    all_lines = []
    try:
        with open(args.input_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_lines.append(line)
    except FileNotFoundError:
        print(f"[Rank {local_rank} FATAL] 输入文件未找到: {args.input_jsonl}")
        raise
        
    if not all_lines:
        raise ValueError(f"No data found in input JSONL: {args.input_jsonl}")
    
    all_idx = list(range(len(all_lines)))
    if args.limit and args.limit > 0:
        all_idx = all_idx[: args.limit]

    all_idx_this_rank = all_idx[local_rank::world_size]
    
    print(f"[Rank {local_rank}] Processing {len(all_idx_this_rank)} items (Total: {len(all_idx)}, Batch: {args.vllm_batch_size})")

    part_file_path = args.output_jsonl 

    progress_bar = tqdm(total=len(all_idx_this_rank), desc=f"vLLM (Rank {local_rank})", position=local_rank)
    
    all_records_to_write = []

    for start in range(0, len(all_idx_this_rank), args.vllm_batch_size):
        batch_indices = all_idx_this_rank[start : start + args.vllm_batch_size]
        vllm_request_list = []
        other_data_batch = []

        for idx in batch_indices:
            item = None 
            try:
                
                line = all_lines[idx]
                item = json.loads(line)

                item_id = item.get("id", f"idx_{idx}")

                solution_string = item.get("solution") 
                
                true_label_text_raw = "UNKNOWN" 
                if isinstance(solution_string, str):
                    # (使用我们之前用过的 regex)
                    match = re.search(r'<verdict>(.*?)</verdict>', solution_string, flags=re.IGNORECASE | re.DOTALL)
                    if match:
                        true_label_text_raw = match.group(1).strip().upper()

                if "FAKE" in true_label_text_raw:
                    true_label = "FAKE"
                    label_map = 1
                elif "REAL" in true_label_text_raw:
                    true_label = "REAL"
                    label_map = 0
                else:
                    true_label = "UNKNOWN"
                    label_map = -1 

                item_messages = item.get("messages", [])
                item_images = item.get("images", []) 
                rag_verdict = item.get("RAG_verdict", "N/A") 
                rag_correct = item.get("rag_correct", "N/A") 
                reference = item.get("reference", []) 
                
                messages = []
                messages.append({"role": "system", "content": system_prompt})

                user_message_content = []
                for img_path in item_images:
                    if not img_path: continue
                    user_message_content.append({"type": "image", "image": img_path})
                
                user_text = ""
                for msg in item_messages:
                    if msg.get("role") == "user":
                        if isinstance(msg.get("content"), str):
                             user_text = msg.get("content", "")
                        break 
                user_message_content.append({"type": "text", "text": user_text})
                
                messages.append({"role": "user", "content": user_message_content})
                
                chat_text = safe_construct_prompt(processor, messages)
                pil_images, _ = process_vision_info(messages)
                
                request_dict = {"prompt": chat_text, "multi_modal_data": {"image": pil_images}}
                vllm_request_list.append(request_dict)
                
                other_data_batch.append(
                    {
                        "id": item_id, 
                        "label": true_label, 
                        "y_label": label_map,
                        "rag_verdict": rag_verdict,
                        "rag_correct": rag_correct,
                        "reference": reference,
                        "query_image": item_images
                    }
                )
                
            except Exception as ex_prep:
                print(f"[Rank {local_rank} ERROR] data prep failed for index {idx}: {repr(ex_prep)}")
                traceback.print_exc()
                item_id = f"idx_{idx}"
                if item: item_id = item.get("id", f"idx_{idx}")
                all_records_to_write.append(
                    {
                        "id": item_id, 
                        "label": "UNKNOWN", "verdict": "ERROR",
                        "y_label": -1, "y_pred": 0.5, 
                        "model_output": f"DATA_PREP_EXCEPTION: {repr(ex_prep)}",
                    }
                )

        if not vllm_request_list:
            progress_bar.update(len(batch_indices))
            continue

        sampling_params = SamplingParams(
            max_tokens=args.max_new_tokens, 
            logprobs=args.logprobs_top_k, 
            temperature=0.000001 
        )

        try:
            raw_outputs_iter = llm.generate(vllm_request_list, sampling_params)
            
            outputs = list(raw_outputs_iter)
        except Exception as gen_e:
            print(f"[Rank {local_rank} EXCEPTION] llm.generate failed for batch starting {start}: {repr(gen_e)}")
            traceback.print_exc()
            for other in other_data_batch:
                all_records_to_write.append(
                    {
                        "id": other.get("id", "N/A"), 
                        "label": other.get("label"), "verdict": "ERROR", 
                        "y_label": other.get("y_label", -1), "y_pred": 0.5,
                        "model_output": f"GEN_EXCEPTION: {repr(gen_e)}",
                    }
                )
            progress_bar.update(len(batch_indices))
            continue

        if args.debug:
            print(f"[Rank {local_rank} DEBUG] VERDICT_PREFIX_IDS={VERDICT_PREFIX_IDS} REAL_PATH_ID={REAL_PATH_ID} FAKE_PATH_ID={FAKE_PATH_ID}")

        
        processed_results = [] 
        
        for out_idx, out_obj in enumerate(outputs):
            prob_fake = 0.5
            verdict = "N/A"
            missing_flag = True
            try:
                out0 = out_obj.outputs[0]
                
                token_ids = out0.token_ids
                if hasattr(token_ids, "tolist"):
                    try: token_ids = token_ids.tolist()
                    except Exception: token_ids = list(token_ids)
                else: token_ids = list(token_ids)
                
                logprobs_list = out0.logprobs
                
                prob_fake, verdict, missing_flag = calculate_fake_prob_vllm(
                    token_ids,
                    logprobs_list,
                    tokenizer,
                    VERDICT_PREFIX_IDS,
                    REAL_PATH_ID,
                    FAKE_PATH_ID
                )
                
                if missing_flag and args.debug: 
                    other = other_data_batch[out_idx]
                    print(f"[Rank {local_rank} INFO] 样本 [id: {other.get('id', 'N/A')}] 因 Real/Fake 路径均未找到, 赋值 0.5。")

            except Exception as ex_proc:
                print(f"[Rank {local_rank} WARN] processing output {out_idx} failed: {repr(ex_proc)}")
                traceback.print_exc()
                prob_fake = 0.5
                verdict = "PROC_ERROR"
                missing_flag = True
                
            processed_results.append((prob_fake, verdict, missing_flag))

        min_len = min(len(outputs), len(other_data_batch))
        if args.debug and len(outputs) != len(other_data_batch):
            print(f"[Rank {local_rank} WARN] outputs len {len(outputs)} != other_data_batch len {len(other_data_batch)}; using min {min_len}")

        for j in range(min_len):
            out_obj = outputs[j]
            other = other_data_batch[j]
            out_text = getattr(out_obj.outputs[0], "text", "").strip()
            
            
            prob_fake, predicted_verdict, _ = processed_results[j] if j < len(processed_results) else (0.5, "N/A", True)
            
            final_verdict = predicted_verdict
            if final_verdict in ["N/A", "PROC_ERROR"]:
                final_verdict = parse_model_response(out_text) 
            label = other.get("label", "UNKNOWN")
            VRAG_correct = (final_verdict.upper() == label)

            record = {
                "id": other.get("id"),
                "label": other.get("label"),
                "verdict": final_verdict,
                "VRAG_correct": VRAG_correct,
                "rag_verdict": other.get("rag_verdict", "N/A"),
                "rag_correct": other.get("rag_correct", "N/A"),
                "y_label": other.get("y_label", -1),
                "y_pred": prob_fake,
                "reference": other.get("reference", []),
                "query_image": other.get("query_image", -1),
                "model_output": out_text,
            }
            all_records_to_write.append(record)

        progress_bar.update(len(batch_indices))

    progress_bar.close()

    out_dir = os.path.dirname(part_file_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    all_records_to_write.sort(key=lambda x: str(x.get("id", "zzzz"))) 
    with open(part_file_path, "w", encoding="utf-8") as fout:
        for rec in all_records_to_write:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    print(f"[Rank {local_rank}] 工作完成, 写入 {len(all_records_to_write)} 条记录到 {part_file_path}")
    
    print(f"[Rank {local_rank}]: 脚本执行完毕。")

if __name__ == "__main__":
    main()