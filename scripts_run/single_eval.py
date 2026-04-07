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

import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info 
from tqdm import tqdm

def find_sublist_index(main_list: List[int], sub_list: List[int]) -> Optional[int]:
    """Return the index after the first occurrence of sub_list in main_list, or None."""
    n = len(main_list)
    m = len(sub_list)
    if m == 0 or m > n:
        return None
    for i in range(n - m + 1):
        if main_list[i : i + m] == sub_list:
            return i + m
    return None

system_prompt = '''
You are a top-tier facial forensics analyst. Your mission is to assess the authenticity of the provided facial image. Your assessment should be conducted from two distinct analytical perspectives:
1.Low-Level Analysis: Examine the image's low-level features to detect any manipulation artifacts.
2.Semantic-Level Analysis: Evaluate the entire face for semantic inconsistencies.
You should respond with only one of the following: \"Verdict: FAKE\" or \"Verdict: REAL\".
'''

FORGERY_TYPE_MAP = {
    "FD": "Facial Distortion",
    "FE": "Facial Expression",
    "FS": "Facial Swapping",
    "OTHER": "Other"
}


def parse_label(raw_label: str) -> Dict[str, str]:
    if raw_label == "FF-real":
        return {"label": "REAL", "type": "N/A"}
    parts = raw_label.split('-')
    if len(parts) == 2:
        type_code = parts[1]
        type_full_name = FORGERY_TYPE_MAP.get(type_code, "Unknown")
        return {"label": "FAKE", "type": type_full_name}
    return {"label": "FAKE", "type": "Unknown"}



def parse_model_response(response_text: str) -> str:
    """
    从模型输出文本中解析最终判决。
    格式: "Verdict: Real" 或 "Verdict: Fake"
    """
    verdict = "PARSE_ERROR"
    try:
        match = re.search(r"Verdict:\s*(Real|Fake)", response_text, re.IGNORECASE)
        if match:
            verdict = match.group(1).upper()
    except Exception as e:
        print(f"Error parsing model response: {e}")
    
    # 返回 "REAL", "FAKE", 或 "PARSE_ERROR"
    return verdict

def safe_get_logprob(step_logprobs: Dict, token_id: int, tokenizer) -> float:
    """
    Try multiple key types in step_logprobs and return logprob or -inf.
    Handles values that are raw floats or vLLM Logprob objects.
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
    except Exception:
        pass

    try:
        s_token_id = str(token_id)
        if s_token_id in step_logprobs:
            return _get_float_from_value(step_logprobs[s_token_id])
    except Exception:
        pass

    try:
        tok_str = tokenizer.convert_ids_to_tokens([token_id])[0]
        if tok_str in step_logprobs:
            return _get_float_from_value(step_logprobs[tok_str])
    except Exception:
        pass

    try:
        for k, v in step_logprobs.items():
            try:
                if int(k) == token_id:
                    return _get_float_from_value(v)
            except Exception:
                continue
    except Exception:
        pass

    return -math.inf

def ensure_token_id(tokenizer, word: str, fallback_id: int) -> int:
    """Return first token id for `word` or fallback."""
    try:
        ids = tokenizer(word, add_special_tokens=False).input_ids
        if ids:
            return ids[0]
    except Exception:
        pass
    return fallback_id

def safe_construct_prompt(processor, messages: List[Dict[str, Any]]) -> str:
    """Try processor.apply_chat_template if available; else fallback to concatenation."""
    try:
        if hasattr(processor, "apply_chat_template"):
            # Some processors use different signature; try common one
            try:
                return processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            except TypeError:
                return processor.apply_chat_template(messages)
        else:
            parts = []
            for msg in messages:
                if isinstance(msg, dict):
                    if msg.get("type") == "text":
                        parts.append(msg.get("text", ""))
                    elif msg.get("type") == "image":
                        parts.append(f"[Image: {msg.get('image')}]")
                elif isinstance(msg, list):
                    for sub in msg:
                        if isinstance(sub, dict) and sub.get("type") == "text":
                            parts.append(sub.get("text", ""))
            return "\n".join(parts)
    except Exception as e:
        print("Warning: constructing prompt via fallback due to:", e)
        return str(messages)

def main():
    parser = argparse.ArgumentParser(description="Multi-GPU vLLM inference (Manual Launch Worker)")

    parser.add_argument("--input_jsonl", type=str, default="")
    
    parser.add_argument("--output_jsonl", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--vllm_batch_size", type=int, default=32, help="vLLM 内部的批处理大小 (可以设得比较大)")
    parser.add_argument("--logprobs_top_k", type=int, default=10, help="在 logprobs 中请求多少个 top tokens (必须大于等于1)")
    parser.add_argument("--debug", action="store_true", help="打印更多调试信息")
    args = parser.parse_args()

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
        max_model_len=4096,
        limit_mm_per_prompt={"image": 6}
    )

    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=True)
    tokenizer = processor.tokenizer

    all_lines = []
    try:
        with open(args.input_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_lines.append(line)
    except FileNotFoundError:
        print(f"[Rank {local_rank} FATAL] 输入文件未找到: {args.input_jsonl}")
        raise

    all_idx = list(range(len(all_lines)))
    
    if args.limit and args.limit > 0:
        all_idx = all_idx[: args.limit]

    all_idx_this_rank = all_idx[local_rank::world_size]
    
    print(f"[Rank {local_rank}] Processing {len(all_idx_this_rank)} items (Total: {len(all_idx)}, Batch: {args.vllm_batch_size})")

    part_file_path = args.output_jsonl 
    
    REAL_ID = 8800
    FAKE_ID = 36965
    TRIGGER_TEXT = "Verdict:"
    trigger_ids = []
    try:
        trigger_ids = tokenizer(TRIGGER_TEXT, add_special_tokens=False).input_ids
    except Exception:
        trigger_ids = []
    if not trigger_ids:
        print(f"[Rank {local_rank} WARN] trigger text tokenized to empty list. TRIGGER_TEXT={TRIGGER_TEXT}")
    
    REAL_ID = ensure_token_id(tokenizer, " Real", REAL_ID)
    FAKE_ID = ensure_token_id(tokenizer, " Fake", FAKE_ID)
    if args.debug:
        print(f"[Rank {local_rank} DEBUG] REAL_ID={REAL_ID} FAKE_ID={FAKE_ID} trigger_ids={trigger_ids}")


    progress_bar = tqdm(total=len(all_idx_this_rank), desc=f"vLLM (Rank {local_rank})", position=local_rank)
    
    all_records_to_write = []

    for start in range(0, len(all_idx_this_rank), args.vllm_batch_size):
        batch_indices = all_idx_this_rank[start : start + args.vllm_batch_size]
        vllm_request_list = []
        other_data_batch = []

        for idx in batch_indices:
            try:
                line = all_lines[idx]
                item = json.loads(line)
                item_id = item.get("id", f"idx_{idx}")

                solution_string = item.get("solution") 
                true_label_text_raw = "UNKNOWN"
                if isinstance(solution_string, str):
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
                messages = [] 
                messages.append({"role": "system", "content": system_prompt})

                user_message_content = []
                for img_path in item_images:
                    if not img_path: continue
                    if not os.path.exists(img_path):
                            print(f"[Rank {local_rank} WARN] 找不到图片: {img_path} (在 index {idx})")
                        
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
                        "query_image": item_images
                    }
                )
            except Exception as ex_prep:
                print(f"[Rank {local_rank} ERROR] data prep failed for index {idx}: {repr(ex_prep)}")
                traceback.print_exc()
                all_records_to_write.append(
                    {
                        "index": idx, "query_image": None, "label": "UNKNOWN", "verdict": "ERROR",
                        "y_label": -1, "y_pred": 0.5, "reference_labels": [],
                        "model_output": f"DATA_PREP_EXCEPTION: {repr(ex_prep)}", "reference_images": [],
                    }
                )

        if not vllm_request_list:
            progress_bar.update(len(batch_indices))
            continue

        sampling_params = SamplingParams(max_tokens=args.max_new_tokens, logprobs=args.logprobs_top_k, temperature=0.000001)

        # run generation (robust)
        try:
            raw_outputs_iter = llm.generate(vllm_request_list, sampling_params)
            outputs = list(raw_outputs_iter)
            if args.debug:
                print(f"[Rank {local_rank} DEBUG] outputs length: {len(outputs)}")
        except Exception as gen_e:
            print(f"[Rank {local_rank} EXCEPTION] llm.generate failed for batch starting {start}: {repr(gen_e)}")
            traceback.print_exc()
            for other in other_data_batch:
                all_records_to_write.append(
                    {
                        "index": other.get("index", -1), "query_image": other.get("query_image"),
                        "label": other.get("label"), "verdict": "ERROR", "y_label": other.get("y_label", -1),
                        "y_pred": 0.5, "reference_labels": other.get("reference_labels", []),
                        "model_output": f"GEN_EXCEPTION: {repr(gen_e)}", "reference_images": [],
                    }
                )
            progress_bar.update(len(batch_indices))
            continue

        if args.debug:
            print(f"[DEBUG] REAL_ID={REAL_ID} FAKE_ID={FAKE_ID} trigger_ids={trigger_ids}")

        # process each output
        prob_fakes_list = []
        if trigger_ids:
            for out_idx, out_obj in enumerate(outputs):
                prob_fake = 0.5
                try:
                    out0 = out_obj.outputs[0]
                    token_ids = out0.token_ids
                    if hasattr(token_ids, "tolist"):
                        try: token_ids = token_ids.tolist()
                        except Exception: token_ids = list(token_ids)
                    else: token_ids = list(token_ids)
                    logprobs_list = out0.logprobs
                    trigger_end_index = find_sublist_index(token_ids, trigger_ids)
                    if (
                        trigger_end_index is not None
                        and isinstance(logprobs_list, (list, tuple))
                        and len(logprobs_list) > trigger_end_index
                    ):
                        step_logprobs = logprobs_list[trigger_end_index]
                        logprob_real = safe_get_logprob(step_logprobs, REAL_ID, tokenizer)
                        logprob_fake = safe_get_logprob(step_logprobs, FAKE_ID, tokenizer)
                        if (logprob_real > -math.inf) or (logprob_fake > -math.inf):
                            lp = torch.tensor([logprob_real, logprob_fake], dtype=torch.float64)
                            probs = torch.softmax(lp, dim=0)
                            prob_fake = float(probs[1].item())
                        else:
                            out_text = getattr(out0, "text", "") or ""
                            parsed = parse_model_response(out_text)
                            prob_fake = 1.0 if parsed == "FAKE" else (0.0 if parsed == "REAL" else 0.5)
                    else:
                        out_text = getattr(out0, "text", "") or ""
                        parsed = parse_model_response(out_text)
                        prob_fake = 1.0 if parsed == "FAKE" else (0.0 if parsed == "REAL" else 0.5)
                except Exception as ex_proc:
                    print(f"[Rank {local_rank} WARN] processing output {out_idx} failed: {repr(ex_proc)}")
                    traceback.print_exc()
                    prob_fake = 0.5
                prob_fakes_list.append(prob_fake)
        else:
            for out_obj in outputs:
                out_text = getattr(out_obj.outputs[0], "text", "") or ""
                parsed = parse_model_response(out_text)
                prob_fakes_list.append(1.0 if parsed == "FAKE" else (0.0 if parsed == "REAL" else 0.5))

        min_len = min(len(outputs), len(other_data_batch))
        if args.debug and len(outputs) != len(other_data_batch):
            print(f"[Rank {local_rank} WARN] outputs len {len(outputs)} != other_data_batch len {len(other_data_batch)}; using min {min_len}")

        for j in range(min_len):
            out_obj = outputs[j]
            other = other_data_batch[j]
            out_text = getattr(out_obj.outputs[0], "text", "").strip()
            prob_fake = prob_fakes_list[j] if j < len(prob_fakes_list) else 0.5
            verdict = parse_model_response(out_text)
            record = {
                "index": other.get("index", -1),
                "query_image": other.get("query_image"),
                "label": other.get("label"),
                "verdict": verdict,
                "y_label": other.get("y_label", -1),
                "y_pred": prob_fake,
                "model_output": out_text,
            }
            all_records_to_write.append(record)

        progress_bar.update(len(batch_indices))

    progress_bar.close()

    out_dir = os.path.dirname(part_file_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    all_records_to_write.sort(key=lambda x: x.get("index", float("inf")))
    with open(part_file_path, "w", encoding="utf-8") as fout:
        for rec in all_records_to_write:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    print(f"[Rank {local_rank}] 工作完成, 写入 {len(all_records_to_write)} 条记录到 {part_file_path}")
    
    print(f"[Rank {local_rank}]: 脚本执行完毕。")

if __name__ == "__main__":
    main()