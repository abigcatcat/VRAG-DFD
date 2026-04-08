#!/usr/bin/env python
# -*- coding: utf-8 -*-

from openai import OpenAI
import base64
import json
import multiprocessing
import os
import time
import sys  # 
from prompt import prompt_anno  
from tqdm import tqdm  

# --- API 配置 ---
client_config = {
    "base_url": "",
    "api_key": "",
    "timeout": 30.0 # 增加超时设置
}

JSON_PATH = ""

# 过程文件（JSON Lines）：用于即时保存和断点续传
OUTPUT_JSONL_PATH = "/ms-swift/jsons/xx.jsonl"
# 最终文件（Pretty JSON）：任务全部完成后生成的美观文件
OUTPUT_JSON_PATH = "/ms-swift/jsons/xx.json"


def encode_image(image_path):
    """将图片文件编码为 base64 字符串"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"错误: 找不到图片文件 {image_path}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"编码图片 {image_path} 时出错: {e}", file=sys.stderr)
        return None

def process_line(data):
    try:
        data_id = data["id"]
        fake_image_path = data["images"][0]
        real_image_path = data["original_images"][0]

        fake_image = encode_image(fake_image_path)
        real_image = encode_image(real_image_path)

        if fake_image is None or real_image is None:
            print(f"ID {data_id} 的图片编码失败，跳过。", file=sys.stderr)
            return None

        client = OpenAI(
            base_url=client_config["base_url"],
            api_key=client_config["api_key"],
            timeout=client_config.get("timeout", 20.0)
        )
    
        completion = client.chat.completions.create(
            model="gemini-2.5-pro",
            messages=[
                {"role": "system", "content": prompt_anno["system_prompt_NT"]},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt_anno["user_prompt_NT"]},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{fake_image}"
                    }},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{real_image}"
                    }}
                ]}
            ]
        )
        response = completion.choices[0].message.content
        
        result = {
            "id": data["id"],
            "annotation": response,
            "images": data["images"],
            "original_images": data["original_images"]
        }
        return result

    except Exception as e:
        print(f"处理 ID {data.get('id', 'N/A')} 时发生错误: {str(e)}", file=sys.stderr)
        time.sleep(1)
        return None

def load_processed_ids(path):
    """
    加载已处理过的 ID，用于断点续传
    """
    processed_ids = set()
    if not os.path.exists(path):
        return processed_ids

    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try: 
                    data = json.loads(line.strip())
                    if 'id' in data:
                        processed_ids.add(data['id'])
                except json.JSONDecodeError:
                    print(f"警告: 无法解析已处理文件中的行: {line.strip()}", file=sys.stderr)
    except Exception as e:
        print(f"警告: 读取已处理文件 {path} 失败: {e}", file=sys.stderr)
    
    return processed_ids


def main():

    print(f"正在检查已处理过的任务... (来自 {OUTPUT_JSONL_PATH})")
    processed_ids = load_processed_ids(OUTPUT_JSONL_PATH)
    if processed_ids:
        print(f"检测到 {len(processed_ids)} 个已处理的任务，将跳过它们。")

    # 1. 从 JSONL 文件读取所有任务
    tasks = []
    print(f"正在从 {JSON_PATH} 加载任务...")
    try:
        with open(JSON_PATH, 'r', encoding='utf-8') as infile:
            for line in infile:
                try:
                    data = json.loads(line.strip())
                    if data.get('id') not in processed_ids:
                        tasks.append(data)
                except json.JSONDecodeError:
                    print(f"跳过无法解析的行: {line.strip()}", file=sys.stderr)
        print(f"总共加载了 {len(tasks)} 个新任务。")
    except FileNotFoundError:
        print(f"错误: 找不到输入文件 {JSON_PATH}", file=sys.stderr)
        return
    except Exception as e:
        print(f"读取输入文件时出错: {e}", file=sys.stderr)
        return

    if not tasks:
       
        print("没有需要处理的新任务。")
        processing_done = True # 
     
    else:
        # 2. 设置进程池
        num_processes = 10
        print(f"使用 {num_processes} 个进程开始处理 {len(tasks)} 个新任务...")
        success_count = 0
        fail_count = 0
        
        # 4. 使用进程池并行处理任务
        processing_done = False # 
      

        try:
            with open(OUTPUT_JSONL_PATH, 'a', encoding='utf-8') as outfile, \
                 multiprocessing.Pool(processes=num_processes) as pool:
                
                with tqdm(total=len(tasks), desc="处理中", unit="项") as pbar:
                    for result in pool.imap_unordered(process_line, tasks):
                        if result is not None:
                            try:
                                outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
                                outfile.flush()
                                success_count += 1
                            except Exception as e:
                                print(f"错误: 写入结果到 {OUTPUT_JSONL_PATH} 失败: {e}", file=sys.stderr)
                                fail_count += 1
                        else:
                            fail_count += 1
                        pbar.update(1)
                
            processing_done = True # 

        except KeyboardInterrupt:
            print("\n检测到用户中断 (Ctrl+C)。正在终止进程...")
            print("已处理的结果已保存，下次运行将自动续传。")
        except Exception as e:
            print(f"\n处理过程中发生未捕获的严重错误: {e}", file=sys.stderr)
            print("已处理的结果已保存，下次运行将自动续传。")
        finally:
            print(f"\n处理完毕。")
            if tasks: # 
                print(f"本次运行成功: {success_count} 个")
                print(f"本次运行失败: {fail_count} 个")
            print(f"所有成功的结果（包括之前运行的）均已保存在: {OUTPUT_JSONL_PATH}")

    # 5. 转换 .jsonl 为 .json （仅当处理正常完成时）
    if processing_done:
        print(f"\n正在将 {OUTPUT_JSONL_PATH} 转换为美观的 JSON 数组格式...")
        all_results = []
        try:
            with open(OUTPUT_JSONL_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        all_results.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        pass # 

            if not all_results:
                print("没有检测到任何结果，跳过最终 JSON 转换。")
                return 
            
            all_results.sort(key=lambda x: x.get('id', 0)) # 
            
            with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as outfile:
                json.dump(all_results, outfile, ensure_ascii=False, indent=4)
            
            print(f"转换完成！最终的美观文件已保存到: {OUTPUT_JSON_PATH}")
            print(f"（{OUTPUT_JSONL_PATH} 文件也已保留，用于断点续传）")

        except Exception as e:
            print(f"错误: 转换 {OUTPUT_JSONL_PATH} 为 {OUTPUT_JSON_PATH} 时失败: {e}", file=sys.stderr)
            print(f"不过不用担心，所有原始数据仍在 {OUTPUT_JSONL_PATH} 中。")

    elif not processing_done:
        print("\n处理未正常完成（可能被中断），跳过最终 JSON 转换。")
        print(f"请重新运行脚本以完成剩余任务。所有已完成的结果安全存储在 {OUTPUT_JSONL_PATH} 中。")


if __name__ == "__main__":
    main()
