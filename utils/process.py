    
import json
from collections import Counter
def build_user_content(rag_results):
    rag_items = []
    for i, ref in enumerate(rag_results, 1):
        # Ensure score is formatted to 2 decimal places for consistency
        item = (
            f"[Reference {i}]\n"
            f"- Similarity: {ref['similarity']:.3f}\n"
            f"- Reference annotation: {ref['annotation']}"
        )
        rag_items.append(item)
    rag_report_str = "\n\n".join(rag_items)

    user_content = (
        f"<image>\n"
        f"The query image to be analyzed is shown above.\n\n"
        f"========================\n"
        f"[Retrieval Report (RAG Context)]\n"
        f"========================\n"
        f"{rag_report_str}\n\n"
        f"========================\n"
        f"Based on the visual evidence and the retrieval report above, strictly execute the three-step cross-validation reasoning process (Preliminary Visual Analysis -> RAG Reference Information Analysis -> Fusion, Reasoning, and Decision) and provide the final verdict."
    )
    
    return user_content

def build_finetune_data_and_write_jsonl(original_json_file, output_jsonl_file):
    with open(original_json_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    id_counter = 0

    with open(output_jsonl_file, 'w', encoding='utf-8') as out_f:
        for entry in original_data.get('detailed_results', []):
            annotation = entry.get('annotation', '')
            
            top_k_results = entry.get('top_k_results', [])
            user_prompt = build_user_content(top_k_results)
            label = entry.get('true_label', '')
            label = "Real" if label in ["FF-real", "CelebDFv2_real","CelebDFv1_real","DFDCP_Real","DFD_real","DF_real","FFIW_Real","WDF_Real"] else "Fake"

            solution_text = f"<verdict>{label}</verdict>"
            labels_in_topk = []
            for item in top_k_results:
                item_label = item.get('label', '')
                item_label = "Real" if item_label in ["FF-real", "CelebDFv2_real","CelebDFv1_real","DFDCP_Real","DFD_real","DF_real","FFIW_Real","WDF_Real"] else "Fake"
                labels_in_topk.append(item_label)
            
            counter = Counter(labels_in_topk)

            RAG_result = counter.most_common(1)[0][0]
            rag_is_correct = (RAG_result == label)
            solution_data = {
                "ground_truth": label,
                "rag_correct": rag_is_correct
            }
            solution_json_str = json.dumps(solution_data)
            
            finetune_entry = {
                "id": id_counter,  # 使用递增的 id
                "messages": [
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                "RAG_verdict": RAG_result,
                "rag_correct": rag_is_correct,
                "reference": labels_in_topk,
                "images": [entry.get("image_path", "")],
                "solution": solution_text
            }
            
            
            out_f.write(json.dumps(finetune_entry, ensure_ascii=False) + "\n")

            id_counter += 1

    print(f"微调数据已写入 {output_jsonl_file}")


original_json_file = "./input_json"
output_jsonl_file = './output_jsonl'  # 输出的微调数据文件路径

build_finetune_data_and_write_jsonl(original_json_file, output_jsonl_file)
