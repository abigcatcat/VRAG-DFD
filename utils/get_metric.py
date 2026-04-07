import json
import numpy as np
from sklearn import metrics
from collections import defaultdict

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

# 计算帧级AUC和Acc
def compute_frame_level_metrics(y_pred, y_true, verdict,label):
    # 计算每一帧的AUC
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred, pos_label=1)
    frame_auc = metrics.auc(fpr, tpr)

    return frame_auc

# 计算视频级AUC和Acc
def compute_video_level_auc(jsonl_data):
    result_dict = defaultdict(list)  
    new_label = []  
    new_pred = []   
    total = 0
    correct = 0
    per_class = {}  

    for item in jsonl_data:
        
        query_image = item.get('query_image', None)

        query_image = query_image[0]  
        
        y_label = item['y_label']
        y_pred = item['y_pred']
        label = item['label']
        verdict = item['verdict']

        verdict = verdict.upper()
        label = label.upper()
        if verdict in ["REAL", "FAKE"] and label in ["REAL", "FAKE"]:
            total += 1
            if label not in per_class:
                per_class[label] = {"total": 0, "correct": 0}
            per_class[label]["total"] += 1
            if verdict == label:
                correct += 1
                per_class[label]["correct"] += 1

        video_name = query_image.split('/')
        video_name = '/'.join(video_name[-5:-1])
        print(video_name)
        
        result_dict[video_name].append((y_label, y_pred, verdict))
 
    if total > 0:
        print(f"Overall accuracy: {correct/total:.4f} ({correct}/{total})")
    else:
        print("No valid samples found.")

    for lbl, stats in per_class.items():
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"{lbl}: {acc:.4f} ({stats['correct']}/{stats['total']})")

    for video, frames in result_dict.items():
        label_sum = 0
        pred_accum = 0
        num_frames = len(frames)

        for frame in frames:
            label_sum += frame[0]  
            pred_accum += frame[1]  

        new_label.append(label_sum / num_frames)
        new_pred.append(pred_accum / num_frames)

    new_label = np.array(new_label)
    new_pred = np.array(new_pred)

    # 计算视频级别AUC
    fpr, tpr, _ = metrics.roc_curve(new_label, new_pred, pos_label=1)
    v_auc = metrics.auc(fpr, tpr)


    return v_auc


def get_test_metrics(jsonl_data):
    
    y_pred = [item['y_pred'] for item in jsonl_data]
    y_true = [item['y_label'] for item in jsonl_data]
    verdict = [item['verdict'] for item in jsonl_data]
    label = [item['label'] for item in jsonl_data]

    # 计算帧级AUC和Acc
    frame_auc = compute_frame_level_metrics(np.array(y_pred), np.array(y_true), np.array(verdict), np.array(label))

    # 计算视频级AUC和Acc
    v_auc = compute_video_level_auc(jsonl_data)

    result = {
        'frame_auc': frame_auc,
        'video_auc': v_auc,
    }   
    

    return result

file_path = '/ms-swift/infer_jsons/FFIW.jsonl'  
jsonl_data = read_jsonl(file_path)
metrics_result = get_test_metrics(jsonl_data)

print(f"Frame-level AUC: {metrics_result['frame_auc']}")
print(f"Video-level AUC: {metrics_result['video_auc']}")
