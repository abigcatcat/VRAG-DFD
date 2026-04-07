import json
import os

input_file = "/Youtu_Pangu_Security_Public_cq11/huihan/projects/ms-swift/anno/metadata_readable.json"
output_dir = "split_by_label"

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 读取 json 文件
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 用字典按 label 分组
groups = {}

for item in data:
    label = item.get("label")
    if label is None:
        continue
    groups.setdefault(label, []).append(item)

# 将每个 label 写出为一个文件
for label, items in groups.items():
    output_path = os.path.join(output_dir, f"label_{label}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

print(f"完成！总共生成 {len(groups)} 个文件，保存在目录：{output_dir}")
