# VRAG-DFD: Verifiable Retrieval-Augmentation for MLLM-based Deepfake Detection

VRAG-DFD is a framework that introduces Verifiable Retrieval-Augmented Generation (RAG) into the Deepfake Detection (DFD) domain. By combining professional forensic knowledge retrieval with Reinforcement Learning (GRPO), we empower Multi-modal Large Language Models (MLLMs) to perform expert-level forensic analysis with critical reasoning.

<a href="https://arxiv.org/pdf/2504.04907" alt="paper"><img src="https://img.shields.io/badge/ArXiv-2504.04907-cc0000.svg?style=flat" /></a>
[![Huggingface Checkpoint](https://img.shields.io/badge/%F0%9F%A4%97-Huggingface%20Checkpoint-0070BA?labelColor=555555)](https://huggingface.co/abigcatcat/VRAG-DFD)


#Overview
<div align=center><img src="https://github.com/abigcatcat/VRAG-DFD/blob/main/asset/VRAG-DFD.png"/></div>

## Contents
* [Phase 1: Deepfake RAG](#phase-1-deepfake_rag)
    * [Installation](#installation)
    * [Build the Forensic Database](#build-the-forensic-database)
    * [Generate Offline Retrieval Results](#generate-offline-retrieval-results)
* [Phase 2: MLLM Training & Evaluation](#phase-2-mllm-training--evaluation)
    * [Installation](#installation-1)
    * [Training Pipeline](#training-pipeline)
    * [Inference & Evaluation](#inference--evaluation)
* [Citation](#citation)

# Phase 1: deepfake_RAG
This phase involves building the Forensic Knowledge Database (FKD) and generating retrieval results for test sets.

## Installation
````bash
   git clone https://github.com/abigcatcat/VRAG-DFD.git
   cd VRAG-DFD/deepfake_RAG
   pip install -r requirements.txt
   ````

## Build the Forensic Database
We use the annotated FF++ dataset to create the vector database.
1. Configure your paths and parameters in `deepfake_RAG/config.py`:

```python
DATASET_CONFIG = {
    'json_path': "/path/to/your/annotations/rag_anno.json",  # 标注文件路径
    'image_root': '/path/to/your/images',  # 图像根目录
    'max_images_per_video': 32,  # 每视频采样帧数
    'batch_size': 32,
}

MODEL_CONFIG = {
    'model_path': '/path/to/effort_model.pth',  # 预训练模型路径
}
```
2. Run the build command:
````bash
python demo.py --mode build
````

## Generate Offline Retrieval Results
Retrieve forgery evidence for public datasets (e.g., Celeb-DF v1/v2) to generate JSON files for the MLLM.
1. Update parameters in `deepfake_RAG/config.py`:
```python
DATASET_CONFIG = {
    'json_path': "/path/to/test_dataset.json",  # 测试数据集JSON
    'image_root': '/path/to/test/images',  # 测试图像目录
}

DATABASE_CONFIG = {
    'save_dir': './deepfake_rag_database',  # 已构建的数据库路径
}
```
2. Run the build command:
````bash
python demo.py --mode test --output_file <output_path/filename>.json
````

# Phase 2: MLLM Training & Evaluation
The training follows a three-stage progressive strategy to cultivate critical reasoning.

## Installation
````bash
   cd VRAG-DFD
   pip install -r requirements.txt
   ````

1. Training Pipeline
The training data is located in datasets_json/, and execution scripts are in scripts_run/.

| Stage | Training Data | Execution Script | Description |
| :--- | :--- | :--- | :--- |
| **Stage 1** | `stage1_train.json` | `bash scripts_run/stage1.sh` | Visual Alignment |
| **Stage 2** | `rag_finetune_data.json` | `bash scripts_run/stage2.sh` | Forensic SFT |
| **Stage 3** | `rag_grpo_data.json` | `bash scripts_run/stage3.sh` | Critical RL (GRPO) |

2. Inference & Evaluation
Step A: Format Processing
Convert Phase 1 retrieval results into the training-compatible format:
````bash
python utils/process.py --input <path_to_retrieval_json> --output <formatted_json>
````

Step B: Batch Inference
Run the batch evaluation script:
````bash
bash scripts_run/eval_batch.sh
````

Step C: Metrics Calculation
Calculate AUC and Acc:
````bash
python utils/get_metrics.py --results <prediction_file>.json
````

# Citation
If you use our dataset, code or find VRAG-DFD useful, please cite our paper in your work as:

```bib

```

