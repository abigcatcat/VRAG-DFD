# VRAG-DFD: Verifiable Retrieval-Augmentation for MLLM-based Deepfake Detection

VRAG-DFD is a framework that introduces Verifiable Retrieval-Augmented Generation (RAG) into the Deepfake Detection (DFD) domain. By combining professional forensic knowledge retrieval with Reinforcement Learning (GRPO), we empower Multi-modal Large Language Models (MLLMs) to perform expert-level forensic analysis with critical reasoning.

# Project Structure

```text
VRAG_DFD/
├── deepfake_RAG/           # Offline RAG Process
│   ├── config.py           # Configuration parameters
│   ├── demo.py               # Main entry point for the RAG system
│   ├── multimodal_rag.py   # RAG system implementation
│   └── effort_detector.py  # DFD detector/encoder implementation
├── ms-swift/               # ms-swift training framework
├── scripts_run/            # Training and inference scripts
│   ├── stage_1.sh          # Stage 1: Alignment training
│   ├── stage_2.sh          # Stage 2: Forensic SFT
│   ├── stage_3.sh          # Stage 3: Critical RL (GRPO)
│   ├── eval_batch.sh       # Batch evaluation script
│   └── eval.py             # Evaluation logic
├── utils/                  # Utility functions
│   ├── process.py          # Data format conversion (RAG JSON to Swift format)
│   └── get_metric.py       # Metrics calculation (AUC, EER, etc.)
├── annotations/            # Raw annotation files
└── datasets_jsons/         # JSON files for training and testing
```

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

