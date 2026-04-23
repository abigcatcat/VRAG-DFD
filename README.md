# VRAG-DFD: Verifiable Retrieval-Augmentation for MLLM-based Deepfake Detection

<div align="center">
  <a href="https://arxiv.org/pdf/2604.13660" alt="paper"><img src="https://img.shields.io/badge/ArXiv-2604.13660-cc0000.svg?style=flat" /></a>
  <a href="https://huggingface.co/abigcatcat/VRAG-DFD" alt="Huggingface Checkpoint"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Huggingface%20Checkpoint-0070BA?labelColor=555555" /></a>
</div>

VRAG-DFD is a framework that introduces Verifiable Retrieval-Augmented Generation (RAG) into the Deepfake Detection (DFD) domain. By combining professional forensic knowledge retrieval with Reinforcement Learning (GRPO), we empower Multi-modal Large Language Models (MLLMs) to perform expert-level forensic analysis with critical reasoning.



# Overview
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
    'json_path': "/path/to/yourjson",  # The corresponding JSON file for the dataset used to build the database
    'image_root': '/path/to/your/images',  # Root directory for images
    'max_images_per_video': 32,  # Number of sampled frames per video
    'batch_size': 32,
}

MODEL_CONFIG = {
    'model_path': '/path/to/pth',  # Path to the pre-trained retrievel model
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
    'json_path': "/path/to/test_dataset.json",  # The JSON file corresponding to the test set.
    'image_root': '/path/to/test/images',  # Directory for test images
}

DATABASE_CONFIG = {
    'save_dir': './deepfake_rag_database',  # Path to the constructed database
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
@article{han2026vragdfd,
  title={VRAG-DFD: Verifiable Retrieval-Augmentation for MLLM-based Deepfake Detection},
  author={Hui Han and Shunli Wang and Yandan Zhao and Taiping Yao and Shouhong Ding},
  journal={arXiv preprint arXiv:2604.13660},
  year={2026},
  url={https://arxiv.org/abs/2604.13660}
}
```

