"""
配置文件 - 多模态RAG系统的参数设置
"""
# 模型相关配置
MODEL_CONFIG = {
    'model_path': './ckpt_best.pth',  # 预训练effort模型路径
    'device': 'cuda' if True else 'cpu',  # 根据GPU可用性设置
    'feature_dim': 1024,  # 特征维度
}
# 数据集配置

DATASET_CONFIG = {
    'json_path': "./dataset_json/DF40.json",
    'image_root': './images',  # 图像根目录
    'max_images_per_video': 32,  # 每个视频均匀采样帧数
    'batch_size': 32,  # 批处理大小
}
# "/Youtu_Pangu_Security_Public/youtu-pangu-public/zhiyuanyan/DeepfakeBenchv2/preprocessing/dataset_json/DFDCP.json"
# 数据库配置
DATABASE_CONFIG = {
    'save_dir': '/deepfake_RAG/deepfake_rag_database',
    'index_type': 'IndexFlatIP',  # FAISS索引类型
}

# RAG系统配置
RAG_CONFIG = {
    'default_k': 5,  # 默认检索的top-k数量
}

# 日志配置
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': 'multimodal_rag.log'
}
