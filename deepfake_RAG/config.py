"""
Configuration File - Parameter settings for the Multimodal RAG system
"""

# Model-related configuration
MODEL_CONFIG = {
    'model_path': './ckpt_best.pth',  # Path to the pre-trained retrievel model
    'device': 'cuda' if True else 'cpu',  
    'feature_dim': 1024,  # Feature dimension
}

# Dataset configuration
DATASET_CONFIG = {
    'json_path': "./dataset_json/DF40.json",  #The corresponding JSON file for the dataset used to build the database, or the JSON file corresponding to the test set.
    'image_root': './images',  # Root directory for images
    'max_images_per_video': 32,  # Number of uniformly sampled frames per video
    'batch_size': 32,  # Batch size
}

# Database configuration
DATABASE_CONFIG = {
    'save_dir': '/deepfake_RAG/deepfake_rag_database',
    'index_type': 'IndexFlatIP',  # FAISS index type
}

# RAG system configuration
RAG_CONFIG = {
    'default_k': 5,  # Default number of top-k to retrieve
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': 'multimodal_rag.log'
}