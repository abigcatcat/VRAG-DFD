import os
import json
import pickle
from pathlib import Path
import numpy as np
import yaml
import random
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn.functional as F
from PIL import Image
import faiss
from torchvision import transforms
from collections import defaultdict
import logging
from tqdm import tqdm
from registry import DETECTOR

from effort_detector import EffortDetector
# from clip_large_lora_detector import CLIP_Large_LoRA_Detector as EffortDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """从预训练的effort模型中提取图像特征"""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = device
        self.model = self.load_model(model_path)
        self.transform = self.get_transform()
        
    def load_model(self, model_path: str) -> EffortDetector:
        """加载预训练的effort模型"""
        config = {}
        model = EffortDetector(config)
        if model_path:
            # model.eval()
            ckpt = torch.load(model_path, map_location=self.device) # module.
            new_state_dict = {k.replace('module.', ''): v 
                    for k, v in ckpt.items()
                    if k.startswith('module.')}
            model.load_state_dict(new_state_dict, strict=False)
            print('===> Load checkpoint done!')
        else:
            print('Fail to load the pre-trained weights')
            
        model.to(self.device)
        model.eval()
        return model
    
    def get_transform(self):
        """获取图像预处理变换"""
        # CLIP模型的标准预处理
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
    
    def extract_features(self, image_path: str) -> torch.Tensor:
        """从单张图像提取特征"""
        try:
            # 加载和预处理图像
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 提取特征
            with torch.no_grad():
                data_dict = {'image': image_tensor}
                features = self.model.features(data_dict)  # 获取1024维特征
                
            return features.cpu().squeeze(0)  # 返回1维tensor
            
        except Exception as e:
            logger.error(f"Error extracting features from {image_path}: {e}")
            return torch.zeros(1024)  # 返回零向量作为fallback
    
    def extract_batch_features(self, image_paths: List[str], batch_size: int = 32) -> torch.Tensor:
        """批量提取特征"""
        all_features = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features"):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            
            # 加载批次图像
            for path in batch_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    image_tensor = self.transform(image)
                    batch_images.append(image_tensor)
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")
                    batch_images.append(torch.zeros(3, 224, 224))
            
            # 批量处理
            if batch_images:
                batch_tensor = torch.stack(batch_images).to(self.device)
                
                with torch.no_grad():
                    data_dict = {'image': batch_tensor}
                    features = self.model.features(data_dict)
                    all_features.append(features.cpu())
        
        if all_features:
            return torch.cat(all_features, dim=0)
        else:
            return torch.empty(0, 1024)


class DeepfakeRAGDatabase:
    """深度伪造检测的RAG数据库"""
    
    def __init__(self, feature_dim: int = 1024):
        self.feature_dim = feature_dim
        self.index = None
        self.metadata = []  # 存储每个向量对应的元数据
        self.label_stats = defaultdict(int)
        
    def build_database(self, features: torch.Tensor, metadata_list: List[Dict]):
        """构建FAISS数据库"""
        assert len(features) == len(metadata_list), "Features and metadata length mismatch"
        
        # 转换为numpy数组并标准化
        features_np = features.numpy().astype('float32')
        features_np = features_np / np.linalg.norm(features_np, axis=1, keepdims=True)
        
        # 创建FAISS索引
        self.index = faiss.IndexFlatIP(self.feature_dim)  # 内积相似度
        self.index.add(features_np)
        
        # 存储元数据
        self.metadata = metadata_list
        
        # 统计标签分布
        for meta in metadata_list:
            self.label_stats[meta['label']] += 1
            
        logger.info(f"Database built with {len(features)} vectors")
        logger.info(f"Label distribution: {dict(self.label_stats)}")
    
    def search(self, query_features: torch.Tensor, k: int = 10) -> Tuple[List[float], List[Dict]]:
        """搜索最相似的k个样本"""
        if self.index is None:
            raise ValueError("Database not built yet")
        
        # 标准化查询特征
        query_np = query_features.numpy().astype('float32').reshape(1, -1)
        query_np = query_np / np.linalg.norm(query_np)
        
        # 搜索
        similarities, indices = self.index.search(query_np, k)
        
        # 返回结果
        results_metadata = [self.metadata[idx] for idx in indices[0]]
        results_similarities = similarities[0].tolist()
        
        return results_similarities, results_metadata
    
    def search_with_features(self, query_features: torch.Tensor, k: int = 10) -> Tuple[List[float], torch.Tensor, List[Dict]]:
        """搜索最相似的k个样本，返回相似度、特征向量和元数据"""
        if self.index is None:
            raise ValueError("Database not built yet")
        
        # 标准化查询特征
        query_np = query_features.numpy().astype('float32').reshape(1, -1)
        query_np = query_np / np.linalg.norm(query_np)
        
        # 搜索
        similarities, indices = self.index.search(query_np, k)
        
        # 直接从FAISS索引获取特征向量（避免重新提取）
        retrieved_features = []
        for idx in indices[0]:
            feat_vector = self.index.reconstruct(int(idx))  # 从索引中重构特征
            retrieved_features.append(feat_vector)
        
        # 转换为tensor
        retrieved_features = torch.tensor(np.array(retrieved_features), dtype=torch.float32)  # [k, 1024]
        
        # 返回结果
        results_metadata = [self.metadata[idx] for idx in indices[0]]
        results_similarities = similarities[0].tolist()
        
        return results_similarities, retrieved_features, results_metadata
    
    def save_database(self, save_dir: str):
        """保存数据库"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存FAISS索引
        faiss.write_index(self.index, os.path.join(save_dir, "faiss_index.bin"))
        
        # 保存元数据
        with open(os.path.join(save_dir, "metadata.pkl"), 'wb') as f:
            pickle.dump(self.metadata, f)
            
        # 保存统计信息
        with open(os.path.join(save_dir, "stats.json"), 'w') as f:
            json.dump({
                'feature_dim': self.feature_dim,
                'num_vectors': len(self.metadata),
                'label_stats': dict(self.label_stats)
            }, f, indent=2)
        
        with open(os.path.join(save_dir, "metadata_readable.json"), 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Database saved to {save_dir}")
    
    def load_database(self, save_dir: str):
        """加载数据库"""
        # 加载FAISS索引
        self.index = faiss.read_index(os.path.join(save_dir, "faiss_index.bin"))
        
        # 加载元数据
        with open(os.path.join(save_dir, "metadata.pkl"), 'rb') as f:
            self.metadata = pickle.load(f)
            
        # 加载统计信息
        with open(os.path.join(save_dir, "stats.json"), 'r') as f:
            stats = json.load(f)
            self.feature_dim = stats['feature_dim']
            self.label_stats = defaultdict(int, stats['label_stats'])
            
        logger.info(f"Database loaded from {save_dir}")


class MultimodalRAGSystem:
    """多模态RAG系统"""
    
    def __init__(self, model_path: str, database_dir: Optional[str] = None, device: str = 'cuda'):
        self.feature_extractor = FeatureExtractor(model_path, device)
        self.database = DeepfakeRAGDatabase()
        
        if database_dir and os.path.exists(database_dir):
            self.database.load_database(database_dir)
            logger.info("Existing database loaded")
        else:
            logger.info("No existing database found, will need to build new one")
    
    def build_database_from_json(self, json_path: str, image_root: str, save_dir: str, 
                                max_images_per_video: int = 8):
        """从FF++的JSON文件构建数据库"""
        logger.info("Loading FaceForensics++ metadata...")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        all_image_paths = []
        all_metadata = []
        
        # 遍历所有类别
        ff_data = data['FaceForensics++']
        for category_name, category_data in ff_data.items():
            logger.info(f"Processing category: {category_name}")
            
            # 遍历训练集
            if 'train' in category_data:
                train_data = category_data['train']['c23']
                
                for video_id, video_info in tqdm(train_data.items(), 
                                               desc=f"Processing {category_name}"):
                    frames = video_info['frames']
                    label = video_info['label']
                    
                    # 均匀采样max_images_per_video帧
                    selected_frames = self._uniform_sample_frames(frames, max_images_per_video)
                    
                    for frame_path in selected_frames:
                        full_path = os.path.join(image_root, frame_path)
                        if os.path.exists(full_path):
                            all_image_paths.append(full_path)
                            all_metadata.append({
                                'image_path': full_path,
                                'label': label,
                                'category': category_name,
                                'video_id': video_id,
                                'frame_path': frame_path
                            })
        
        logger.info(f"Found {len(all_image_paths)} valid images")
        
        # 提取特征
        logger.info("Extracting features...")
        features = self.feature_extractor.extract_batch_features(all_image_paths)
        
        # 构建数据库
        logger.info("Building database...")
        self.database.build_database(features, all_metadata)
        
        # 保存数据库
        self.database.save_database(save_dir)
        
        return len(all_image_paths)
    
    def build_database_from_annojson(self, json_path: str, image_root: str, save_dir: str, 
                                max_images_per_video: int = 8):
        """从anno的JSON文件构建数据库"""
        logger.info("Loading anno metadata...")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        all_image_paths = []
        all_metadata = []
        
        for item in data:
            images_path = item['image_path']
            label = item['label']
            video_id = item['video_id']
            # original_path = item.get('original_images', "")
            annotation = item['annotation']
            if os.path.exists(images_path):
                all_image_paths.append(images_path)
                all_metadata.append({
                    'image_path': images_path,
                    # 'original_path': original_path,
                    'label': label,
                    'video_id': video_id,
                    'annotation': annotation
                })
            
        logger.info(f"Found {len(all_image_paths)} valid images")
        
        # 提取特征
        logger.info("Extracting features...")
        features = self.feature_extractor.extract_batch_features(all_image_paths)
        
        # 构建数据库
        logger.info("Building database...")
        self.database.build_database(features, all_metadata)
        
        # 保存数据库
        self.database.save_database(save_dir)
        
        return len(all_image_paths)
    
    def _uniform_sample_frames(self, frames: List[str], num_samples: int) -> List[str]:
        """从帧列表中均匀采样指定数量的帧"""
        total_frames = len(frames)
        if total_frames <= num_samples:
            return frames
        
        # 计算采样间隔
        indices = []
        step = total_frames / num_samples
        for i in range(num_samples):
            idx = int(i * step)
            indices.append(idx)
        
        return [frames[i] for i in indices]

    
    # def query_similar_images(self, query_image_path: str, k: int = 10) -> Dict:
    #     """查询相似图像"""
    #     # 提取查询图像特征
    #     path_parts = query_image_path.split('/')
    #     part_to_compare = '/'.join(path_parts[-5:-1])  # 提取用于比较的路径部分
    #     query_features = self.feature_extractor.extract_features(query_image_path)
        
    #     # 搜索相似图像
    #     similarities, metadata_list = self.database.search(query_features, k)
        
    #     # 保留与查询路径相同的图像
    #     same_path_items = []
    #     other_items = []

    #     for sim, item in zip(similarities, metadata_list):
    #         image_path_parts = item['image_path'].split('/')
    #         image_part_to_compare = '/'.join(image_path_parts[-5:-1])
            
    #         # 如果路径相同，加入 same_path_items，否则加入 other_items
    #         if image_part_to_compare == part_to_compare:
    #             same_path_items.append((sim, item))  # 保存相似度和条目
    #         else:
    #             other_items.append((sim, item))

    #     # 对与查询图像路径相同的条目随机选择 2 个
    #     same_path_items = random.sample(same_path_items, min(2, len(same_path_items)))

    #     # 对与查询图像路径不同的条目按相似度排序，选择 k-2 个
    #     other_items = sorted(other_items, key=lambda x: x[0], reverse=True)[:(5 - len(same_path_items))]

    #     # 合并结果
    #     final_items = same_path_items + other_items
        
    #     # 按相似度对最终结果进行排序
    #     final_items = sorted(final_items, key=lambda x: x[0], reverse=True)
        
    #     # 提取图像路径并保证返回的数量为 k
    #     final_metadata_list = [item for _, item in final_items][:5]
    #     final_similarities = [sim for sim, _ in final_items][:5]
        
    #     # 分析结果
    #     label_counts = defaultdict(int)
        
    #     for meta in final_metadata_list:
    #         label_counts[meta['label']] += 1
        
    #     # 预测结果（基于top-k投票）
    #     predicted_label = max(label_counts.items(), key=lambda x: x[1])[0]
    #     confidence = label_counts[predicted_label] / k
        
    #     return {
    #         'query_image': query_image_path,
    #         'predicted_label': predicted_label,
    #         'confidence': confidence,
    #         'top_k_results': [
    #             {
    #                 'similarity': sim,
    #                 'image_path': meta['image_path'],
    #                 'original_path': meta['original_path'],
    #                 'label': meta['label'],
    #                 'video_id': meta['video_id'],
    #                 'annotation': meta['annotation']
    #             }
    #             for sim, meta in zip(final_similarities, final_metadata_list)
    #         ],
    #         'label_distribution': dict(label_counts)
    #     }
    
    def query_similar_images(self, query_image_path: str, k: int = 10) -> Dict:
        """查询相似图像"""
        # 提取查询图像特征
        path_parts = query_image_path.split('/')
        part_to_compare = '/'.join(path_parts[-4:-1])
        query_features = self.feature_extractor.extract_features(query_image_path)
        
        # 搜索相似图像
        similarities, metadata_list = self.database.search(query_features, k)
        
        # 分析结果
        label_counts = defaultdict(int)
        category_counts = defaultdict(int)
        
        for meta in metadata_list:
            label_counts[meta['label']] += 1
            # category_counts[meta['category']] += 1
        
        # 预测结果（基于top-k投票）
        predicted_label = max(label_counts.items(), key=lambda x: x[1])[0]
        confidence = label_counts[predicted_label] / k
        
        return {
            'query_image': query_image_path,
            'predicted_label': predicted_label,
            'confidence': confidence,
            'top_k_results': [
                {
                    'similarity': sim,
                    'image_path': meta['image_path'],
                    'original_path': meta['original_path'],
                    'label': meta['label'],
                    # 'category': meta['category'],
                    'video_id': meta['video_id'],
                    'annotation': meta['annotation']
                }
                for sim, meta in zip(similarities, metadata_list)
            ],
            
        }
        
    
    def test_on_ff_test_set(self, json_path: str, image_root: str, k: int = 10, 
                           output_file: str = None) -> Dict:
        """在FF++的test子集上测试检索性能"""
        
        ###记得改数据集名字，以及测试集、验证集
        logger.info("Loading FaceForensics++ test set...")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        test_results = []
        category_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
        
        # 遍历所有类别的test子集
        # ff_data = data['FaceForensics++']
        # ff_data = data['DeepFakeDetection']
        # ff_data = data['Celeb-DF-v2']
        # ff_data = data['Celeb-DF-v1']
        # ff_data = data['DFDC']
        # ff_data = data['test_FFIW']
        # ff_data = data['test_WDF']
        
        # 1. 获取文件名（去掉路径）
        file_name = os.path.basename(json_path)      # Celeb-DF-v2.json

        # 2. 去掉后缀 .json 得到 key
        dataset_key = os.path.splitext(file_name)[0]  # Celeb-DF-v2

        # 3. 动态读取 ff_data
        ff_data = data.get(dataset_key)

        if ff_data is None:
            raise ValueError(f"无法在 JSON 文件中找到 key: {dataset_key}")

        print(f"成功获取 ff_data，key = {dataset_key}")
        
        
        for category_name, category_data in ff_data.items():
            logger.info(f"Testing category: {category_name}")
            
            # 遍历测试集
            if 'test' in category_data:
                # test_data = category_data['test']['c23']
                test_data = category_data['test']
                
                for video_id, video_info in tqdm(test_data.items(), 
                                               desc=f"Testing {category_name}"):
                    frames = video_info['frames']
                    true_label = video_info['label']
                    # 从每个测试视频均匀采样32帧进行测试
                    selected_frames = self._uniform_sample_frames(frames, 32)
                    
                    image_root_clean = image_root.replace("\\", "/").rstrip('/') + '/'

                    for frame_path in selected_frames:
                        
                        frame_path_clean = frame_path.replace("\\", "/")

                        if frame_path_clean.startswith(image_root_clean):
                            relative_path = frame_path_clean[len(image_root_clean):]
                        else:
                            relative_path = frame_path_clean

                        full_path = os.path.join(image_root, relative_path)
                        
                        if os.path.exists(full_path):
                            try:
                                # 查询相似图像
                                result = self.query_similar_images(full_path, k)
                                predicted_label = result['predicted_label']
                                confidence = result['confidence']
                                
                                is_correct = (predicted_label == true_label)
                                
                                # 统计结果
                                category_stats[category_name]['total'] += 1
                                if is_correct:
                                    category_stats[category_name]['correct'] += 1
                                
                                test_results.append({
                                    'image_path': full_path,
                                    'true_label': true_label,
                                    'predicted_label': predicted_label,
                                    'confidence': confidence,
                                    'correct': is_correct,
                                    'category': category_name,
                                    'video_id': video_id,
                                    'frame_path': frame_path,
                                    'top_k_results': result['top_k_results']
                                })
                                
                            except Exception as e:
                                logger.warning(f"Failed to process {full_path}: {e}")
        
        # 计算整体统计
        total_samples = len(test_results)
        
        # 构建完整结果
        final_results = {
            'total_samples': total_samples,
            'k_value': k,
            'detailed_results': test_results
        }
        
        # 保存结果到JSON文件
        if output_file:
             # 获取输出文件的目录路径
            output_dir = os.path.dirname(output_file)

            # 如果目录不存在，逐级创建目录
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Test results saved to {output_file}")
        
        return final_results

    def test_on_ff_auto_set(self, json_path: str, image_root: str, k: int = 10,
                           output_file: str = None) -> Dict:
        """在FF++的test子集上测试检索性能"""
        logger.info("Loading FaceForensics++ test set...")
        test_results = []
        category_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
        
        with open(json_path, 'r') as f:
            for line in f:
                items = json.loads(line.strip())
    
                image_path = items.get("images", [])[0]
                if image_path:
                # 从路径中提取类别和视频 ID
                    path_parts = image_path.split('/')
                    category = path_parts[-5]  # 倒数第五项是类别
                    video_id = path_parts[-2]  # 倒数第二项是视频ID
                if category == 'youtube':
                    true_label = 'FF-real'
                elif category == 'Deepfakes':
                    true_label = 'FF-DF'
                elif category == 'FaceSwap':
                    true_label = 'FF-FS'
                elif category == 'Face2Face':
                    true_label = 'FF-F2F'
                elif category == 'NeuralTextures':
                    true_label = 'FF-NT'
                
                result = self.query_similar_images(image_path, k)
                predicted_label = result['predicted_label']
                confidence = result['confidence']
                                    
                is_correct = (predicted_label == true_label)
                
                # 统计结果
                category_stats[category]['total'] += 1
                if is_correct:
                    category_stats[category]['correct'] += 1

                test_results.append({
                    'image_path': image_path,
                    'true_label': true_label,
                    'predicted_label': predicted_label,
                    'confidence': confidence,
                    'correct': is_correct,
                    'category': category,
                    'video_id': video_id,
                    'top_k_results': result['top_k_results']
                })
            
        # 计算整体统计
        total_samples = len(test_results)
        
        # 构建完整结果
        final_results = {
            'total_samples': total_samples,
            'k_value': k,
            'detailed_results': test_results
        }
        
        # 保存结果到JSON文件
        if output_file:
            # 获取输出文件的目录路径
            output_dir = os.path.dirname(output_file)

            # 如果目录不存在，逐级创建目录
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Test results saved to {output_file}")
        
        return final_results


def main():
    """主程序示例"""
    # 配置参数
    MODEL_PATH = "path/to/your/effort_model.pth"  # 您的预训练模型路径
    JSON_PATH = "FaceForensics++.json"
    IMAGE_ROOT = "path/to/your/image/root"  # 图像文件的根目录
    DATABASE_DIR = "deepfake_rag_database"
    
    # 创建RAG系统
    rag_system = MultimodalRAGSystem(MODEL_PATH, DATABASE_DIR)
    
    # 如果数据库不存在，则构建数据库
    if not os.path.exists(DATABASE_DIR):
        logger.info("Building database from FaceForensics++ dataset...")
        num_images = rag_system.build_database_from_json(
            JSON_PATH, IMAGE_ROOT, DATABASE_DIR, max_images_per_video=5
        )
        logger.info(f"Database built with {num_images} images")
    
    # 查询示例
    query_image = "path/to/test/image.jpg"
    if os.path.exists(query_image):
        result = rag_system.query_similar_images(query_image, k=10)
        
        print(f"Query Image: {result['query_image']}")
        print(f"Predicted Label: {result['predicted_label']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Label Distribution: {result['label_distribution']}")
        
        print("\nTop-5 Similar Images:")
        for i, item in enumerate(result['top_k_results'][:5]):
            print(f"{i+1}. Similarity: {item['similarity']:.3f}, "
                  f"Label: {item['label']}, Path: {item['image_path']}")


if __name__ == "__main__":
    main()