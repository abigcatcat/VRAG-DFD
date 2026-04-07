#!/usr/bin/env python3
"""
多模态RAG系统演示程序
"""

import os
import argparse
import logging
from typing import List, Tuple
import torch

from multimodal_rag import MultimodalRAGSystem
from config import MODEL_CONFIG, DATASET_CONFIG, DATABASE_CONFIG, RAG_CONFIG, LOGGING_CONFIG


def setup_logging(log_file: str = None, level: str = 'INFO'):
    """设置日志配置"""
    log_level = getattr(logging, level.upper())
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def print_performance_summary(performance_data: dict):
    """打印性能摘要"""
    print("=" * 50)
    print("PERFORMANCE SUMMARY")
    print("=" * 50)
    
    if 'accuracy' in performance_data:
        print(f"Overall Accuracy: {performance_data['accuracy']:.3f}")
    
    if 'total_samples' in performance_data:
        print(f"Total Samples: {performance_data['total_samples']}")
    
    if 'correct_predictions' in performance_data:
        print(f"Correct Predictions: {performance_data['correct_predictions']}")
    
    # 显示各种k值的准确率
    for key, value in performance_data.items():
        if key.startswith('accuracy@'):
            print(f"{key}: {value:.3f}")
    
    print("=" * 50)


def save_results_to_json(results: dict, file_path: str):
    """保存结果到JSON文件"""
    import json
    import numpy as np
    
    # 处理numpy类型，使其可以序列化
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        return obj
    
    # 递归转换所有numpy对象
    def recursive_convert(data):
        if isinstance(data, dict):
            return {key: recursive_convert(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [recursive_convert(item) for item in data]
        else:
            return convert_numpy(data)
    
    converted_results = recursive_convert(results)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(converted_results, f, indent=2, ensure_ascii=False)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Multimodal RAG for Deepfake Detection')
    
    parser.add_argument('--mode', type=str, choices=['build', 'query', 'evaluate', 'test', 'grpo'], 
                       default='query', help='运行模式')
    
    parser.add_argument('--model_path', type=str, 
                       default=MODEL_CONFIG['model_path'],
                       help='预训练effort模型路径')
    
    parser.add_argument('--json_path', type=str, 
                       default=DATASET_CONFIG['json_path'],
                       help='FaceForensics++ JSON文件路径')
    
    parser.add_argument('--image_root', type=str, 
                       default=DATASET_CONFIG['image_root'],
                       help='图像根目录')
    
    parser.add_argument('--database_dir', type=str, 
                       default=DATABASE_CONFIG['save_dir'],
                       help='数据库保存目录')
    
    parser.add_argument('--query_image', type=str, 
                       help='待查询的图像路径')
    
    parser.add_argument('--test_images_file', type=str,
                       help='测试图像列表文件（用于评估）')
    
    parser.add_argument('--k', type=int, default=RAG_CONFIG['default_k'],
                       help='检索的top-k数量')
    
    parser.add_argument('--max_images_per_video', type=int, 
                       default=DATASET_CONFIG['max_images_per_video'],
                       help='每个视频均匀采样帧数')
    
    parser.add_argument('--batch_size', type=int, 
                       default=DATASET_CONFIG['batch_size'],
                       help='批处理大小')
    
    parser.add_argument('--output_file', type=str, 
                       help='结果输出文件路径')
    
    return parser.parse_args()


def build_database_mode(args):
    """构建数据库模式"""
    logger = logging.getLogger(__name__)
    logger.info("开始构建RAG数据库...")
    
    # 检查必要的路径
    if not os.path.exists(args.json_path):
        raise FileNotFoundError(f"JSON文件不存在: {args.json_path}")
    
    if not os.path.exists(args.image_root):
        raise FileNotFoundError(f"图像根目录不存在: {args.image_root}")
    
    # 创建RAG系统
    rag_system = MultimodalRAGSystem(args.model_path, device=MODEL_CONFIG['device'])
    
    # 构建数据库
    # num_images = rag_system.build_database_from_json(
    #     args.json_path, 
    #     args.image_root, 
    #     args.database_dir,
    #     max_images_per_video=args.max_images_per_video
    # )
    
    num_images = rag_system.build_database_from_annojson(
        args.json_path, 
        args.image_root, 
        args.database_dir,
        max_images_per_video=args.max_images_per_video
    )
    
    logger.info(f"数据库构建完成，共处理 {num_images} 张图像")
    print(f"✅ 数据库已保存到: {args.database_dir}")


def query_mode(args):
    """查询模式"""
    logger = logging.getLogger(__name__)
    
    if not args.query_image:
        raise ValueError("查询模式需要指定 --query_image 参数")
    
    if not os.path.exists(args.query_image):
        raise FileNotFoundError(f"查询图像不存在: {args.query_image}")
    
    # 创建RAG系统
    rag_system = MultimodalRAGSystem(args.model_path, args.database_dir)
    
    logger.info(f"查询图像: {args.query_image}")
    
    # 执行查询
    result = rag_system.query_similar_images(args.query_image, k=args.k)
    
    # 显示结果
    print(f"\n🔍 查询结果:")
    print(f"📸 查询图像: {result['query_image']}")
    print(f"🏷️  预测标签: {result['predicted_label']}")
    print(f"📊 置信度: {result['confidence']:.3f}")
    print(f"📈 标签分布: {result['label_distribution']}")
    print(f"📂 类别分布: {result['category_distribution']}")
    
    print(f"\n🔝 Top-{min(5, args.k)} 相似图像:")
    for i, item in enumerate(result['top_k_results'][:5]):
        print(f"{i+1:2d}. 相似度: {item['similarity']:.3f} | "
              f"标签: {item['label']:12s} | "
              f"类别: {item['category']:8s} | "
              f"路径: {os.path.basename(item['image_path'])}")
    
    # 保存结果
    if args.output_file:
        save_results_to_json(result, args.output_file)
        print(f"💾 结果已保存到: {args.output_file}")


def evaluate_mode(args):
    """评估模式"""
    logger = logging.getLogger(__name__)
    
    if not args.test_images_file:
        raise ValueError("评估模式需要指定 --test_images_file 参数")
    
    if not os.path.exists(args.test_images_file):
        raise FileNotFoundError(f"测试图像文件不存在: {args.test_images_file}")
    
    # 读取测试图像列表
    test_images = []
    with open(args.test_images_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split('\t')
                if len(parts) >= 2:
                    image_path, true_label = parts[0], parts[1]
                    test_images.append((image_path, true_label))
    
    if not test_images:
        raise ValueError("测试图像列表为空")
    
    logger.info(f"加载了 {len(test_images)} 张测试图像")
    
    # 创建RAG系统
    rag_system = MultimodalRAGSystem(args.model_path, args.database_dir)
    
    # 执行评估
    performance = rag_system.analyze_detection_performance(test_images, k=args.k)
    
    # 显示结果
    print_performance_summary(performance)
    
    # 保存结果
    if args.output_file:
        save_results_to_json(performance, args.output_file)
        print(f"💾 评估结果已保存到: {args.output_file}")


def test_mode(args):
    """FF++测试集模式"""
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(args.json_path):
        raise FileNotFoundError(f"JSON文件不存在: {args.json_path}")
    
    if not os.path.exists(args.image_root):
        raise FileNotFoundError(f"图像根目录不存在: {args.image_root}")
    
    # 创建RAG系统
    rag_system = MultimodalRAGSystem(args.model_path, args.database_dir)
    
    logger.info(f"在FF++测试集上进行检索测试...")
    
    # 设置默认输出文件名
    if not args.output_file:
        args.output_file = f"ff_test_results_k{args.k}.json"
    
    # 执行测试
    results = rag_system.test_on_ff_test_set(
        args.json_path, 
        args.image_root, 
        k=args.k, 
        output_file=args.output_file
    )
    
    # 显示结果
    print(f"\n🎯 FF++测试集结果:")
    print(f"📊 整体准确率: {results['overall_accuracy']:.3f}")
    print(f"📈 总测试样本: {results['total_samples']}")
    print(f"✅ 正确预测: {results['correct_predictions']}")
    print(f"🔍 检索k值: {results['k_value']}")
    
    print(f"\n📂 各类别准确率:")
    for category, accuracy in results['category_accuracies'].items():
        total = results['category_stats'][category]['total']
        correct = results['category_stats'][category]['correct']
        print(f"  {category:12s}: {accuracy:.3f} ({correct}/{total})")
    
    print(f"\n💾 详细结果已保存到: {args.output_file}")
    
def auto_mode(args):
    """FF++测试集模式"""
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(args.json_path):
        raise FileNotFoundError(f"JSON文件不存在: {args.json_path}")
    
    if not os.path.exists(args.image_root):
        raise FileNotFoundError(f"图像根目录不存在: {args.image_root}")
    
    # 创建RAG系统
    rag_system = MultimodalRAGSystem(args.model_path, args.database_dir)
    
    logger.info(f"在FF++测试集上进行检索测试...")
    
    # 设置默认输出文件名
    if not args.output_file:
        args.output_file = f"ff_test_results_k{args.k}.json"
    
    # 执行测试
    results = rag_system.test_on_ff_auto_set(
        args.json_path, 
        args.image_root, 
        k=args.k, 
        output_file=args.output_file
    )
    
    # 显示结果
    print(f"\n🎯 FF++测试集结果:")
    print(f"📊 整体准确率: {results['overall_accuracy']:.3f}")
    print(f"📈 总测试样本: {results['total_samples']}")
    print(f"✅ 正确预测: {results['correct_predictions']}")
    print(f"🔍 检索k值: {results['k_value']}")
    
    print(f"\n📂 各类别准确率:")
    for category, accuracy in results['category_accuracies'].items():
        total = results['category_stats'][category]['total']
        correct = results['category_stats'][category]['correct']
        print(f"  {category:12s}: {accuracy:.3f} ({correct}/{total})")
    
    print(f"\n💾 详细结果已保存到: {args.output_file}")


def main():
    """主函数"""
    args = parse_arguments()
    
    # 设置日志
    setup_logging(LOGGING_CONFIG.get('log_file'), LOGGING_CONFIG.get('level', 'INFO'))
    logger = logging.getLogger(__name__)
    
    try:
        # 检查CUDA可用性
        if torch.cuda.is_available():
            logger.info(f"CUDA可用，使用GPU: {torch.cuda.get_device_name()}")
        else:
            logger.info("CUDA不可用，使用CPU")
        
        # 根据模式执行相应操作
        if args.mode == 'build':
            build_database_mode(args)
        elif args.mode == 'query':
            query_mode(args)
        elif args.mode == 'evaluate':
            evaluate_mode(args)
        elif args.mode == 'test':
            test_mode(args)
        elif args.mode == 'auto':
            auto_mode(args)
        else:
            raise ValueError(f"未知的运行模式: {args.mode}")
            
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        print(f"❌ 错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    main()