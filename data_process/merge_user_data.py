"""
合并用户数据脚本

从多个batch结果文件或user_data.json文件合并生成最终的.user.json文件。
支持根据preference_history_end和target_item_pos区分train/test模式。

使用方法:
    python merge_user_data.py \
        --dataset Instruments \
        --root ./data \
        --input_dir ./data/Instruments_batch_results \
        --preference_history_end_train -3 \
        --target_item_pos_train -3 \
        --preference_history_end_test -3 \
        --target_item_pos_test -1
"""

import argparse
import os
import json
import glob
from collections import defaultdict
from typing import Dict, List

from utils import load_json, write_json_file


def load_user_data_from_batch_files(input_dir: str) -> List[Dict]:
    """
    从batch结果文件或user_data.json加载用户数据
    
    Args:
        input_dir: 包含batch结果文件或user_data.json的目录
    
    Returns:
        用户数据列表
    """
    # 优先加载user_data.json（如果存在）
    user_data_file = os.path.join(input_dir, "user_data.json")
    if os.path.exists(user_data_file):
        print(f"Loading user data from: {user_data_file}")
        return load_json(user_data_file)
    
    # 如果user_data.json不存在，提示用户
    print(f"Error: user_data.json not found in {input_dir}")
    print("Please run vllm_batch_inference.py first to generate user_data.json")
    return []


def merge_user_data_to_final_format(
    user_data_list: List[Dict],
    preference_history_end_train: int = -3,
    target_item_pos_train: int = -3,
    preference_history_end_test: int = -3,
    target_item_pos_test: int = -1
) -> Dict:
    """
    合并用户数据到最终格式
    
    Args:
        user_data_list: 用户数据列表（包含位置信息）
        preference_history_end_train: train模式的偏好历史终点位置
        target_item_pos_train: train模式的目标商品位置
        preference_history_end_test: test模式的偏好历史终点位置
        target_item_pos_test: test模式的目标商品位置
    
    Returns:
        最终的用户数据字典
    """
    user_dict = {
        "user_explicit_preference": {},
        "user_vague_intention": {
            "train": {},
            "test": {}
        }
    }
    
    # 按位置信息分类数据
    train_data = []
    test_data = []
    
    for data in user_data_list:
        pref_end = data.get("preference_history_end")
        item_pos = data.get("target_item_pos")
        
        # 判断是train还是test
        if (pref_end == preference_history_end_train and 
            item_pos == target_item_pos_train):
            train_data.append(data)
        elif (pref_end == preference_history_end_test and 
              item_pos == target_item_pos_test):
            test_data.append(data)
        else:
            # 如果位置不匹配，根据target_item_pos判断（-3通常是train，-1通常是test）
            if item_pos == target_item_pos_train:
                train_data.append(data)
            elif item_pos == target_item_pos_test:
                test_data.append(data)
    
    # 处理训练集数据（preference从train数据中提取，但每个用户只保存一次）
    for data in train_data:
        user = data["user"]
        user_dict["user_explicit_preference"][user] = [
            data["general_preference"],
            data["long_term_preference"],
            data["short_term_preference"]
        ]
        user_dict["user_vague_intention"]["train"][user] = {
            "item": data["item"],
            "inters": data["inters"],
            "querys": [data["user_related_intention"], data["item_related_intention"]]
        }
    
    # 处理测试集数据（preference已经在train中设置，这里只设置intention）
    for data in test_data:
        user = data["user"]
        # 如果用户不在训练集中，也设置preference
        if user not in user_dict["user_explicit_preference"]:
            user_dict["user_explicit_preference"][user] = [
                data["general_preference"],
                data["long_term_preference"],
                data["short_term_preference"]
            ]
        
        user_dict["user_vague_intention"]["test"][user] = {
            "item": data["item"],
            "inters": data["inters"],
            "querys": [data["user_related_intention"], data["item_related_intention"]]
        }
    
    return user_dict


def parse_args():
    parser = argparse.ArgumentParser(description='合并用户数据脚本')
    parser.add_argument('--dataset', type=str, required=True, help='数据集名称，如Instruments')
    parser.add_argument('--root', type=str, default='', help='数据根目录')
    parser.add_argument('--input_dir', type=str, required=True, 
                       help='输入目录，包含batch结果文件或user_data.json')
    parser.add_argument('--output_file', type=str, help='输出文件路径（.user.json格式）')
    parser.add_argument('--preference_history_end_train', type=int, default=-3,
                       help='train模式的偏好历史终点位置（如-3表示[:-3]）')
    parser.add_argument('--target_item_pos_train', type=int, default=-3,
                       help='train模式的目标商品位置（如-3表示list[-3]）')
    parser.add_argument('--preference_history_end_test', type=int, default=-3,
                       help='test模式的偏好历史终点位置（如-3表示[:-3]）')
    parser.add_argument('--target_item_pos_test', type=int, default=-1,
                       help='test模式的目标商品位置（如-1表示list[-1]）')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 设置路径
    if args.root:
        args.root = os.path.join(args.root, args.dataset)
    else:
        args.root = args.dataset
    
    # 加载用户数据
    print(f"\n{'='*60}")
    print("Loading user data...")
    print(f"{'='*60}")
    
    user_data_list = load_user_data_from_batch_files(args.input_dir)
    
    if not user_data_list:
        print(f"Error: No user data found in {args.input_dir}")
        return
    
    print(f"Loaded {len(user_data_list)} user data entries")
    
    # 合并数据
    print(f"\n{'='*60}")
    print("Merging user data to final format...")
    print(f"{'='*60}")
    print(f"Train: preference_history_end={args.preference_history_end_train}, target_item_pos={args.target_item_pos_train}")
    print(f"Test: preference_history_end={args.preference_history_end_test}, target_item_pos={args.target_item_pos_test}")
    
    user_dict = merge_user_data_to_final_format(
        user_data_list,
        args.preference_history_end_train,
        args.target_item_pos_train,
        args.preference_history_end_test,
        args.target_item_pos_test
    )
    
    # 确定输出文件路径
    if args.output_file:
        output_file = args.output_file
    else:
        output_file = os.path.join(args.root, f'{args.dataset}.user.json')
    
    # 保存最终文件
    write_json_file(user_dict, output_file)
    print(f"\nSuccessfully generated {output_file}")
    print(f"  - Total users with preferences: {len(user_dict['user_explicit_preference'])}")
    print(f"  - Train samples: {len(user_dict['user_vague_intention']['train'])}")
    print(f"  - Test samples: {len(user_dict['user_vague_intention']['test'])}")


if __name__ == "__main__":
    main()

