"""
合并多个 prompt_part*_results.jsonl 文件为 user.json 格式
只使用 train 数据，test 部分设置为空
"""

import argparse
import os
import json
import re
import glob
from typing import Optional
from utils import write_json_file


# 默认值常量
DEFAULT_PREFERENCES = {
    "general_preference": "The user enjoys high-quality items.",
    "long_term_preference": "The user prefers high-quality items.",
    "short_term_preference": "The user has been focusing on quality items recently.",
    "user_related_intention": "I enjoy high-quality items.",
    "item_related_intention": "High-quality item with good features."
}


def extract_json_from_text(text: str) -> Optional[str]:
    """
    从文本中提取JSON字符串（可能包含markdown代码块或其他文本）
    
    Args:
        text: 可能包含JSON的文本
    
    Returns:
        提取的JSON字符串，如果提取失败返回None
    """
    if not text:
        return None
    
    # 尝试直接解析（如果整个文本就是JSON）
    text = text.strip()
    if text.startswith('{') and text.endswith('}'):
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass
    
    # 尝试从markdown代码块中提取
    # 匹配 ```json ... ``` 或 ``` ... ```
    json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # 尝试找到第一个 { 到最后一个 } 之间的内容
    start_idx = text.find('{')
    if start_idx != -1:
        # 从后往前找最后一个 }
        end_idx = text.rfind('}')
        if end_idx != -1 and end_idx > start_idx:
            json_candidate = text[start_idx:end_idx+1]
            try:
                json.loads(json_candidate)
                return json_candidate
            except json.JSONDecodeError:
                pass
    
    return None


def load_results_files(input_dir: str, pattern: str = "prompt_part*_results.jsonl"):
    """
    加载所有匹配的结果文件
    
    Args:
        input_dir: 输入目录路径
        pattern: 文件匹配模式
    
    Returns:
        所有结果记录的列表
    """
    # 构建完整的文件路径模式
    if os.path.isdir(input_dir):
        file_pattern = os.path.join(input_dir, pattern)
    else:
        # 如果 input_dir 本身就是文件路径模式
        file_pattern = input_dir
    
    # 查找所有匹配的文件
    result_files = sorted(glob.glob(file_pattern))
    
    if not result_files:
        raise ValueError(f"未找到匹配的文件: {file_pattern}")
    
    print(f"找到 {len(result_files)} 个结果文件:")
    for f in result_files:
        print(f"  - {f}")
    
    # 读取所有文件
    all_results = []
    for result_file in result_files:
        print(f"正在读取: {result_file}")
        with open(result_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    all_results.append(record)
                except json.JSONDecodeError as e:
                    print(f"警告: {result_file} 第 {line_num} 行解析失败: {e}")
                    continue
    
    print(f"总共加载 {len(all_results)} 条记录")
    return all_results


def parse_output(output: Optional[str], extracted_json: Optional[dict] = None):
    """
    解析输出，提取所需的字段
    
    Args:
        output: 原始输出文本
        extracted_json: 已提取的JSON（如果存在）
    
    Returns:
        包含所需字段的字典
    """
    # 如果已经有 extracted_json，直接使用
    if extracted_json:
        return {
            "general_preference": extracted_json.get("general_preference", DEFAULT_PREFERENCES["general_preference"]),
            "long_term_preference": extracted_json.get("long_term_preference", DEFAULT_PREFERENCES["long_term_preference"]),
            "short_term_preference": extracted_json.get("short_term_preference", DEFAULT_PREFERENCES["short_term_preference"]),
            "user_related_intention": extracted_json.get("user_related_intention", DEFAULT_PREFERENCES["user_related_intention"]),
            "item_related_intention": extracted_json.get("item_related_intention", DEFAULT_PREFERENCES["item_related_intention"])
        }
    
    # 如果没有 extracted_json，尝试从 output 中提取
    if not output:
        return DEFAULT_PREFERENCES.copy()
    
    # 尝试提取 JSON
    json_str = extract_json_from_text(output)
    if json_str:
        try:
            json_data = json.loads(json_str)
            return {
                "general_preference": json_data.get("general_preference", DEFAULT_PREFERENCES["general_preference"]),
                "long_term_preference": json_data.get("long_term_preference", DEFAULT_PREFERENCES["long_term_preference"]),
                "short_term_preference": json_data.get("short_term_preference", DEFAULT_PREFERENCES["short_term_preference"]),
                "user_related_intention": json_data.get("user_related_intention", DEFAULT_PREFERENCES["user_related_intention"]),
                "item_related_intention": json_data.get("item_related_intention", DEFAULT_PREFERENCES["item_related_intention"])
            }
        except json.JSONDecodeError:
            pass
    
    # 如果无法解析，返回默认值
    print(f"警告: 无法从输出中提取JSON，使用默认值")
    return DEFAULT_PREFERENCES.copy()


def merge_results_to_user_json(results: list):
    """
    将结果合并为 user.json 格式
    
    Args:
        results: 结果记录列表
    
    Returns:
        合并后的用户字典
    """
    # 构建最终的用户字典
    user_dict = {
        "user_explicit_preference": {},
        "user_vague_intention": {
            "train": {},
            "test": {}
        }
    }
    
    # 处理每条结果
    for record in results:
        user = record.get("user")
        item = record.get("item")
        history = record.get("history", [])
        output = record.get("output")
        extracted_json = record.get("extracted_json")
        
        if user is None:
            print(f"警告: 记录缺少 user 字段，跳过: {record.get('id', 'unknown')}")
            continue
        
        # 解析输出
        parsed_data = parse_output(output, extracted_json)
        
        # 设置 preference（每个用户只保存一次）
        if user not in user_dict["user_explicit_preference"]:
            user_dict["user_explicit_preference"][user] = [
                parsed_data["general_preference"],
                parsed_data["long_term_preference"],
                parsed_data["short_term_preference"]
            ]
        
        # 设置 train 的 intention
        user_dict["user_vague_intention"]["train"][user] = {
            "item": item,
            "inters": history,
            "querys": [
                parsed_data["user_related_intention"],
                parsed_data["item_related_intention"]
            ]
        }
    
    return user_dict


def parse_args():
    parser = argparse.ArgumentParser(description='合并 prompt_part*_results.jsonl 文件为 user.json')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='输入目录路径（包含 prompt_part*_results.jsonl 文件）或文件路径模式')
    parser.add_argument('--dataset', type=str, required=True,
                       help='数据集名称，例如: Instruments / Arts / Games')
    parser.add_argument('--root', type=str, default='',
                       help='根目录路径（如果不指定，使用当前目录）')
    parser.add_argument('--output', type=str, default='',
                       help='输出文件路径（如果不指定，使用默认路径）')
    parser.add_argument('--pattern', type=str, default='prompt_part*_results.jsonl',
                       help='文件匹配模式（默认: prompt_part*_results.jsonl）')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # 设置路径
    if args.root:
        root_path = os.path.join(args.root, args.dataset)
    else:
        root_path = args.dataset
    
    # 加载所有结果文件
    print(f"\n{'='*60}")
    print("加载结果文件...")
    print(f"{'='*60}")
    results = load_results_files(args.input_dir, args.pattern)
    
    if not results:
        print("错误: 未找到任何结果记录")
        exit(1)
    
    # 合并结果
    print(f"\n{'='*60}")
    print("合并结果...")
    print(f"{'='*60}")
    user_dict = merge_results_to_user_json(results)
    
    # 确定输出文件路径
    if args.output:
        output_file = args.output
    else:
        output_file = os.path.join(root_path, f'{args.dataset}.user.json')
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 保存最终文件
    print(f"\n{'='*60}")
    print("保存结果...")
    print(f"{'='*60}")
    write_json_file(user_dict, output_file)
    
    print(f"\n成功生成 {output_file}")
    print(f"  - 总用户数（preference）: {len(user_dict['user_explicit_preference'])}")
    print(f"  - Train 样本数: {len(user_dict['user_vague_intention']['train'])}")
    print(f"  - Test 样本数: {len(user_dict['user_vague_intention']['test'])}")

