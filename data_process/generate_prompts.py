"""
生成prompt文件，支持将prompt文件切分成n份
只考虑vLLM和prompt_only模式
"""

import argparse
import os
import json
from utils import load_json, unified_user_analysis_prompt, amazon18_dataset2fullname


def format_history_items(item_list, item2feature, max_his_len):
    """格式化历史商品信息"""
    formatted_items = []
    for j, item_id in enumerate(item_list):
        item_feat = item2feature[str(item_id)]
        item_str = f"{j+1}. Title: {item_feat['title']}"
        if item_feat.get('description'):
            item_str += f"\n   Description: {item_feat['description'][:200]}..."  # 截断过长的描述
        if item_feat.get('brand'):
            item_str += f"\n   Brand: {item_feat['brand']}"
        formatted_items.append(item_str)
    
    if max_his_len > 0 and len(formatted_items) > max_his_len:
        formatted_items = formatted_items[-max_his_len:]
    
    return "\n".join(formatted_items)


def generate_prompts(args, inters, item2feature, reviews, mode='train'):
    """
    生成prompt列表
    
    Args:
        args: 命令行参数
        inters: 交互数据
        item2feature: 商品特征字典
        reviews: 评论数据
        mode: 'train' 或 'test'
    
    Returns:
        prompt_list: prompt列表
        metadata_list: 元数据列表，每个元素包含(user, item, history)
    """
    dataset_full_name = amazon18_dataset2fullname[args.dataset]
    dataset_full_name = dataset_full_name.replace("_", " ").lower()
    
    prompt_list = []
    metadata_list = []  # 存储(user, item, history)用于后续处理
    
    for user, item_list in inters.items():
        user = int(user)
        
        # 根据模式选择目标item
        if mode == 'train':
            if len(item_list) < 3:
                continue
            target_item = int(item_list[-3])
            history = item_list[:-3]
        else:  # test
            if len(item_list) < 1:
                continue
            target_item = int(item_list[-1])
            history = item_list[:-1]
        
        # 获取目标商品特征
        item_feat = item2feature[str(target_item)]
        item_title = item_feat.get('title', '')
        item_description = item_feat.get('description') or 'N/A'
        
        # 获取review（如果存在）
        review = reviews.get(str((user, target_item)), {}).get('review', '')
        review_section = f"\nUser Review: {review}" if review else "\nNote: No user review available."
        
        # 对于preference，使用去掉最后3个的历史
        preference_history = item_list[:-3] if len(item_list) >= 3 else []
        preference_history_items = format_history_items(preference_history, item2feature, args.max_his_len) if preference_history else "No purchase history available."
        
        # 构建prompt
        prompt = unified_user_analysis_prompt.format(
            dataset_full_name=dataset_full_name,
            history_items=preference_history_items,
            item_title=item_title,
            item_description=item_description,
            review_section=review_section
        )
        
        prompt_list.append(prompt)
        metadata_list.append({
            "user": user,
            "item": target_item,
            "history": history
        })
    
    return prompt_list, metadata_list


def save_prompts(prompt_list, metadata_list, output_file):
    """
    保存prompt到文件（JSONL格式，每行一个JSON对象）
    
    Args:
        prompt_list: prompt列表
        metadata_list: 元数据列表
        output_file: 输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, (prompt, metadata) in enumerate(zip(prompt_list, metadata_list)):
            record = {
                "id": idx,
                "prompt": prompt,
                **metadata
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(prompt_list)} prompts to {output_file}")


def split_prompts_file(input_file, n_splits):
    """
    将prompt文件切分成n份
    
    Args:
        input_file: 输入的prompt文件路径
        n_splits: 切分的份数
    """
    # 读取所有prompt
    prompts = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
    
    total = len(prompts)
    print(f"Total prompts: {total}")
    
    # 计算每份的大小
    chunk_size = (total + n_splits - 1) // n_splits  # 向上取整
    
    # 切分并保存
    base_name = os.path.splitext(input_file)[0]
    for i in range(n_splits):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total)
        
        if start_idx >= total:
            break
        
        chunk = prompts[start_idx:end_idx]
        output_file = f"{base_name}_part{i+1}.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in chunk:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        print(f"Part {i+1}: {len(chunk)} prompts saved to {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(description='生成prompt文件')
    parser.add_argument('--dataset', type=str, default='Instruments', help='数据集名称')
    parser.add_argument('--root', type=str, default='', help='数据根目录')
    parser.add_argument('--max_his_len', type=int, default=20, help='最大历史长度')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='模式: train或test')
    parser.add_argument('--output', type=str, default='', help='输出文件路径（如果不指定，使用默认路径）')
    parser.add_argument('--split', type=int, default=0, help='将prompt文件切分成n份（0表示不切分）')
    parser.add_argument('--split_file', type=str, default='', help='要切分的prompt文件路径（用于单独切分已有文件）')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # 如果指定了split_file，只进行切分操作
    if args.split_file:
        if args.split <= 0:
            print("Error: --split must be > 0 when using --split_file")
            exit(1)
        split_prompts_file(args.split_file, args.split)
        exit(0)
    
    # 设置数据路径
    args.root = os.path.join(args.root, args.dataset)
    
    # 加载数据
    inter_path = os.path.join(args.root, f'{args.dataset}.inter.json')
    inters = load_json(inter_path)
    
    item2feature_path = os.path.join(args.root, f'{args.dataset}.item.json')
    item2feature = load_json(item2feature_path)
    
    reviews_path = os.path.join(args.root, f'{args.dataset}.review.json')
    if os.path.exists(reviews_path):
        reviews = load_json(reviews_path)
    else:
        print(f"Warning: {reviews_path} not found, using empty reviews")
        reviews = {}
    
    # 生成prompt
    print(f"Generating {args.mode} prompts...")
    prompt_list, metadata_list = generate_prompts(args, inters, item2feature, reviews, mode=args.mode)
    
    # 确定输出文件路径
    if args.output:
        output_file = args.output
    else:
        output_file = os.path.join(args.root, f'{args.dataset}_{args.mode}_prompts.jsonl')
    
    # 保存prompt文件
    save_prompts(prompt_list, metadata_list, output_file)
    
    # 如果需要切分
    if args.split > 0:
        print(f"\nSplitting prompts file into {args.split} parts...")
        split_prompts_file(output_file, args.split)

