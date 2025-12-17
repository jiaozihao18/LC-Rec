"""
vLLM 离线批量推理脚本

使用vLLM的离线API（LLM类）直接进行批量推理，从原始数据生成prompts并推理。

使用方法:
    python vllm_batch_inference.py \
        --dataset Instruments \
        --root ./data \
        --model_path meta-llama/Llama-3.1-8B-Instruct \
        --mode both
"""

import argparse
import os
import json
import inspect
from typing import Optional, Dict, List, Tuple
from vllm import LLM, SamplingParams
from pydantic import ValidationError

from utils import load_json, write_json_file, unified_user_analysis_prompt, amazon18_dataset2fullname
from get_llm_output import (
    UserAnalysisResponse, DEFAULT_PREFERENCES, extract_json_from_text, 
    parse_json_object_response, format_history_items
)


def generate_prompts_from_data(inters: Dict, item2feature: Dict, reviews: Dict, 
                                dataset: str, preference_history_end: int = -3,
                                target_item_pos: int = -3, max_his_len: int = 20) -> Tuple[List[str], List[Dict]]:
    """
    直接从原始数据生成prompts（复用get_llm_output.py的逻辑）
    
    Args:
        inters: 交互数据
        item2feature: 商品特征字典
        reviews: 评论数据
        dataset: 数据集名称
        preference_history_end: 偏好总结取的历史终点位置（如-3表示[:-3]）
        target_item_pos: 意图提取取的商品位置（如-3表示list[-3]）
        max_his_len: 最大历史长度
    
    Returns:
        (prompt_list, user_data_list) 元组
        - prompt_list: prompt列表
        - user_data_list: 包含user, item, history, preference_history_end, target_item_pos的字典列表
    """
    dataset_full_name = amazon18_dataset2fullname.get(dataset, dataset)
    dataset_full_name = dataset_full_name.replace("_", " ").lower()
    
    prompt_list = []
    user_data_list = []  # 存储用户数据信息
    
    for user, item_list in inters.items():
        user = int(user)
        
        # 检查是否有足够的item
        if len(item_list) < abs(target_item_pos):
            continue
        
        # 根据target_item_pos选择目标item
        target_item = int(item_list[target_item_pos])
        # history是去掉target_item之后的所有历史
        if target_item_pos < 0:
            # 负数索引：如-3，需要去掉item_list[-3]
            # item_list[:target_item_pos] 取到target_item之前的所有元素
            # item_list[target_item_pos+1:] 取target_item之后的所有元素
            history = item_list[:target_item_pos] + item_list[target_item_pos+1:]
        else:
            # 正数索引：如3，需要去掉item_list[3]
            history = item_list[:target_item_pos] + item_list[target_item_pos+1:]
        
        # 获取目标商品特征
        item_feat = item2feature.get(str(target_item), {})
        item_title = item_feat.get('title', '')
        item_description = item_feat.get('description') or 'N/A'
        
        # 获取review（如果存在）
        review = reviews.get(str((user, target_item)), {}).get('review', '')
        review_section = f"\nUser Review: {review}" if review else "\nNote: No user review available."
        
        # 对于preference，使用preference_history_end指定的历史终点
        if preference_history_end < 0:
            preference_history = item_list[:preference_history_end] if len(item_list) >= abs(preference_history_end) else []
        else:
            preference_history = item_list[:preference_history_end]
        
        preference_history_items = format_history_items(preference_history, item2feature, max_his_len) if preference_history else "No purchase history available."
        
        # 构建prompt
        prompt = unified_user_analysis_prompt.format(
            dataset_full_name=dataset_full_name,
            history_items=preference_history_items,
            item_title=item_title,
            item_description=item_description,
            review_section=review_section
        )
        
        prompt_list.append(prompt)
        user_data_list.append({
            "user": user,
            "item": target_item,
            "history": history,
            "preference_history_end": preference_history_end,
            "target_item_pos": target_item_pos
        })
    
    return prompt_list, user_data_list


def format_prompt_for_chat_model(messages: List[Dict], tokenizer) -> str:
    """
    将messages格式化为模型可接受的prompt格式（标准HF模型，使用chat template）
    
    Args:
        messages: 消息列表，包含system、user等角色
        tokenizer: tokenizer对象（标准HF模型肯定有apply_chat_template）
    
    Returns:
        格式化后的prompt字符串
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def run_batch_inference_from_prompts(
    llm: LLM,
    prompts: List[str],
    custom_ids: List[str],
    user_data_list: List[Dict],
    json_schema: Optional[Dict] = None,
    guided_decoding_backend: str = "outlines",
    response_format: str = 'json_schema',
    batch_size: int = 100,
    system_message: str = "You are a helpful assistant.",
    output_file: str = None
) -> List[Dict]:
    """
    使用vLLM进行批量推理（从prompt列表）
    
    Args:
        llm: vLLM LLM实例
        prompts: prompt列表
        custom_ids: 对应的custom_id列表
        user_data_list: 用户数据列表，包含user, item, history等信息
        json_schema: JSON Schema（用于结构化输出）
        guided_decoding_backend: 引导解码后端
        response_format: 响应格式类型
        batch_size: 批量大小
        system_message: 系统消息
        output_file: 输出文件路径（JSONL格式，每个batch追加写入）
    
    Returns:
        响应列表，每个元素包含custom_id和生成的文本
    """
    responses = []
    total = len(prompts)
    
    # 准备采样参数
    sampling_params = SamplingParams(
        temperature=0.4,
        max_tokens=1024,
        stop=None
    )
    
    # 如果使用结构化输出，提示信息
    if response_format == 'json_schema' and json_schema:
        print(f"Using JSON Schema for structured output (backend: {guided_decoding_backend})")
    
    # 批量处理
    for i in range(0, total, batch_size):
        batch_prompts = prompts[i:i + batch_size]
        batch_custom_ids = custom_ids[i:i + batch_size]
        batch_user_data = user_data_list[i:i + batch_size]
        
        print(f"Processing batch {i // batch_size + 1}/{(total + batch_size - 1) // batch_size} "
              f"({i + 1}-{min(i + batch_size, total)}/{total})")
        
        # 格式化prompts（使用chat template）
        tokenizer = llm.llm_engine.tokenizer
        formatted_prompts = []
        for prompt in batch_prompts:
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = format_prompt_for_chat_model(messages, tokenizer)
            formatted_prompts.append(formatted_prompt)
        
        # 执行批量推理
        try:
            outputs = llm.generate(formatted_prompts, sampling_params)
            
            # 处理输出并写入文件（JSONL格式，每行一个JSON对象）
            batch_items = []
            for j, output in enumerate(outputs):
                custom_id = batch_custom_ids[j]
                generated_text = output.outputs[0].text.strip()
                
                response_item = {
                    "custom_id": custom_id,
                    "text": generated_text,
                    "prompt": formatted_prompts[j]
                }
                responses.append(response_item)
                
                # 组合推理结果和user data
                batch_item = {
                    "custom_id": custom_id,
                    "vllm_output": generated_text,
                    "user_data": batch_user_data[j]
                }
                batch_items.append(batch_item)
            
            # 每个batch完成后立即追加写入文件（JSONL格式）
            if output_file:
                with open(output_file, 'a', encoding='utf-8') as f:
                    for item in batch_items:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                print(f"Appended batch results to: {output_file} ({len(batch_items)} items)")
                
        except Exception as e:
            print(f"Batch inference failed: {e}")
            # 为失败的请求添加None响应
            batch_items = []
            for j, custom_id in enumerate(batch_custom_ids):
                response_item = {
                    "custom_id": custom_id,
                    "text": None,
                    "error": str(e)
                }
                responses.append(response_item)
                
                # 组合推理结果（失败）和user data
                batch_item = {
                    "custom_id": custom_id,
                    "vllm_output": None,
                    "error": str(e),
                    "user_data": batch_user_data[j]
                }
                batch_items.append(batch_item)
            
            # 即使失败也保存batch结果
            if output_file:
                with open(output_file, 'a', encoding='utf-8') as f:
                    for item in batch_items:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                print(f"Appended batch results (with errors) to: {output_file} ({len(batch_items)} items)")
    
    return responses


def parse_batch_responses(responses: List[Dict], response_format: str = 'json_schema') -> Dict[str, Optional[UserAnalysisResponse]]:
    """
    解析批量推理结果
    
    Args:
        responses: 响应列表
        response_format: 响应格式类型
    
    Returns:
        解析后的结果字典，key为custom_id，value为解析后的Pydantic对象
    """
    parsed_results = {}
    
    for resp in responses:
        custom_id = resp.get("custom_id")
        text = resp.get("text")
        
        if text is None:
            parsed_results[custom_id] = None
            continue
        
        try:
            if response_format == 'prompt_only':
                # prompt_only模式：从文本中提取JSON
                json_str = extract_json_from_text(text)
                if json_str:
                    parsed_results[custom_id] = parse_json_object_response(json_str)
                else:
                    parsed_results[custom_id] = None
            else:
                # json_schema或json_object模式：直接解析JSON
                parsed_results[custom_id] = parse_json_object_response(text)
                
        except Exception as e:
            print(f"Failed to parse response for {custom_id}: {e}")
            parsed_results[custom_id] = None
    
    return parsed_results


def generate_user_data_from_results(parsed_results: Dict[str, Optional[UserAnalysisResponse]], 
                                   user_data_list: List[Dict]) -> List[Dict]:
    """
    从解析结果生成用户数据（包含位置信息）
    
    Args:
        parsed_results: 解析后的结果字典，key为custom_id（格式：index）
        user_data_list: 包含user, item, history, preference_history_end, target_item_pos的字典列表
    
    Returns:
        用户数据列表，每个元素包含完整信息
    """
    results = []
    
    # parsed_results的key是custom_id（如"0", "1"），需要提取索引
    for custom_id, parsed_response in parsed_results.items():
        # 从custom_id中提取索引
        try:
            idx = int(custom_id)
        except (ValueError, IndexError):
            print(f"Warning: Cannot extract index from custom_id {custom_id}, skipping")
            continue
        
        if idx >= len(user_data_list):
            continue
        
        user_info = user_data_list[idx]
        user = user_info["user"]
        target_item = user_info["item"]
        history = user_info["history"]
        preference_history_end = user_info["preference_history_end"]
        target_item_pos = user_info["target_item_pos"]
        
        # 使用解析后的响应
        if parsed_response is None:
            print(f"Empty response for user {user} (index {idx}), using defaults")
            parsed_data = DEFAULT_PREFERENCES.copy()
        else:
            # 将Pydantic对象转换为字典
            parsed_data = parsed_response.model_dump()
        
        results.append({
            "user": user,
            "item": target_item,
            "inters": history,
            "preference_history_end": preference_history_end,
            "target_item_pos": target_item_pos,
            **{k: parsed_data.get(k, DEFAULT_PREFERENCES[k]) for k in DEFAULT_PREFERENCES.keys()}
        })
    
    return results


def parse_args():
    parser = argparse.ArgumentParser(description='vLLM离线批量推理脚本')
    parser.add_argument('--dataset', type=str, required=True, help='数据集名称，如Instruments')
    parser.add_argument('--root', type=str, default='', help='数据根目录')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径（HuggingFace模型名称或本地路径）')
    parser.add_argument('--output_file', type=str, help='输出文件路径（JSONL格式，每个batch追加写入，默认：当前目录下的batch_results.jsonl）')
    parser.add_argument('--response_format', type=str, default='json_schema',
                       choices=['json_schema', 'json_object', 'prompt_only'],
                       help='响应格式类型')
    parser.add_argument('--batch_size', type=int, default=100,
                       help='批量推理时的批次大小')
    parser.add_argument('--guided_decoding_backend', type=str, default='outlines',
                       choices=['outlines', 'lm-format-enforcer', 'xgrammar'],
                       help='引导解码后端（用于结构化输出）')
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                       help='张量并行大小（用于多GPU）')
    parser.add_argument('--max_model_len', type=int, default=None,
                       help='最大模型长度（用于减少内存使用）')
    parser.add_argument('--max_num_seqs', type=int, default=None,
                       help='最大序列数（用于减少内存使用）')
    parser.add_argument('--max_his_len', type=int, default=20,
                       help='最大历史长度')
    parser.add_argument('--preference_history_end', type=int, default=-3,
                       help='偏好总结取的历史终点位置（如-3表示[:-3]）')
    parser.add_argument('--target_item_pos', type=int, default=-3,
                       help='意图提取取的商品位置（如-3表示list[-3]）')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 设置路径
    if args.root:
        args.root = os.path.join(args.root, args.dataset)
    else:
        args.root = args.dataset
    
    # 设置输出文件（当前目录下的JSONL文件）
    if args.output_file is None:
        args.output_file = f'{args.dataset}_batch_results.jsonl'
    # 确保输出文件路径是绝对路径（如果提供的是相对路径，则相对于当前目录）
    if not os.path.isabs(args.output_file):
        args.output_file = os.path.join(os.getcwd(), args.output_file)
    
    # 如果文件已存在，清空它（重新开始）
    if os.path.exists(args.output_file):
        os.remove(args.output_file)
        print(f"Removed existing output file: {args.output_file}")
    
    # 准备JSON Schema（如果需要）
    json_schema = None
    if args.response_format == 'json_schema':
        json_schema = UserAnalysisResponse.model_json_schema()
    
    # 初始化vLLM
    print(f"\n{'='*60}")
    print(f"Initializing vLLM with model: {args.model_path}")
    print(f"{'='*60}")
    
    # 构建LLM初始化参数
    llm_kwargs = {
        "model": args.model_path,
        "tensor_parallel_size": args.tensor_parallel_size,
    }
    
    if args.max_model_len:
        llm_kwargs["max_model_len"] = args.max_model_len
    if args.max_num_seqs:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    
    # 如果使用结构化输出，尝试配置guided decoding
    if args.response_format == 'json_schema':
        sig = inspect.signature(LLM.__init__)
        if 'guided_decoding_backend' in sig.parameters:
            llm_kwargs["guided_decoding_backend"] = args.guided_decoding_backend
            print(f"Configured guided decoding backend: {args.guided_decoding_backend}")
    
    try:
        llm = LLM(**llm_kwargs)
        print("vLLM initialized successfully")
    except Exception as e:
        print(f"Failed to initialize vLLM: {e}")
        return
    
    # 加载原始数据
    print(f"\n{'='*60}")
    print("Loading raw data...")
    print(f"{'='*60}")
    
    inter_path = os.path.join(args.root, f'{args.dataset}.inter.json')
    item2feature_path = os.path.join(args.root, f'{args.dataset}.item.json')
    reviews_path = os.path.join(args.root, f'{args.dataset}.review.json')
    
    if not os.path.exists(inter_path):
        print(f"Error: Interaction file not found: {inter_path}")
        return
    if not os.path.exists(item2feature_path):
        print(f"Error: Item feature file not found: {item2feature_path}")
        return
    
    inters = load_json(inter_path)
    item2feature = load_json(item2feature_path)
    
    if os.path.exists(reviews_path):
        reviews = load_json(reviews_path)
    else:
        print(f"Warning: {reviews_path} not found, using empty reviews")
        reviews = {}
    
    print(f"Loaded {len(inters)} users, {len(item2feature)} items")
    
    # 直接从原始数据生成prompts（使用位置参数，不区分train/test）
    print(f"\n{'='*60}")
    print(f"Processing data with preference_history_end={args.preference_history_end}, target_item_pos={args.target_item_pos}")
    print(f"{'='*60}")
    
    prompt_list, user_data_list = generate_prompts_from_data(
        inters, item2feature, reviews, args.dataset,
        args.preference_history_end, args.target_item_pos, args.max_his_len
    )
    print(f"Generated {len(prompt_list)} prompts")
    
    # 生成custom_ids（使用索引）
    custom_ids = [str(i) for i in range(len(prompt_list))]
    
    # 执行批量推理（每个batch会追加写入到JSONL文件）
    print(f"Running batch inference for {len(prompt_list)} prompts...")
    print(f"Batch results will be appended to: {args.output_file}")
    responses = run_batch_inference_from_prompts(
        llm, prompt_list, custom_ids, user_data_list, json_schema,
        args.guided_decoding_backend, args.response_format, args.batch_size,
        output_file=args.output_file
    )
    
    print(f"\n{'='*60}")
    print(f"Batch inference completed!")
    print(f"  - Output file: {args.output_file}")
    print(f"  - Total samples: {len(responses)}")
    print(f"  - Preference history end: {args.preference_history_end}")
    print(f"  - Target item position: {args.target_item_pos}")
    print(f"\nNote: The output file contains vLLM outputs and user data for each sample.")
    print(f"      Use a merge script to process each batch and generate final .user.json file")


if __name__ == "__main__":
    main()
