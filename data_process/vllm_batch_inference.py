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
from typing import Optional, Dict, List
from vllm import LLM, SamplingParams
from pydantic import ValidationError

from utils import load_json, write_json_file, unified_user_analysis_prompt, amazon18_dataset2fullname
from get_llm_output import (
    UserAnalysisResponse, DEFAULT_PREFERENCES, extract_json_from_text, 
    parse_json_object_response, format_history_items
)


def generate_prompts_from_data(inters: Dict, item2feature: Dict, reviews: Dict, 
                                dataset: str, mode: str, max_his_len: int = 20) -> tuple:
    """
    直接从原始数据生成prompts（复用get_llm_output.py的逻辑）
    
    Args:
        inters: 交互数据
        item2feature: 商品特征字典
        reviews: 评论数据
        dataset: 数据集名称
        mode: 'train' 或 'test'
        max_his_len: 最大历史长度
    
    Returns:
        (prompt_list, user_data_list) 元组
        - prompt_list: prompt列表
        - user_data_list: (user, item, history) 元组列表
    """
    dataset_full_name = amazon18_dataset2fullname.get(dataset, dataset)
    dataset_full_name = dataset_full_name.replace("_", " ").lower()
    
    prompt_list = []
    user_data_list = []  # 存储(user, item, history)用于后续处理
    
    for user, item_list in inters.items():
        user = int(user)
        
        # 根据模式选择目标item
        if mode == 'train':
            if len(item_list) < 3:
                continue  # 跳过历史不足的用户
            target_item = int(item_list[-3])
            history = item_list[:-3]
        else:  # test
            if len(item_list) < 1:
                continue
            target_item = int(item_list[-1])
            history = item_list[:-1]
        
        # 获取目标商品特征
        item_feat = item2feature.get(str(target_item), {})
        item_title = item_feat.get('title', '')
        item_description = item_feat.get('description') or 'N/A'
        
        # 获取review（如果存在）
        review = reviews.get(str((user, target_item)), {}).get('review', '')
        review_section = f"\nUser Review: {review}" if review else "\nNote: No user review available."
        
        # 对于preference，使用去掉最后3个的历史
        preference_history = item_list[:-3] if len(item_list) >= 3 else []
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
        user_data_list.append((user, target_item, history))
    
    return prompt_list, user_data_list


def format_prompt_for_chat_model(messages: List[Dict], tokenizer) -> str:
    """
    将messages格式化为模型可接受的prompt格式
    
    Args:
        messages: 消息列表，包含system、user等角色
        tokenizer: tokenizer对象（用于获取chat template）
    
    Returns:
        格式化后的prompt字符串
    """
    # 尝试使用tokenizer的apply_chat_template方法
    if hasattr(tokenizer, 'apply_chat_template'):
        try:
            # 直接传递完整的messages列表，让tokenizer处理
            # apply_chat_template会自动处理system message（如果tokenizer支持）
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return prompt
        except Exception as e:
            # 如果apply_chat_template失败，可能是tokenizer不支持system message
            # 尝试只传递user消息
            try:
                chat_messages = [msg for msg in messages if msg.get("role") != "system"]
                if chat_messages:
                    prompt = tokenizer.apply_chat_template(
                        chat_messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    # 如果有system message，添加到开头
                    system_msg = next((msg.get("content", "") for msg in messages if msg.get("role") == "system"), "")
                    if system_msg:
                        prompt = f"{system_msg}\n\n{prompt}"
                    return prompt
            except Exception as e2:
                print(f"Warning: Failed to use apply_chat_template: {e2}")
    
    # 回退到简单拼接
    prompt_parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            prompt_parts.append(f"System: {content}")
        elif role == "user":
            prompt_parts.append(f"User: {content}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}")
    
    return "\n\n".join(prompt_parts)


def run_batch_inference_from_prompts(
    llm: LLM,
    prompts: List[str],
    custom_ids: List[str],
    json_schema: Optional[Dict] = None,
    guided_decoding_backend: str = "outlines",
    response_format: str = 'json_schema',
    batch_size: int = 100,
    system_message: str = "You are a helpful assistant."
) -> List[Dict]:
    """
    使用vLLM进行批量推理（从prompt列表）
    
    Args:
        llm: vLLM LLM实例
        prompts: prompt列表
        custom_ids: 对应的custom_id列表
        json_schema: JSON Schema（用于结构化输出）
        guided_decoding_backend: 引导解码后端
        response_format: 响应格式类型
        batch_size: 批量大小
        system_message: 系统消息
    
    Returns:
        响应列表，每个元素包含custom_id和生成的文本
    """
    responses = []
    total = len(prompts)
    
    # 准备采样参数
    # 注意：vLLM的结构化输出可能需要通过LLM初始化时的参数或SamplingParams来配置
    # 这里先使用标准生成，然后从输出中解析JSON
    sampling_params = SamplingParams(
        temperature=0.4,
        max_tokens=1024,
        stop=None
    )
    
    # 如果使用结构化输出，尝试配置guided decoding
    # vLLM的guided decoding配置方式可能因版本而异
    # 如果LLM初始化时已配置guided decoding，这里不需要额外设置
    # 否则，我们会在生成后从输出中解析JSON
    if response_format == 'json_schema' and json_schema:
        print(f"Using JSON Schema for structured output (backend: {guided_decoding_backend})")
        print("Note: If guided decoding is not configured in LLM initialization, JSON will be parsed from output text")
    
    # 批量处理
    for i in range(0, total, batch_size):
        batch_prompts = prompts[i:i + batch_size]
        batch_custom_ids = custom_ids[i:i + batch_size]
        
        print(f"Processing batch {i // batch_size + 1}/{(total + batch_size - 1) // batch_size} "
              f"({i + 1}-{min(i + batch_size, total)}/{total})")
        
        # 格式化prompts（如果是chat模型）
        formatted_prompts = []
        for prompt in batch_prompts:
            # 尝试使用tokenizer格式化prompt（如果是chat模型）
            if hasattr(llm, 'llm_engine') and hasattr(llm.llm_engine, 'tokenizer'):
                tokenizer = llm.llm_engine.tokenizer
                # 构建messages格式
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ]
                formatted_prompt = format_prompt_for_chat_model(messages, tokenizer)
            else:
                # 简单拼接system message和prompt
                formatted_prompt = f"{system_message}\n\n{prompt}" if system_message else prompt
            
            formatted_prompts.append(formatted_prompt)
        
        # 执行批量推理
        try:
            outputs = llm.generate(formatted_prompts, sampling_params)
            
            # 处理输出
            # outputs是一个RequestOutput对象的列表，每个对象包含一个prompt的生成结果
            for j, output in enumerate(outputs):
                custom_id = batch_custom_ids[j]
                
                # 检查outputs结构
                if hasattr(output, 'outputs') and output.outputs:
                    # output.outputs是一个列表，通常只有一个元素
                    generated_text = output.outputs[0].text.strip()
                elif hasattr(output, 'text'):
                    # 某些版本可能直接有text属性
                    generated_text = output.text.strip()
                else:
                    print(f"Warning: Unexpected output format for {custom_id}")
                    generated_text = ""
                
                responses.append({
                    "custom_id": custom_id,
                    "text": generated_text,
                    "prompt": formatted_prompts[j]
                })
                
        except Exception as e:
            print(f"Batch inference failed: {e}")
            import traceback
            traceback.print_exc()
            # 为失败的请求添加None响应
            for custom_id in batch_custom_ids:
                responses.append({
                    "custom_id": custom_id,
                    "text": None,
                    "error": str(e)
                })
    
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
                                   user_data_list: List[tuple]) -> Dict:
    """
    从解析结果生成用户数据
    
    Args:
        parsed_results: 解析后的结果字典，key为custom_id（格式：mode_index）
        user_data_list: (user, item, history) 元组列表
    
    Returns:
        用户数据字典
    """
    results = {}
    
    # parsed_results的key是custom_id（如"train_0", "train_1"），需要提取索引
    for custom_id, parsed_response in parsed_results.items():
        # 从custom_id中提取索引（格式：mode_index）
        try:
            # custom_id格式：train_0, test_1等
            parts = custom_id.split('_')
            if len(parts) >= 2:
                idx = int(parts[-1])  # 取最后一部分作为索引
            else:
                # 如果格式不对，尝试直接转换
                idx = int(custom_id)
        except (ValueError, IndexError):
            print(f"Warning: Cannot extract index from custom_id {custom_id}, skipping")
            continue
        
        if idx >= len(user_data_list):
            continue
        
        user, target_item, history = user_data_list[idx]
        
        # 使用解析后的响应
        if parsed_response is None:
            print(f"Empty response for user {user} (index {idx}), using defaults")
            parsed_data = DEFAULT_PREFERENCES.copy()
        else:
            # 将Pydantic对象转换为字典
            parsed_data = parsed_response.model_dump()
        
        results[user] = {
            "item": target_item,
            "inters": history,
            **{k: parsed_data.get(k, DEFAULT_PREFERENCES[k]) for k in DEFAULT_PREFERENCES.keys()}
        }
    
    return results


def merge_train_test_results(train_results: Dict, test_results: Dict) -> Dict:
    """
    合并训练集和测试集结果，生成最终的用户数据文件格式
    
    Args:
        train_results: 训练集结果
        test_results: 测试集结果
    
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
    
    # 处理训练集数据（preference从train数据中提取，但每个用户只保存一次）
    for user, data in train_results.items():
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
    for user, data in test_results.items():
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
    parser = argparse.ArgumentParser(description='vLLM离线批量推理脚本')
    parser.add_argument('--dataset', type=str, required=True, help='数据集名称，如Instruments')
    parser.add_argument('--root', type=str, default='', help='数据根目录')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径（HuggingFace模型名称或本地路径）')
    parser.add_argument('--output_file', type=str, help='输出文件路径（.user.json格式）')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'both'], default='both',
                       help='处理模式：train（仅训练集）、test（仅测试集）、both（训练集和测试集）')
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
    parser.add_argument('--skip_inference', action='store_true',
                       help='跳过推理步骤，直接解析已有的结果文件')
    parser.add_argument('--results_file', type=str,
                       help='已有结果文件路径（JSON格式），如果提供则直接使用该文件')
    parser.add_argument('--max_his_len', type=int, default=20,
                       help='最大历史长度')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 设置路径
    if args.root:
        args.root = os.path.join(args.root, args.dataset)
    else:
        args.root = args.dataset
    
    # 处理模式
    modes_to_process = []
    if args.mode == 'both':
        modes_to_process = ['train', 'test']
    else:
        modes_to_process = [args.mode]
    
    train_results = {}
    test_results = {}
    
    # 准备JSON Schema（如果需要）
    json_schema = None
    if args.response_format == 'json_schema':
        json_schema = UserAnalysisResponse.model_json_schema()
    
    # 初始化vLLM（如果不需要跳过推理）
    llm = None
    if not args.skip_inference:
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
        # 注意：vLLM的guided decoding配置方式可能因版本而异
        # 某些版本可能需要通过guided_decoding_backend参数配置
        if args.response_format == 'json_schema':
            # 尝试设置guided decoding后端（如果vLLM支持）
            try:
                # 检查LLM.__init__是否支持guided_decoding_backend参数
                sig = inspect.signature(LLM.__init__)
                if 'guided_decoding_backend' in sig.parameters:
                    llm_kwargs["guided_decoding_backend"] = args.guided_decoding_backend
                    print(f"Configured guided decoding backend: {args.guided_decoding_backend}")
            except Exception:
                # 如果不支持，将在生成后从输出中解析JSON
                pass
        
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
    
    for mode in modes_to_process:
        print(f"\n{'='*60}")
        print(f"Processing {mode} data...")
        print(f"{'='*60}")
        
        # 直接从原始数据生成prompts
        print(f"Generating prompts from raw data for {mode} mode...")
        prompt_list, user_data_list = generate_prompts_from_data(
            inters, item2feature, reviews, args.dataset, mode, args.max_his_len
        )
        print(f"Generated {len(prompt_list)} prompts")
        
        # 生成custom_ids（使用索引）
        custom_ids = [f"{mode}_{i}" for i in range(len(prompt_list))]
        
        # 执行批量推理或加载已有结果
        if args.skip_inference and args.results_file:
            print(f"Loading results from: {args.results_file}")
            with open(args.results_file, 'r', encoding='utf-8') as f:
                responses_data = json.load(f)
            responses = responses_data.get(mode, [])
        else:
            if llm is None:
                print("Error: vLLM not initialized. Cannot perform inference.")
                continue
            
            print(f"Running batch inference for {len(prompt_list)} prompts...")
            responses = run_batch_inference_from_prompts(
                llm, prompt_list, custom_ids, json_schema,
                args.guided_decoding_backend, args.response_format, args.batch_size
            )
            
            # 保存结果（可选）
            if args.results_file:
                results_data = {mode: responses}
                if os.path.exists(args.results_file):
                    existing_data = load_json(args.results_file)
                    results_data.update(existing_data)
                write_json_file(results_data, args.results_file)
                print(f"Saved results to: {args.results_file}")
        
        # 解析结果
        print(f"Parsing {len(responses)} responses...")
        parsed_results = parse_batch_responses(responses, args.response_format)
        print(f"Successfully parsed {sum(1 for v in parsed_results.values() if v is not None)}/{len(parsed_results)} responses")
        
        # 生成用户数据
        print(f"Generating user data for {mode}...")
        mode_results = generate_user_data_from_results(parsed_results, user_data_list)
        
        if mode == 'train':
            train_results = mode_results
        else:
            test_results = mode_results
        
        print(f"Generated data for {len(mode_results)} users in {mode} mode")
    
    # 合并结果并生成最终文件
    if args.mode == 'both' or (args.mode == 'train' and train_results) or (args.mode == 'test' and test_results):
        print(f"\n{'='*60}")
        print("Merging results and generating final output...")
        print(f"{'='*60}")
        
        user_dict = merge_train_test_results(train_results, test_results)
        
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
    else:
        print("\nNo results to merge.")


if __name__ == "__main__":
    main()
